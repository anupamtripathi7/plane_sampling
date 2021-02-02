import copy
import os
import cv2
import kornia
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils, projector, transforms
from .camera import PerspectiveCameras


EPS = 1e-8


class Config(object):
    """object for passing parameters to models
    """
    def __init__(self):
        pass


class View(object):
    """ Class to hold an image and the camera
    """

    @classmethod
    def from_params(cls, params, base_dir, shape=None, load_depth=False):
        camera = PerspectiveCameras.from_params(params)
        image_path = os.path.join(base_dir, params["relative_path"])

        image = cv2.imread(image_path, -1)
        image = kornia.image_to_tensor(image, False)
        image = image / 255.0
        image = kornia.bgr_to_rgb(image)

        if shape is not None:
            image = utils.pad_or_crop(image, shape)

        if load_depth:
            depth_path = os.path.join(base_dir, params["relative_depth_path"])
            depth = torch.load(depth_path)
        else:
            depth = None

        return cls(camera, image, depth)

    @classmethod
    def collate(cls, views):
        camera = PerspectiveCameras.collate([view.camera for view in views])
        image = torch.cat([view.image for view in views], dim=0)

        if views[0].depth is not None:
            depth = torch.cat([view.depth for view in views], dim=0)
        else:
            depth = None

        return cls(camera, image, depth)

    @classmethod
    def from_tensors(cls, tensor, camera):
        camera = PerspectiveCameras.from_tensors(camera)

        if tensor.shape[1] == 3:
            image = tensor
            depth = None
        elif tensor.shape[1] == 4:
            image = tensor[:, :3]
            depth = tensor[:, 3:]

        return cls(camera, image, depth)

    def __init__(self, camera, image, depth=None):
        """initialize class

        Args:
            params (dict): dictionary with camera params and info for image
            base_dir (string): base path used for loading image
        """
        self.camera = camera
        self.image = image
        self.depth = depth

    def cv2image(self):
        return (self.image.permute(0, 2, 3, 1) * 255.0).detach().cpu().long().numpy()

    def to(self, device):
        self.image = self.image.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        self.camera.to(device)

        return self

    def tensors(self):
        if self.depth is None:
            image_tensor = self.image
        else:
            image_tensor = torch.cat([self.image, self.depth], dim=1)

        return image_tensor, self.camera.tensors()


class MultiSurfaceImage(object):
    """
    A class which stores the Multi Surface Image and its operations
    """
    def __init__(self, features, disparity, disparity_bins, camera):
        """initalize MultiSurfaceImage class

        Args:
            features (torch.tensor): features (RGBA) values for each surface. shape (B, M, C, H, W)
            disparity (torch.tensor): normalized disparity values for each surface. shape (B, M, H, W)
            disparity_bins (torch.tensor): the disparity range for each layer. shape (B, M, 2)
            camera (PerspectiveCameras): reference camera for 3D projection
        """
        super().__init__()

        # disparity = (torch.tanh(disparity) + 1.0) / 2.0  # (B, M, H, W)
        # self.features = (torch.tanh(features) + 1.0) / 2.0  # (B, M, C, H, W)
        self.features = features  # (B, M, C, H, W)
        self.surfaces = projector.disparity_to_surface(disparity, disparity_bins)
        self.camera = camera

    def view(self, camera, size=2, resolution=None):
        if resolution is None:
            resolution = self.features.shape[-2:]
        B = camera.intrinsics.shape[0]
        features = self.features.expand(B, -1, -1, -1, -1)
        surfaces = self.surfaces.expand(B, -1, -1, -1, -1)
        projection = projector.perspective_view(surfaces, features, self.camera, camera, size, resolution)  # (B, C-1, H, W)

        return projection

    def translated_view(self, t, size=2):
        dest_camera = self.camera.translate(t)
        return self.view(dest_camera, size)

    def composite(self, mode="alpha"):
        if mode == "alpha":
            output = projector.alpha_composite(self.features)
        elif mode == "max":
            output = projector.max_composite(self.features)
        elif mode == "softmax":
            output = projector.softmax_composite(self.features)

        return output


class MultiPlaneImage(object):
    """Multi Plane Image class
    """
    @classmethod
    def from_weights(cls, weights, features, depths, camera, n=None):
        B, M, V, C, H, W = features.shape
        alpha = torch.sigmoid(weights[:, :, :1])  # (B, M, 1, H, W)
        weights = torch.cat([torch.zeros((B, M, 1, H, W), device=weights.device), weights[:, :, 1:]], dim=2)  # (B, M, V, H, W)
        weights = torch.softmax(weights, dim=2)
        features = torch.sum(weights[:, :, :, None] * features, dim=2)  # (B, M, C, H, W)
        features = torch.cat([features, alpha], dim=2)  # (B, M, C+1, H, W)
        return cls(features, depths, camera, n)

    def __init__(self, features, depths, camera, n=None):
        """Initialization

        Args:
            features (torch.tensor): features (RGBA) values for each surface. shape (B, M, C, H, W)
            disparity_planes (torch.tensor): disparity values for each layer. shape (M)
            camera (PerspectiveCameras): reference camera for 3D projection
        """

        B, M, C, H, W = features.shape
        self.features = features  # (B, M, C, H, W)
        self.depths = depths  # (M)
        self.camera = camera  # (B, V)
        # self.n = n if n is not None else torch.tensor([0.0, 0.0, 1.0], device=features.device).expand(B, 1, -1)  # (B, 1, 3)
        self.n = torch.tensor([0.0, 0.0, 1.0], device=features.device).expand(B, 1, -1)  # (B, 1, 3)
        self.device = features.device

    def view(self, camera, compositor):
        B, V, _, _, _ = camera.tensors().shape
        # B, M, C, H, W = self.features.shape
        features = self.features[:, :, None].expand(-1, -1, V, -1, -1, -1)
        warped_features = self.camera.homographic_warp(camera, features, self.depths, self.n, s2t=False)  # (B, M, V, C, H, W)
        out = compositor(warped_features)  # (B, V, C-1, H, W)

        return out
