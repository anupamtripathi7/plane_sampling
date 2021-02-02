import logging
import torch
import torch.nn.functional as F
from . import projector
from . import transforms
from . import utils
from . import warping
from .camera import PerspectiveCameras


EPS = 1e-8


class PSV(object):
    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.depths, cfg.patch_size, cfg.patch_range, cfg.save_grid, cfg.psv_per_view)

    def __init__(
        self,
        depths,
        patch_size,
        patch_range,
        save_grid=True,
        psv_per_view=True,
    ):

        self.depths = depths
        self.patch_size = patch_size
        self.patch_range = patch_range
        self.save_grid = save_grid
        self.psv_per_view = psv_per_view
        self.device = depths.device
        self.n = torch.tensor([0.0, 0.0, 1.0], device=self.device)[None, None]  # (1, 1, 3)

        self.regular_grid = utils.regular_meshgrid(self.patch_size, device=self.device)  # (H, W, 2)
        self.start = None

    def from_data(self, data):
        pass

    def __call__(self, x, x_cams, y_cam, y=None):
        B, V, C, _, _ = x.shape
        H, W = self.patch_size
        M = self.depths.shape[0]

        sH = torch.randint(self.patch_range[0][0], self.patch_range[0][1] - H + 1, size=(B,), device=self.device)
        sW = torch.randint(self.patch_range[1][0], self.patch_range[1][1] - W + 1, size=(B,), device=self.device)
        self.start = torch.stack([sH, sW], dim=1)  # (B, 2)
        src_grid = self.regular_grid + self.start[:, None, None]  # (B, H, W, 2)

        # if y is not None:
        #     y_patch = torch.zeros((B, 1, 1, C, H, W), device=self.device)
        #     for i in range(B):
        #         y_patch[i, 0, 0] = y[i, 0, :, sH[i]: sH[i] + H, sW[i]: sW[i] + W]

        #     y_patch = y_patch.expand(-1, M, -1, -1, -1, -1)
        # else:
        #     y_patch = None

        if y is not None:
            y_patch = torch.zeros((B, C, H, W), device=self.device)
            for i in range(B):
                y_patch[i] = y[i, 0, :, sH[i]: sH[i] + H, sW[i]: sW[i] + W]
        else:
            y_patch = None

        x = x[:, None].expand(-1, M, -1, -1, -1, -1)  # (B, M, V, C, H, W)

        if self.psv_per_view:
            x_patch = torch.zeros((B, M, V, C, H, W), device=self.device)

            for v in range(V):
                x_cams_v = PerspectiveCameras.from_tensors(x_cams[:, v:v + 1])  # (B, 1, 4, 4)
                y_cam_v = PerspectiveCameras.from_tensors(y_cam)  # (B, 1, 4, 4)
                x_patch[:, :, v:v + 1] = y_cam_v.homographic_warp(x_cams_v, x[:, :, v:v + 1], self.depths, self.n, uv=src_grid, s2t=True)

        else:
            x_cams = PerspectiveCameras.from_tensors(x_cams)  # (B, V, 4, 4)
            y_cam = PerspectiveCameras.from_tensors(y_cam.expand(-1, V, -1, -1, -1))  # (B, V, 4, 4)
            x_patch = y_cam.homographic_warp(x_cams, x, self.depths, self.n, uv=src_grid, s2t=True)

        return x_patch, y_patch


class AlphaCompositing(object):
    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.normalize_weights)

    def __init__(self, normalize_weights=True):
        self.normalize_weights = normalize_weights

    def __call__(self, weights, features=None, grid=None):
        """Combines a list of RGBA images using the over operation.
        Combines RGBA images from back to front with the over operation.
        The alpha image of the first image is ignored and assumed to be 1.0.

        Args:
            rgba (torch.tensor): RGBA images. shape (B, M, C, H, W)

        Returns:
            torch.tensor: combined image. shape (B, C-1, H, W)
        """
        rgba = weights
        B, M, V, C, H, W = rgba.shape

        if self.normalize_weights:
            rgba = torch.sigmoid(rgba)

        for i in range(M - 1, -1, -1):
            rgb = rgba[:, i, :, :-1]  # (B, V, C-1, H, W)
            alpha = rgba[:, i, :, -1:]  # (B, V, 1, H, W)

            if i == M - 1:
                out = rgb  # (B, V, C-1, H, W)
            else:
                out = rgb * alpha + out * (1.0 - alpha)  # (B, V, C-1, H, W)

        return out


class SoftmaxCompositing(object):
    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.normalize_weights)

    def __init__(self, normalize_weights=True):
        self.normalize_weights = normalize_weights

    def __call__(self, weights, features=None, grid=None):
        """Combines a tensor of RGBA images using the softmax operation.
        The RGB values are weighted with softmax(alpha) for each pixel

        Args:
            rgba (torch.tensor): RGBA images. shape (B, M, C, H, W)

        Returns:
            torch.tensor: combined image. shape (B, C-1, H, W)
        """
        rgba = weights
        # B, M, C, H, W = rgba.shape

        if self.normalize_weights:
            rgb = torch.sigmoid(rgba[:, :, :-1])
            weights = torch.softmax(rgba[:, :, -1:], dim=1)  # (B, M, 1, H, W)
        else:
            rgb = rgba[:, :, :-1]
            weights = rgba[:, :, -1:]  # (B, M, 1, H, W)

        out = torch.sum(rgb * weights, dim=1)  # (B, C-1, H, W)

        return out


class WeightedAlphaCompositing(object):
    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.normalize_features, cfg.normalize_weights)

    def __init__(self, normalize_features=False, normalize_weights=True):
        self.normalize_features = normalize_features
        self.normalize_weights = normalize_weights
        self.alpha_compisitor = AlphaCompositing(normalize_weights=False)

    def __call__(self, weights, features, grid=None):
        B, M, V, C, H, W = features.shape

        if self.normalize_features:
            features = torch.sigmoid(features)

        if self.normalize_weights:
            alpha = torch.sigmoid(weights[:, :, -1:])  # (B, M, 1, H, W)
            rgb_weights = torch.cat([torch.zeros((B, M, 1, H, W), device=weights.device), weights[:, :, :-1]], dim=2)  # (B, M, V, H, W)
            rgb_weights = torch.softmax(rgb_weights, dim=2)  # (B, M, V, H, W)
        else:
            alpha = weights[:, :, -1:]
            rgb_weights = weights[:, :, :-1]

        rgb = torch.sum(rgb_weights[:, :, :, None] * features, dim=2)  # (B, M, C, H, W)
        rgba = torch.cat([rgb, alpha], dim=2)
        out = self.alpha_compisitor(rgba)

        return out


class AccountableAlphaCompositing(object):
    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.acc_alpha_mode, cfg.normalize_features)

    def __init__(self, mode="mean", normalize_features=False):
        self.mode = mode
        self.normalize_features = normalize_features
        self.alpha_compisitor = AlphaCompositing(normalize_weights=False)

    def __call__(self, weights, features, grid):
        B, M, V, C, H, W = features.shape

        if self.normalize_features:
            features = torch.sigmoid(features)

        if self.mode is not None:
            weights = torch.exp(weights).permute(0, 2, 1, 3, 4).reshape(B * V, M * H * W)  # (B*V, M*H*W)
            grid = grid.permute(0, 2, 1, 3, 4, 5).reshape(B * V, M * H * W, 2)  # (B*V, M*H*W, 2)
            unwarped_normalization = warping.forward_warp_nn(grid, weights)  # (B*V, H, W)
            warped_normalization = warping.backward_warp(
                grid.reshape(B * V, M, H, W, 2),
                unwarped_normalization[:, None]
            )  # (B*V*M, 1, H, W)
            weights = weights.reshape(-1) / (warped_normalization.reshape(-1) + EPS)
            weights = weights.reshape(B, M, V, H, W)

        alpha = torch.sum(weights, dim=2, keepdim=True)  # (B, M, 1, H, W)
        weights = weights / (alpha + EPS)  # (B, M, V, H, W)

        if self.mode is None:
            alpha = alpha / (V * M)
        elif self.mode == "mean":
            alpha = alpha / V  # (B, M, 1, H, W)
        elif self.mode == "softmax":
            alpha = torch.softmax(alpha, dim=1)  # (B, M, 1, H, W)
        elif self.mode == "sigmoid":
            alpha = torch.sigmoid(alpha - V / 2)  # (B, M, 1, H, W)

        rgb = torch.sum(weights[:, :, :, None] * features, dim=2)  # (B, M, C, H, W)
        rgba = torch.cat([rgb, alpha], dim=2)
        out = self.alpha_compisitor(rgba)

        return out
