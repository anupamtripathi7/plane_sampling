import logging
import torch
from random import randint
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from . import transforms
from .camera import PerspectiveCameras


EPS = 1e-8


def alpha_composite(rgba, normalize=True):
    """Combines a list of RGBA images using the over operation.
    Combines RGBA images from back to front with the over operation.
    The alpha image of the first image is ignored and assumed to be 1.0.

    Args:
        rgba (torch.tensor): RGBA images. shape (B, M, C, H, W)

    Returns:
        torch.tensor: combined image. shape (B, C-1, H, W)
    """
    B, M, C, H, W = rgba.shape

    if normalize:
        rgba = torch.sigmoid(rgba)

    for i in range(M - 1, -1, -1):
        rgb = rgba[:, i, :-1]  # (B, C-1, H, W)
        alpha = rgba[:, i, -1:]  # (B, 1, H, W)

        if i == M - 1:
            output = rgb  # (B, C-1, H, W)
        else:
            output = rgb * alpha + output * (1.0 - alpha)  # (B, C-1, H, W)

    return output


def alpha_correction(alpha):
    B, M, H, W = alpha.shape

    corrected_alpha = torch.zeros(alpha.shape, device=alpha.device)
    accumulated_alpha = torch.zeros(alpha.shape, device=alpha.device)

    for i in range(M):
        if i == 0:
            corrected_alpha[:, i] = alpha[:, i]
            accumulated_alpha[:, i] = alpha[:, i]
        else:
            corrected_alpha[:, i] = (1 - accumulated_alpha[:, i - 1]) * alpha[:, i]
            accumulated_alpha[:, i] = accumulated_alpha[:, i - 1] + corrected_alpha[:, i]

    return corrected_alpha, accumulated_alpha


def max_composite(rgba, normalize=True):
    """Combines a tensor of RGBA images using the max operation.
    The RGB values with max alpha is selected for each pixel

    Args:
        rgba (torch.tensor): RGBA images. shape (B, M, C, H, W)

    Returns:
        torch.tensor: combined image. shape (B, C-1, H, W)
    """
    B, M, C, H, W = rgba.shape

    if normalize:
        rgb = (torch.tanh(rgba[:, :, :-1]) + 1.0) / 2.0
    else:
        rgb = rgba[:, :, :-1]

    _, idxs = torch.max(rgba[:, :, -1:], dim=1, keepdim=True)  # (B, 1, 1, H, W)
    idxs = idxs.expand(-1, -1, C - 1, -1, -1)  # (B, 1, C-1, H, W)
    output = torch.gather(rgb, 1, idxs)[:, 0]  # (B, C-1, H, W)

    return output


def softmax_composite(rgba, normalize=True):
    """Combines a tensor of RGBA images using the softmax operation.
    The RGB values are weighted with softmax(alpha) for each pixel

    Args:
        rgba (torch.tensor): RGBA images. shape (B, M, C, H, W)

    Returns:
        torch.tensor: combined image. shape (B, C-1, H, W)
    """
    B, M, C, H, W = rgba.shape

    if normalize:
        rgb = torch.sigmoid(rgba[:, :, :-1])
        weights = F.softmax(rgba[:, :, -1:], dim=1)  # (B, M, 1, H, W)
    else:
        rgb = rgba[:, :, :-1]
        weights = rgba[:, :, -1:]  # (B, M, 1, H, W)

    output = torch.sum(rgb * weights, dim=1)  # (B, C-1, H, W)

    return output


def weighted_alpha_composting(features, weights):
    B, M, V, C, H, W = features.shape

    alpha = torch.sigmoid(weights[:, :, -1:])  # (B, M, 1, H, W)
    rgb_weights = torch.cat([torch.zeros((B, M, 1, H, W), device=weights.device), weights[:, :, :-1]], dim=2)  # (B, M, V, H, W)

    rgb_weights = torch.softmax(rgb_weights, dim=2)  # (B, M, V, H, W)
    rgb = torch.sum(rgb_weights[:, :, :, None] * features, dim=2)  # (B, M, C, H, W)
    rgba = torch.cat([rgb, alpha], dim=2)
    out = alpha_composite(rgba, False)

    return out


def accountable_normalization(features, weights):
    pass


def composite(features, weights=None, mode="alpha"):
    if mode == "alpha":
        output = alpha_composite(features)
    elif mode == "max":
        output = max_composite(features)
    elif mode == "softmax":
        output = softmax_composite(features)
    elif mode == "weighted_alpha":
        assert weights is not None, "weights cannot be None for this argument"
        output = weighted_alpha_composting(features, weights)

    return output


def disparity_to_surface(disparity, disparity_bins):
    """compose xyz coordinates from disparity

    Args:
        disparity (torch.tensor): normalized disparity for each layer of shape (B, M, H, W)
        disparity_bins (torch.tensor): range of disparity for each layer. shape (B, M, 2)

    Returns:
        torch.tensor: xyz coordinate values of shape (B, M, H, W, 3)
    """
    B, M, H, W = disparity.shape
    device = disparity.device

    xy = utils.regular_meshgrid([H, W], device).expand(B, M, -1, -1, -1).float()  # (B, M, H, W, 2)
    disparity_bins = disparity_bins.clone()[..., None, None]  # (B, M, 2, 1, 1)

    # normalized_disparity = (torch.tanh(disparity) + 1.0) / 2.0
    scaled_disparity = disparity_bins[:, :, 0] + (disparity_bins[:, :, 1] - disparity_bins[:, :, 0]) * disparity  # (B, M, H, W)
    depth = 1.0 / (scaled_disparity + EPS)  # (B, M, H, W)
    depth = depth[..., None]  # (B, M, H, W, 1)
    surfaces = torch.cat([xy * depth, depth], dim=-1)  # (B, M, H, W, 3)

    return surfaces


def generate_psv(images, src_camera, dest_camera, disparity_planes, batch=True):
    B, C, H, W = images.shape
    M = disparity_planes.shape[0]

    if torch.is_tensor(src_camera):
        src_camera = PerspectiveCameras.from_tensors(src_camera).to(src_camera.device)

    if torch.is_tensor(dest_camera):
        dest_camera = PerspectiveCameras.from_tensors(dest_camera).to(dest_camera.device)

    depth_planes = 1.0 / disparity_planes
    depths = depth_planes[None, :, None, None, None].expand(B, -1, H, W, 1)  # (B, M, H, W, 1)
    src_xy = utils.regular_meshgrid([H, W], images.device)  # (H, W, 2)
    src_xyz = torch.cat([src_xy, torch.ones((H, W, 1), device=images.device)], dim=-1)  # (H, W, 3)
    src_xyz = (src_xyz * depths).reshape(B, M * H * W, 3)  # (B, M*H*W, 3)
    del src_xy, depths

    hw = torch.tensor([H, W], device=images.device)
    dest_xy = dest_camera.view(src_xyz, src_camera, False, batch).reshape(B * M, H, W, 2)  # (B*M, H, W, 2)
    # dest_xy = torch.ones((B * M, H, W, 2), device=images.device)  # (B*M, H, W, 2)
    del src_xyz

    dest_xy = (dest_xy / hw - 0.5) * 2.0  # (B*M, H, W, 2)
    images = images[:, None].expand(-1, M, -1, -1, -1).reshape(B * M, C, H, W)  # (B*M, C, H, W)

    psv = F.grid_sample(images, dest_xy[..., [1, 0]]).reshape(B, M, C, H, W)  # (B, M, C, H, W)
    # psv = torch.zeros((B, M, C, H, W), device=images.device)  # (B, M, C, H, W)

    return psv

def fast_psv(images, src_camera, dest_camera, disparity_planes, batch=True):
    B, C, H, W = images.shape
    M = disparity_planes.shape[0]

    if torch.is_tensor(src_camera):
        src_camera = PerspectiveCameras.from_tensors(src_camera).to(src_camera.device)

    if torch.is_tensor(dest_camera):
        dest_camera = PerspectiveCameras.from_tensors(dest_camera).to(dest_camera.device)

    R, T = dest_camera.relateive_transform(src_camera)  # (B, 1, 3, 3), (B, 1, 3, 1)

    depth_planes = 1.0 / disparity_planes  # (M)
    src_xy = utils.regular_meshgrid([H, W], images.device)  # (H, W, 2)
    src_xyz = torch.cat([src_xy, torch.ones((H, W, 1), device=images.device)], dim=-1)
    del src_xy
    src_xyz = src_xyz[None].expand(B, -1, -1, -1).reshape(B, H * W, 3, 1)  # (B, H*W, 3, 1)
    # src_xyz = (src_xyz * depths).reshape(B, M * H * W, 3, 1)  # (B, M*H*W, 3, 1)

    dest_R = torch.matmul(R, src_xyz)  # (B, H*W, 3, 1)
    dest_xyz = (dest_R[:, None] * depth_planes[..., None, None, None] + T[:, None]).reshape(B * M, H, W, 3, 1)  # (B*M, H, W, 3, 1)
    dest_xy = transforms.from_mm(dest_xyz, 2)  # (B*M, H, W, 2)
    del dest_R, dest_xyz

    hw = torch.tensor([H, W], device=images.device)
    dest_xy = (dest_xy / hw - 0.5) * 2.0  # (B*M, H, W, 2)
    images = images[:, None].expand(-1, M, -1, -1, -1).reshape(B * M, C, H, W)  # (B*M, C, H, W)

    psv = F.grid_sample(images, dest_xy[..., [1, 0]]).reshape(B, M, C, H, W)  # (B, M, C, H, W)
    # psv = torch.zeros((B, M, C, H, W), device=images.device)  # (B, M, C, H, W)

    return psv


def patch_psv(
    images,
    src_camera,
    dest_camera,
    disparity_planes,
    patch_size,
    random_patch=True,
    patch_range=None,
    patch_start=None,
):
    B, C, h, w = images.shape
    H, W = patch_size
    M = disparity_planes.shape[0]

    if torch.is_tensor(src_camera):
        src_camera = PerspectiveCameras.from_tensors(src_camera).to(src_camera.device)

    if torch.is_tensor(dest_camera):
        dest_camera = PerspectiveCameras.from_tensors(dest_camera).to(dest_camera.device)

    if random_patch:
        dH = torch.randint(patch_range[0][0], patch_range[0][1] - H, size=(B, 1), device=images.device)
        dW = torch.randint(patch_range[1][0], patch_range[1][1] - W, size=(B, 1), device=images.device)
    else:
        dH = torch.ones((B,), device=images.device) * patch_start[0]
        dW = torch.ones((B,), device=images.device) * patch_start[1]

    dHW = torch.cat([dH, dW], dim=1)[:, None, None]  # (B, 1, 1, 2)
    R, T = dest_camera.relateive_transform(src_camera)  # (B, 1, 3, 3), (B, 1, 3, 1)

    depth_planes = 1.0 / disparity_planes  # (M)
    src_xy = utils.regular_meshgrid([H, W], images.device)
    src_xy = src_xy + dHW  # (B, H, W, 2)
    src_xyz = torch.cat([src_xy, torch.ones((B, H, W, 1), device=images.device)], dim=-1)  # (B, H, W, 3)
    src_xyz = src_xyz.reshape(B, H * W, 3, 1)  # (B, H*W, 3, 1)
    # src_xyz = (src_xyz * depths).reshape(B, M * H * W, 3, 1)  # (B, M*H*W, 3, 1)

    dest_R = torch.matmul(R, src_xyz)  # (B, H*W, 3, 1)
    dest_xyz = (dest_R[:, None] * depth_planes[..., None, None, None] + T[:, None]).reshape(B * M, H, W, 3, 1)  # (B*M, H, W, 3, 1)
    dest_xy = transforms.from_mm(dest_xyz, 2)  # (B*M, H, W, 2)
    del dest_R, dest_xyz

    hw = torch.tensor([H, W], device=images.device)
    dest_xy = (dest_xy[..., [1, 0]] / hw - 0.5) * 2.0  # (B*M, H, W, 2)
    images = images[:, None].expand(-1, M, -1, -1, -1).reshape(B * M, C, H, W)  # (B*M, C, H, W)

    psv = F.grid_sample(images, dest_xy).reshape(B, M, C, H, W)  # (B, M, C, H, W)
    # psv = torch.zeros((B, M, C, H, W), device=images.device)  # (B, M, C, H, W)

    return psv


def fgbg_blending(fg, bg, weights):
    pass
