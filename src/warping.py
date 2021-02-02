import logging
import torch
import torch.nn.functional as F
from . import utils
from . import transforms
from .camera import PerspectiveCameras


EPS = 1e-8


def generate_splatter_idxs(coordinates, size):
    """generate indices for each splatter coordinate based on splatter size

    Args:
        coordinates (torch.tensor): coordinates to splatter. shape (B, N, 2)
        size (int): size of splatter

    Returns:
        torch.tensor: splattered indices for each coordinate. shape (B, N, size * size, 2)
    """
    # coordinates: (B, N, 1, 2)
    splatter_window = utils.regular_meshgrid([size, size]).reshape(size**2, 2).to(coordinates.device)  # (S*S, 2)
    splatter_idxs = coordinates + splatter_window   # (B, N, S*S, 2)

    return splatter_idxs


def filter_valid_points(coordinates, resolution):
    """filter coordinates that lie outsize the rendering window.
    Clips value to min and max resolution


    Args:
        coordinates (torch.tensor): coordinates to splatter. shape (B, N, 2)
        resolution (torch.tensor): size of the rendering window

    Returns:
        torch.tensor: clipped coordinates. shape (B, N, 2)
    """
    H, W = resolution
    coordinates[coordinates <= 0.0] = 0 + EPS
    coordinates[..., 0][coordinates[..., 0] >= H - 1] = H - 1 - EPS
    coordinates[..., 1][coordinates[..., 1] >= W - 1] = W - 1 - EPS

    return coordinates


def splat(features, coordinates, resolution, size=2):
    """splatters features to grid values based on coordinates

    Args:
        features (torch.tensor): features (RGBA) values for each surface. shape (B, N, C)
        coordinates (torch.tensor): coordinates to splatter. shape (B, N, 2)
        resolution (torch.tensor): size of the rendering window/grid
        size (int, optional): size of splatter. Defaults to 2.

    Returns:
        torch.tensor: splattered tensor of shape (B, H, W, C)
    """
    # features: (B, N, C)
    # coordinates: (B, N, 2)
    H, W = resolution
    B, N, C = features.shape
    device = coordinates.device

    coordinates = filter_valid_points(coordinates, resolution)
    nearest_grid = coordinates.long()[:, :, None, :]  # (B, N, 1, 2)
    splatter_idxs = generate_splatter_idxs(nearest_grid, size)  # (B, N, S*S, 2)
    coordinate_distance = torch.abs(coordinates[:, :, None, :] - (splatter_idxs - size // 2 + 1))  # (B, N, S*S, 2)
    del coordinates
    # distance = torch.sqrt(torch.sum(coordinate_distance**2, dim=-1) + EPS).reshape(B, -1, size**2)  # (B, N, S*S)
    distance = torch.prod(coordinate_distance, dim=-1).reshape(B, -1, size**2)  # (B, N, S*S)
    del coordinate_distance
    splatter_weights = 1 / (distance + EPS)  # (B, N, S*S)
    del distance

    raveled_spalatter_idxs = utils.ravel_multi_index(splatter_idxs, [H + size - 1, W + size - 1]).reshape(B, -1)  # (B, N*S*S)

    weighted_attributes = features[:, :, None, :] * splatter_weights[..., None]  # (B, N, S*S, C)
    weighted_attributes = weighted_attributes.reshape(B, -1, C)  # (B, N*S*S, C)

    splattered_attributes = torch.zeros((B, (H + size - 1) * (W + size - 1), C), device=device)  # (B, (H+S-1)*(W+S-1), C)
    splattered_attributes.scatter_add_(1, raveled_spalatter_idxs[..., None].expand(-1, -1, C), weighted_attributes)  # (B, (H+S-1)*(W+S-1), C)

    splattered_weights = torch.zeros((B, (H + size - 1) * (W + size - 1)), device=device)  # (B, (H+S-1)*(W+S-1))
    splattered_weights.scatter_add_(1, raveled_spalatter_idxs, splatter_weights.reshape(B, -1))  # (B, (H+S-1)*(W+S-1))

    final = splattered_attributes / (splattered_weights[..., None] + EPS)  # (B, (H+S-1)*(W+S-1), C)
    final = final.reshape(B, H + size - 1, W + size - 1, C)  # (B, (H+S-1), (W+S-1), C)
    final = final[:, size // 2 - 1:- size // 2, size // 2 - 1: - size // 2]  # (B, H, W, C)

    return final


def patch_psv(
    images,
    src_grid,
    src_camera,
    dest_camera,
    depths,
    save_grid=True
):

    B, C, h, w = images.shape
    B, H, W, _ = src_grid.shape
    M = depths.shape[0]
    device = images.device

    if torch.is_tensor(src_camera):
        src_camera = PerspectiveCameras.from_tensors(src_camera).to(device)

    if torch.is_tensor(dest_camera):
        dest_camera = PerspectiveCameras.from_tensors(dest_camera).to(device)

    dest_grid = grid_transform(src_grid, src_camera, dest_camera, depths)  # (B, M, H, W, 2)
    psv = backward_warp(dest_grid, images)  # (B, M, C, H, W)

    if save_grid:
        return psv, dest_grid
    else:
        return psv, None


def grid_transform(grid, src_camera, dest_camera, depth_planes):
    B, H, W, _ = grid.shape
    M = depth_planes.shape[0]
    device = grid.device

    src_xyz = torch.cat([grid, torch.ones((B, H, W, 1), device=device)], dim=-1)  # (B, H, W, 3)
    src_xyz = src_xyz.reshape(B, H * W, 3, 1)  # (B, H*W, 3, 1)
    R, T = dest_camera.relative_transform(src_camera)  # (B, 1, 3, 3), (B, 1, 3, 1)
    # R, T = src_camera.relative_transform(dest_camera)  # (B, 1, 3, 3), (B, 1, 3, 1)
    dest_xyz = torch.matmul(R, src_xyz)  # (B, H*W, 3, 1)
    dest_xyz = (dest_xyz[:, None] * depth_planes[..., None, None, None] + T[:, None])   # (B, M, H*W, 3, 1)
    dest_xyz = dest_xyz.reshape(B, M, H, W, 3, 1)  # (B, M, H, W, 3, 1)
    dest_xy = transforms.from_mm(dest_xyz, 2)  # (B, M, H, W, 2)

    return dest_xy


def backward_warp(grid, images):
    B, M, H, W, _ = grid.shape
    B, C, h, w = images.shape
    device = images.device

    hw = torch.tensor([h, w], device=device)
    grid = grid.reshape(B * M, H, W, 2)  # (B*M, H, W, 2)
    grid = (grid / hw - 0.5) * 2.0  # (B*M, H, W, 2)
    grid = grid[..., [1, 0]]
    images = images[:, None].expand(-1, M, -1, -1, -1).reshape(B * M, C, h, w)  # (B*M, C, H, W)
    psv = F.grid_sample(images, grid).reshape(B, M, C, H, W)  # (B, M, C, H, W)

    return psv


def forward_warp_nn(grid, images, fill=0.0):
    B, N = images.shape
    h, w = 1000, 1000
    device = grid.device

    grid = filter_valid_points(grid, [h, w])  # (B, N, 2)
    grid = torch.round(grid).type(torch.int64)

    grid = utils.ravel_multi_index(grid, [h, w])  # (B, N)

    output = torch.ones((B, h * w), device=device) * fill  # (B, h*w)
    output.scatter_add_(1, grid, images)  # (B, H*W)
    output = output.reshape(B, h, w)  # (B, h, w)

    return output
