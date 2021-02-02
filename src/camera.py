import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from . import transforms
from . import utils
from .garbage_collector import log_tensor_sizes


EPS = 1e-8


class PerspectiveCameras(nn.Module):
    @classmethod
    def from_params(cls, params, device=None):
        return cls(params["intrinsics"][:, 0], params["extrinsics"][:, 0])

    @classmethod
    def collate(cls, cameras):
        intrinsics = torch.cat([camera.intrinsics for camera in cameras], dim=0)
        extrinsics = torch.cat([camera.extrinsics for camera in cameras], dim=0)

        return cls(intrinsics, extrinsics)

    @classmethod
    def from_tensors(cls, tensors):
        return cls(tensors[..., 0], tensors[..., 1])

    def __init__(self, intrinsics, extrinsics=None):
        super().__init__()
        self.intrinsics = intrinsics  # (*, 4, 4)
        self.extrinsics = extrinsics if extrinsics is not None else torch.eye(4).expand_as(intrinsics)  # (*, 4, 4)
        # print('camera: ', intrinsics.shape, self.extrinsics.shape)
        self.device = intrinsics.device
        self.derived_matrices()

    def derived_matrices(self):
        self.K = self.intrinsics[..., :3, :3]  # (*, 3, 3)
        self.R = self.extrinsics[..., :3, :3]  # (*, 3, 3)
        self.T = self.extrinsics[..., :3, 3:]  # (*, 3, 1)

    def to(self, device):
        self.device = device
        self.intrinsics = self.intrinsics.to(device)
        self.extrinsics = self.extrinsics.to(device)

        return self

    def clone(self):
        return copy.deepcopy(self)

    def tensors(self):
        return torch.stack([self.intrinsics, self.extrinsics], dim=-1)

    def relative_extrinsics(self, camera):
        return torch.matmul(camera.extrinsics, torch.inverse(self.extrinsics))

    def t2s_homography(self, uv, camera, depths, n):
        rel_ext = torch.inverse(self.relative_extrinsics(camera))
        R, T = rel_ext[..., :3, :3], rel_ext[..., :3, 3:]

        kRk = utils.chain_mm([self.K, R, torch.inverse(camera.K)])  # (*, 3, 3)
        kRtnRk = utils.chain_mm([self.K, R, T, n, R, torch.inverse(camera.K)])  # (*, 3, 3)
        knRtk = utils.chain_mm([n, R, T])  # (*, 3, 3)

        kRk_grid = utils.matmul(kRk, uv)  # (*, 3, 1)
        kRtnRk_grid = utils.matmul(kRtnRk, uv)  # (*, 3, 1)

        grid = kRk_grid + (kRtnRk_grid / (depths - knRtk + EPS))  # (*, 3, 3)
        return grid

    def s2t_homography(self, uv, camera, depths, n):
        rel_ext = self.relative_extrinsics(camera)
        R, T = rel_ext[..., :3, :3], rel_ext[..., :3, 3:]

        kRk = utils.chain_mm([camera.K, R, torch.inverse(self.K)])  # (*, 3, 3)
        ktnk = utils.chain_mm([camera.K, T, n, torch.inverse(self.K)])  # (*, 3, 3)

        kRk_grid = utils.matmul(kRk, uv)  # (*, 3, 1)
        ktnk_grid = utils.matmul(ktnk, uv)  # (*, 3, 1)
        grid = kRk_grid + (ktnk_grid / (depths + EPS))  # (*, 3, 1)
        return grid

    def affine_grid(self, camera, uv, depths, n, s2t=True):
        B, V, _, _, _ = camera.tensors().shape
        B, H, W, _ = uv.shape
        M = depths.shape[0]

        camera = camera.expand_dim([1, 3, 4])  # (B, 1, V, 1, 1, 3, 3)
        self = self.expand_dim([1, 3, 4])  # (B, 1, V, 1, 1, 3, 3)
        uv = transforms.euclid_to_homo(uv)  # (B, H, W, 3)
        uv = uv[:, None, None, :, :, :, None]  # (B, 1, 1, H, W, 3, 1)
        depths = depths[None, :, None, None, None, None, None]  # (1, M, 1, 1, 1, 1, 1)
        n = n[:, None, :, None, None, None]  # (B, 1, V, 1, 1, 1, 3)

        if s2t:
            grid = self.s2t_homography(uv, camera, depths, n)  # (B, M, V, H, W, 3, 1)
        else:
            grid = self.t2s_homography(uv, camera, depths, n)  # (B, M, V, 3, 3)
        grid = transforms.homo_to_euclid(grid[..., 0])  # (*, 2)
        return grid

    def homographic_warp(self, camera, features, depths, n, uv=None, s2t=True):
        B, M, V, C, H, W = features.shape
        if uv is None:
            uv = utils.regular_meshgrid([H, W], device=self.device).expand(B, -1, -1, -1)  # (B, H, W, 2)
        hw = torch.tensor([H, W], device=self.device)  # (H, W, 2)

        grid = self.affine_grid(camera, uv, depths, n, s2t)  # (B, M, V, H, W, 2)
        grid = (grid / hw - 0.5) * 2.0  # (B, M, V, H, W, 2)
        grid = grid[..., [1, 0]]  # (B, M, V, H, W, 2)

        features = utils.merge_dims(features, [0, 2])  # (B*M*V, C, H, W)
        grid = utils.merge_dims(grid, [0, 2])  # (B*M*V, H, W, 2)

        out = F.grid_sample(features, grid.float())  # (B*M*V, C, H, W)
        out = utils.unmerge_dims(out, 0, [B, M, V])  # (B, M, V, C, H, W)

        return out

    def expand_dim(self, dim):
        if isinstance(dim, int):
            tensors = self.tensors().unsqueeze(dim)
        else:
            tensors = self.tensors()
            for d in dim:
                tensors = tensors.unsqueeze(d)

        return PerspectiveCameras.from_tensors(tensors)

    def translate(self, t):
        dest_extrinsics = self.extrinsics.clone()
        dest_extrinsics += t
        return PerspectiveCameras(self.intrinsics.clone(), dest_extrinsics)

    def translate_in_x(self, t):
        dest_extrinsics = self.extrinsics.clone()
        dest_extrinsics[..., 1, -1] += t/100
        return PerspectiveCameras(self.intrinsics.clone(), dest_extrinsics)
