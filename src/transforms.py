import torch


EPS = 1e-8


def to_mm(v):
    B, N, C = v.shape

    if C < 4:
        v = torch.cat([v, torch.ones((B, N, 4 - C), device=v.device)], dim=-1)

    v = v[..., None]  # (B, N, 4, 1)

    return v


def from_mm(v, C):
    B, N = v.shape[:2]
    v = v[..., 0]

    if C == 2:
        v = homo_to_euclid(v[..., :3])
    elif C == 3:
        v = homo_to_euclid(v)

    return v


def homo_to_euclid(points):
    return points[..., :-1] / (points[..., -1:] + EPS)


def euclid_to_homo(points):
    points_shape = list(points.shape)
    points_shape[-1] = 1
    return torch.cat([points, torch.ones(points_shape, device=points.device)], dim=-1)


def hw2xy(points):
    points = points[..., [1, 0]]

    return points
