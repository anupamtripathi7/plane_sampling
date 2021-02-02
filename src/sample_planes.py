from src.msi import MultiPlaneImage
from src.camera import PerspectiveCameras
from src.components import AlphaCompositing
from copy import deepcopy
from src.mpi import alpha_correction_on_mpi
import torch
from pytorch_msssim import ssim
import numpy as np


mse = torch.nn.MSELoss()


def sample_planes(mpi_object, n, order, gt, camera):
    """
    Sample planes by removing top n planes in idx
    Args:
        mpi_object (MultiPlaneImage): MPI object
        n (int): Number of planes to remove
        order (torch.Tensor): Argsort indices for planes
        gt (torch.Tensor): Ground truth image
        camera (PerspectiveCameras): Camera object

    Returns:
        (dict): SSIM and MSE loss
        (torch.Tensor): MPI features with removed planes
    """
    idx = torch.argsort(order)
    mpi = mpi_object.features
    original = deepcopy(gt).float().to(mpi.device)

    losses = {'ssim': [], 'mse': []}
    for i in range(0, n):
        mpi[0, idx[:i]] = torch.zeros_like(mpi[0, 0]).to(mpi.device)
        mpi_object.features = mpi
        img = mpi_object.view(camera, AlphaCompositing(normalize_weights=False))
        print(idx[:i])
        losses['ssim'].append(1 - ssim(original[None, :], img[0].permute(0, 2, 3, 1), data_range=1.0))
        losses['mse'].append(mse(original[None, :], img[0].permute(0, 2, 3, 1)))
    return losses, mpi


def sample_planes_on_correlation(mpi_object, num_planes_to_remove, order, gt, camera):
    """
    Sample planes by removing top n adjacent planes in idx
    Args:
        mpi_object (MultiPlaneImage): MPI object
        num_planes_to_remove (int): Number of planes to remove
        order (torch.Tensor): Argsort indices for adjacent planes
        gt (torch.Tensor): Ground truth image
        camera (PerspectiveCameras): Camera object

    Returns:
        (dict): SSIM and MSE loss
        (torch.Tensor): MPI features with removed planes
    """
    idx = np.argsort(order)
    mpi = mpi_object.features
    original = deepcopy(gt).float().to(mpi.device)

    losses = {'ssim': [], 'mse': []}
    for i in range(0, num_planes_to_remove):
        indices_to_remove = idx[-i:] if i != 0 else []
        # for n, x in enumerate(idx):
        #     if x < i:
        #         indices_to_remove.append(n)
                # indices_to_remove.append(n) if idx[n] > idx[n + 2] else indices_to_remove.append(n + 1)
        mpi[0, indices_to_remove] = torch.zeros_like(mpi[0, 0])
        mpi_object.features = mpi
        img = mpi_object.view(camera, AlphaCompositing(normalize_weights=False))
        losses['ssim'].append(1 - ssim(original[None, :], img[0].permute(0, 2, 3, 1), data_range=1.0))
        losses['mse'].append(mse(original[None, :], img[0].permute(0, 2, 3, 1)))
    return losses, mpi


def sample_planes_using_sum(mpi_object_og, n, gt):
    """
    Remove planes using the sum of alpha channels method
    Args:
        mpi_object_og (MultiPlaneImage): MPI object
        n (int): Number of planes to remove
        gt (dict/torch.Tensor): Ground truth image

    Returns:
        (dict): Loss
    """
    mpi_object = deepcopy(mpi_object_og)
    mpi_object_with_correction = alpha_correction_on_mpi(mpi_object)
    alpha = mpi_object_with_correction.features[0, :, -1]               # (M, H, W)
    alpha_flat = alpha.view(alpha.size(0), -1)
    alpha_sum = torch.sum(alpha_flat, dim=1)
    idx = torch.argsort(alpha_sum)
    print(idx)

    if isinstance(gt, dict):
        loss = {}
        for shift, img in gt.items():
            camera = mpi_object.camera.translate_in_x(shift)
            loss[shift], _ = sample_planes(deepcopy(mpi_object), n, alpha_sum, img, camera)
    else:
        loss, _ = sample_planes(mpi_object, n, alpha_sum, gt, camera=mpi_object.camera)
    return loss, mpi_object, idx


def sample_planes_using_max(mpi_object, n, gt):
    """
    Remove planes using the max in alpha channels method
    Args:
        mpi_object (MultiPlaneImage): MPI object
        n (int): Number of planes to remove
        gt (dict/torch.Tensor): Ground truth image

    Returns:
        (dict): Loss
    """
    mpi_object = deepcopy(mpi_object)
    mpi = mpi_object.features
    mpi_object_with_correction = alpha_correction_on_mpi(mpi_object)
    alpha = mpi_object_with_correction.features[0, :, -1]  # (M, H, W)
    values, counts = torch.argmax(alpha, dim=0).unique(return_counts=True)
    extra_vals, extra_counts = [], []
    for x in range(mpi.shape[1]):
        if x not in values:
            extra_vals.append(x)
            extra_counts.append(0)
    counts = torch.cat((counts.cpu(), torch.tensor(extra_counts)))
    idx = np.argsort(counts)
    print(idx)

    if isinstance(gt, dict):
        loss = {}
        for shift, img in gt.items():
            camera = mpi_object.camera.translate_in_x(shift)
            loss[shift], _ = sample_planes(deepcopy(mpi_object), n, counts, img, camera)
    else:
        loss, _ = sample_planes(mpi_object, n, counts, gt, camera=mpi_object.camera)
    return loss, mpi_object, idx


def sample_planes_using_correlation(mpi_object, n, gt):
    """
    Remove planes using the correlation between adjacent planes method
    Args:
        mpi_object (MultiPlaneImage): MPI object
        n (int): Number of planes to remove
        gt (dict/torch.Tensor): Ground truth image

    Returns:
        (dict): Loss
    """
    mpi_object = deepcopy(mpi_object)
    correlation = []
    mpi_object_with_correction = alpha_correction_on_mpi(mpi_object)
    mpi_with_correction = mpi_object_with_correction.features.cpu()
    for n1, plane in enumerate(mpi_with_correction[0, 1:]):
        prev_plane = mpi_with_correction[0, n1]
        correlation.append(np.corrcoef(prev_plane.numpy().flatten(), plane.numpy().flatten())[0, 1])  # between x and y in correlation matrix
    idx = np.argsort(correlation)
    print(idx)

    if isinstance(gt, dict):
        loss = {}
        for shift, img in gt.items():
            camera = mpi_object.camera.translate_in_x(shift)
            loss[shift], _ = sample_planes_on_correlation(deepcopy(mpi_object), n, correlation, img, camera)
    else:
        loss, _ = sample_planes_on_correlation(mpi_object, n, correlation, gt, camera=mpi_object.camera)
    return loss, mpi_object, idx


def sample_planes_using_ssim_correlation(mpi_object, n, gt):
    """
    Remove planes using the ssim correlation between adjacent planes method
    Args:
        mpi_object (MultiPlaneImage): MPI object
        n (int): Number of planes to remove
        gt (dict/torch.Tensor): Ground truth image

    Returns:
        (dict): Loss
    """
    mpi_object = deepcopy(mpi_object)
    mpi_object_with_correction = alpha_correction_on_mpi(mpi_object)
    mpi_with_correction = mpi_object_with_correction.features.cpu()
    losses = []
    for n1, plane in enumerate(mpi_with_correction[0, 1:]):
        prev_plane = mpi_with_correction[0, n1]
        losses.append(ssim(prev_plane.unsqueeze(0), plane.unsqueeze(0), data_range=1.0))
    idx = np.argsort(losses)
    print(idx)

    if isinstance(gt, dict):
        loss = {}
        for shift, img in gt.items():
            camera = mpi_object.camera.translate_in_x(shift)
            loss[shift], _ = sample_planes_on_correlation(deepcopy(mpi_object), n, losses, img, camera)

    else:
        loss, _ = sample_planes_on_correlation(mpi_object, n, losses, gt, camera=mpi_object.camera)
    return loss, mpi_object, idx