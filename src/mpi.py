import math
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from src.msi import MultiPlaneImage
from src.camera import PerspectiveCameras
from src.utils import image_to_tensor, Config
from src.projector import alpha_correction
import glob


cfg = Config()


def load_mpi(mpi_folder='data/banana', M=32):
    """
    Loads mpi images into MPI object
    Args:
        mpi_folder (str): Folder name
        M (int):  Number of planes

    Returns:
        (MultiPlaneImage): MPI object
        (torch.Tensor): MPI features of size (1, M, C, H, W)
        (dict): Dict of ground truth image of size (H, W, 3)

    """
    depth_range = np.array([1.0, 100.0])
    disparity_range = 1 / depth_range
    mpi = []
    disparities = []
    for num in range(M):
        rgb = cv2.cvtColor(cv2.imread(os.path.join(mpi_folder, 'mpi_rgb_{}.png'.format(str(num).zfill(2)))),
                           cv2.COLOR_BGR2RGB)
        alpha = cv2.imread(os.path.join(mpi_folder, 'mpi_alpha_{}.png'.format(str(num).zfill(2))), 0)
        plane = np.append(rgb / 255.0, np.expand_dims(alpha / 255.0, -1), axis=-1)
        mpi.append(plane)
        disparities.append((M - num - 1) * (disparity_range[0] - disparity_range[1]) / (M - 1))  # depth
    mpi = image_to_tensor(np.array(mpi)).float()[None, :]
    gt, _, intrinsics = load_scene(mpi_folder, pad_to_size=mpi)

    print('MPI features: {}, gt size: {}'.format(mpi.size(), gt[0].size()))
    print('MPI features max: {}, gt max: {}'.format(mpi.max(), gt[0].max()))

    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
    camera = PerspectiveCameras(intrinsics, extrinsics)

    disparities = torch.tensor(disparities).double().to(cfg.device)
    mpi_object = MultiPlaneImage(mpi, disparities, camera)

    return mpi_object, mpi, gt


def plot_planes(mpi, alpha=False, hist=False, alpha_corrected=None):
    """
    Plot all the planes from mpi
    Args:
        mpi (torch.tensor): mpi array (1, M, 4, H, W)
        m (int): number of planes
        alpha (bool): if true, plots alpha planes too
        hist (bool): if true, plots histogram of alpha channels
    """
    m = mpi.size(1)
    fig, axs = plt.subplots(5, math.ceil(m / 5), figsize=(60, 40), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)

    axs = axs.ravel()

    for i, img in enumerate(mpi[0]):
        img = img[:-1].permute(1, 2, 0).cpu()
        img = img.numpy()
        axs[i].imshow(img)
        axs[i].set_title('sum: {} max: {}'.format(img.sum(), round(img.max().item(), 2)), fontsize=40)
    plt.show()

    if alpha_corrected is not None:
        mpi = alpha_corrected
    if alpha:
        fig, axs = plt.subplots(5, math.ceil(m / 5), figsize=(60, 40), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)

        axs = axs.ravel()
        for i, img in enumerate(mpi[0]):
            img = img[-1].cpu().numpy()
            im = axs[i].imshow(img)
            axs[i].set_title('sum: {} max: {}'.format(img.sum(), round(img.max().item(), 2)), fontsize=40)
            # fig.colorbar(im, ax=axs[i])
        plt.show()

        fig, axs = plt.subplots(5, math.ceil(m / 5), figsize=(60, 40), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)

        axs = axs.ravel()
        for i, img in enumerate(mpi[0]):
            img = img.permute(1, 2, 0).cpu().numpy()
            axs[i].imshow(img)
            axs[i].set_title('sum: {} max: {}'.format(img.sum(), round(img.max().item(), 2)), fontsize=40)
        plt.show()

    if hist:
        fig, axs = plt.subplots(5, math.ceil(m / 5), figsize=(15, 15), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)

        axs = axs.ravel()
        for i, img in enumerate(mpi[0]):
            img = img[-1].cpu().numpy()
            (n, _, _) = axs[i].hist(img.flatten(), bins=25)
        plt.show()


def alpha_correction_on_mpi(mpi_object):
    """
    Return an mpi object with corrected alphas
    Args:
        mpi_object (MultiPlaneImage): MPI object

    Returns:
        (MultiPlaneImage): MPI object with corrected alphas
    """
    mpi_object = deepcopy(mpi_object)
    mpi_with_correction = mpi_object.features
    alpha = mpi_with_correction[:, :, -1, :, :].flip(dims=(1,))
    alpha, _ = alpha_correction(alpha)
    mpi_with_correction[:, :, -1, :, :] = alpha.flip(dims=(1,))
    mpi_object.features = mpi_with_correction
    return mpi_object


def load_scene(folder='data/banana', device=cfg.device, pad_to_size=None):
    gt = {}
    for img in glob.glob(os.path.join(folder, 'render_*')):
        gt_np = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        if pad_to_size is not None:
            pad1 = int((pad_to_size.size(-2) - gt_np.shape[-3]) / 2)
            pad2 = int((pad_to_size.size(-1) - gt_np.shape[-2]) / 2)
            gt_np = np.pad(gt_np, ((pad1, pad1), (pad2, pad2), (0, 0)))
        gt[float(img[22:-4])] = torch.from_numpy(gt_np / 255.0).to(device)

    src_img = []
    for img in glob.glob(os.path.join(folder, 'src_image*')):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        src_img.append(img)

    f = open(os.path.join(folder, "intrinsics.txt"), "r")
    intrinsics = f.read().strip().split(' ')
    intrinsics = list(map(float, intrinsics))
    intrinsics = torch.tensor([[intrinsics[0], 0, intrinsics[2], 0],
                               [0, intrinsics[1], intrinsics[3], 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).unsqueeze(0).unsqueeze(0).float().to(device)

    return gt, np.array(src_img), intrinsics


def get_disparity_planes(depth_range, M, keep=None):
    if keep is None:
        keep = list(range(M))
    depth_range = np.array(depth_range)
    disparity_range = 1 / depth_range
    disparities = []
    for num in keep:
        disparities.append((M - num - 1) * (disparity_range[0] - disparity_range[1]) / (M - 1))
    return torch.tensor(disparities)


if __name__ == "__main__":
    # mpi, features = load_mpi()
    # plot_planes(features, True)
    print(get_disparity_planes([1, 100], 32, keep=[0, 20,  6, 18, 19,  1]))
