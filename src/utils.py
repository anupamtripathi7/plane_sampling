import torch.nn as nn
import torch
import cv2
import math
import copy
import importlib
from random import randint
import numpy as np
from torch.nn.functional import grid_sample
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_to_tensor(img):
    """
    Convert image to tensor and permute
    Args:
        img (nd-array): numpy image (B, H, W, C)
    Returns:
        Tensor image
    """
    img = torch.from_numpy(img).to(Config().device)
    return img.permute(0, 3, 1, 2)


def tensor_to_image(img_tensor):
    """
    Convert image to tensor and permute
    Args:
        img_tensor (torch.tensor): tensor (B, C, H, W)
    Returns:
        Tensor image
    """
    img_tensor = img_tensor.cpu().numpy()
    return img_tensor.permute(0, 2, 3, 1)


def name_to_class(class_name, module_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def z(i, fill=4):
    if isinstance(i, int):
        i = str(i)

    i = i.zfill(fill)

    return i


def color_flip(images, dim=1):
    if dim == 1:
        return images[:, [2, 1, 0]]
    elif dim == -1:
        return images[..., [2, 1, 0]]


def regular_meshgrid(shape, device="cpu"):
    grid = torch.meshgrid(*[torch.arange(i) for i in shape])
    grid = torch.stack(grid, dim=-1).to(device)
    return grid


def ravel_multi_index(indices, shape):
    assert indices.shape[-1] == len(shape), f"indices.shape[-1] and len(shape) must be equal. Found {indices.shape[-1]} and {len(shape)}"

    out = torch.zeros((indices.shape[:-1]), dtype=indices.dtype, device=indices.device)
    cum_prod = torch.ones((1,), dtype=torch.int64, device=indices.device)
    for i in range(len(shape) - 1, -1, -1):
        out += indices[..., i] * cum_prod
        cum_prod = cum_prod * int(shape[i])

    return out

def tocv(tensor, to_int=False, cf=True):
    tensor = tensor.clone().detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]

    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[0] == 1:
        tensor = tensor[0]

    if tensor.ndim > 2 and tensor.shape[2] == 3 and cf:
        tensor = tensor[..., [2, 1, 0]]

    if tensor.dtype == torch.float32:
        tensor = torch.clamp(tensor, min=0.0, max=1.0)

    if to_int:
        tensor = (tensor * 255.0).byte()

    tensor = tensor.numpy()

    return tensor

def pad_or_crop(tensor, shape):
    if len(shape) > 2:
        shape = shape[-2:]
    B, C, H, W = tensor.shape

    if H == shape[0] and W == shape[1]:
        return tensor

    ret_tensor = torch.zeros((B, C, *shape), device=tensor.device, dtype=tensor.dtype)
    ret_tensor[:, :, :H, :W] = tensor[:, :, :shape[0], :shape[1]]

    return ret_tensor


def coordinate_samling(src, xy):
    B, H, W, _ = xy.shape
    size = torch.tensor([H, W], dtype=torch.float32, device=xy.device)
    grid = (xy / size) * 2 - 1.0
    sampled = grid_sample(src, grid)

    return sampled


def update_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def crop_nd(tensor, slc_limits, dim=[-2, -1]):
    assert len(slc_limits) == len(dim), f"len(slc_limits) and len(dim) should match. found {len(slc_limits)} and {len(dim)}"
    slc = [slice(None)] * tensor.ndim
    for i in len(dim):
        slc[dim] = slice(slc_limits[i][0], slc_limits[i][1])

    return tensor[slc]

def symmetric_crop(tensor, c, dim):
    slc_limits = [[c, -c]] * len(dim)

    return crop_nd(tensor, slc_limits, dim=dim)

def center_crop(tensor, template, dim):
    tensor_shape = list(tensor.shape)
    template_shape = list(template.shape)

    slc_limits = []
    for d in dim:
        slc_limits.append([(tensor_shape[d] - template_shape[d]) // 2, -1 * ((tensor_shape[d] - template_shape[d]) - ((tensor_shape[d] - template_shape[d]) // 2))])

    return crop_nd(tensor, slc_limits, dim)

def random_crop(tensor, dim, patch_size, patch_range=None, batch_dim=0):
    assert len(patch_size) == len(dim), f"len(patch_size) and len(dim) should match. found {len(patch_size)} and {len(dim)}"
    tensor_shape = list(tensor.shape)

    if patch_range is None:
        patch_range = []
        for d in dim:
            patch_range.append([0, tensor_shape[d]])
    else:
        assert len(patch_range) == len(dim), f"len(patch_range) and len(dim) should match. found {len(patch_range)} and {len(dim)}"

    B = tensor_shape[batch_dim]
    out_tensors = []

    for i in range(B):
        slc_limits = []
        for j in len(dim):
            start = randint(patch_range[j][0], patch_range[j][1] - patch_size[j])
            slc_limits.append([start, start + patch_size[j]])

        slc = [slice(None)] * len(tensor_shape)
        slc[batch_dim] = slice(i, i + 1)
        out_tensors.append(crop_nd(tensor[slc], slc_limits, dim))

    out_tensor = torch.stack(out_tensors, dim=batch_dim)


def uniform_disparity_planes(depth_range, num_planes):
    disparity_range = 1.0 / depth_range
    step = (disparity_range[0] - disparity_range[1]) / (num_planes - 1)
    disparity_planes = torch.tensor([disparity_range[0] - i * step for i in range(num_planes)])

    return disparity_planes


def uniform_disparity_depth_planes(depth_range, num_planes):
    disparity_range = 1.0 / depth_range
    step = (disparity_range[0] - disparity_range[1]) / (num_planes - 1)
    disparity_planes = torch.tensor([disparity_range[0] - i * step for i in range(num_planes)])
    depth_planes = 1 / disparity_planes

    return depth_planes


def depth_range_to_disparity_bins(depth_range, num_planes):
    disparity_range = 1.0 / depth_range
    step = (disparity_range[0] - disparity_range[1]) / num_planes
    disparity_bins = torch.tensor([[disparity_range[0], disparity_range[0] - (i + 1) * step] for i in range(num_planes)])

    return disparity_bins


def batch_idx_generator(idxs, batch_size, random=True):
    num_batches = idxs.shape[0] // batch_size
    if random:
        idxs = idxs[torch.randperm(idxs.shape[0])]

    for i in range(num_batches):
        yield idxs[i * batch_size: (i + 1) * batch_size]


def video_from_frames(filename, frames, fps, codec, cf=True):
    B, C, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(filename, fourcc, fps, (W, H))

    for frame in frames:
        frame = tocv(frame, to_int=True, cf=cf)
        video.write(frame)

    video.release()

def circular_translation(c, r, N):
    theta = torch.arange(N) * 2 * np.pi / N
    r_vector = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    t = torch.zeros((N, 4, 4))
    t[:, :2, 3] = r_vector * r + c

    return t


def hist_match_grey(source, template, to_int=True):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments:
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    output = interp_t_values[bin_idx].reshape(oldshape)

    if to_int:
        output = output.astype(np.uint8)

    return output


def hist_match(source, template, channel_dim=0, to_int=True):
    equalized_img = []

    for channel in range(source.shape[channel_dim]):
        if channel_dim == 0:
            equalized_img.append(hist_match_grey(source[channel], template[channel], to_int=to_int))
        elif channel_dim == 2:
            equalized_img.append(hist_match_grey(source[:, :, channel], template[:, :, channel], to_int=to_int))
        else:
            print("channel dimension not proper !! ")
            return

    equalized_img = np.array(equalized_img)

    if channel_dim == 2:
        equalized_img = equalized_img.transpose(1, 2, 0)

    return equalized_img


def plot(img, figsize=10, pytorch=True, colorbar=False, grid=True, cmap=None):
    if pytorch:
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(img, cmap=cmap)
    if colorbar:
        plt.colorbar()
    if grid:
        plt.grid()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def merge_dims(tensor, dim):
    out_shape = list(tensor.shape)
    out_shape = out_shape[: dim[0]] + [np.prod(tensor.shape[dim[0]:dim[1] + 1])] + out_shape[dim[1] + 1:]
    return tensor.reshape(out_shape)

def unmerge_dims(tensor, dim, shape):
    out_shape = list(tensor.shape)
    out_shape = out_shape[: dim] + list(shape) + out_shape[dim + 1:]
    return tensor.reshape(out_shape)

def matmul(a, b):
    shape_a = a.shape
    shape_b = b.shape

    if a.ndim < b.ndim:
        for _ in range(b.ndim - a.ndim):
            a = a.unsqueeze(0)
    else:
        for _ in range(a.ndim - b.ndim):
            b = b.unsqueeze(0)

    expanded_shape = [max([a.shape[i], b.shape[i]]) for i in range(a.ndim - 2)] + [-1, -1]

    a = a.expand(expanded_shape).reshape(-1, a.shape[-2], a.shape[-1])
    b = b.expand(expanded_shape).reshape(-1, b.shape[-2], b.shape[-1])

    out = torch.bmm(a, b).reshape(expanded_shape[:-2] + [shape_a[-2]] + [shape_b[-1]])
    return out


def chain_mm(matrics):
    out = matrics[0]
    for i in range(1, len(matrics)):
        out = matmul(out, matrics[i])
    return out
