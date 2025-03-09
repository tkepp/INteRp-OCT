#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch.nn.functional as F
import numpy as np
import json
from typing import Tuple
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.explicit_start = True
yaml.default_flow_style = False
yaml.boolean_representation = ['False', 'True']


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


# Load the config
def read_yaml(config_path: str) -> (obj, dict):
    with open(config_path) as f:
        config_dict = yaml.load(f)
        config_obj = dict2obj(config_dict)
    return config_obj, config_dict


def write_dict_as_yaml(config_dict: dict, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(config_dict, yaml_file)
        #yaml_file.write(yaml_dump)
    yaml_file.close()


def set_gpu(gpu_num, verbose=True):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    total_memory = round(torch.cuda.get_device_properties(0).total_memory*1e-9, 3)
    if verbose:
        print('Using GPU '+str(gpu_num), '- Total Memory: '+str(total_memory)+' GB')


def remove_subjects(path_list, ignored_subjects):
    return [path for path in path_list if not any(subject in path for subject in ignored_subjects)]


def divide_list(data, sizes):
    result = []
    index = 0
    for size in sizes:
        if index + size <= len(data):
            result.append(data[index:index + size])
            index += size
        else:
            # If there are not enough elements left in `data` for the current chunk size,
            # take the remaining elements.
            result.append(data[index:])
            break
    return result


def save_as_itk(img_np: np.array, path: str, spacing: Tuple[float, ...]) -> None:
    img_itk = sitk.GetImageFromArray(img_np)
    img_itk.SetSpacing(spacing)
    sitk.WriteImage(img_itk, path)


def normalize(x, old_min=None, old_max=None, new_min=-1, new_max=1):
    if old_min is None:
        x_min = np.nanmin(x)
    else:
        x_min = old_min
    if old_max is None:
        x_max = np.nanmax(x)
    else:
        x_max = old_max
    return (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords  # * torch.pi


def flattening_with_label(image, label, boundaries=None, mask=None):
    D, H, W = image.shape

    if mask is None:
        mask = label

    bm = (H - np.argmax((mask > 0)[:, ::-1, :], axis=1)).astype(float) - 1
    new_level = np.floor(H * 0.65)
    img_flat = np.zeros(image.shape)
    label_flat = np.zeros(label.shape)
    if boundaries is not None:
        boundaries_flat = np.zeros(boundaries.shape)

    for d in range(D):
        if boundaries is not None:
            bms = boundaries[-1, d, ...].astype(float)
        else:
            bms = bm[d, ...].astype(float)


        # remove empty slices
        bms[bms > H * 0.95] = np.nan
        bms[bms < H * 0.05] = np.nan

        # fill nan values with closest non-nan value
        nan_mask = np.isnan(bms)
        bms[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), bms[~nan_mask])
        bms = np.round(bms).astype(int)

        # compute shift
        diff = bms - new_level

        # Convert diff to matrix
        uy = np.tile(diff[None, ...], [H, 1])

        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        indices = np.reshape(Y + uy, (-1, 1)), np.reshape(X, (-1, 1)).astype(float)

        # apply column shifts
        img_flat[d, ...] = map_coordinates(image[d, ...], indices, order=3, mode='constant', cval=0.0).reshape((H, W))
        label_flat[d, ...] = map_coordinates(label[d, ...], indices, order=0, mode='nearest').reshape((H, W))
        if boundaries is not None:
            boundaries_flat[:, d, ...] = boundaries[:, d, ...] - diff


    if boundaries is not None:
        return img_flat, label_flat, boundaries_flat
    else:
        return img_flat, label_flat


class AverageMeter(object):
    def __init__(self, is_tensor=False):
        self.reset()
        self.is_tensor = is_tensor

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            self.is_tensor = True
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def turn_to_onehot(seg, n_classes):

    H, W = seg.shape

    onehot = torch.zeros((H, W, n_classes))
    for c in range(n_classes):
        #if c == 1:  # Retina
        #    onehot[:, :, c] = (seg >= c).float()
        #else:  # Background and pathologies
        #    onehot[:, :, c] = (seg == c).float()
        onehot[:, :, c] = (seg == c).float()

    return onehot.clone()


def get_train_indices(num_subjects, num_slices):
    subject_and_slice_ids = []
    subject_ids = np.arange(num_subjects)
    for subject_id in subject_ids:
        slice_ids = np.arange(num_slices)
        for slice_id in slice_ids:
            subject_and_slice_ids.append((subject_id, slice_id))

    return subject_and_slice_ids


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.00001, min_epochs=500):
        """
        Args:
            patience (int): How many epochs to wait after last time training loss improved.
            min_delta (float): Minimum change in the training loss to qualify as an improvement.
            min_epochs (int): Minimum number of epochs to run before considering early stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch, train_loss):
        if epoch < self.min_epochs:
            return
        if self.best_loss is None:
            self.best_loss = train_loss
        elif train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True):
    if window is None:
        window = create_window(window_size, img1.size(1)).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class COMBILoss(torch.nn.Module):
    def __init__(self):
        super(COMBILoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.mse_los = torch.nn.MSELoss()

    def forward(self, img1, img2):
        ssim = self.ssim_loss(img1, img2)
        mse = self.mse_los(img1, img2)

        return 0.1 * ssim + mse