import cv2
import torch
import pathlib

import numpy as np
import sigpy as sp

from mri_utils.espirit import ifft, fft
from mri_utils.fftc import ifft2c_new, fft2c_new
from mri_utils.math import complex_abs

from torch.utils.data import DataLoader
from mri_data import transforms
from mri_data.mri_data import SelectiveSliceData, SelectiveSliceData_Val


class DataTransform:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mask = get_mask(cfg.data.num_coils)

    def __call__(self, kspace, target, attrs, fname, slice, sense_maps=None):
        kspace = kspace.transpose(1, 2, 0)
        x = ifft(kspace, (0, 1))  # (768, 396, 16)
        coil_compressed_x = ImageCropandKspaceCompression(x)  # (384, 384, 8)

        x = transforms.to_tensor(coil_compressed_x).permute(2, 0, 1, 3)
        y = ifft2c_new(fft2c_new(x) * self.mask)

        normalized_y, mean, std = transforms.normalize_instance(y)
        normalized_x = transforms.normalize(x, mean, std)

        final_input = torch.zeros(self.cfg.data.num_coils*2, 384, 384)
        final_input[0:self.cfg.data.num_coils, :, :] = normalized_y[:, :, :, 0]
        final_input[self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :] = normalized_y[:, :, :, 1]

        final_gt = torch.zeros(self.cfg.data.num_coils*2, 384, 384)
        final_gt[0:self.cfg.data.num_coils, :, :] = normalized_x[:, :, :, 0]
        final_gt[self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :] = normalized_x[:, :, :, 1]

        return (final_gt.float(), final_input.float(), mean.float(), std.float(), self.mask), self.mask # Weirdness here to work generally with inpainting loader


def create_datasets(cfg, big_test=False):
    train_data = SelectiveSliceData(
        root=pathlib.Path(cfg.data.path) / 'multicoil_train',
        transform=DataTransform(cfg),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=cfg.data.num_of_top_slices,
        restrict_size=False,
    )

    dev_data = SelectiveSliceData_Val(
        root=pathlib.Path(cfg.data.path) / 'multicoil_val',
        transform=DataTransform(cfg),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=cfg.data.num_of_top_slices,
        restrict_size=False,
        big_test=big_test # Set to true to use entire fastMRI validation set
    )

    return dev_data, train_data


def create_data_loaders_mri(cfg, rank, world_size, big_test=False, drop_last_val=True):
    dev_data, train_data = create_datasets(cfg, big_test=big_test)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True, # Helps with training time on multiple GPUs
        sampler=train_sampler
    )

    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True, # Helps with training time on multiple GPUs - set false for testing
        sampler=dev_sampler
    )

    return train_loader, dev_loader


# Helper functions for Transform
def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def unflatten(t, shape_t):
    t = t.reshape(shape_t)
    return t


def ImageCropandKspaceCompression(x):
    w_from = (x.shape[0] - 384) // 2  # crop images into 384x384
    h_from = (x.shape[1] - 384) // 2
    w_to = w_from + 384
    h_to = h_from + 384
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > 8:
        x_tocompression = cropped_x.reshape(384 ** 2, cropped_x.shape[-1])
        U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:, 0:8].reshape(384, 384, 8)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x

def get_mask(numcoil):
    # R = 4 GRO sampling mask
    a = np.array(
        [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
         151, 155, 158, 161, 164,
         167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
         223, 226, 229, 233, 236,
         240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
         374])
    m = np.zeros((384, 384))
    m[:, a] = 1
    m[:, 176:208] = 1

    samp = m
    mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
    mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    return mask.float()