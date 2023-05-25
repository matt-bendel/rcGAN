import torch
import numpy as np

from utils.mri import transforms


def get_mask(resolution):
    m = np.zeros((resolution, resolution))
    a = np.array(
        [1, 23, 42, 60, 77, 92, 105, 117, 128, 138, 147, 155, 162, 169, 176, 182, 184, 185, 186, 187, 188, 189, 190,
         191, 192, 193, 194, 195,
         196, 197, 198, 199, 200, 204, 210, 217, 224, 231, 239, 248, 258, 269, 281, 294, 309, 326, 344, 363])
    m[:, a] = True

    samp = m
    numcoil = 8
    mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
    mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    return mask
