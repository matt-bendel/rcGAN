import torch

import sigpy as sp
import sigpy.mri as mr

from .fftc import ifft2c_new, fft2c_new
from .math import tensor_to_complex_np

def get_sense_operator(cfg, y, rank):
    new_y = torch.zeros(
        size=(cfg.data.num_coils, cfg.data.resolution, cfg.data.resolution, 2),
        device=y.device)
    new_y[:, :, :, 0] = y[0:cfg.data.num_coils, :, :]
    new_y[:, :, :, 1] = y[cfg.data.num_coils:cfg.data.num_coils*2, :, :]
    new_y = fft2c_new(new_y)

    maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y.cpu()), calib_width=cfg.sense.calib_width,
                               device=sp.Device(rank), show_pbar=False, crop=cfg.sense.crop,
                               kernel_width=cfg.sense.kernel_width).run().get()
    S = sp.linop.Multiply((cfg.data.resolution, cfg.data.resolution), maps)

    return S