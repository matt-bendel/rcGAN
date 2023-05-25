import os
import torch
import random
import pickle
import yaml
import json
import types

import numpy as np
import pytorch_lightning as pl

from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.lightning.rcGAN import rcGAN
from pytorch_lightning import seed_everything
import sigpy as sp
import sigpy.mri as mr
from utils.mri.fftc import ifft2c_new, fft2c_new
from utils.mri.math import complex_abs, tensor_to_complex_np

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()

    seed_everything(1, workers=True)

    with open('configs/mri.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MRIDataModule(cfg, big_test=True)
    dm.setup()

    # TODO: Automatically create sense map dir

    if args.sense_maps_val:
        loader = dm.val_dataloader()
        maps_path = cfg.sense_maps_path + "val_full_res"
    else:
        loader = dm.test_dataloader()
        maps_path = cfg.sense_maps_path + "test_full_res"

    for i, data in enumerate(loader):
        y, x, mask, mean, std, maps, fname, slice = data
        new_y = torch.zeros(y.size(0), 8, 384, 384, 2)
        new_y[:, :, :, :, 0] = y[:, 0:8, :, :]
        new_y[:, :, :, :, 1] = y[:, 8:16, :, :]

        for j in range(y.size(0)):
            new_y_true = fft2c_new(new_y[j] * std[j] + mean[j])
            maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y_true.cpu()), calib_width=cfg.calib_width,
                                       device=sp.Device(0), crop=0.70,
                                       kernel_width=6).run().get()

            with open(f'{maps_path}/{fname[j]}_{slice[j]}.pkl', 'wb') as outp:
                pickle.dump(maps, outp, pickle.HIGHEST_PROTOCOL)
