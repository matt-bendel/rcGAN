import torch
import yaml
import os
import types
import json
import pathlib
import lpips

import numpy as np

from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN import rcGAN
from utils.mri.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.metrics import psnr, ssim
from utils.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from utils.mri.fftc import ifft2c_new, fft2c_new
import sigpy as sp
import sigpy.mri as mr
from utils.mri.transforms import to_tensor
from DISTS_pytorch import DISTS

def load_object(dct):
    return types.SimpleNamespace(**dct)


def rgb(im, unit_norm=False):
    embed_ims = torch.zeros(size=(3, 384, 384))
    tens_im = torch.tensor(im)

    if unit_norm:
        tens_im = (tens_im - torch.min(tens_im)) / (torch.max(tens_im) - torch.min(tens_im))
    else:
        tens_im = 2 * (tens_im - torch.min(tens_im)) / (torch.max(tens_im) - torch.min(tens_im)) - 1

    embed_ims[0, :, :] = tens_im
    embed_ims[1, :, :] = tens_im
    embed_ims[2, :, :] = tens_im

    return embed_ims.unsqueeze(0)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.mask_type = 1

    if args.default_model_descriptor:
        args.num_noise = 1

    if args.mri:
        with open('configs/mri.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        cfg.batch_size = cfg.batch_size * 4
        dm = MRIDataModule(cfg, args.mask_type, big_test=True)

        dm.setup()
        test_loader = dm.test_dataloader()
        if args.rcgan:
            model_alias = rcGAN
        else:
            model_alias = L1SSIMMRI
    else:
        print("No valid application selected. Please include one of the following args: --mri")
        exit()

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    lpips_met = lpips.LPIPS(net='alex')
    dists_met = DISTS()

    with torch.no_grad():
        model = model_alias.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        model.cuda()
        model.eval()

        n_samps = [1, 2, 4, 8, 16, 32]

        for n in n_samps:
            print(f"\n\n{n} SAMPLES")
            psnrs = []
            ssims = []
            apsds = []
            lpipss = []
            distss = []

            for i, data in enumerate(test_loader):
                y, x, mask, mean, std, maps, _, _ = data
                y = y.cuda()
                x = x.cuda()
                mask = mask.cuda()
                mean = mean.cuda()
                std = std.cuda()

                gens = torch.zeros(size=(y.size(0), n, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()
                for z in range(n):
                    gens[:, z, :, :, :, :] = model.reformat(model.forward(y, mask))

                avg = torch.mean(gens, dim=1)

                gt = model.reformat(x)

                for j in range(y.size(0)):
                    single_samps = np.zeros((n, cfg.im_size, cfg.im_size))

                    S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))
                    gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                        (avg[j] * std[j] + mean[j]).cpu())

                    avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                    gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                    for z in range(n):
                        np_samp = tensor_to_complex_np((gens[j, z, :, :, :, :] * std[j] + mean[j]).cpu())
                        single_samps[z, :, :] = torch.tensor(S.H * np_samp).abs().numpy()

                    med_np = np.median(single_samps, axis=0)

                    apsds.append(np.mean(np.std(single_samps, axis=0), axis=(0, 1)))
                    psnrs.append(psnr(gt_np, avg_gen_np))
                    ssims.append(ssim(gt_np, avg_gen_np))
                    lpipss.append(lpips_met(rgb(gt_np), rgb(avg_gen_np)).numpy())
                    distss.append(dists_met(rgb(gt_np, unit_norm=True), rgb(avg_gen_np, unit_norm=True)).numpy())

            print('AVG Recon')
            print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
            print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
            print(f'LPIPS: {np.mean(lpipss)} \pm {np.std(lpipss) / np.sqrt(len(lpipss))}')
            print(f'DISTS: {np.mean(distss)} \pm {np.std(distss) / np.sqrt(len(distss))}')
            print(f'APSD: {np.mean(apsds)}')
