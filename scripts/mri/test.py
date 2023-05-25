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

    with open('configs/mri.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    cfg.batch_size = cfg.batch_size * 4
    dm = MRIDataModule(cfg, big_test=True)

    dm.setup()

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    lpips_met = lpips.LPIPS(net='alex')
    dists_met = DISTS()

    with torch.no_grad():
        model = rcGAN.load_from_checkpoint(
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
            print(f'PSNR: {np.mean(psnrs):.2f} \pm {np.std(psnrs) / np.sqrt(len(psnrs)):.2f}')
            print(f'SSIM: {np.mean(ssims):.4f} \pm {np.std(ssims) / np.sqrt(len(ssims)):.4f}')
            print(f'LPIPS: {np.mean(lpipss):.4f} \pm {np.std(lpipss) / np.sqrt(len(lpipss)):.4f}')
            print(f'DISTS: {np.mean(distss):.4f} \pm {np.std(distss) / np.sqrt(len(distss)):.4f}')
            print(f'APSD: {np.mean(apsds):.1f}')

    cfids = []
    m_comps = []
    c_comps = []

    inception_embedding = VGG16Embedding(parallel=True)
    # CFID_1
    cfid_metric = CFIDMetric(gan=model,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=32)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    # CFID_2
    cfid_metric = CFIDMetric(gan=model,
                             loader=val_dataloader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=8)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    # CFID_3
    cfid_metric = CFIDMetric(gan=model,
                             loader=val_dataloader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=train_dataloader,
                             num_samps=1)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    print("\n\n")
    for l in range(3):
        print(f'CFID_{l+1}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')
