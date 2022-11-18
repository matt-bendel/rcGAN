import torch
import numpy as np
import time

from mri_utils import get_sense_operator
from mri_utils.math import tensor_to_complex_np
from models import load_best_gen
from metrics import psnr, ssim

class Tester:
    def __init__(self, cfg, mri):
        self.cfg = cfg
        self.mri = mri

        self.G = load_best_gen(cfg, mri)

    def update_gen_status(self, val):
        self.G.update_gen_status(val=val)

    def test_batch_mri(self, x, y, mean, std, mask):
        losses = {
            'psnr': [],
            'ssim': []
        }

        gens, t = self._get_P_recons(y, mask, P=self.cfg.test.P)

        x = x * std[:, None, None, None] + mean[:, None, None, None]
        x_hat_avg = torch.mean(gens, dim=1) * std[:, None, None, None]+ mean[:, None, None, None]

        x_reshaped = torch.zeros(
            size=(y.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2),
            device=y.device)
        x_reshaped[:, :, :, :, 0] = x[:, 0:self.cfg.data.num_coils, :, :]
        x_reshaped[:, :, :, :, 1] = x[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        x_hat_avg_reshaped = torch.zeros(size=(y.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2), device=y.device)
        x_hat_avg_reshaped[:, :, :, :, 0] = x_hat_avg[:, 0:self.cfg.data.num_coils, :, :]
        x_hat_avg_reshaped[:, :, :, :, 1] = x_hat_avg[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        for j in range(y.size(0)):
            S = get_sense_operator(self.cfg, y[j] * std[j] + mean[j])

            x_mc_np = tensor_to_complex_np((x_reshaped[j]).cpu())
            x_hat_mc_avg_np = tensor_to_complex_np((x_hat_avg_reshaped[j]).cpu())

            x_np = torch.tensor(S.H * x_mc_np).abs().numpy()
            x_hat_avg_np = torch.tensor(S.H * x_hat_mc_avg_np).abs().numpy()

            losses['psnr'].append(psnr(x_np, x_hat_avg_np))
            losses['ssim'].append(ssim(x_np, x_hat_avg_np, mri=True))

        return np.mean(losses['psnr']), np.mean(losses['ssim']), t

    def test_batch_inpaint(self, x, y, mean, std, mask):
        losses = {
            'psnr': [],
            'ssim': [],
        }

        gens, t = self._get_P_recons(None, mask, self.cfg.test.P, x=x)
        apsd = torch.std(gens, dim=1).mean().cpu().numpy()

        avg = torch.mean(gens, dim=1) * std[:, :, None, None] + mean[:, :, None, None]
        x = x * std[:, :, None, None] + mean[:, :, None, None]

        for j in range(y.size(0)):
            losses['ssim'].append(ssim(x[j].cpu().numpy(), avg[j].cpu().numpy(), mri=False))
            losses['psnr'].append(psnr(x[j].cpu().numpy(), avg[j].cpu().numpy()))

        return np.mean(losses['psnr']), np.mean(losses['ssim']), apsd, t

    def _get_P_recons(self, y, mask, P=2, x=None):
        if x is not None:
            b_size = x.size(0)
        else:
            b_size = y.size(0)

        gens = torch.zeros(b_size, P, self.cfg.data.channels, self.cfg.data.resolution, self.cfg.data.resolution).cuda()

        times = []
        for z in range(P):
            start = time.time()
            with torch.no_grad():
                if x is not None:
                    gens[:, z, :, :, :] = self.G(x, mask)
                else:
                    gens[:, z, :, :, :] = self.G(y, mask)
            times.append(time.time() - start)

        return gens, np.mean(times)