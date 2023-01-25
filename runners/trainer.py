import shutil
import torch
import numpy as np
import torch.autograd as autograd

from mri_utils import get_sense_operator
from mri_utils.math import tensor_to_complex_np
from models import get_gan
from metrics import psnr
from torch.nn import functional as F

class Trainer:
    def __init__(self, cfg, mri, resume, rank, world_size):
        self.cfg = cfg
        self.mri = mri
        self.beta_std_mult = 1
        self.beta_std_0 = np.sqrt(2 / (np.pi * cfg.train.P * (cfg.train.P + 1)))

        self.G, self.D, self.G_opt, self.D_opt, self.best_loss, self.start_epoch = get_gan(cfg, mri, resume, rank, world_size)

    def update_gen_status(self, val):
        self.G.update_gen_status(val=val)

    def generator_update(self, x, y, mask):
        # More efficient way to zero gradient
        for param in self.G.gen.parameters():
            param.grad = None

        if self.mri:
            gens = self._get_P_recons(y, mask, P=self.cfg.train.P)
        else:
            gens = self._get_P_recons(y, mask, P=self.cfg.train.P, x=x)

        fake_pred = []
        for k in range(y.shape[0]):
            fake_pred.append(self.D(gens[k], y[k].unsqueeze(0).repeat(self.cfg.train.P, 1, 1, 1)))

        fake_pred = torch.cat(fake_pred, dim=0)
        x_hat_avg = torch.mean(gens, dim=1)

        g_loss = - self.cfg.train.adv_weight * fake_pred.mean()
        g_loss += self.cfg.train.l1_weight * F.l1_loss(x_hat_avg, x)
        g_loss += - self.beta_std_mult * self.beta_std_0 * torch.mean(torch.std(gens, dim=1), dim=(0, 1, 2, 3))

        g_loss.backward()
        self.G_opt.step()

        return g_loss.item()

    def discriminator_update(self, x, y, mask):
        # More efficient way to zero gradient
        self.D.train()
        for param in self.D.parameters():
            param.grad = None

        if self.mri:
            x_hat = self.G(y, mask)
        else:
            x_hat = self.G(x, mask)

        real_pred = self.D(x, y)
        real_pred_sq = real_pred ** 2

        fake_pred = self.D(x_hat, y)

        d_loss = fake_pred.mean() - real_pred.mean() # Wasserstein Loss
        d_loss += self.cfg.train.gp_weight * self._compute_gradient_penalty(x.data, x_hat.data, y.data) # Gradient Penalty
        d_loss += self.cfg.train.drift_weight * real_pred_sq.mean() # Drift loss

        d_loss.backward()
        self.D_opt.step()
        self.D.eval()

        return d_loss.item()

    def validate_mri(self, x, y, mean, std, mask, rank):
        losses = {
            'psnr_1': [],
            'psnr_8': [],
        }

        gens = self._get_P_recons(y, mask, P=self.cfg.validate.P)

        x = x * std[:, None, None, None] + mean[:, None, None, None]
        x_hat_1 = gens[:, 0, :, :, :] * std[:, None, None, None] + mean[:, None, None, None]
        x_hat_avg = torch.mean(gens, dim=1) * std[:, None, None, None] + mean[:, None, None, None]

        x_reshaped = torch.zeros(
            size=(y.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2),
            device=y.device)
        x_reshaped[:, :, :, :, 0] = x[:, 0:self.cfg.data.num_coils, :, :]
        x_reshaped[:, :, :, :, 1] = x[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        x_hat_1_reshaped = torch.zeros(
            size=(y.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2),
            device=y.device)
        x_hat_1_reshaped[:, :, :, :, 0] = x_hat_1[:, 0:self.cfg.data.num_coils, :, :]
        x_hat_1_reshaped[:, :, :, :, 1] = x_hat_1[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        x_hat_avg_reshaped = torch.zeros(size=(y.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2), device=y.device)
        x_hat_avg_reshaped[:, :, :, :, 0] = x_hat_avg[:, 0:self.cfg.data.num_coils, :, :]
        x_hat_avg_reshaped[:, :, :, :, 1] = x_hat_avg[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        for j in range(y.size(0)):
            S = get_sense_operator(self.cfg, y[j] * std[j] + mean[j], rank)

            x_mc_np = tensor_to_complex_np((x_reshaped[j]).cpu())
            x_mc_hat_1 = tensor_to_complex_np((x_hat_avg_reshaped[j]).cpu())
            x_hat_mc_avg_np = tensor_to_complex_np((x_hat_avg_reshaped[j]).cpu())

            x_np = torch.tensor(S.H * x_mc_np).abs().numpy()
            x_hat_1_np = torch.tensor(S.H * x_mc_hat_1).abs().numpy()
            x_hat_avg_np = torch.tensor(S.H * x_hat_mc_avg_np).abs().numpy()

            losses['psnr_8'].append(psnr(x_np, x_hat_avg_np))
            losses['psnr_1'].append(psnr(x_np, x_hat_1_np))

        return np.mean(losses['psnr_8']), np.mean(losses['psnr_1'])

    def validate_inpaint(self, x, y, mean, std, mask):
        losses = {
            'psnr_1': [],
            'psnr_8': [],
        }

        gens = self._get_P_recons(y, mask, P=self.cfg.validate.P, x=x)

        avg = torch.mean(gens, dim=1) * std[:, :, None, None] + mean[:, :, None, None]
        x = x * std[:, :, None, None] + mean[:, :, None, None]

        for j in range(y.size(0)):
            losses['psnr_8'].append(psnr(x[j].cpu().numpy(), avg[j].cpu().numpy()))
            losses['psnr_1'].append(
                psnr(x[j].cpu().numpy(), (gens[j, 0] * std[j, :, None, None] + mean[j, :, None, None]).cpu().numpy()))

        return np.mean(losses['psnr_8']), np.mean(losses['psnr_1'])

    def save_model(self, is_best, epoch):
        self._save_generator_weights(is_best, epoch)
        self._save_discriminator_weights(epoch)

    def _save_generator_weights(self, is_best, epoch):
        torch.save(
            {
                'epoch': epoch,
                'cfg': self.cfg,
                'model': self.G.gen.state_dict(),
                'optimizer': self.G_opt.state_dict(),
                'best_loss': self.best_loss,
            },
            f=self.cfg.checkpoint_dir / 'generator_model.pt'
        )

        if is_best:
            shutil.copyfile(fpath / 'generator_model.pt',
                            fpath / 'generator_best_model.pt')

    def _save_discriminator_weights(self, epoch):
        torch.save(
            {
                'epoch': epoch,
                'cfg': self.cfg,
                'model': self.D.state_dict(),
                'optimizer': self.D_opt.state_dict(),
            },
            f=self.cfg.checkpoint_dir / 'discriminator_model.pt'
        )

    def _get_P_recons(self, y, mask, P=2, x=None):
        gens = torch.zeros(y.size(0), P, self.cfg.data.channels, self.cfg.data.resolution, self.cfg.data.resolution).cuda()

        for z in range(P):
            if x is not None:
                gens[:, z, :, :, :] = self.G(x, mask)
            else:
                gens[:, z, :, :, :] = self.G(y, mask)

        return gens


    def _compute_gradient_penalty(self, x, x_hat, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((x.size(0), 1, 1, 1))).to(x.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * x + ((1 - alpha) * x_hat)).requires_grad_(True)
        d_interpolates = self.D(interpolates, y)
        if self.mri:
            fake = Tensor(x.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(x.device)
        else:
            fake = Tensor(x.shape[0], 1).fill_(1.0).to(x.device)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty