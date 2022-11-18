import torch
import numpy as np
import matplotlib.pyplot as plt

from mri_utils import get_sense_operator
from mri_utils.math import tensor_to_complex_np
from models import load_best_gen
from metrics import psnr, ssim
from scipy import ndimage

class Plotter:
    def __init__(self, cfg, mri, plot_dir):
        self.cfg = cfg
        self.mri = mri
        self.plot_dir = plot_dir
        self.count = 0
        self.mod_num = 1

        self.G = load_best_gen(cfg, mri)

    def update_gen_status(self, val):
        self.G.update_gen_status(val=val)

    def generate_mri_plot(self, x, y, mean, std, mask):
        gens = self._get_P_recons(y, mask, mean, std, P=self.cfg.test.P)

        x = x * std[:, None, None, None] + mean[:, None, None, None]
        x_hat_avg = torch.mean(gens, dim=1)

        x_reshaped = torch.zeros(
            size=(y.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2),
            device=y.device)
        x_reshaped[:, :, :, :, 0] = x[:, 0:self.cfg.data.num_coils, :, :]
        x_reshaped[:, :, :, :, 1] = x[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        x_hat_avg_reshaped = torch.zeros(size=(y.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2), device=y.device)
        x_hat_avg_reshaped[:, :, :, :, 0] = x_hat_avg[:, 0:self.cfg.data.num_coils, :, :]
        x_hat_avg_reshaped[:, :, :, :, 1] = x_hat_avg[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        gens_np = np.zeros((gens.shape[1], self.cfg.data.resolution, self.cfg.data.resolution))

        for j in range(y.size(0)):
            if self.count % self.mod_num == 0:
                S = get_sense_operator(self.cfg, y[j] * std[j] + mean[j])

                for z in range(gens.shape[1]):
                    gen_reshaped = torch.zeros(
                        size=(self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2),
                        device=y.device)
                    gen_reshaped[:, :, :, 0] = gens[j, z, 0:self.cfg.data.num_coils, :, :]
                    gen_reshaped[:, :, :, 1] = gens[j, z, self.cfg.data.num_coils:self.cfg.data.num_coils * 2, :, :]
                    gen = tensor_to_complex_np((gen_reshaped).cpu())
                    gens_np[z] = torch.tensor(S.H * gen).abs().numpy()

                x_mc_np = tensor_to_complex_np((x_reshaped[j]).cpu())
                x_hat_mc_avg_np = tensor_to_complex_np((x_hat_avg_reshaped[j]).cpu())

                x_np = torch.tensor(S.H * x_mc_np).abs().numpy()
                x_hat_avg_np = torch.tensor(S.H * x_hat_mc_avg_np).abs().numpy()

                std_dev_np = np.std(gens_np, axis=0)

                fig = plt.figure()
                fig.subplots_adjust(wspace=0, hspace=0.05)
                plt.axis('off')
                self._generate_image(fig, x_np, x_np, 'GT', 1, 1, 4)
                self._generate_image(fig, x_np, x_hat_avg_np, 'rcGAN', 2, 1, 4)
                im_er, ax_er = self._generate_error_map(fig, x_np, x_hat_avg_np, 3, 1, 4)
                self._get_colorbar(fig, im_er, ax_er)

                im_std, ax_std = self._generate_image(fig, x_np, std_dev_np, 'Std. Dev', 4, 1, 4)
                self._get_colorbar(fig, im_std, ax_std)

                plt.savefig(f'{self.plot_dir}mri_recon_{self.count}',
                            bbox_inches='tight', dpi=300)
                plt.close(fig)

            self.count += 1

    def generate_inpaint_plot(self, x, y, mean, std, mask):
        gens = self._get_P_recons(None, mask, mean, std, self.cfg.test.P, x=x)

        x = x * std[:, :, None, None] + mean[:, :, None, None]
        y = y * std[:, :, None, None] + mean[:, :, None, None]

        for j in range(y.size(0)):
            if self.count % self.mod_num == 0:
                fig = plt.figure()
                plt.axis('off')
                plt.imshow(x[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                plt.savefig(f'{self.plot_dir}inpaint_recon_gt_{self.count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                fig = plt.figure()
                plt.axis('off')
                plt.imshow(y[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                plt.savefig(f'{self.plot_dir}inapint_recon_masked_{self.count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                fig = plt.figure()
                fig.subplots_adjust(wspace=0, hspace=0.05)

                for r in range(5):
                    ax = fig.add_subplot(1, 5, r + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(gens[j, r, :, :, :].cpu().numpy().transpose(1, 2, 0))

                plt.savefig(f'{self.plot_dir}inpaint_recon_rcgan_{self.count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

            self.count += 1

    def _get_P_recons(self, y, mask, mean, std, P=2, x=None):
        if x is not None:
            b_size = x.size(0)
        else:
            b_size = y.size(0)

        gens = torch.zeros(b_size, P, self.cfg.data.channels, self.cfg.data.resolution, self.cfg.data.resolution).cuda()

        for z in range(P):
            with torch.no_grad():
                if x is not None:
                    gens[:, z, :, :, :] = self.G(x, mask) * std[:, :, None, None] + mean[:, :, None, None]
                else:
                    gens[:, z, :, :, :] = self.G(y, mask) * std[:, None, None, None] + mean[:, None, None, None]

        return gens

    def _generate_image(self, fig, target, image, method, image_ind, rows, cols):
        # rows and cols are both previously defined ints
        ax = fig.add_subplot(rows, cols, image_ind)
        if method != 'GT' and method != 'Std. Dev':
            psnr_val = psnr(target, image)
            ssim_val = ssim(target, image, True)
            ax.text(0.46, 0.04, f'PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}',
                    horizontalalignment='center', verticalalignment='center', fontsize=3.5, color='yellow',
                    transform=ax.transAxes)

        if method == 'Std. Dev':
            im = ax.imshow(ndimage.rotate(image, 180), cmap='viridis', vmin=0, vmax=3e-5)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            im = ax.imshow(np.abs(ndimage.rotate(image, 180)), cmap='gray', vmin=0, vmax=np.max(target))
            ax.set_xticks([])
            ax.set_yticks([])

        return im, ax

    def _generate_error_map(self, fig, target, recon, image_ind, rows, cols, k=1.5, title=None):
        # Assume rows and cols are available globally
        # rows and cols are both previously defined ints
        ax = fig.add_subplot(rows, cols, image_ind)  # Add to subplot

        # Normalize error between target and reconstruction

        error = np.abs(target - recon)
        im = ax.imshow(ndimage.rotate(k * error, 180), cmap='jet', vmin=0, vmax=0.0001)

        if title != None:
            ax.set_title(title, size=10)
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Return plotted image and its axis in the subplot
        return im, ax

    def _get_colorbar(self, fig, im, ax, left=False, top=False):
        # Get position of final error map axis
        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

        # Appropriately rescale final axis so that colorbar does not effect formatting
        pad = 0.01
        width = 0.02
        cbar_ax = fig.add_axes(
            [x10, y11 + pad, x11 - x10, width])  # if not left else fig.add_axes([x10 - 2*pad, y10, width, y11 - y10])
        cbar = fig.colorbar(im, cax=cbar_ax, format='%.0e', orientation='horizontal')  # Generate colorbar
        cbar.ax.locator_params(nbins=3)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.tick_params(rotation=90)
        tl = cbar.ax.get_xticklabels()

        tl[0].set_horizontalalignment('left')
        tl[-1].set_horizontalalignment('right')

        return cbar