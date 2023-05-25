import torch
import yaml
import os
import types
import json
import pathlib

import numpy as np
import matplotlib.patches as patches

from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN import rcGAN
from utils.mri.math import complex_abs, tensor_to_complex_np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sigpy as sp
from scipy import ndimage

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    with open('configs/mri.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MRIDataModule(cfg, big_test=True)
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        rcGAN_model = rcGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        rcGAN_model.cuda()

        rcGAN_model.eval()

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std, maps, fname, slice = data
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens_rcgan = torch.zeros(
                size=(y.size(0), cfg.num_z_test, cfg.in_chans // 2, cfg.im_size, cfg.im_size, 2)).cuda()

            for z in range(cfg.num_z_test):
                gens_rcgan[:, z, :, :, :, :] = rcGAN_model.reformat(rcGAN_model.forward(y, mask))

            avg_rcgan = torch.mean(gens_rcgan, dim=1)

            gt = rcGAN_model.reformat(x)
            zfr = rcGAN_model.reformat(y)

            for j in range(y.size(0)):
                np_avgs = {
                    'rcgan': None,
                }

                np_samps = {
                    'rcgan': [],
                }

                np_stds = {
                    'rcgan': None,
                }

                np_gt = None

                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), tensor_to_complex_np(maps[j].cpu()))

                np_gt = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)
                np_zfr = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((zfr[j] * std[j] + mean[j]).cpu())).abs().numpy(), 180)

                np_avgs['rcgan'] = ndimage.rotate(
                    torch.tensor(S.H * tensor_to_complex_np((avg_rcgan[j] * std[j] + mean[j]).cpu())).abs().numpy(),
                    180)

                for z in range(cfg.num_z_test):
                    np_samps['rcgan'].append(ndimage.rotate(torch.tensor(
                        S.H * tensor_to_complex_np((gens_rcgan[j, z] * std[j] + mean[j]).cpu())).abs().numpy(), 180))

                np_stds['rcgan'] = np.std(np.stack(np_samps['rcgan']), axis=0)

                method = 'rcgan'
                zoom_startx = np.random.randint(120, 250)
                zoom_starty1 = np.random.randint(30, 80)
                zoom_starty2 = np.random.randint(260, 300)

                p = np.random.rand()
                zoom_starty = zoom_starty1
                if p <= 0.5:
                    zoom_starty = zoom_starty2

                zoom_length = 80

                x_coord = zoom_startx + zoom_length
                y_coords = [zoom_starty, zoom_starty + zoom_length]

                # Global recon, error, std
                nrow = 1
                ncol = 4

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Truth")

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_avgs[method], cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(method)


                ax = plt.subplot(gs[0, 2])
                im = ax.imshow(2 * np.abs(np_avgs[method] - np_gt), cmap='jet', vmin=0,
                               vmax=np.max(np.abs(np_avgs['rcgan'] - np_gt)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Error")


                ax = plt.subplot(gs[0, 3])
                ax.imshow(np_stds[method], cmap='viridis', vmin=0, vmax=np.max(np_stds['rcgan']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Std. Dev.")

                plt.savefig(f'figures/mri/mri_fig_avg_err_std_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                nrow = 1
                ncol = 8

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np_gt, cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                ax1 = ax

                rect = patches.Rectangle((zoom_startx, zoom_starty), zoom_length, zoom_length, linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np_gt[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                          cmap='gray',
                          vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Truth')

                connection_path_1 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty],
                                                            [0, 0], coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_1)
                connection_path_2 = patches.ConnectionPatch([zoom_startx + zoom_length, zoom_starty + zoom_length], [0, zoom_length],
                                                            coordsA=ax1.transData,
                                                            coordsB=ax.transData, color='r')
                fig.add_artist(connection_path_2)

                ax = plt.subplot(gs[0, 2])
                ax.imshow(
                    np_avgs[method][zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('32-Avg.')

                ax = plt.subplot(gs[0, 3])
                avg = np.zeros((384, 384))
                for l in range(4):
                    avg += np_samps[method][l]

                avg = avg / 4

                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('4-Avg.')

                ax = plt.subplot(gs[0, 4])
                avg = np.zeros((384, 384))
                for l in range(2):
                    avg += np_samps[method][l]

                avg = avg / 2
                ax.imshow(
                    avg[zoom_starty:zoom_starty + zoom_length, zoom_startx:zoom_startx + zoom_length],
                    cmap='gray', vmin=0, vmax=0.7 * np.max(np_gt))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('2-Avg.')

                for samp in range(2):
                    ax = plt.subplot(gs[0, samp + 5])
                    ax.imshow(np_samps[method][samp][zoom_starty:zoom_starty + zoom_length,
                              zoom_startx:zoom_startx + zoom_length], cmap='gray', vmin=0,
                              vmax=0.7 * np.max(np_gt))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Sample {samp + 1}')


                ax = plt.subplot(gs[0, 7])
                ax.imshow(np_stds[method][zoom_starty:zoom_starty + zoom_length,
                          zoom_startx:zoom_startx + zoom_length], cmap='viridis', vmin=0,
                          vmax=np.max(np_stds['rcgan']))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Std. Dev.')

                plt.savefig(f'figures/mri/zoomed_avg_samps_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                if fig_count == args.num_figs:
                    exit()
                fig_count += 1
