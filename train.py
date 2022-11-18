import random
import torch
import yaml
import argparse
import os
import pathlib

import numpy as np

from runners.trainer import Trainer
from data_loaders import create_data_loaders_train
from parse_args import create_arg_parser
from metrics import cfid

def train(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if args.is_mri:
        with open(os.path.join('config', 'mri.yml'), 'r') as f:
            cfg = yaml.load(f)
    else:
        with open(os.path.join('config', 'inpaint.yml'), 'r') as f:
            cfg = yaml.load(f)

    cfg = dict2namespace(cfg)

    cfg.checkpoint_dir = pathlib.Path(cfg.checkpoint_dir)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg, args.is_mri, args.resume)

    std_mults = [trainer.beta_std_mult]
    psnr_diffs = []

    if args.resume:
        std_mults = []
        psnr_diffs = []

        with open("std_weights.txt", "r") as file1:
            for line in file1.readlines():
                for i in line.split(","):
                    std_mults.append(float(i.strip().replace('[', '').replace(']', '').replace(' ', '')))

        with open("psnr_diffs.txt", "r") as file1:
            for line in file1.readlines():
                for i in line.split(","):
                    psnr_diffs.append(float(i.strip().replace('[', '').replace(']', '').replace(' ', '')))

        trainer.beta_std_mult = std_mults[-1]

    if args.resume:
        trainer.start_epoch += 1

    train_loader, dev_loader = create_data_loaders_train(args.is_mri, cfg)

    for epoch in range(trainer.start_epoch, cfg.train.n_epochs):
        trainer.update_gen_status(val=False)
        for i, data in enumerate(train_loader):
            x, y, _, _, mask = data[0]
            y = y.to(args.device)
            x = x.to(args.device)
            mask = mask.to(args.device)

            for j in range(cfg.train.num_iters_discriminator):
                d_loss = trainer.discriminator_update(x, y, mask)

            g_loss = trainer.generator_update(x, y, mask)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch + 1, cfg.train.n_epochs, i, len(train_loader.dataset) / cfg.train.batch_size, d_loss,
                   g_loss)
            )

        losses = {
            'psnr_1': [],
            'psnr_8': []
        }

        trainer.update_gen_status(val=True)
        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                with torch.no_grad():
                    x, y, mean, std, mask = data[0]
                    y = y.to(args.device)
                    x = x.to(args.device)
                    mean = mean.to(args.device)
                    std = std.to(args.device)
                    mask = mask.to(args.device)

                    if args.is_mri:
                        psnr_8, psnr_1 = trainer.validate_mri(x, y, mean, std, mask)
                    else:
                        psnr_8, psnr_1 = trainer.validate_inpaint(x, y, mean, std, mask)

                    # TODO: Implement this
                    if args.train_gif:
                        pass

                    losses['psnr_1'].append(psnr_1)
                    losses['psnr_8'].append(psnr_8)

        psnr_diff = (np.mean(losses['psnr_1']) + 2.5) - np.mean(losses['psnr_8'])
        trainer.beta_std_mult = trainer.beta_std_mult + cfg.validate.mu_std * psnr_diff

        CFID = cfid(cfg, trainer.G, dev_loader, args.is_mri)

        best_model = CFID < trainer.best_loss and (np.abs(psnr_diff) <= cfg.validate.psnr_threshold)
        trainer.best_loss = CFID if best_model else trainer.best_loss

        trainer.save_model(best_model, epoch)

        std_mults.append(trainer.beta_std_mult)
        psnr_diffs.append(psnr_diff)
        file = open("std_weights.txt", "w+")

        # Saving the 2D array in a text file
        content = str(std_mults)
        file.write(content)
        file.close()

        file = open("psnr_diffs.txt", "w+")

        # Saving the 2D array in a text file
        content = str(psnr_diffs)
        file.write(content)
        file.close()

        print(f"END OF EPOCH {epoch + 1}: \n")
        print(f"[Validation 8-PSNR: {np.mean(losses['psnr_8']):.2f}] [Validation CFID: {CFID:.2f}]")

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = True

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train(args)
