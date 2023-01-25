import yaml
import torch
import random
import os
import argparse

import numpy as np

from parse_args import create_arg_parser
from data_loaders import create_data_loaders_test
from runners.tester import Tester
from metrics import cfid, fid

def test(args):
    losses = {
        'psnr': [],
        'ssim': [],
        'time': []
    }

    if args.is_mri:
        with open(os.path.join('config', 'mri.yml'), 'r') as f:
            cfg = yaml.load(f)
    else:
        with open(os.path.join('config', 'inpaint.yml'), 'r') as f:
            cfg = yaml.load(f)

    cfg = dict2namespace(cfg)
    tester = Tester(cfg, args.is_mri)

    ref_loader, test_loader = create_data_loaders_test(args.is_mri, cfg)
    tester.update_gen_status(val=True)

    print("\nComputing CFID...")
    cfid_val = cfid(cfg, tester.G, test_loader, args.is_mri)
    print("\nComputing FID...")
    fid_val = fid(cfg, tester.G, ref_loader, test_loader, args.is_mri)

    print("\nComputing PSNR and SSIM")
    for i, data in enumerate(test_loader):
        x, y, mean, std, mask = data[0]
        y = y.cuda()
        x = x.cuda()
        mean = mean.cuda()
        std = std.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            if args.is_mri:
                psnr, ssim, time = tester.test_batch_mri(x, y, mean, std, mask)
            else:
                psnr, ssim, time = tester.test_batch_inpaint(x, y, mean, std, mask)

        losses['psnr'].append(psnr)
        losses['ssim'].append(ssim)
        losses['time'].append(time)

    print("Test Results:")
    print(f"CFID: {cfid_val:.2f}\nFID: {fid_val:.2f}\n{cfg.test.P}-PSNR: {np.mean(losses['psnr']):.2f}\n{cfg.test.P}-SSIM: {np.mean(losses['ssim']):.4f}\nTIME({cfg.test.batch_size}): {np.mean(losses['time'])}")

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

    test(args)
