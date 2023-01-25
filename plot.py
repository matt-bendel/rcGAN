import yaml
import torch
import random
import os
import argparse

import numpy as np

from parse_args import create_arg_parser
from data_loaders import create_data_loaders_test
from runners.plotter import Plotter

def plot(args):
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
    plotter = Plotter(cfg, args.is_mri, args.plot_dir)

    _, test_loader = create_data_loaders_test(args.is_mri, cfg)
    plotter.update_gen_status(val=True)

    print(f"\nGenerating {args.num_plots} plots")
    plotter.mod_num = len(test_loader.dataset) // args.num_plots

    for i, data in enumerate(test_loader):
        x, y, mean, std, mask = data[0]
        y = y.cuda()
        x = x.cuda()
        mean = mean.cuda()
        std = std.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            if args.is_mri:
                plotter.generate_mri_plot(x, y, mean, std, mask)
            else:
                plotter.generate_inpaint_plot(x, y, mean, std, mask)

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

    plot(args)