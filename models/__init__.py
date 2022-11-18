import torch
import pathlib

from generator_wrappers import get_gen_wrapper
from .inpainting.comodgan.co_mod_gan import CoModGANGenerator, CoModGANDisc
from .mri.generator import MRIGenerator
from .mri.discriminator import MRIDiscriminator

def get_gan(cfg, mri, resume):
    if resume:
        return get_gan_resume(cfg, mri)
    else:
        return get_gan_fresh(cfg, mri)

def get_gan_fresh(cfg, mri):
    generator = build_generator(cfg, mri)
    discriminator = build_discriminator(cfg, mri)

    if cfg.data_parallel:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    generator = get_gen_wrapper(mri, cfg, generator)

    # Optimizers
    opt_gen = build_optim(cfg, generator.gen.parameters())
    opt_disc = build_optim(cfg, discriminator.parameters())

    best_loss = 100000
    start_epoch = 0

    return generator, discriminator, opt_gen, opt_disc, best_loss, start_epoch

def get_gan_resume(cfg, mri):
    checkpoint_file_gen = pathlib.Path(
        f'{cfg.checkpoint_dir}/generator_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    checkpoint_file_dis = pathlib.Path(
        f'{cfg.checkpoint_dir}/discriminator_model.pt')
    checkpoint_disc = torch.load(checkpoint_file_dis, map_location=torch.device('cuda'))

    generator = build_generator(cfg, mri)
    discriminator = build_discriminator(cfg, mri)

    if cfg.data_parallel:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    generator.load_state_dict(checkpoint_gen['model'])
    discriminator.load_state_dict(checkpoint_disc['model'])

    generator = get_gen_wrapper(mri, cfg, generator)

    opt_gen = build_optim(cfg, generator.gen.parameters())
    opt_disc = build_optim(cfg, discriminator.parameters())

    opt_gen.load_state_dict(checkpoint_gen['optimizer'])
    opt_disc.load_state_dict(checkpoint_disc['optimizer'])

    best_loss = checkpoint_gen['best_loss']
    start_epoch = checkpoint_gen['epoch']

    return generator, discriminator, opt_gen, opt_disc, best_loss, start_epoch

def load_best_gen(cfg, mri):
    checkpoint_file_gen = pathlib.Path(
        f'{cfg.checkpoint_dir}/generator_best_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_generator(cfg, mri)

    if cfg.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    generator = get_gen_wrapper(mri, cfg, generator)

    return generator

def build_discriminator(cfg, mri):
    if mri:
        model = MRIDiscriminator(
            input_nc=cfg.data.channels * 2,
        ).to(torch.device('cuda'))
    else:
        model = CoModGANDisc(cfg.data.resolution).to(torch.device('cuda'))

    return model

def build_generator(cfg, mri):
    if mri:
        model = MRIGenerator(
            in_chans=cfg.data.channels + 2,
            out_chans=cfg.data.channels,
        ).to(torch.device('cuda'))
    else:
        model = CoModGANGenerator(cfg.data.resolution).to(torch.device('cuda'))

    return model

def build_optim(cfg, params):
    return torch.optim.Adam(params, lr=cfg.optimizer.lr, betas=(cfg.optimizer.beta_1, cfg.optimizer.beta_2))