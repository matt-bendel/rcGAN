import torch
import numpy as np

class InpaintGANWrapper:
    def __init__(self, gen, cfg):
        self.cfg = cfg
        self.gen = gen

    def get_noise(self, num_vectors, device):
        return [torch.randn(num_vectors, 512, device=device)]

    def update_gen_status(self, val):
        self.gen.eval() if val else self.gen.train()

    def __call__(self, x, mask):
        # Data consistency baked into CoModGAN
        samples = self.gen(x, mask, self.get_noise(x.size(0), x.device), return_latents=False, truncation=None, truncation_latent=None)
        return samples
