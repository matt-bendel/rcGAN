import torch
import numpy as np

from mri_utils.fftc import ifft2c_new, fft2c_new

class MRIGANWrapper:
    def __init__(self, gen, cfg):
        self.cfg = cfg
        self.gen = gen

    def get_noise(self, num_vectors):
        return torch.randn(num_vectors, 2, self.cfg.data.resolution, self.cfg.data.resolution).cuda()

    def update_gen_status(self, val):
        self.gen.eval() if val else self.gen.train()

    def reformat(self, samples):
        reformatted_tensor = torch.zeros(size=(samples.size(0), self.cfg.data.num_coils, self.cfg.data.resolution, self.cfg.data.resolution, 2),
                                         device=samples.device)
        reformatted_tensor[:, :, :, :, 0] = samples[:, 0:self.cfg.data.num_coils, :, :]
        reformatted_tensor[:, :, :, :, 1] = samples[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

        return reformatted_tensor

    def data_consistency(self, samples, y, mask):
        reformatted_samples = self.reformat(samples)
        kspace_samples = fft2c_new(reformatted_samples)

        reformatted_y = self.reformat(y)
        kspace_y = fft2c_new(reformatted_y)

        data_consistent_kspace = kspace_y + (1 - mask) * kspace_samples

        image = ifft2c_new(data_consistent_kspace)

        output_im = torch.zeros(size=samples.shape, device=y.device)
        output_im[:, 0:self.cfg.data.num_coils, :, :] = image[:, :, :, :, 0]
        output_im[:, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :] = image[:, :, :, :, 1]

        return output_im

    def __call__(self, y, mask):
        num_vectors = y.size(0)
        z = self.get_noise(num_vectors)
        samples = self.gen(torch.cat([y, z], dim=1))

        return self.data_consistency(samples, y, mask)
