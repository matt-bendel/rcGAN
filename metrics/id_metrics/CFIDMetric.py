# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch

import numpy as np
import sigpy as sp
import sigpy.mri as mr

import torchvision.transforms as transforms
from mri_utils.fftc import fft2c_new, ifft2c_new
from mri_utils.math import complex_abs, tensor_to_complex_np
from tqdm import tqdm
from mri_utils import get_sense_operator

def symmetric_matrix_square_root_torch(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    u, s, v = torch.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = s
    si[torch.where(si >= eps)] = torch.sqrt(si[torch.where(si >= eps)])

    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return torch.matmul(torch.matmul(u, torch.diag(si)), v)


def trace_sqrt_product_torch(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root_torch(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))

    return torch.trace(symmetric_matrix_square_root_torch(sqrt_a_sigmav_a))


class CFIDMetric:
    """Helper function for calculating CFID metric.

    Note: This code is adapted from Facebook's FJD implementation in order to compute
    CFID in a streamlined fashion.

    """
    def __init__(self,
                 cfg,
                 gan,
                 loader,
                 image_embedding,
                 condition_embedding,
                 mri,
                 cuda=False):
        self.cfg = cfg
        self.gan = gan
        self.loader = loader
        self.image_embedding = image_embedding
        self.condition_embedding = condition_embedding
        self.mri = mri
        self.cuda = cuda
        self.num_samps = cfg.test.cfid_P
        self.vgg_transforms = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def _get_embed_im_mri(self, multi_coil_inp, mean, std, maps):
        embed_ims = torch.zeros(size=(multi_coil_inp.size(0), 3, 384, 384),
                                device=multi_coil_inp.device)
        for i in range(multi_coil_inp.size(0)):
            reformatted = torch.zeros(size=(self.cfg.data.num_coils, 384, 384, 2),
                                      device=multi_coil_inp.device)
            reformatted[:, :, :, 0] = multi_coil_inp[i, 0:self.cfg.data.num_coils, :, :]
            reformatted[:, :, :, 1] = multi_coil_inp[i, self.cfg.data.num_coils:self.cfg.data.num_coils*2, :, :]

            unnormal_im = tensor_to_complex_np((reformatted * std[i] + mean[i]).cpu())

            im = torch.tensor(maps[i].H * unnormal_im).abs()

            im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))

            embed_ims[i, 0, :, :] = im
            embed_ims[i, 1, :, :] = im
            embed_ims[i, 2, :, :] = im

        return self.vgg_transforms(embed_ims)

    def _get_embed_im_inpaint(self, inp, mean, std, maps):
        embed_ims = torch.zeros(size=(inp.size(0), 3, 128, 128),
                                device=inp.device)
        for i in range(inp.size(0)):
            im = inp[i, :, :, :] * std[i, :, None, None] + mean[i, :, None, None]
            im = 2 * (im - torch.min(im)) / (torch.max(im) - torch.min(im)) - 1
            embed_ims[i, :, :, :] = im

        return embed_ims

    def _get_embed_im(self, inp, mean, std, maps):
        if self.mri:
            return self._get_embed_im_mri(inp, mean, std, maps)
        else:
            return self._get_embed_im_inpaint(inp, mean, std, maps)

    def _get_generated_distribution(self):
        image_embed = []
        cond_embed = []
        true_embed = []

        for i, data in tqdm(enumerate(self.loader),
                            desc='Computing generated distribution',
                            total=len(self.loader)):
            gt, condition, mean, std, mask = data[0]
            if self.cuda:
                condition = condition.cuda()
                gt = gt.cuda()
                mean = mean.cuda()
                std = std.cuda()
                mask = mask.cuda()

            maps = []

            with torch.no_grad():
                if self.mri:
                    for j in range(condition.shape[0]):
                        maps.append(get_sense_operator(self.cfg, condition[j] * std[j] + mean[j]))

                for l in range(self.num_samps):
                    if self.mri:
                        recon = self.gan(condition, mask)
                    else:
                        recon = self.gan(gt, mask)

                    image = self._get_embed_im(recon, mean, std, maps)
                    condition_im = self._get_embed_im(condition, mean, std, maps)
                    true_im = self._get_embed_im(gt, mean, std, maps)

                    img_e = self.image_embedding(image)
                    cond_e = self.condition_embedding(condition_im)
                    true_e = self.image_embedding(true_im)

                    if self.cuda:
                        true_embed.append(true_e)
                        image_embed.append(img_e)
                        cond_embed.append(cond_e)
                    else:
                        true_embed.append(true_e.cpu().numpy())
                        image_embed.append(img_e.cpu().numpy())
                        cond_embed.append(cond_e.cpu().numpy())

        if self.cuda:
            true_embed = torch.cat(true_embed, dim=0)
            image_embed = torch.cat(image_embed, dim=0)
            cond_embed = torch.cat(cond_embed, dim=0)
        else:
            true_embed = np.concatenate(true_embed, axis=0)
            image_embed = np.concatenate(image_embed, axis=0)
            cond_embed = np.concatenate(cond_embed, axis=0)

        return image_embed.to(dtype=torch.float64), cond_embed.to(dtype=torch.float64), true_embed.to(
            dtype=torch.float64)

    def get_cfid_torch(self):
        y_predict, x_true, y_true = self._get_generated_distribution()

        # mean estimations
        y_true = y_true.to(x_true.device)
        m_y_predict = torch.mean(y_predict, dim=0)
        m_x_true = torch.mean(x_true, dim=0)
        m_y_true = torch.mean(y_true, dim=0)

        no_m_y_true = y_true - m_y_true
        no_m_y_pred = y_predict - m_y_predict
        no_m_x_true = x_true - m_x_true

        c_y_predict_x_true = torch.matmul(no_m_y_pred.t(), no_m_x_true) / y_predict.shape[0]
        c_y_predict_y_predict = torch.matmul(no_m_y_pred.t(), no_m_y_pred) / y_predict.shape[0]
        c_x_true_y_predict = torch.matmul(no_m_x_true.t(), no_m_y_pred) / y_predict.shape[0]

        c_y_true_x_true = torch.matmul(no_m_y_true.t(), no_m_x_true) / y_predict.shape[0]
        c_x_true_y_true = torch.matmul(no_m_x_true.t(), no_m_y_true) / y_predict.shape[0]
        c_y_true_y_true = torch.matmul(no_m_y_true.t(), no_m_y_true) / y_predict.shape[0]

        inv_c_x_true_x_true = torch.linalg.pinv(torch.matmul(no_m_x_true.t(), no_m_x_true) / y_predict.shape[0])

        c_y_true_given_x_true = c_y_true_y_true - torch.matmul(c_y_true_x_true,
                                                               torch.matmul(inv_c_x_true_x_true, c_x_true_y_true))
        c_y_predict_given_x_true = c_y_predict_y_predict - torch.matmul(c_y_predict_x_true,
                                                                        torch.matmul(inv_c_x_true_x_true,
                                                                                     c_x_true_y_predict))
        c_y_true_x_true_minus_c_y_predict_x_true = c_y_true_x_true - c_y_predict_x_true
        c_x_true_y_true_minus_c_x_true_y_predict = c_x_true_y_true - c_x_true_y_predict

        # Distance between Gaussians
        m_dist = torch.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
        c_dist1 = torch.trace(
            torch.matmul(torch.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_c_x_true_x_true),
                         c_x_true_y_true_minus_c_x_true_y_predict))
        c_dist_2_1 = torch.trace(c_y_true_given_x_true + c_y_predict_given_x_true)
        c_dist_2_2 = - 2 * trace_sqrt_product_torch(
            c_y_predict_given_x_true, c_y_true_given_x_true)

        c_dist2 = c_dist_2_1 + c_dist_2_2
        c_dist = c_dist1 + c_dist2

        cfid = m_dist + c_dist

        print(f"M ERROR: {m_dist.cpu().numpy()}")
        print(f"C ERROR: {c_dist.cpu().numpy()}")

        return cfid.cpu().numpy()
