import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import time

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)
    snr = 10 * np.log10(np.mean(gt ** 2) / noise_mse)

    return snr


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)

    return noise_mse


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None, multichannel: bool = False
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    if multichannel:
        ssim = structural_similarity(
            gt, pred, data_range=maxval, channel_axis=2
        )
    else:
        ssim = structural_similarity(
            gt, pred, data_range=maxval
        )

    return ssim
