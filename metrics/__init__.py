import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional
from .id_metrics.embeddings import VGG16Embedding, InceptionEmbedding
from .id_metrics.FIDMetric import FIDMetric
from .id_metrics.CFIDMetric import CFIDMetric

def fid(cfg, G, ref_loader, dev_loader, mri):
    print("GETTING EMBEDDING NETWORK")
    embedding = get_embedding(mri)

    fid_metric = FIDMetric(cfg, G, ref_loader, dev_loader, embedding, embedding, mri, True)

    return fid_metric.get_fid()

def cfid(cfg, G, dev_loader, mri):
    print("GETTING EMBEDDING NETWORK")
    embedding = get_embedding(mri)

    cfid_metric = CFIDMetric(cfg, G, dev_loader, embedding, embedding, mri, True)

    return cfid_metric.get_cfid_torch()

def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def ssim(
        gt: np.ndarray, pred: np.ndarray, mri, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    if mri:
        ssim = structural_similarity(
            gt, pred, data_range=maxval
        )
    else:
        ssim = structural_similarity(
            gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), data_range=maxval, multichannel=True
        )

    return ssim

def get_embedding(mri):
    if mri:
        print("Got VGG16")
        embedding = VGG16Embedding(parallel=True)
    else:
        print("Got InceptionV3")
        embedding = InceptionEmbedding(parallel=True)

    return embedding