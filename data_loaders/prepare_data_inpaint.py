import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset

class DataTransform:
    def __init__(self, cfg):
        np.random.seed(0)

        arr = np.ones((cfg.data.resolution, cfg.data.resolution))
        arr[cfg.data.resolution // 4: 3 *cfg.data.resolution//4, cfg.data.resolution // 4: 3 *cfg.data.resolution//4] = 0
        self.mask = torch.tensor(np.reshape(arr, (cfg.data.resolution, cfg.data.resolution)), dtype=torch.float).repeat(3, 1, 1)

    def __call__(self, gt_im):
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        gt = (gt_im - mean[:, None, None]) / std[:, None, None]
        masked_im = gt * self.mask

        return gt, masked_im, mean, std, self.mask



def create_datasets(cfg):
    transform = transforms.Compose([transforms.ToTensor(), DataTransform(cfg)])
    dataset = datasets.ImageFolder(cfg.data.path, transform=transform)
    train_data, dev_data, test_data = torch.utils.data.random_split(
        dataset, [27000, 2000, 1000],
        generator=torch.Generator().manual_seed(0)
    )

    return test_data, dev_data, train_data


def create_data_loaders_inpaint(cfg):
    test_data, dev_data, train_data = create_datasets(cfg)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=cfg.validate.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=cfg.test.batch_size,
        num_workers=16,
        pin_memory=True,
    )

    return train_loader, dev_loader, test_loader
