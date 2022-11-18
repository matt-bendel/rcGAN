from .prepare_data_inpaint import create_data_loaders_inpaint
from .prepare_data_mri import create_data_loaders_mri

def create_data_loaders_train(is_mri, cfg):
    if is_mri:
        train_loader, dev_loader = create_data_loaders_mri(cfg, big_test=False)
    else:
        train_loader, dev_loader, _ = create_data_loaders_inpaint(cfg)

    return train_loader, dev_loader

def create_data_loaders_test(is_mri, cfg):
    if is_mri:
        train_loader, test_loader = create_data_loaders_mri(cfg, big_test=True)
    else:
        train_loader, _, test_loader = create_data_loaders_inpaint(cfg)

    return train_loader, test_loader