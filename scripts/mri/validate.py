import torch
import os
import yaml
import types
import json
import numpy as np
import pytorch_lightning as pl

from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.lightning.rcGAN import rcGAN
from pytorch_lightning import seed_everything
from utils.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    with open('configs/mri.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MRIDataModule(cfg)
    dm.setup()
    val_loader = dm.val_dataloader()
    best_epoch = -1
    inception_embedding = VGG16Embedding()
    best_cfid = 10000000
    start_epoch = 50
    end_epoch = 100

    with torch.no_grad():
        for epoch in range(start_epoch, end_epoch):
            print(f"VALIDATING EPOCH: {epoch + 1}")
            try:
                model = rcGAN.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={epoch}.ckpt')
            except Exception as e:
                print(e)
                continue

            if model.is_good_model == 0:
                print("NO GOOD: SKIPPING...")
                continue

            model = model.cuda()
            model.eval()

            cfid_metric = CFIDMetric(gan=model,
                                     loader=val_loader,
                                     image_embedding=inception_embedding,
                                     condition_embedding=inception_embedding,
                                     cuda=True,
                                     args=cfg,
                                     ref_loader=False,
                                     num_samps=1)

            cfids = cfid_metric.get_cfid_torch_pinv()

            cfid_val = np.mean(cfids)

            if cfid_val < best_cfid:
                best_epoch = epoch
                best_cfid = cfid_val

    print(f"BEST EPOCH: {best_epoch}")

    for epoch in range(50, end_epoch):
        if epoch != best_epoch:
            os.remove(cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={epoch}.ckpt')

    os.rename(cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={best_epoch}.ckpt', cfg.checkpoint_dir + args.exp_name + f'/checkpoint_best.ckpt')
