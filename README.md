# A Regularized Conditional GAN for Posterior Sampling in Inverse Problems [[arXiv]](https://arxiv.org/abs/2210.13389)
## Utilization Instructions
First, install the required modules via
```
pip install -r requirements.txt
```

Next, set the location of the checkpoint directory in both config files, found in the ``config/`` folder. Then,
download [the fastMRI dataset](https://fastmri.med.nyu.edu/) or [the celebA-HQ 256x256 dataset](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P).

There are three primary scripts: ``train.py``, ``test.py``, and ``plot.py``. To train models, run
```python train.py```. To parralelize models on multiple GPUs, be sure to include the ``--data-parallel`` argument.
To run any of the scripts on MRI data, include the ``--is-mri`` argument. Other arguments and
their function can be found in ``parse_args.py``.

E.g., to run the training for MRI on multiple GPUs you would use
```
python train.py --data-parallel --is-mri
```

### Extending the Code
To extend the code to work for other datasets, you will want to create a new generator
wrapper in ``generator_wrappers/``. This should handle all reconstruction logic (e.g., giving inputs to the generator,
generating latent vectors, data-consistency, etc.). Then, you will want to create a new data loader
for your new dataset in ``data_loaders/``. Finally, small modifications will need to be made to the files in
the ``runners/`` directory for application specific logic (see the MRI vs. inpainting examples).

## References
This repository contains code from the following works, which should be cited:

```
@article{zbontar2018fastmri,
  title={fastMRI: An open dataset and benchmarks for accelerated MRI},
  author={Zbontar, Jure and Knoll, Florian and Sriram, Anuroop and Murrell, Tullie and Huang, Zhengnan and Muckley, Matthew J and Defazio, Aaron and Stern, Ruben and Johnson, Patricia and Bruno, Mary and others},
  journal={arXiv preprint arXiv:1811.08839},
  year={2018}
}

@article{devries2019evaluation,
  title={On the evaluation of conditional GANs},
  author={DeVries, Terrance and Romero, Adriana and Pineda, Luis and Taylor, Graham W and Drozdzal, Michal},
  journal={arXiv preprint arXiv:1907.08175},
  year={2019}
}

@inproceedings{Karras2020ada,
  title={Training Generative Adversarial Networks with Limited Data},
  author={Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle={Proc. NeurIPS},
  year={2020}
}

@inproceedings{zhao2021comodgan,
  title={Large Scale Image Completion via Co-Modulated Generative Adversarial Networks},
  author={Zhao, Shengyu and Cui, Jonathan and Sheng, Yilun and Dong, Yue and Liang, Xiao and Chang, Eric I and Xu, Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@misc{zeng2022github,
    howpublished = {Downloaded from \url{https://github.com/zengxianyu/co-mod-gan-pytorch}},
    month = sep,
    author={Yu Zeng},
    title = {co-mod-gan-pytorch},
    year = 2022
}
```

## Citation
If you find this code helpful, please cite our paper:
```
@journal{bendel2022arxiv,
  author = {Bendel, Matthew and Ahmad, Rizwan and Schniter, Philip},
  title = {A Regularized Conditional {GAN} for Posterior Sampling in Inverse Problems},
  year = {2022},
  journal={arXiv:2210.13389}
}
```