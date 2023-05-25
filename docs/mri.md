# Reproducing our MRI results
## Update Config
Update ```configs/mri.yml``` with the path to your data, where you want to store checkpoints, and with the path
to estimated coil sensitivities (see below).

## Data Setup
First you need to create the test set:
```python
python /scripts/mri/create_test_set.py
```
This script will automatically remove the volumes we used during testing from
the validation set, and place them in a new directory called ```small_T2_test```.

Next, create a directory called ```sense_maps``` with subdirectories ```val_full_res``` and ```test_full_res```.

Your file structure should
look like this (wherever you stored your fastMRI data):
```
├── .../fastMRI_brain/data
│   ├── multicoil_train
│   ├── multicoil_val
│   ├── small_T2_test
├── .../fastMRI_brain/sense_maps
│   ├── val_full_res
│   ├── test_full_res
```

## Estimating Sensitivity Maps
Next, you will need to generate the sensitivity maps. To do so, run the following commands:
```python
python /scripts/mri/estimate_maps.py --sense-maps-val
python /scripts/mri/estimate_maps.py
```
This will estimate and store the sensitivity maps for the validation and test sets - there is no
need to do so for the training set.

#### Note: You will need to update some hard-coded paths in ```data/mri_data.py```. Incorporating these into an argument remains a TODO.

## Training
Training is as simple as running the following command:
```python
python train.py --mri --exp-name rcgan_test --num-gpus X
```
where ```X``` is the number of GPUs you plan to use. Note that this project uses Weights and Biases (wandb) for logging.
See [their documentation](https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

If you need to resume training, use the following command:
```python
python train.py --mri --exp-name rcgan_test --num-gpus X --resume --resume-epoch Y
```
where ```Y``` is the epoch to resume from.

By default, we save the previous 50 epochs. Ensure that your checkpoint path points to a location with sufficient disk space.
If disk space is a concern, 50 can be reduced to 25.
This is important for the next step, validation.

## Validation
During training, validation is necessary in order to update the weight applied to
the standard deviation reward. However, we select our best model according to validation
Conditonal Frechet Inception Distance ([CFID](https://arxiv.org/abs/2103.11521)). So, we must also retrospectively validate.

To do so, run the following command:
```python
python /scripts/mri/validate.py --exp-name rcgan_test
```
This script will select the model which has an acceptable PSNR gain (see our paper for details)
and the lowest CFID. Models which lie outside the acceptable PSNR gain are automatically
skipped. 

Once completed, all other checkpoints will automatically be deleted.

## Testing
To test the model's PSNR, SSIM, LPIPS, DISTS, and APSD, execute the following command:
```python
python /scripts/mri/test.py --exp-name rcgan_test
```
This will test all aforementioned metrics on the average reconstruction for 1, 2, 4, 8, 16, and 32 samples.

## Plot
To generate figures similar to those found in our paper, execute the following command:
```python
python /scripts/mri/test.py --exp-name rcgan_test --num-figs 5
```
where ```--num-figs``` controls the number of figures to generate. This script generates two different
kinds of figures:
1. A global figure which features the 32-sample avg. reconstruction, error map, and std. deviation map.
2. A figure which focuses on a randomly selected zoomed region. This figure shows the 32-, 4-, and 2-sample average
reconstructions, as well as 2 individual samples.

The figures are saved in ```figures/mri```.

## Questions and Concerns
If you have any questions, or run into any issues, don't hesitate to reach out at bendel.8@osu.edu.