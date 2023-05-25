# Extending rcGAN
#### Note: This process is likely to change in the near future. Whenever it does, these docs will be updated.

## Config
First, create a new config folder for your application:
```
├── configs
│   ├── mri.yml
│   ├── myapplication.yml
```
How you set this up is up to you!

## Data Module
Next, you will need to create a DataModule for your application. To do so,
create a new file in ```data/lightning```:
```
├── data
│   ├── lightning
│       ├── MRIDataModule.py
│       ├── MyApplicationDataModule.py
```
See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) for how to structure this file.
```MRIDataModule.py``` is a good reference for this.

You may need additional files to work with your data. You may either place these in ```data/datasets```, or a new directory.

## Lightning Module
Create a new folder in ```models/archs``` for architectures related to your application:
```
├── models
│   ├── archs
│       ├── mri
│       ├── myapplication
```
In here you should store the architectures for your model (e.g., generator, discriminator, etc.).

Next, create the Lightning Module. Create a new python file in ```models/lightning```. This contains all training/validation
logic for your model. See [the docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) for more information.
```models/lightning/rcGAN.py``` is a good reference/starting point.

#### Note: In the near future, I plan to change this procedure so that new models inherit from a base rcGAN Lightning Module. This remains a TODO.

## Add Arguments
In ```utils/parse_args.py``` add a new argument for your application:
```python
...
parser.add_argument('--mri', action='store_true',
                        help='If the application is MRI')
parser.add_argument('--myapplication', action='store_true',
                        help='If the application is your application')
...
```
You may also add any additional arguments required by your application, however it is suggested that you keep
most application-specific variables confined to the config file.

## Modify the Training Script
Next, add your model to ```train.py```:
```python
...
from data.lightning.MyApplicationDataModule import MyApplicationDataModule
from models.lightning.myapplication_rcGAN import rcGAN
...
if args.mri:
    with open('configs/mri.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MRIDataModule(cfg)

    model = rcGAN(cfg, args.exp_name, args.num_gpus)
elif args.myapplication:
    with open('configs/myapplication.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MyApplicationDataModule(cfg)

    model = rcGAN(cfg, args.exp_name, args.num_gpus)
else:
    print("No valid application selected. Please include one of the following args: --mri")
    exit()
...
```
Training is as simple as running the following command:
```python
python train.py --mri --exp-name rcgan_test --num-gpus X
```
where ```X``` is the number of GPUs you plan to use. Note that this project uses Weights and Biases (wandb) for logging.
See [their documentation](https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

## Retrospective Validation, Testing, Other
For anything else, create a new directory in ```utils```:
```
├── utils
│   ├── mri
│   ├── myapplication
```

In here you can create any other scripts.

### Retrospective Validation
This is not necessary, as you may want to select your best model some other way. If this is the case,
I suggest modifying the ```ModelCheckpoint``` object in ```train.py```. You can automatically save your best
model based on some tracked metric (e.g., psnr, ssim, etc.). See the [lightning docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html) for more information and 
see [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) for options which would be easy to integrate.

## Questions and Concerns
If you have any questions, or run into any issues, don't hesitate to reach out at bendel.8@osu.edu.