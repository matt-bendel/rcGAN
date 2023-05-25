import pathlib

from utils.args import Args


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    parser.add_argument('--resume', action='store_true',
                        help='Whether or not to resume training.')
    parser.add_argument('--resume-epoch', default=0, type=int, help='Epoch to resume training from')
    parser.add_argument('--mri', action='store_true',
                        help='If the application is MRI')
    parser.add_argument('--myapplication', action='store_true',
                        help='If the application is your custom application')
    parser.add_argument('--exp-name', type=str, default="", help='Name for the run.')
    parser.add_argument('--num-gpus', default=1, type=int, help='The number of GPUs to use during training.')
    parser.add_argument('--num-figs', default=1, type=int, help='The number of figures to generate while plotting.')

    parser.add_argument('--sense-maps-val', action='store_true',
                        help='Whether or not to generate sense maps for validation data.')

    return parser
