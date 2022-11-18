import argparse
import pathlib

class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Override defaults with passed overrides
        self.set_defaults(**overrides)

def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    # DATA ARGS
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--is-mri', action='store_true',
                        help='If set, train/test using MRI logic')

    # LOGISTICAL ARGS
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. ')
    parser.add_argument('--train-gif', action='store_true',
                        help='If set, save gif of posterior samples during training.'
                             '"--checkpoint" should be set with this')
    parser.add_argument('--device', default=0,
                        help='Which device to train on. Use idx of cuda device, or -1 for CPU')
    parser.add_argument('--plot-dir', type=str, default="",
                        help='The directory to save the plots to. This can be relative. Include a trailing slash')
    parser.add_argument('--num-plots', type=int,
                        help='The number of plots to generate when running plot.py')
    return parser
