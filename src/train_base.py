import argparse
from typing import Union, Type

import yaml

from aphynity.aphinity_options.aphinityoptions import AphinityOptions
from mcmc_time_series.options.mcmc_options import MCMCOptions
from time_series.time_series_options.experiment_options import ExperimentOptions


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the experiment to run.")
    parser.add_argument("--output_dir", type=str, help="Name of the experiment to run.")
    parser.add_argument(
        "--options_file", type=str, help="Name of the experiment to run."
    )
    return parser


def load_experiment_dict(options_file: str) -> dict:
    exp_dir = options_file
    with open(exp_dir, "r") as yaml_in:
        opts_dict = yaml.safe_load(yaml_in)
    return opts_dict


def init_options(
    options: Union[Type[MCMCOptions], Type[ExperimentOptions], Type[AphinityOptions]]
) -> Union[MCMCOptions, ExperimentOptions, AphinityOptions]:
    parser = train_parser()
    args = parser.parse_args()
    opts = options()
    if args.options_file:
        opts_dict = load_experiment_dict(args.options_file)
        opts = options(**opts_dict).parse_obj(opts_dict)
    if args.output_dir:
        opts.output_dir = args.output_dir
        opts.name = args.name
    opts.initialize_setup()
    return opts
