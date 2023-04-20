from __future__ import annotations

import os
from dataclasses import asdict

# from argparse import ArgumentParser
from pathlib import Path

from mila_datamodules.clusters.env_variables import (
    run_job_step_to_get_slurm_env_variables,
)
from mila_datamodules.clusters.utils import (
    in_job_but_not_in_job_step_so_no_slurm_env_vars,
)

# Load the SLURM environment variables into the current environment, if we're running inside a job
# but don't have the SLURM variables set.
if in_job_but_not_in_job_step_so_no_slurm_env_vars():
    run_job_step_to_get_slurm_env_variables()

from mila_datamodules.cli.prepare_huggingface import (
    HfDatasetsEnvVariables,
    prepare_huggingface_datasets,
)
from mila_datamodules.cli.prepare_torchvision import prepare_torchvision_datasets
from mila_datamodules.clusters.cluster import Cluster

SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])

current_cluster = Cluster.current_or_error()


# TODO: For the datasets we don't have archives for, we could either list the locations where the
# extracted version can be found, or we could download the archive in $SCRATCH.


def add_prepare_arguments(parser):
    subparsers = parser.add_subparsers(
        title="dataset", description="Which dataset to prepare", dest="dataset"
    )
    dataset_preparation_functions = {
        dataset_type.__name__.lower(): prepare_dataset_fns[current_cluster]
        for dataset_type, prepare_dataset_fns in prepare_torchvision_datasets.items()
        if current_cluster in prepare_dataset_fns
    }
    dataset_preparation_functions = dict(
        sorted(
            (dataset_name, prepare_dataset_fn)
            for dataset_name, prepare_dataset_fn in dataset_preparation_functions.items()
        )
    )

    # TODO: Add preparation function for HuggingFace datasets.

    for dataset_name, prepare_dataset_fn in dataset_preparation_functions.items():
        dataset_parser = subparsers.add_parser(
            dataset_name, help=f"Prepare the {dataset_name} dataset"
        )
        dataset_parser.add_argument("--root", type=Path, default=SLURM_TMPDIR / "datasets")
        # IDEA: Add a way for the dataset preparation thingy to add its own arguments
        # (e.g. --split='train'/'val')
        # prepare_dataset_fn.add_arguments(dataset_parser)
        dataset_parser.set_defaults(function=prepare_dataset_fn)

    huggingface_preparation_functions = {
        dataset_name: prepare_dataset_fns[current_cluster]
        for dataset_name, prepare_dataset_fns in prepare_huggingface_datasets.items()
        if current_cluster in prepare_dataset_fns
    }

    for dataset_name, prepare_dataset_fn in huggingface_preparation_functions.items():
        dataset_parser = subparsers.add_parser(
            dataset_name, help=f"Prepare the {dataset_name} dataset from HuggingFace"
        )
        dataset_parser.add_arguments(prepare_dataset_fn, dest="function")
        # dataset_parser.add_argument(
        #     "name",
        #     type=str,
        #     # required=False,
        #     # positional=True,
        #     default="",
        #     help="Dataset config name.",
        # )
        # dataset_parser.set_defaults(function=prepare_dataset_fn)


def prepare(args):
    """Prepare a dataset"""
    args_dict = vars(args)

    assert args_dict.pop("command_name") == "prepare"
    assert args_dict.pop("command") is prepare
    dataset = args_dict.pop("dataset")
    function = args_dict.pop("function")
    kwargs = args_dict

    output = function(**kwargs)
    if isinstance(output, (str, Path)):
        new_root = output
        print(f"The {dataset} dataset can now be read from the following directory: {new_root}")
    else:
        assert isinstance(output, HfDatasetsEnvVariables)
        print(
            "The following environment variables have been set in this process, but will "
            "probably also need to also be added in the job script:"
        )
        for key, value in asdict(output).items():
            print(f"export {key}={value}")


def get_env_variables_to_use():
    """IDEA: Output only the environment variables that need to be set for the current job."""
