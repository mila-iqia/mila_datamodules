from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import rich.logging
import simple_parsing
from simple_parsing import ArgumentParser

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.clusters.env_variables import run_job_step_to_get_slurm_env_variables
from mila_datamodules.clusters.utils import (
    get_slurm_tmpdir,
    in_job_but_not_in_job_step_so_no_slurm_env_vars,
)

# Load the SLURM environment variables into the current environment, if we're running inside a job
# but don't have the SLURM variables set.
if in_job_but_not_in_job_step_so_no_slurm_env_vars():
    run_job_step_to_get_slurm_env_variables()

from mila_datamodules.cli.huggingface import (
    HfDatasetsEnvVariables,
    command_line_args_for_hf_dataset,
    prepare_huggingface_dataset_fns,
)
from mila_datamodules.cli.torchvision import (
    command_line_args_for_dataset,
    prepare_torchvision_datasets,
)
from mila_datamodules.clusters.cluster import Cluster

current_cluster = Cluster.current_or_error()


def add_prepare_arguments(parser: ArgumentParser):
    parser.add_argument("-v", "--verbose", action="count")
    subparsers = parser.add_subparsers(
        title="dataset", description="Which dataset to prepare", dest="dataset"
    )

    add_torchvision_prepare_args(subparsers)
    add_huggingface_prepare_arguments(subparsers)


def add_torchvision_prepare_args(
    subparsers: argparse._SubParsersAction[simple_parsing.ArgumentParser],
):
    # Only add commands for the datasets that we know how to prepare on the current cluster.
    dataset_names_types_and_functions = {
        dataset_type: prepare_dataset_fns[current_cluster]
        for dataset_type, prepare_dataset_fns in prepare_torchvision_datasets.items()
        if current_cluster in prepare_dataset_fns
    }
    dataset_names_types_and_functions = [
        (dataset_type.__name__.lower(), dataset_type, preparation_fn)
        for dataset_type, preparation_fn in dataset_names_types_and_functions.items()
    ]

    for dataset_name, dataset_type, prepare_dataset_fn in sorted(
        dataset_names_types_and_functions
    ):
        dataset_parser = subparsers.add_parser(
            dataset_name, help=f"Prepare the {dataset_name} dataset"
        )

        # IDEA: Add a way for the dataset preparation thingy to add its own arguments
        # (e.g. --split='train'/'val')
        command_line_args = command_line_args_for_dataset.get(dataset_type)
        if command_line_args:
            dataset_parser.add_arguments(command_line_args, dest="dataset_kwargs")
        else:
            dataset_parser.add_argument(
                "--root", type=Path, default=get_slurm_tmpdir() / "datasets"
            )

        # prepare_dataset_fn.add_arguments(dataset_parser)
        dataset_parser.set_defaults(function=prepare_dataset_fn)


def add_huggingface_prepare_arguments(
    subparsers: argparse._SubParsersAction[simple_parsing.ArgumentParser],
):
    huggingface_preparation_functions = {
        dataset_name: prepare_dataset_fns[current_cluster]
        for dataset_name, prepare_dataset_fns in prepare_huggingface_dataset_fns.items()
        if current_cluster in prepare_dataset_fns
    }

    for dataset_name, prepare_hf_dataset_fn in huggingface_preparation_functions.items():
        dataset_parser = subparsers.add_parser(
            dataset_name, help=f"Prepare the {dataset_name} dataset from HuggingFace"
        )
        assert isinstance(dataset_parser, simple_parsing.ArgumentParser)
        command_line_args = command_line_args_for_hf_dataset[dataset_name]
        dataset_parser.add_arguments(command_line_args, dest="dataset_kwargs")
        dataset_parser.set_defaults(function=prepare_hf_dataset_fn)


def prepare(args):
    """Prepare a dataset."""
    args_dict = vars(args)

    assert args_dict.pop("command_name") == "prepare"
    assert args_dict.pop("command") is prepare
    verbose: int = args_dict.pop("verbose") or 0

    def _level(verbose: int) -> int:
        return (
            logging.DEBUG
            if verbose > 1
            else logging.INFO
            if verbose == 1
            else logging.WARNING
            if verbose == 0
            else logging.ERROR
        )

    logger = logging.getLogger("mila_datamodules")
    logger.addHandler(rich.logging.RichHandler(markup=True, tracebacks_width=100))
    logger.setLevel(_level(verbose))

    hf_logger = logging.getLogger("datasets")
    hf_logger.setLevel(_level(verbose - 2))
    hf_logger.addHandler(rich.logging.RichHandler(markup=True, tracebacks_width=100))

    # logging.basicConfig(
    #     level=_level(verbose),
    #     format=f"[node{get_node_index()}-task{get_local_rank()}] %(name)s - %(message)s ",
    #     handlers=[rich.logging.RichHandler(markup=True, tracebacks_width=100)],
    # )

    # TODO: Dispatch what to do with `args` (and the output of the function) in a smarter way,
    # based on which module was selected.

    dataset: str = args_dict.pop("dataset")
    function: Callable = args_dict.pop("function")
    dataset_kwargs = args_dict.pop("dataset_kwargs", None)

    kwargs = args_dict

    if dataset_kwargs:
        assert isinstance(dataset_kwargs, DatasetArguments)
        additional_kwargs = dataset_kwargs.to_dataset_kwargs()
        assert not any(k in kwargs for k in additional_kwargs)
        kwargs.update(additional_kwargs)

    output = function(**kwargs)

    if isinstance(output, (str, Path)):
        new_root = output
        # TODO: For some datasets (e.g. Coco), it's not actually true! We would like to tell users
        # how exactly the dataset can be created, for instance:
        # ```
        # tvd.CocoCaptions(root=SLURM_TMPDIR/"datasets/train2017",
        #                  annFile=SLURM_TMPDIR/"datasets/annotations/stuff_train2017.json")
        # ```
        # TODO: Perhaps we should return a dictionary of **kwargs to use to create the dataset?
        print(f"The {dataset} dataset can now be read by using {new_root} as the 'root' argument.")
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
