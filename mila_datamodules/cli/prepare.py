from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from logging import getLogger as get_logger
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
    VisionDatasetArgs,
    command_line_args_for_dataset,
    prepare_torchvision_datasets,
)
from mila_datamodules.clusters.cluster import Cluster

current_cluster = Cluster.current_or_error()
logger = get_logger(__name__)


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
    dataset_types_and_functions = {
        dataset_type: prepare_dataset_fns[current_cluster]
        for dataset_type, prepare_dataset_fns in prepare_torchvision_datasets.items()
        if current_cluster in prepare_dataset_fns
    }
    dataset_names_types_and_functions = [
        (dataset_type.__name__.lower(), dataset_type, preparation_fn)
        for dataset_type, preparation_fn in dataset_types_and_functions.items()
    ]

    for dataset_name, dataset_type, prepare_dataset_fn in sorted(
        dataset_names_types_and_functions
    ):
        dataset_parser = subparsers.add_parser(
            dataset_name, help=f"Prepare the {dataset_name} dataset"
        )
        command_line_args = command_line_args_for_dataset.get(dataset_type, VisionDatasetArgs)
        dataset_parser.add_arguments(command_line_args, dest="dataset_preparation")

        # prepare_dataset_fn.add_arguments(dataset_parser)
        dataset_parser.set_defaults(function=prepare_dataset_fn)

        dataset_parser.set_defaults(prepare_fn=prepare_torchvision)


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
        dataset_parser.add_arguments(command_line_args, dest="dataset_preparation")
        dataset_parser.set_defaults(function=prepare_hf_dataset_fn)

        dataset_parser.set_defaults(prepare_fn=prepare_huggingface)


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

    # TODO: Dispatch what to do with `args` (and the output of the function) in a smarter way,
    # based on which module was selected.
    if "prepare_fn" not in args_dict or not callable(args_dict["prepare_fn"]):
        raise RuntimeError(
            "The parsing function isn't configured correctly. It should call "
            "parser.set_defaults(prepare_fn=some_callable) where "
            "some_callable: Callable[[dict], Any]"
        )
    prepare_fn: Callable[[dict], None] = args_dict.pop("prepare_fn")
    prepare_fn(args_dict)


def prepare_torchvision(args_dict: dict):
    dataset: str = args_dict.pop("dataset")
    function: Callable = args_dict.pop("function")
    dataset_arguments = args_dict.pop("dataset_preparation")

    kwargs = args_dict

    if dataset_arguments:
        assert isinstance(dataset_arguments, DatasetArguments)
        additional_kwargs = dataset_arguments.to_dataset_kwargs()
        assert not any(k in kwargs for k in additional_kwargs)
        kwargs.update(additional_kwargs)

    output = function(**kwargs)

    logger.setLevel(logging.INFO)

    new_root = output

    import torchvision.datasets

    # FIXME: Ugly AF, just there for the demo:
    dataset_class = {
        k: v for k, v in vars(torchvision.datasets).items() if k.lower() == dataset
    }.popitem()[1]

    kwargs.update(root=new_root)
    code_snippet = (
        f"{dataset_class.__name__}(" + ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) + ")"
    )
    slurm_tmpdir = get_slurm_tmpdir()
    code_snippet = (
        (code_snippet)
        .replace(f'"{slurm_tmpdir}', 'os.environ["SLURM_TMPDIR"] + "')
        .replace(f"'{slurm_tmpdir}", "os.environ['SLURM_TMPDIR'] + '")
    )
    # fn = logger.info if verbose > 0 else print
    logger.info(
        "Here's how you can use this prepared dataset in your job:\n"
        + "\n"
        + "```python\n"
        + "import os\n"
        + code_snippet
        + "\n```"
    )


def prepare_huggingface(args_dict: dict):
    args_dict.pop("dataset")
    function: Callable = args_dict.pop("function")
    dataset_arguments = args_dict.pop("dataset_preparation", DatasetArguments())

    kwargs = args_dict

    if dataset_arguments:
        assert isinstance(dataset_arguments, DatasetArguments)
        additional_kwargs = dataset_arguments.to_dataset_kwargs()
        assert not any(k in kwargs for k in additional_kwargs)
        kwargs.update(additional_kwargs)

    output = function(**kwargs)

    logger.setLevel(logging.INFO)

    assert isinstance(output, HfDatasetsEnvVariables)
    logger.info(
        "The following environment variables have been set in this process, but will "
        "probably also need to also be added in the job script:"
    )
    for key, value in asdict(output).items():
        print(f"export {key}={value}")


def get_env_variables_to_use():
    """IDEA: Output only the environment variables that need to be set for the current job."""
