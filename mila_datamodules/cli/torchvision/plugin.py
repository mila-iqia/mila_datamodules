from __future__ import annotations

import argparse
import logging
from logging import getLogger as get_logger
from typing import Callable

import simple_parsing

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.prepare import PreparePlugin
from mila_datamodules.cli.torchvision import (
    VisionDatasetArgs,
    command_line_args_for_dataset,
    prepare_torchvision_datasets,
)
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import (
    get_slurm_tmpdir,
)

current_cluster = Cluster.current_or_error()
logger = get_logger(__name__)


class TorchVisionPlugin(PreparePlugin):
    def add_prepare_args(
        self,
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
            dataset_parser.set_defaults(function=prepare_dataset_fn)
            # dataset_parser.set_defaults(prepare_fn=self.prepare)

    def prepare(self, args_dict: dict):
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
