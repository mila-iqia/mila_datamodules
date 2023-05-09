from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from logging import getLogger as get_logger
from typing import Callable

import simple_parsing

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.huggingface import (
    HfDatasetsEnvVariables,
    command_line_args_for_hf_dataset,
    prepare_huggingface_dataset_fns,
)
from mila_datamodules.cli.prepare import PreparePlugin
from mila_datamodules.clusters.cluster import Cluster

current_cluster = Cluster.current_or_error()
logger = get_logger(__name__)


class HuggingFacePlugin(PreparePlugin):
    def add_prepare_args(
        self,
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

            # dataset_parser.set_defaults(prepare_fn=self.prepare)

    def prepare(self, args_dict: dict):
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
