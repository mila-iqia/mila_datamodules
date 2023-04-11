from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path

from mila_datamodules.cli.prepare_torchvision import prepare_torchvision_datasets
from mila_datamodules.clusters.cluster import Cluster

SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])

current_cluster = Cluster.current_or_error()


# TODO: For the datasets we don't have archives for, we could either list the locations where the
# extracted version can be found, or we could download the archive in $SCRATCH.


def prepare(argv: list[str] | None = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(
        title="dataset", description="Which dataset to prepare", dest="dataset"
    )
    dataset_preparation_functions = {
        dataset_type.__name__.lower(): prepare_dataset_fns[current_cluster]
        for dataset_type, prepare_dataset_fns in prepare_torchvision_datasets.items()
        if current_cluster in prepare_dataset_fns
    }

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

    args = parser.parse_args(argv)

    args_dict = vars(args)
    dataset = args_dict.pop("dataset")
    function = args_dict.pop("function")
    kwargs = args_dict

    new_root = function(**kwargs)
    print(f"The {dataset} dataset can now be read from the following directory: {new_root}")


if __name__ == "__main__":
    prepare()
