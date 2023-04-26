"""TODO: Write some dataset preparation functions for HuggingFace datasets.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Literal

from datasets import DownloadConfig, load_dataset, load_dataset_builder
from simple_parsing import field

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.huggingface.base import HfDatasetsEnvVariables, use_variables
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_scratch_dir, get_slurm_tmpdir
from mila_datamodules.utils import cpus_per_node

logger = get_logger(__name__)

WikitextName = Literal[
    "wikitext-103-v1", "wikitext-2-v1", "wikitext-103-raw-v1", "wikitext-2-raw-v1"
]


@dataclass
class PrepareWikitextArgs(DatasetArguments):
    """Options for preparing the wikitext dataset."""

    name: WikitextName = field(positional=True)


def prepare_wikitext(name: WikitextName) -> HfDatasetsEnvVariables:
    """Prepare the wikitext dataset."""
    path = "wikitext"
    cluster = Cluster.current_or_error()
    scratch = get_scratch_dir()
    slurm_tmpdir = get_slurm_tmpdir()
    # Number of processes to use for preparing the dataset. Note, since this is expected to be only
    # executed once per node, we can use all the CPUs
    num_proc = cpus_per_node()

    # Load the dataset under $SCRATCH/cache/huggingface first, since we don't have a shared copy
    # on the cluster.
    scratch_hf_datasets_cache = scratch / "cache/huggingface/datasets"
    with use_variables(HF_DATASETS_CACHE=scratch_hf_datasets_cache):
        # load_dataset(path, name, num_proc=num_proc)
        dataset_builder = load_dataset_builder(path, name)
        # TODO: There are tons of arguments that we could probably pass here.
        dataset_builder.download_and_prepare(num_proc=num_proc)

    # Copy the dataset from $SCRATCH to $SLURM_TMPDIR
    dataset_dir = scratch_hf_datasets_cache / path
    relative_path = dataset_dir.relative_to(scratch)
    logger.info(f"Copying dataset from {dataset_dir} -> {slurm_tmpdir / relative_path}")
    shutil.copytree(
        scratch / relative_path,
        slurm_tmpdir / relative_path,
        symlinks=True,  # TODO: Keep symlinks in source dir as symlinks in destination dir?
        dirs_exist_ok=True,
    )

    logger.info("Checking that the dataset was copied correctly to SLURM_TMPDIR...")
    with use_variables(
        HfDatasetsEnvVariables(
            HF_DATASETS_CACHE=slurm_tmpdir / "cache/huggingface/datasets",
            HF_DATASETS_OFFLINE=1,  # Disabling internet just to be sure everything is setup.
        )
    ):
        # download_config = None
        download_config = (
            DownloadConfig(
                local_files_only=True,
            )
            # if hf_env_variables.HF_DATASETS_OFFLINE
            # else None
        )
        dataset_builder = load_dataset_builder(path, name, download_config=download_config)
        dataset_builder.download_and_prepare(
            download_config=download_config,
            num_proc=num_proc,
        )
        load_dataset(
            path,
            name,
            download_config=download_config,
            num_proc=num_proc,
        )

    offline_bit = 0 if cluster.internet_access_on_compute_nodes else 1

    return HfDatasetsEnvVariables(
        HF_HOME=f"{scratch}/cache/huggingface",
        HF_DATASETS_CACHE=f"{slurm_tmpdir}/cache/huggingface/datasets",
        HF_DATASETS_OFFLINE=offline_bit,
    )
