from __future__ import annotations

from logging import getLogger as get_logger

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.huggingface.base import (
    HfDatasetsEnvVariables,
    PrepareGenericDatasetArgs,
    PrepareHfDatasetFn,
    prepare_generic,
)
from mila_datamodules.cli.huggingface.wikitext import PrepareWikitextArgs, prepare_wikitext
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_scratch_dir, get_slurm_tmpdir

logger = get_logger(__name__)

prepare_huggingface_dataset_fns: dict[str, dict[Cluster, PrepareHfDatasetFn]] = {
    "wikitext": {Cluster.Mila: prepare_wikitext},
    "huggingface": {Cluster.Mila: prepare_generic},
}

command_line_args_for_hf_dataset: dict[str, DatasetArguments | type[DatasetArguments]] = {
    "wikitext": PrepareWikitextArgs,
    "huggingface": PrepareGenericDatasetArgs,
}


def env_vars_to_set(dataset: str) -> HfDatasetsEnvVariables:
    """IDEA: Write a method that can be called to just get the environment variables."""
    return HfDatasetsEnvVariables(
        HF_HOME=f"{get_scratch_dir()}/cache/huggingface",
        HF_DATASETS_CACHE=f"{get_slurm_tmpdir()}/cache/huggingface/datasets",
    )
