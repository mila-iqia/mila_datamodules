"""TODO: Write some dataset preparation functions for HuggingFace datasets.
"""
from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import asdict, dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal, Protocol

from datasets import DownloadConfig, load_dataset
from simple_parsing import field

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_scratch_dir, get_slurm_tmpdir

Path().write_text
SLURM_TMPDIR = get_slurm_tmpdir()
SCRATCH = get_scratch_dir()

logger = get_logger(__name__)


@dataclass
class HfDatasetsEnvVariables:
    HF_HOME: str | Path = SCRATCH / "cache/huggingface"
    HF_DATASETS_CACHE: str | Path = SCRATCH / "cache/huggingface/datasets"

    # When running on a cluster where compute nodes don't have internet access, we copy what we can
    # from $SCRATCH to $SLURM_TMPDIR, and set these variables to 1 to avoid attempting to
    # downloading anything that is missing:
    HF_DATASETS_OFFLINE: Literal[0, 1] = 0

    # # TODO: Seems to be used for model weights.
    # HUGGINGFACE_HUB_CACHE: str | Path = SCRATCH / "cache/huggingface/hub"
    # TRANSFORMERS_OFFLINE: Literal[0, 1] = 0


def set_hf_variables(hf_variables: HfDatasetsEnvVariables):
    env_vars = asdict(hf_variables)
    for key, value in env_vars.items():
        os.environ[key] = str(value)

    # NOTE: If the datasets module is already imported, changing the environment variables will
    # have no effect on how HuggingFace works. Therefore we also need to modify the attributes of
    # the `datasets.config` module instead.
    import datasets.config

    config_module_modified_attributes = {}
    for key, value in env_vars.items():
        if key == "HF_HOME":
            # The attribute name is different than the env variable name:
            config_module_modified_attributes["HF_CACHE_HOME"] = datasets.config.HF_CACHE_HOME
            setattr(datasets.config, "HF_CACHE_HOME", value)
        elif hasattr(datasets.config, key):
            config_module_modified_attributes[key] = getattr(datasets.config, key)
            setattr(datasets.config, key, value)
        else:
            logger.debug(f"There is no {key!r} attribute in `datasets.config`. Not modifying it.")
    return config_module_modified_attributes


@contextlib.contextmanager
def use_variables(hf_variables: HfDatasetsEnvVariables):
    backup = {key: os.environ.get(key) for key in asdict(hf_variables)}

    previous_module_attributes = set_hf_variables(hf_variables)
    import datasets.config

    yield

    for key, previous_value_or_none in backup.items():
        if previous_value_or_none is None:
            # The environment variable wasn't set before, so we remove it:
            os.environ.pop(key)
        else:
            os.environ[key] = previous_value_or_none

    for key, value in previous_module_attributes.items():
        setattr(datasets.config, key, value)


@dataclass
class PrepareWikitext:
    """Prepares the wikitext dataset."""

    name: Literal[
        "wikitext-103-v1", "wikitext-2-v1", "wikitext-103-raw-v1", "wikitext-2-raw-v1"
    ] = field(positional=True)

    def __call__(self) -> HfDatasetsEnvVariables:
        prepare_wikitext(self.name)
        return self.env_vars_to_set()

    def env_vars_to_set(self) -> HfDatasetsEnvVariables:
        """IDEA: Write a method that can be called to just get the environment variables."""
        offline_bit = 0 if Cluster.current_or_error().internet_access_on_compute_nodes else 1
        return HfDatasetsEnvVariables(
            HF_HOME=f"{SCRATCH}/cache/huggingface",
            HF_DATASETS_CACHE=f"{SLURM_TMPDIR}/cache/huggingface/datasets",
            HF_DATASETS_OFFLINE=offline_bit,
        )


def prepare_wikitext(
    name: Literal["wikitext-103-v1", "wikitext-2-v1", "wikitext-103-raw-v1", "wikitext-2-raw-v1"],
) -> HfDatasetsEnvVariables:
    """Prepare the wikitext dataset."""
    path = "wikitext"
    cluster = Cluster.current_or_error()
    with use_variables(HfDatasetsEnvVariables()):
        load_dataset(path, name)

    dataset_dir = SCRATCH / "cache/huggingface/datasets" / path
    relative_path = dataset_dir.relative_to(SCRATCH)
    logger.info(f"Copying dataset from {dataset_dir} -> {SLURM_TMPDIR / relative_path}")
    shutil.copytree(
        SCRATCH / relative_path,
        SLURM_TMPDIR / relative_path,
        symlinks=True,  # TODO: Keep symlinks in source dir as symlinks in destination dir?
        dirs_exist_ok=True,
    )
    logger.info("Checking that the dataset was copied correctly to SLURM_TMPDIR...")
    with use_variables(
        HfDatasetsEnvVariables(
            # HF_HOME=SCRATCH / "cache/huggingface",
            HF_DATASETS_CACHE=SLURM_TMPDIR / "cache/huggingface/datasets",
            HF_DATASETS_OFFLINE=1,
        )
    ):
        load_dataset(
            path,
            name,
            download_config=DownloadConfig(
                local_files_only=True,
            ),
        )

    offline_bit = 0 if cluster.internet_access_on_compute_nodes else 1

    return HfDatasetsEnvVariables(
        HF_HOME=f"{SCRATCH}/cache/huggingface",
        HF_DATASETS_CACHE=f"{SLURM_TMPDIR}/cache/huggingface/datasets",
        HF_DATASETS_OFFLINE=offline_bit,
    )


def prepare_the_pile(
    path: str,
    name: str | None = None,
    subsets=("all",),
    version="0.0.0",
) -> HfDatasetsEnvVariables:
    ...


class PrepareHfDatasetFn(Protocol):
    """A function that prepares a HuggingFace dataset, and returns the environment variables that
    should be set in the user job."""

    def __call__(self) -> HfDatasetsEnvVariables:
        ...


prepare_huggingface_datasets: dict[str, dict[Cluster, type[PrepareHfDatasetFn]]] = {
    "wikitext": {Cluster.Mila: PrepareWikitext},
}
