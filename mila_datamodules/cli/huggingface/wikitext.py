"""TODO: Write some dataset preparation functions for HuggingFace datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Literal

from simple_parsing import field

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.huggingface.base import HfDatasetsEnvVariables, prepare_generic

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
    return prepare_generic(path="wikitext", name=name)

    return prepare_generic(path="wikitext", name=name)
