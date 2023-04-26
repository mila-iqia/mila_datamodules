from __future__ import annotations

from mila_datamodules.cli.huggingface.base import (
    HfDatasetsEnvVariables,
)


def prepare_the_pile(
    path: str,
    name: str | None = None,
    subsets=("all",),
    version="0.0.0",
) -> HfDatasetsEnvVariables:
    ...
