from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
from typing import Protocol

from mila_datamodules.types import D_co, P

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser

# TODO: Remove D_co here. We basically just want this to be typed as
# Callable[Concatenate[str, P], str].


class PrepareDatasetFn(Protocol[D_co, P]):
    def __call__(
        self,
        root: str | Path,
        /,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        raise NotImplementedError
