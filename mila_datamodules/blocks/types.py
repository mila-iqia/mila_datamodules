from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Protocol

from typing_extensions import Concatenate

from mila_datamodules.types import D, P

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser

# TODO: Remove D_co here. We basically just want this to be typed as
# Callable[Concatenate[str, P], str].


# TODO: This is too complicated.
class PrepareDatasetFn(Protocol[D, P]):
    dataset_fn: Callable[Concatenate[str, P], D] | None = None

    def __call__(
        self,
        root: str | Path,
        /,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        raise NotImplementedError
