from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable

from typing_extensions import Concatenate

from mila_datamodules.blocks.types import PrepareDatasetFn
from mila_datamodules.cli.utils import runs_on_local_main_process_first
from mila_datamodules.types import D, D_co, P

logger = get_logger(__name__)


class Compose(PrepareDatasetFn[D_co, P]):
    class Stop(Exception):
        pass

    def __init__(self, *callables: PrepareDatasetFn[D_co, P]) -> None:
        self.callables = callables

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        try:
            for c in self.callables:
                # TODO: Check that nesting `runs_on_local_main_process_first` decorators isn't a
                # problem.
                root = c(root, *dataset_args, **dataset_kwargs)
        except self.Stop:
            pass
        return str(root)


class SkipRestIfThisWorks(PrepareDatasetFn[D, P]):
    """Raises a special Stop exception when running the given callable doesn't raise an exception.

    If an exception of a type matching one in `exceptions` is raised by the function, the exception
    is ignored. Other exceptions are raised.

    This is used to short-cut the list of operations to perform inside a `Compose` block.
    """

    def __init__(
        self,
        function: PrepareDatasetFn[D, P] | Callable[Concatenate[str, P], Any],
        continue_if_raised: type[Exception] | tuple[type[Exception], ...] = RuntimeError,
    ):
        self.function = function
        self.exceptions = (
            [continue_if_raised] if isinstance(continue_if_raised, type) else continue_if_raised
        )

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        try:
            # logger.info(f"Checking if the dataset has been prepared in {root}")
            function_output = self.function(str(root), *dataset_args, **dataset_kwargs)
            if isinstance(function_output, (str, Path)):
                root = function_output
        except tuple(self.exceptions) as expected_exception:
            logger.info(
                f"Failed: dataset has not been prepared in {root}, continuing with dataset "
                f"preparation."
            )
            logger.debug(f"Exceptions: {expected_exception}")
        else:
            logger.info("Success!")
            raise Compose.Stop()
        return str(root)
