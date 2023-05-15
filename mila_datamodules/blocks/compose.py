from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable

from typing_extensions import Concatenate

from mila_datamodules.blocks.types import PrepareDatasetFn
from mila_datamodules.cli.utils import runs_on_local_main_process_first
from mila_datamodules.types import D, D_co, DatasetFnWithStrArg, P

logger = get_logger(__name__)


class Compose(PrepareDatasetFn[D, P]):
    """Calls functions in order.

    The functions take a string as their first positional argument.

    If one of the functions returns a string, it is passed as the positional argument to the next
    function.
    If the output of one of the functions isn't a string, it is ignored, and same argument is
    passed to the next function.
    """

    class Stop(Exception):
        pass

    def __init__(self, *callables: DatasetFnWithStrArg[Any, P]) -> None:
        self.callables = callables

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        root = str(root)
        try:
            for c in self.callables:
                # TODO: Check that nesting `runs_on_local_main_process_first` decorators isn't a
                # problem.
                # logger.debug(f"Calling {c} with {root}, {dataset_args=}, {dataset_kwargs=}")
                output = c(root, *dataset_args, **dataset_kwargs)
                if isinstance(output, str):
                    root = output
        except self.Stop:
            pass
        return root

    @property
    def dataset_fn(self) -> DatasetFnWithStrArg[D_co, P] | None:
        """Gets the dataset_fn from one of the callables, if it exists."""
        dataset_fn = None
        for fn in self.callables:
            dataset_fn = dataset_fn or getattr(fn, "dataset_fn", None)
        return dataset_fn


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


class SkipRestIf(PrepareDatasetFn[D_co, P]):
    """Raises a special Stop exception when the output of the function is truthy.

    This is used to skip the rest of the operations in a `Compose` block.
    """

    def __init__(
        self,
        function: Callable[Concatenate[str, P], Any],
    ):
        self.function = function

    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        output = self.function(str(root), *dataset_args, **dataset_kwargs)
        if output:
            raise Compose.Stop()
        return str(root)
