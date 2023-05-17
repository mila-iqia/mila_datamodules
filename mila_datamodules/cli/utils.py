from __future__ import annotations

import contextlib
import functools
import logging
import os
import warnings
from logging import getLogger as get_logger
from typing import Callable, Iterable, TypeVar

import torch
import torch.distributed
import tqdm
from tqdm.rich import tqdm_rich
from tqdm.std import TqdmExperimentalWarning
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir

logger = get_logger(__name__)

C = TypeVar("C", bound=Callable)

current_cluster = Cluster.current_or_error()


def get_node_index() -> int:
    return int(os.environ["SLURM_NODEID"])


def get_rank() -> int:
    return int(os.environ["SLURM_PROCID"])


def get_local_rank() -> int:
    return int(os.environ["SLURM_LOCALID"])


def is_main():
    return get_rank() == 0


def is_local_main():
    return get_local_rank() == 0


@contextlib.contextmanager
def _goes_first(is_first: bool):
    if is_first:
        yield
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        yield


@contextlib.contextmanager
def main_process_first():
    if not torch.distributed.is_initialized():
        yield
        return
    with _goes_first(is_main()):
        yield


@contextlib.contextmanager
def local_main_process_first():
    if not torch.distributed.is_initialized():
        yield
        return
    with _goes_first(is_local_main()):
        yield


def runs_on_main_process_first(function: C) -> C:
    @functools.wraps(function)
    def _inner(*args, **kwargs):
        with main_process_first():
            return function(*args, **kwargs)

    return _inner  # type: ignore


def runs_on_local_main_process_first(function: C) -> C:
    @functools.wraps(function)
    def _inner(*args, **kwargs):
        with local_main_process_first():
            return function(*args, **kwargs)

    return _inner  # type: ignore


def replace_dir_name_with_SLURM_TMPDIR(some_string: str) -> str:
    slurm_tmpdir = get_slurm_tmpdir()

    new_name = some_string.replace(str(slurm_tmpdir), "os.environ['SLURM_TMPDIR']")
    return new_name


_P = ParamSpec("_P")
PbarType = TypeVar("PbarType", bound=tqdm.std.tqdm)

T = TypeVar("T")


def _tqdm_rich_pbar(
    fn: Callable[Concatenate[Iterable[T], _P], tqdm_rich] = tqdm_rich,
) -> Callable[Concatenate[Iterable[T], _P], tqdm_rich[T]]:
    @functools.wraps(fn)
    def _fn(
        seq: Iterable[T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> tqdm_rich[T]:
        # TODO: Disable all the output when a --quiet/-q flag is passed on the command-line.
        kwargs.setdefault("disable", logger.level == logging.NOTSET)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
            return fn(seq, *args, **kwargs)

    return _fn


pbar = _tqdm_rich_pbar()
