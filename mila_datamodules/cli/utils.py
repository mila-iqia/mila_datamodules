from __future__ import annotations

import contextlib
import functools
import os
from pathlib import Path
from typing import Callable, TypeVar

import torch
import torch.distributed

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir

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


def replace_dir_name_with_SLURM_TMPDIR(dir_name: str | Path) -> str:
    slurm_tmpdir = get_slurm_tmpdir()
    if Path(dir_name).is_relative_to(slurm_tmpdir):
        new_name = str(dir_name).replace(str(slurm_tmpdir), "$SLURM_TMPDIR")
        return new_name
    return str(dir_name)
