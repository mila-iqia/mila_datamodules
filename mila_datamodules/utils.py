from __future__ import annotations

import functools
import inspect
import os
import shutil
import socket
import subprocess
import tempfile
from logging import getLogger as get_logger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import torchvision.datasets as tvd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = Callable[P, D]


def get_slurm_tmpdir() -> Path:
    """Returns the SLURM temporary directory.

    Also works when using `mila code`.
    """
    if "SLURM_TMPDIR" in os.environ:
        return Path(os.environ["SLURM_TMPDIR"])
    if "SLURM_JOB_ID" in os.environ:
        # Running with `mila code`, so we don't have all the SLURM environment variables.
        # However, we're lucky, because we can reliably predict the SLURM_TMPDIR given the job id.
        job_id = os.environ["SLURM_JOB_ID"]
        return Path(f"/Tmp/slurm.{job_id}.0")
    raise RuntimeError(
        "None of SLURM_TMPDIR or SLURM_JOB_ID are set! "
        "Cannot locate the SLURM temporary directory."
    )


def get_cpus_on_node() -> int:
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    return cpu_count()


def all_files_exist(
    required_files: Sequence[str],
    base_dir: Path,
) -> bool:
    return all((base_dir / f).exists() for f in required_files)


def replace_root(dataset_type: Callable[Concatenate[str, P], D], root: str | Path):
    fn = replace_kwargs(dataset_type, root=str(root))

    def wrapped(root: Path | None = None, *args: P.args, **kwargs: P.kwargs) -> D:
        return fn(*args, **kwargs)

    return wrapped


def replace_kwargs(dataset_type: Callable[P, D], **fixed_arguments):
    """Returns a callable where the given argument values are fixed.

    NOTE: Simply using functools.partial wouldn't work, since passing one of the fixed arguments
    would raise an error.
    """
    init_signature = inspect.signature(dataset_type)

    @functools.wraps(dataset_type)
    def _wrap(*args: P.args, **kwargs: P.kwargs) -> D:
        bound_signature = init_signature.bind_partial(*args, **kwargs)
        for key, value in fixed_arguments.items():
            bound_signature.arguments[key] = value
        args = bound_signature.args  # type: ignore
        kwargs = bound_signature.kwargs  # type: ignore
        return dataset_type(*args, **kwargs)

    return _wrap


def copy_dataset_files(files_to_copy: Sequence[str], source_dir: Path, dest_dir: Path) -> None:
    assert all_files_exist(files_to_copy, base_dir=source_dir)
    for source_file in files_to_copy:
        source_path = source_dir / source_file
        destination_path = dest_dir / source_file
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {source_path} -> {destination_path}")
        if source_path.is_dir():
            # Copy the folder over.
            # TODO: Check that this doesn't overwrite existing files.
            # TODO: Test this out with symlinks?
            shutil.copytree(
                src=source_path,
                dst=destination_path,
                symlinks=False,
                dirs_exist_ok=True,
            )
        elif not destination_path.exists():
            # Copy the file over.
            shutil.copy(src=source_path, dst=destination_path, follow_symlinks=False)
