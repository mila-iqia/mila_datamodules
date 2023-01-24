from __future__ import annotations

import functools
import inspect
import os
import shutil
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from torch.utils.data import Dataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters.env_variables import setup_slurm_env_variables
from mila_datamodules.clusters.utils import on_slurm_cluster

T = TypeVar("T")

logger = get_logger(__name__)

T = TypeVar("T")
OutT = TypeVar("OutT")
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = Callable[P, D]


def in_job_process_without_slurm_env_vars() -> bool:
    """Returns `True` if this process is being executed inside another shell of the job (e.g. when
    using `mila code`, the vscode shell doesn't have the SLURM environment variables set)."""
    return on_slurm_cluster() and "SLURM_JOB_ID" in os.environ and "SLURM_TMPDIR" not in os.environ


# Load the SLURM environment variables into the current environment, if we're running inside a job
# but don't have the SLURM variables set.
if in_job_process_without_slurm_env_vars():
    setup_slurm_env_variables()


def all_files_exist(
    required_files: Sequence[str],
    base_dir: str | Path,
) -> bool:
    return all((Path(base_dir) / f).exists() for f in required_files)


def replace_root(dataset_type: Callable[Concatenate[str, P], D], root: str | Path):
    fn = replace_kwargs(dataset_type, root=str(root))

    def wrapped(root: Path | None = None, *args: P.args, **kwargs: P.kwargs) -> D:
        return fn(*args, **kwargs)

    return wrapped


def replace_kwargs(function: Callable[P, OutT], **fixed_arguments):
    """Returns a callable where the given argument values are fixed.

    NOTE: Simply using functools.partial wouldn't work, since passing one of the fixed arguments
    would raise an error.
    """
    init_signature = inspect.signature(function)

    @functools.wraps(function)
    def _wrap(*args: P.args, **kwargs: P.kwargs) -> OutT:
        bound_signature = init_signature.bind_partial(*args, **kwargs)
        for key, value in fixed_arguments.items():
            bound_signature.arguments[key] = value
        args = bound_signature.args  # type: ignore
        kwargs = bound_signature.kwargs  # type: ignore
        return function(*args, **kwargs)

    return _wrap


def replace_arg_defaults(
    dataset_type: Callable[P, T], *new_default_args: P.args, **new_default_kwargs: P.kwargs
) -> Callable[P, T]:
    """Returns a callable where the given argument have a different default value.

    NOTE: Simply using functools.partial wouldn't work, since passing one of the fixed arguments
    would raise an error.
    """
    init_signature = inspect.signature(dataset_type)
    new_defaults = init_signature.bind_partial(*new_default_args, **new_default_kwargs)

    @functools.wraps(dataset_type)
    def _wrap(*args: P.args, **kwargs: P.kwargs) -> T:
        bound_signature = init_signature.bind_partial(*args, **kwargs)
        for key, value in new_defaults.arguments.items():
            bound_signature.arguments.setdefault(key, value)
        args = bound_signature.args  # type: ignore
        kwargs = bound_signature.kwargs  # type: ignore
        return dataset_type(*args, **kwargs)

    return _wrap


def copy_fn(src: str | Path, dest: str | Path):
    """Copies a file/dir from `src` to `dest` and sets the mode of the copy to 644."""
    src_path = Path(src).resolve()
    if src_path.is_dir():
        os.mkdir(dest, mode=0o644)
    else:
        shutil.copyfile(src_path, dest, follow_symlinks=False)
        os.chmod(dest, 0o644)


def chmod_recursive(path: str | Path, mode: int):
    """Sets the mode of a file/dir and all its subdirectories and files to `mode`."""
    path = Path(path).resolve()
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)


def extract_archive(archive_path: str | Path, dest: str | Path):
    """Extracts an archive to `dest` and sets the mode of all extracted files to 644."""
    shutil.unpack_archive(archive_path, dest)
    chmod_recursive(dest, 0o644)


def copy_dataset_files(
    files_to_copy: Sequence[str], source_dir: str | Path, dest_dir: str | Path
) -> None:
    """TODO: If the file is an archive, extract it into the destination directory, rather than
    copy the files? (see https://github.com/lebrice/mila_datamodules/issues/4).
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    assert all_files_exist(files_to_copy, base_dir=source_dir)

    for source_file in files_to_copy:
        source_path = source_dir / source_file
        destination_path = dest_dir / source_file
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir():
            print(f"Copying {source_path} -> {destination_path}")
            # Copy the folder over.
            # TODO: Getting a weird error with CIFAR100. Seems to also be related to SCRATCH
            # directory contents.
            destination_path.mkdir(parents=True, exist_ok=True)
            # TODO: Check that this doesn't overwrite existing files.
            # TODO: Test this out with symlinks?
            shutil.copytree(
                src=source_path,
                dst=destination_path,
                symlinks=False,
                dirs_exist_ok=True,
                copy_function=copy_fn,
            )
        elif source_path.suffix in [".tar", ".zip", ".gz"]:
            print(f"Extracting {source_path} -> {destination_path}")
            # Extract the archive.
            extract_archive(source_path, destination_path)

        elif not destination_path.exists():
            print(f"Copying {source_path} -> {destination_path}")
            # Copy the file over.
            try:
                copy_fn(source_path, destination_path)
            except FileExistsError:
                # Weird. Getting a FileExistsError for SVHN, even though we checked that the
                # destination path didn't already exist...
                pass
