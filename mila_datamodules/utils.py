from __future__ import annotations

import functools
import inspect
import os
import shutil
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import tqdm
from torch.utils.data import Dataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters.utils import on_slurm_cluster

logger = get_logger(__name__)

V = TypeVar("V")
_T = TypeVar("_T", bound=type)
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
# if in_job_process_without_slurm_env_vars():
#     setup_slurm_env_variables()


def all_files_exist(
    required_files: Sequence[str | Path],
    base_dir: str | Path,
) -> bool:
    return all((Path(base_dir) / f).exists() for f in required_files)


def replace_root(dataset_type: Callable[Concatenate[str, P], D], root: str | Path):
    fn = replace_kwargs(dataset_type, root=str(root))

    def wrapped(root: Path | None = None, *args: P.args, **kwargs: P.kwargs) -> D:
        return fn(*args, **kwargs)

    return wrapped


def replace_kwargs(function: C, **fixed_arguments) -> C:
    """Returns a callable where the given argument values are fixed.

    NOTE: Simply using functools.partial wouldn't work, since passing one of the fixed arguments
    would raise an error.
    TODO: Double-check that functools.partial isn't enough here.
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
    TODO: Double-check that functools.partial isn't enough here.
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
    """Sets the mode of a file/dir and all its subdirectories and files to `mode`.

    TODO: Doesn't seem to be completely working.
    """
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


def copytree_with_symlinks(
    src_dir_with_files: Path,
    dst_dir_with_links: Path,
    replace_real_files_with_symlinks: bool = False,
    disable_pbar: bool = False,
):
    """same as sshutil.copytree, but creates symlinks instead of copying the files.

    For every file in `dir_with_files`, create a link to it in `dir_with_links`.
    """
    pbar = tqdm.tqdm(disable=disable_pbar)

    def _copy_fn(src: str, dst: str) -> None:
        # NOTE: This also overwrites the files in the user directory with symlinks to the same
        # files in the shared directory. We might not necessarily want to do that.
        # For instance, we might want to do a checksum or something first, to check that they have
        # exactly the same contents.
        src_path = Path(src)
        dst_path = Path(dst)
        rel_s = src_path.relative_to(src_dir_with_files)
        # rel_d = dst_path.relative_to(dst_dir_with_links)

        if dst_path.exists():
            if dst_path.is_symlink():
                # From a previous run.
                return
            if replace_real_files_with_symlinks:
                # Replace "real" files with symlinks.
                dst_path.unlink()

        # print(f"Linking {rel_s}")
        pbar.set_description(f"Linking {rel_s}")
        pbar.update(1)
        dst_path.symlink_to(src_path)

    shutil.copytree(
        src_dir_with_files,
        dst_dir_with_links,
        symlinks=True,
        copy_function=_copy_fn,
        dirs_exist_ok=True,
    )
