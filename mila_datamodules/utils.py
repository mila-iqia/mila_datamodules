from __future__ import annotations

import functools
import inspect
import os
import shutil
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar, overload

import tqdm
from torch.utils.data import Dataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters.env_variables import setup_slurm_env_variables
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
if in_job_process_without_slurm_env_vars():
    setup_slurm_env_variables()


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


def replace_kwargs(function: Callable[P, OutT], **fixed_arguments):
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
    files_to_copy: Sequence[str | Path], source_dir: str | Path, dest_dir: str | Path
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


def _get_key_to_use_for_indexing(potential_classes: Mapping[_T, Any], key: _T) -> _T:
    if key in potential_classes:
        return key
    # Return the entry with the same name, if `some_type` is a subclass of it.
    parent_classes_with_same_name = [
        cls for cls in potential_classes if cls.__name__ == key.__name__ and issubclass(key, cls)
    ]
    if len(parent_classes_with_same_name) == 0:
        # Can't find a key to use.
        raise KeyError(key)
    elif len(parent_classes_with_same_name) > 1:
        raise ValueError(
            f"Multiple parent classes with the same name: {parent_classes_with_same_name}"
        )
    key = parent_classes_with_same_name[0]
    return key


_V = TypeVar("_V")

_MISSING = object()


@overload
def getitem_with_subclasscheck(potential_classes: Mapping[_T, V], key: _T) -> V:
    ...


@overload
def getitem_with_subclasscheck(potential_classes: Mapping[_T, V], key: _T, default: _V) -> V | _V:
    ...


def getitem_with_subclasscheck(
    potential_classes: Mapping[_T, V], key: _T, default: _V = _MISSING
) -> V | _V:
    if key in potential_classes:
        return potential_classes[key]
    try:
        key = _get_key_to_use_for_indexing(potential_classes, key=key)
        return potential_classes[key]
    except KeyError:
        if default is not _MISSING:
            return default  # type: ignore
        raise


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
        # NOTE: This also overwrites the files in the user directory with symlinks to the same files in
        # the shared directory. We might not necessarily want to do that.
        # For instance, we might want to do a checksum or something first, to check that they have
        # exactly the same contents.
        src_path = Path(src)
        dst_path = Path(dst)
        rel_s = src_path.relative_to(src_dir_with_files)
        rel_d = dst_path.relative_to(dst_dir_with_links)

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
