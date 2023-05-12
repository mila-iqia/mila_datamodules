from __future__ import annotations

import shlex
import shutil
import subprocess
from logging import getLogger as get_logger
from pathlib import Path
from shutil import unpack_archive
from typing import Callable, Iterable
from zipfile import ZipFile

from typing_extensions import Concatenate

from mila_datamodules.blocks.types import PrepareDatasetFn
from mila_datamodules.cli.utils import is_local_main, runs_on_local_main_process_first
from mila_datamodules.types import D, D_co, P
from mila_datamodules.utils import copy_fn, dataset_name

logger = get_logger(__name__)


class CallDatasetFn(PrepareDatasetFn[D_co, P]):
    """Function that calls the dataset constructor with the given arguments.

    Parameters
    ----------
    dataset_type :
        The dataset type or callable.
    verify : bool, optional
        If `verify` is `True` and the `dataset_type` takes a `download` argument, then `download`
        is set to `False`. This is used to skip downloading files or verifying checksums of the
        archives, making things faster if we just want to check that the dataset is setup properly.
    get_index : int, optional
        If passed, the dataset instance is also indexed (`dataset[get_index]`) to check that the
        dataset is properly set up.
    """

    def __init__(
        self,
        dataset_fn: type[D_co] | Callable[Concatenate[str, P], D_co],
        extract_and_verify_archives: bool = False,
        get_index: int | None = 0,
    ):
        self.dataset_fn = dataset_fn
        self.extract_and_verify_archives = extract_and_verify_archives
        self.get_index = get_index

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        """Use the dataset constructor to prepare the dataset in the `root` directory.

        If the dataset has a `download` argument in its constructor, it will be set to `True` so
        the archives are extracted.

        NOTE: This should only really be called after the actual dataset preparation has been done
        in a subclass's `__call__` method.

        Returns `root` (as a string).
        """
        dataset_kwargs = dataset_kwargs.copy()  # type: ignore
        if self.extract_and_verify_archives:
            dataset_kwargs["download"] = True
            Path(root).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Extracting the dataset archives in {root}."
            if self.extract_and_verify_archives
            else f"Checking if the dataset is properly set up in {root}."
        )
        fn_name = dataset_name(self.dataset_fn)
        fn_call_message = f"{fn_name}({root!r}"
        if dataset_args or dataset_kwargs:
            fn_call_message += ", "
        if dataset_args:
            fn_call_message += ", ".join(f"{v!r}" for v in dataset_args)
        if dataset_kwargs:
            fn_call_message += ", ".join(f"{k}={v!r}" for k, v in dataset_kwargs.items())
        fn_call_message += ")"
        logger.debug(f"Calling {fn_call_message}")

        dataset_instance = self.dataset_fn(str(root), *dataset_args, **dataset_kwargs)
        if is_local_main():
            logger.info(f"Successfully read dataset:\n{dataset_instance}")

        if self.get_index is not None:
            _ = dataset_instance[self.get_index]
            if is_local_main():
                logger.debug(f"Sample at index {self.get_index}:\n{_}")
        return str(root)


class ExtractArchives(PrepareDatasetFn[D_co, P]):
    """Extract some archives files in a subfolder of the `root` directory."""

    def __init__(self, archives: dict[str, str | Path]):
        """
        Parameters
        ----------

        - archives:
            A mapping from an archive name to path where the archive
            should be extracted (relative to the 'root' dir).
            The destination paths need to be relative.
        """
        self.archives = {glob: Path(path) for glob, path in archives.items()}

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        logger.info(f"Extracting archives in {root}...")
        for archive, dest in self.archives.items():
            archive = Path(archive)
            assert not dest.is_absolute()

            dest = root / dest
            logger.debug(f"Extracting {archive} in {dest}")
            if archive.suffix == ".zip":
                with ZipFile(root / archive) as zf:
                    zf.extractall(str(dest))
            else:
                unpack_archive(archive, extract_dir=dest)

        return str(root)


class MoveFiles(PrepareDatasetFn[D, P]):
    """Reorganize datasets' files in the `root` directory."""

    def __init__(self, files: dict[str, str | Path]):
        """
        Parameters
        ----------

        - files:
            A mapping from an archive and a destination's path where the result
            should be moved and replaced.

            If the destination path's leaf is "*", the destination's parent will be used to hold
            the file. If not, the destination will be used as the target for the move.
            The files are moved in sequence. The destination's path should be relative.
        """
        self.files = {source: Path(destination) for source, destination in files.items()}

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        root = Path(root)
        move_files(root, self.files)
        return str(root)


def move_files(root: Path, files: dict[str, Path]) -> None:
    logger.info(f"Moving files in {root}...")
    for source, dest in files.items():
        assert not dest.is_absolute()
        dest = root / dest

        logger.debug(f"Moving {source} to {dest}")

        # Move a single file or directory. Simple.
        if "*" not in source and "*" not in dest.name:
            source = root / source
            logger.debug(f"Moving {source} to {dest}")
            shutil.move(source, dest)
            continue

        # TODO: Debugging.
        assert source.endswith("*"), source
        dest_dir = dest.parent if dest.name == "*" else dest
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Move things with a glob pattern. A bit more difficult.
        for source_file_or_dir in list(root.glob(str(source))):
            assert source_file_or_dir.exists(), source_file_or_dir
            dest_path = dest_dir / source_file_or_dir.name
            if dest_path.exists():
                continue
            if dest_dir.is_relative_to(source_file_or_dir):
                # Don't move a directory into itself.
                # (e.g. with KMNIST, there's a MoveFiles({"*": "KMNIST/raw/*"}).
                continue
            logger.debug(f"Moving {source_file_or_dir} in {dest_dir}")
            shutil.move(source_file_or_dir, dest_dir)


class CopyFiles(PrepareDatasetFn[D, P]):
    """Copies some files from the cluster to the `root` directory."""

    def __init__(
        self,
        relative_paths_to_files: dict[str, str | Path],
        ignore_dirs: Iterable[str] = (".git"),
    ):
        self.relative_paths_to_cluster_path = {
            relative_path: Path(path) for relative_path, path in relative_paths_to_files.items()
        }
        # Note; this could be a lazy Path().glob object perhaps?
        self.ignore_dirs = ignore_dirs

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path,
        *constructor_args: P.args,
        **constructor_kwargs: P.kwargs,
    ):
        root = Path(root)
        logger.info(f"Copying files from the network filesystem to {root}.")
        copy_files(root, self.relative_paths_to_cluster_path, self.ignore_dirs)
        return str(root)


def copy_files(root: Path, relative_paths_to_cluster_paths: dict, ignore_dirs: Iterable[str]):
    assert all(
        path_on_cluster.exists() for path_on_cluster in relative_paths_to_cluster_paths.values()
    )

    for relative_path, path_on_cluster in relative_paths_to_cluster_paths.items():
        dest_path = root / relative_path
        if relative_path == ".":
            logger.warning(
                RuntimeWarning(
                    f"Don't set '.' as the relative path for a copy. Use the folder name."
                    f"(Using {path_on_cluster.name} as the destination."
                )
            )
            relative_path = path_on_cluster.name

        if dest_path.exists():
            logger.debug(
                f"Skipping copying {path_on_cluster} as it already exists at {dest_path}."
            )
            continue

        if path_on_cluster.is_dir():
            logger.debug(f"Copying directory {path_on_cluster} -> {dest_path}.")

            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                path_on_cluster,
                dest_path,
                ignore=shutil.ignore_patterns(*ignore_dirs),
                dirs_exist_ok=True,
                copy_function=copy_fn,
            )
            # TODO: Figure out the right way to do this here in Python directly. Nothing seems
            # to work.
            subprocess.check_call(shlex.split(f"chmod -R a+w {dest_path}"))
        else:
            logger.debug(f"Copying file {path_on_cluster} -> {dest_path}.")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path_on_cluster, dest_path)
            dest_path.chmod(0o755)
