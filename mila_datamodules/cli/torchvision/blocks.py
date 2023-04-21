from __future__ import annotations

import contextlib
import inspect
import shutil
from logging import getLogger as get_logger
from pathlib import Path
from shutil import unpack_archive
from typing import (
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
)
from zipfile import ZipFile

from typing_extensions import Concatenate

from mila_datamodules.cli.utils import is_local_main, runs_on_local_main_process_first
from mila_datamodules.clusters.utils import get_slurm_tmpdir

from ._types import VD, P, VD_co

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser
SLURM_TMPDIR = get_slurm_tmpdir()


class PrepareVisionDataset(Protocol[VD_co, P]):
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        raise NotImplementedError


class CallDatasetConstructor(PrepareVisionDataset[VD_co, P]):
    def __init__(self, dataset_type: Callable[Concatenate[str, P], VD_co], verify=False):
        self.dataset_type = dataset_type
        self.verify = verify

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        """Use the dataset constructor to prepare the dataset in the `root` directory.

        If the dataset has a `download` argument in its constructor, it will be set to `True` so
        the archives are extracted.

        NOTE: This should only really be called after the actual dataset preparation has been done
        in a subclass's `__call__` method.

        Returns `root` (as a string).
        """
        Path(root).mkdir(parents=True, exist_ok=True)

        dataset_kwargs = dataset_kwargs.copy()  # type: ignore
        if "download" in inspect.signature(self.dataset_type).parameters:
            dataset_kwargs["download"] = not self.verify

        logger.debug(
            f"Using dataset constructor: {self.dataset_type} with args {dataset_args}, and "
            f"kwargs {dataset_kwargs}"
        )
        dataset_instance = self.dataset_type(str(root), *dataset_args, **dataset_kwargs)
        if is_local_main():
            print(dataset_instance)
        return str(root)


def _recursive_list_files(root: Path, ignore_prefix: tuple[str, ...] = (".",)) -> Iterable[Path]:
    if not root.exists():
        return []

    for entry in root.iterdir():
        if entry.name.startswith(ignore_prefix):
            continue
        if entry.is_file():
            yield entry
        if entry.is_dir():
            # NOTE: The Path objects here will have the right prefix (including `root`). No need
            # to add it.
            yield from _recursive_list_files(entry, ignore_prefix=ignore_prefix)


def dataset_files_in_source_dir(
    source: str | Path, ignore_prefixes=(".", "scripts", "README")
) -> dict[str, Path]:
    source = Path(source).expanduser().resolve()
    return {
        str(file.relative_to(source)): file
        for file in _recursive_list_files(Path(source), ignore_prefix=ignore_prefixes)
    }


class MakeSymlinksToDatasetFiles(PrepareVisionDataset[VD, P]):
    """Creates symlinks to the datasets' files in the `root` directory."""

    def __init__(
        self,
        source_dir_or_relative_paths_to_files: str | Path | Mapping[str, str | Path],
    ):
        """
        Parameters
        ----------

        - source_or_relative_paths_to_files:
            Either a source directory, in which case all the files under that directory are
            symlinked, or a mapping from filenames (relative to the 'root' directory) where the
            symlink should be created, to the absolute path to the file on the cluster.
        """
        self.relative_paths_to_files: dict[str, Path]
        if isinstance(source_dir_or_relative_paths_to_files, (str, Path)):
            source = source_dir_or_relative_paths_to_files
            self.relative_paths_to_files = dataset_files_in_source_dir(source)
        else:
            self.relative_paths_to_files = {
                str(k): Path(v) for k, v in source_dir_or_relative_paths_to_files.items()
            }

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        for relative_path, dataset_file in self.relative_paths_to_files.items():
            assert dataset_file.exists()
            # Make a symlink in the local scratch directory to the archive on the network.
            archive_symlink = root / relative_path
            if archive_symlink.exists():
                continue

            archive_symlink.parent.mkdir(parents=True, exist_ok=True)
            archive_symlink.symlink_to(dataset_file)
            print(f"Making link from {archive_symlink} -> {dataset_file}")

        return str(root)


class ExtractArchives(PrepareVisionDataset[VD, P]):
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
        for archive, dest in self.archives.items():
            archive = Path(archive)
            assert not dest.is_absolute()

            dest = root / dest
            print(f"Extracting {archive} in {dest}")
            if archive.suffix == ".zip":
                with ZipFile(root / archive) as zf:
                    zf.extractall(str(dest))
            else:
                unpack_archive(archive, extract_dir=dest)

        return str(root)


class MoveFiles(PrepareVisionDataset[VD, P]):
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
        self.files = [(glob, Path(path)) for glob, path in files.items()]

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        root = Path(root)
        for glob, dest in self.files:
            assert not dest.is_absolute()
            dest = root / dest
            for entry in root.glob(glob):
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Avoid replacing dest by itself
                if dest.name == "*" and entry != dest.parent:
                    entry.replace(dest.parent / entry.name)
                elif dest.name != "*" and entry != dest:
                    entry.replace(dest)

        return str(root)


class CopyTree(CallDatasetConstructor[VD, P]):
    """Copies a tree of files from the cluster to the `root` directory."""

    def __init__(
        self,
        dataset_type: Callable[Concatenate[str, P], VD],
        relative_paths_to_dirs: dict[str, str | Path],
        ignore_filenames: Sequence[str] = (".git",),
    ):
        self.dataset_type = dataset_type
        self.relative_paths_to_dirs = {
            relative_path: Path(path) for relative_path, path in relative_paths_to_dirs.items()
        }
        self.ignore_dirs = ignore_filenames

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *constructor_args: P.args,
        **constructor_kwargs: P.kwargs,
    ):
        assert all(directory.exists() for directory in self.relative_paths_to_dirs.values())

        root = Path(root)
        for relative_path, tree in self.relative_paths_to_dirs.items():
            dest_dir = root / relative_path
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                tree,
                dest_dir,
                ignore=shutil.ignore_patterns(*self.ignore_dirs),
                dirs_exist_ok=True,
            )

        return super()(root, *constructor_args, **constructor_kwargs)


class Compose(PrepareVisionDataset[VD_co, P]):
    class Stop(Exception):
        pass

    def __init__(self, *callables: PrepareVisionDataset[VD_co, P]) -> None:
        self.callables = callables

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
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


class StopOnSucess(PrepareVisionDataset[VD, P]):
    """Raises a special Stop exception when running the given callable doesn't raise an exception.

    If an exception of a type matching one in `exceptions` is raised by the function, the exception
    is ignored. Other exceptions are raised.

    This is used to short-cut the list of operations to perform inside a `Compose` block.
    """

    def __init__(
        self,
        function: PrepareVisionDataset[VD, P] | Callable[Concatenate[str, P], VD],
        exceptions: Sequence[type[Exception]] = (),
    ):
        self.function = function
        self.exceptions = exceptions

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        with contextlib.suppress(*self.exceptions):
            self.function(str(root), *dataset_args, **dataset_kwargs)
            raise Compose.Stop()
        return str(root)
