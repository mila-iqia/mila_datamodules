from __future__ import annotations

import shutil
from logging import getLogger as get_logger
from pathlib import Path
from shutil import unpack_archive
from typing import Callable, Iterable, Mapping, Protocol
from zipfile import ZipFile

from typing_extensions import Concatenate

from mila_datamodules.cli.utils import is_local_main, runs_on_local_main_process_first

from .types import D, D_co, P

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser


class PrepareDatasetFn(Protocol[D_co, P]):
    def __call__(
        self,
        root: str | Path,
        /,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        raise NotImplementedError


class CallDatasetConstructor(PrepareDatasetFn[D_co, P]):
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
        dataset_type: Callable[Concatenate[str, P], D_co],
        extract_and_verify_archives: bool = False,
        get_index: int | None = 0,
    ):
        self.dataset_type = dataset_type
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
        Path(root).mkdir(parents=True, exist_ok=True)

        dataset_kwargs = dataset_kwargs.copy()  # type: ignore
        if self.extract_and_verify_archives:
            dataset_kwargs["download"] = True

        logger.info(
            f"Extracting the dataset archives in {root}."
            if self.extract_and_verify_archives
            else f"Checking if the dataset is properly set up in {root}."
        )
        fn_name = getattr(self.dataset_type, "__name__", str(self.dataset_type))
        fn_call_message = f"{fn_name}({root!r}"
        if dataset_args or dataset_kwargs:
            fn_call_message += ", "
        if dataset_args:
            fn_call_message += ", ".join(f"{v!r}" for v in dataset_args)
        if dataset_kwargs:
            fn_call_message += ", ".join(f"{k}={v!r}" for k, v in dataset_kwargs.items())
        fn_call_message += ")"
        logger.debug(f"Calling {fn_call_message}")

        dataset_instance = self.dataset_type(str(root), *dataset_args, **dataset_kwargs)
        if is_local_main():
            logger.info(f"Successfully read dataset:\n{dataset_instance}")

        if self.get_index is not None:
            _ = dataset_instance[self.get_index]
            if is_local_main():
                logger.debug(f"Sample at index {self.get_index}:\n{_}")
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


class MakeSymlinksToDatasetFiles(PrepareDatasetFn[D_co, P]):
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
        logger.info(f"Making symlinks in {root} pointing to the dataset files on the network.")
        for relative_path, dataset_file in self.relative_paths_to_files.items():
            assert dataset_file.exists(), dataset_file
            # Make a symlink in the local scratch directory to the archive on the network.
            archive_symlink = root / relative_path
            if archive_symlink.exists():
                continue

            archive_symlink.parent.mkdir(parents=True, exist_ok=True)
            archive_symlink.symlink_to(dataset_file)
            logger.debug(f"Making link from {archive_symlink} -> {dataset_file}")

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
            # TODO: Does this assume that the keys are globs? IF so, that's not intended, we should
            # be able to pass {"a.zip": "b.zip"}, not just globs.
            for entry in root.glob(glob):
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Avoid replacing dest by itself
                if dest.name == "*" and entry != dest.parent:
                    entry.replace(dest.parent / entry.name)
                elif dest.name != "*" and entry != dest:
                    entry.replace(dest)

        return str(root)


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
        assert all(
            path_on_cluster.exists()
            for path_on_cluster in self.relative_paths_to_cluster_path.values()
        )

        for relative_path, path_on_cluster in self.relative_paths_to_cluster_path.items():
            dest_path = root / relative_path
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
                    ignore=shutil.ignore_patterns(*self.ignore_dirs),
                    dirs_exist_ok=True,
                )
            else:
                logger.debug(f"Copying file {path_on_cluster} -> {dest_path}.")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path_on_cluster, dest_path)
                dest_path.chmod(0o511)

        return str(root)


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


class StopOnSuccess(PrepareDatasetFn[D, P]):
    """Raises a special Stop exception when running the given callable doesn't raise an exception.

    If an exception of a type matching one in `exceptions` is raised by the function, the exception
    is ignored. Other exceptions are raised.

    This is used to short-cut the list of operations to perform inside a `Compose` block.
    """

    def __init__(
        self,
        function: PrepareDatasetFn[D, P] | Callable[Concatenate[str, P], D],
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
            self.function(str(root), *dataset_args, **dataset_kwargs)
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
