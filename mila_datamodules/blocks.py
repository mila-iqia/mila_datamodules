from __future__ import annotations

import shlex
import shutil
import subprocess
from logging import getLogger as get_logger
from pathlib import Path
from shutil import unpack_archive
from typing import Callable, Iterable, Mapping, Protocol
from zipfile import ZipFile

from typing_extensions import Concatenate

from mila_datamodules.cli.types import D, D_co, P
from mila_datamodules.cli.utils import is_local_main, rich_pbar, runs_on_local_main_process_first
from mila_datamodules.clusters.utils import get_slurm_tmpdir
from mila_datamodules.utils import copy_fn

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
        dataset_kwargs = dataset_kwargs.copy()  # type: ignore
        if self.extract_and_verify_archives:
            dataset_kwargs["download"] = True
            Path(root).mkdir(parents=True, exist_ok=True)

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


PREPARED_DATASETS_FILE = "prepared_datasets.txt"


# TODO: Check if the dataset is already setup in another SLURM_TMPDIR, and if so,
# create hard links to the dataset files.
class AddDatasetNameToPreparedDatasetsFile(PrepareDatasetFn):
    def __init__(self, dataset_name: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name

    def __call__(self, root: str | Path, /, *args, **kwargs) -> str:
        prepared_datasets_file = Path(root) / PREPARED_DATASETS_FILE

        if prepared_datasets_file.exists():
            datasets = prepared_datasets_file.read_text().splitlines(keepends=False)
        else:
            datasets = []

        if self.dataset_name in datasets:
            logger.debug(f"Dataset {self.dataset_name} is already in the prepared datasets file.")
            return str(root)

        with open(prepared_datasets_file, "a") as f:
            logger.info(
                f"Adding the '{self.dataset_name}' to the prepared datasets file at "
                f"{prepared_datasets_file}."
            )
            f.write(self.dataset_name + "\n")
        return str(root)


def get_prepared_datasets_from_file(p: Path) -> list[str]:
    with open(p, "r") as f:
        return [line.strip() for line in f.readlines()]


class MakePreparedDatasetUsableByOthersOnSameNode(PrepareDatasetFn[D_co, P]):
    def __init__(self, readable_files_or_directories: list[str | Path] | None) -> None:
        super().__init__()
        self.readable_files_or_directories = readable_files_or_directories

    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        root = Path(root)
        files_to_make_readonly_to_others = (
            list(_tree(root))
            if not self.readable_files_or_directories
            else [
                root / f
                for file_or_dir in self.readable_files_or_directories
                for f in _tree(file_or_dir)
            ]
        )
        # TODO: Make the

        parent_dirs: set[Path] = set()
        for file in files_to_make_readonly_to_others:
            parent_dirs.update(file.parents)

        user = root.owner()
        for parent_dir in parent_dirs:
            if parent_dir.owner() == user:
                logger.debug(f"Making dir {parent_dir} readable by others on the same node.")
                parent_dir.chmod(parent_dir.stat().st_mode | 0o755)

        for file in rich_pbar(files_to_make_readonly_to_others, desc="Making files readable..."):
            file.chmod(0o755)

        # raise NotImplementedError(
        #     "TODO: Make the `root` directory and (only) the dataset files within it readable by "
        #     "others"
        # )
        return str(root)


class ReuseAlreadyPreparedDatasetOnSameNode(PrepareDatasetFn[D_co, P]):
    """Load the dataset by reusing a previously-prepared copy of the dataset on the same node.

    If no copy is available, raises a `RuntimeError`.
    NOTE: This is meant to be used wrapped by a `StopOnSuccess` inside a `Compose`. For example:

    ```python
    prepare_imagenet = Compose(
        # Try creating the dataset from the root directory. Stop if this works, else continue.
        StopOnSuccess(CallDatasetConstructor(tvd.ImageNet)),
        # Try creating the dataset by reusing a previously prepared copy on the same node.
        # Stop if this works, otherwise continue.
        StopOnSuccess(
            ReuseAlreadyPreparedDatasetOnSameNode(
                tvd.ImageNet,
                prepared_dataset_files_or_directories=[
                    "ILSVRC2012_devkit_t12.tar.gz",
                    "ILSVRC2012_img_train.tar",
                    "ILSVRC2012_img_val.tar",
                    "md5sums",
                    "meta.bin",
                    "train",
                ],
            )
        ),
        MakeSymlinksToDatasetFiles(f"{datasets_dir}/imagenet"),
        CallDatasetConstructor(tvd.ImageNet),
        AddDatasetToPreparedDatasetsFile(tvd.ImageNet),
    )
    """

    def __init__(
        self,
        dataset_type: type[D_co] | Callable[Concatenate[str, P], D_co],
        prepared_dataset_files_or_directories: list[str],
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset_files_or_directories = prepared_dataset_files_or_directories
        # TODO: Make this less torchvision-specific.
        self.dataset_name = getattr(dataset_type, "__name__", str(dataset_type)).lower()

    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        potential_dirs = cache_dirs_on_same_node_with_dataset_already_prepared(
            self.dataset_name, cache_dir=str(Path(root).relative_to(get_slurm_tmpdir()))
        )

        for potential_dir in potential_dirs:
            all_files_or_dirs_that_should_exist = [
                potential_dir / relative_path_to_file_or_dir
                for relative_path_to_file_or_dir in self.dataset_files_or_directories
            ]

            if not all(p.exists() for p in all_files_or_dirs_that_should_exist):
                logger.debug(
                    f"The SLURM_TMPDIR at {potential_dir} doesn't contain all the necessary files "
                    f"for this dataset."
                )
                continue

            logger.debug(f"Listing all the dataset files in {potential_dir}")
            all_files_to_link: set[Path] = set()
            for file_or_dir in all_files_or_dirs_that_should_exist:
                if file_or_dir.is_dir():
                    all_files_to_link.update(_tree(file_or_dir))
                else:
                    all_files_to_link.add(file_or_dir)

            link_paths_to_file_paths = {
                root / file.relative_to(potential_dir): file for file in all_files_to_link
            }
            if all(link_path.exists() for link_path in link_paths_to_file_paths):
                logger.debug(f"Links all already present in {root}!")
            else:
                logger.info(
                    f"Creating hard links in {root} pointing to the files in {potential_dir}."
                )
                make_links_to_dataset_files(link_paths_to_file_paths)

            root = CallDatasetConstructor(self.dataset_type, extract_and_verify_archives=False)(
                root, *dataset_args, **dataset_kwargs
            )
            logger.info(f"SUCCESS! Dataset was already prepared in {potential_dir}!")
            # TODO: If calling the dataset constructor doesn't work for some reason, perhaps we
            # should remove all the hard links we just created?
            return root

        logger.info("Unable to find an already prepared version of this dataset on this node.")
        raise RuntimeError()


def make_links_to_dataset_files(link_path_to_file_path: dict[Path, Path]):
    pbar = rich_pbar(list(link_path_to_file_path.items()), unit="Files", desc="Making links")
    for link_path, file_path in pbar:
        assert file_path.exists(), file_path
        # Make a symlink in the local scratch directory to the archive on the network.
        if link_path.exists():
            continue
        link_path.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: Inverse order of arguments compared to `Path.symlink_to`:
        # Make `link_path` a hard link to `file_path`.
        # logger.debug(f"Making hard link from {link_path} -> {file_path}")
        file_path.link_to(link_path)


def cache_dirs_on_same_node_with_dataset_already_prepared(
    dataset: str, cache_dir="cache"
) -> list[Path]:
    # TODO: `cache` is currently only created when using HuggingFace datasets.
    slurm_tmpdir = get_slurm_tmpdir()
    other_slurm_tmpdirs = [
        p for p in slurm_tmpdir.parent.iterdir() if p.is_dir() and p != slurm_tmpdir
    ]

    def _can_be_read(d: Path) -> bool:
        try:
            next(d.iterdir())
            logger.debug(f"Able to read from other {d} (owned by {d.owner()})!")
            return True
        except IOError as err:
            logger.debug(f"Unable to read from {d}: {err}")
            return False

    logger.debug(f"Other slurm TMPDIRS: {other_slurm_tmpdirs}")
    directories_that_can_be_read = [d for d in other_slurm_tmpdirs if _can_be_read(d)]
    # Look in those to check if any have a `cache` folder and possibly a file that shows which
    # dataset was prepared.
    usable_dirs = [
        d
        for d in directories_that_can_be_read
        if (d / cache_dir).is_dir() and (d / cache_dir / "prepared_datasets.txt").exists()
    ]

    prepared_datasets_per_dir = {
        (d / cache_dir): get_prepared_datasets_from_file(d / cache_dir / "prepared_datasets.txt")
        for d in usable_dirs
    }
    potential_directories = []
    for other_prepared_datasets_dir, prepared_datasets_in_dir in prepared_datasets_per_dir.items():
        if dataset in prepared_datasets_in_dir:
            potential_directories.append(other_prepared_datasets_dir)
    return potential_directories


def _tree(root: str | Path, ignore_prefix: tuple[str, ...] = (".",)) -> Iterable[Path]:
    root = Path(root)
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
            yield from _tree(entry, ignore_prefix=ignore_prefix)


def all_files_in_dir(
    source: str | Path, ignore_prefixes=(".", "scripts", "README")
) -> dict[str, Path]:
    source = Path(source).expanduser().resolve()
    return {
        str(file.relative_to(source)): file
        for file in _tree(Path(source), ignore_prefix=ignore_prefixes)
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
            self.relative_paths_to_files = all_files_in_dir(source)
        else:
            self.relative_paths_to_files = {
                str(k): Path(v) for k, v in source_dir_or_relative_paths_to_files.items()
            }

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        root = Path(root)
        make_symlinks_to_dataset_files(root, self.relative_paths_to_files)
        return str(root)


def make_symlinks_to_dataset_files(root: Path, relative_paths_to_files: dict[str, Path]):
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Making symlinks in {root} pointing to the dataset files on the network.")
    for relative_path, dataset_file in relative_paths_to_files.items():
        assert dataset_file.exists(), dataset_file
        # Make a symlink in the local scratch directory to the archive on the network.
        archive_symlink = root / relative_path
        if archive_symlink.exists():
            continue

        archive_symlink.parent.mkdir(parents=True, exist_ok=True)
        archive_symlink.symlink_to(dataset_file)
        logger.debug(f"Making link from {archive_symlink} -> {dataset_file}")


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


class SkipRestIfThisWorks(PrepareDatasetFn[D, P]):
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
