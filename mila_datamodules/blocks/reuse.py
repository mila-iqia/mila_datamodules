from __future__ import annotations

import dataclasses
import inspect
import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, Union

import yaml

from mila_datamodules.blocks.base import CallDatasetFn, DatasetFnWithStrArg
from mila_datamodules.blocks.compose import Compose
from mila_datamodules.blocks.path_utils import has_permission, tree
from mila_datamodules.blocks.types import PrepareDatasetFn
from mila_datamodules.cli.utils import rich_pbar
from mila_datamodules.clusters.utils import get_slurm_tmpdir
from mila_datamodules.types import D_co, P
from mila_datamodules.utils import dataset_name

logger = get_logger(__name__)
PREPARED_DATASETS_FILE = "prepared_datasets.yaml"

DatasetFn = Union[type[D_co], Callable[P, D_co]]


@dataclass(frozen=True)
class PreparedDatasetEntry:
    """Simple schema for the entries in the prepared datasets file.

    Each entry represents a dataset that has already been prepared.
    """

    # NOTE: Storing the dataset function, just to keep this as general as possible.
    # dataset_name: str

    dataset_fn: Callable

    dataset_args: list[Any]
    """The positional arguments that are passed to the dataset function."""

    dataset_kwargs: dict[str, Any]
    """The keyword arguments that are passed to the dataset function (e.g. split="val")."""

    # TODO: Actually make use of this, for example if there is the dataset is already prepared in a
    # different dir, then reuse it.
    prepared_at: str


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
        dataset_fn: DatasetFn[D_co, P],
        prepared_dataset_files_or_directories: list[str],
        extra_files_depending_on_kwargs: dict[str, dict[Any, str | list[str]]] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_fn = dataset_fn
        self.dataset_files_or_directories = prepared_dataset_files_or_directories
        self.extra_files_depending_on_kwargs = extra_files_depending_on_kwargs or {}
        # TODO: Make this less torchvision-specific.
        self.dataset_name = dataset_name(dataset_fn)

    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        root = Path(root)
        dataset_files_or_directories = self.dataset_files_or_directories.copy()
        for kwarg, value_to_extra_files_or_dirs in self.extra_files_depending_on_kwargs.items():
            if dataset_kwargs.get(kwarg) in value_to_extra_files_or_dirs:
                extra_path_or_paths = value_to_extra_files_or_dirs[dataset_kwargs.get(kwarg)]
                logger.info(f"Also sharing {extra_path_or_paths} with other users on this node.")
                if isinstance(extra_path_or_paths, (str, Path)):
                    dataset_files_or_directories.append(str(root / extra_path_or_paths))
                else:
                    dataset_files_or_directories.extend(str(root / p) for p in extra_path_or_paths)

        success = reuse_already_prepared_dataset_on_same_node(
            root=Path(root),
            dataset_name=self.dataset_name,
            dataset_fn=self.dataset_fn,
            dataset_files_or_directories=dataset_files_or_directories,
            *dataset_args,
            **dataset_kwargs,
        )
        if success:
            return str(root)
        else:
            # Raise an error so the SkipIf... block stops.
            raise RuntimeError()


# TODO: Check if the dataset is already setup in another SLURM_TMPDIR, and if so,
# create hard links to the dataset files.
class AddDatasetNameToPreparedDatasetsFile(PrepareDatasetFn[D_co, P]):
    def __init__(self, dataset_fn: type[D_co] | Callable[P, D_co]) -> None:
        super().__init__()
        self.dataset_fn = dataset_fn
        self.dataset_name = dataset_name(dataset_fn)

    def __call__(self, root: str | Path, /, *args: P.args, **kwargs: P.kwargs) -> str:
        # TODO: Use a yaml file so we can save more information than just the dataset name.
        # TODO: Also save the *args and **kwargs in the file so we can check if the chosen split
        # was already processed.
        add_dataset_to_prepared_datasets_file(self.dataset_fn, root=root, *args, **kwargs)
        return str(root)


class SkipIfAlreadyPrepared(PrepareDatasetFn[D_co, P]):
    def __init__(self, dataset_fn: type[D_co] | Callable[P, D_co]) -> None:
        super().__init__()
        self.dataset_fn = dataset_fn
        self.dataset_name = dataset_name(dataset_fn)

    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        if is_already_prepared(self.dataset_fn, root=root, *dataset_args, **dataset_kwargs):
            raise Compose.Stop
        return str(root)


class MakePreparedDatasetUsableByOthersOnSameNode(PrepareDatasetFn[D_co, P]):
    def __init__(
        self,
        readable_files_or_directories: list[str] | None,
        dataset_fn: type[D_co] | Callable[P, D_co] | None = None,
        extra_files_depending_on_kwargs: dict[str, dict[Any, str | list[str]]] | None = None,
    ) -> None:
        super().__init__()
        if readable_files_or_directories is None and extra_files_depending_on_kwargs:
            raise RuntimeError(
                f"Cannot use {readable_files_or_directories=} and extra_files_depending_on_kwargs"
            )

        self.readable_files_or_directories = readable_files_or_directories
        self.extra_files_depending_on_kwargs = extra_files_depending_on_kwargs or {}
        self.dataset_fn = dataset_fn

    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        root = Path(root)
        if self.readable_files_or_directories is None:
            readable_files_or_directories = None
        else:
            readable_files_or_directories = self.readable_files_or_directories.copy()
            if self.extra_files_depending_on_kwargs:
                readable_files_or_directories.extend(
                    _extra_values_based_on_kwargs(
                        root,
                        extra_files_depending_on_kwargs=self.extra_files_depending_on_kwargs,
                        dataset_fn=self.dataset_fn,
                        *dataset_args,
                        **dataset_kwargs,
                    )
                )

        make_prepared_dataset_usable_by_others_on_same_node(root, readable_files_or_directories)
        return str(root)


def reuse_already_prepared_dataset_on_same_node(
    root: Path,
    dataset_name: str,
    dataset_fn: DatasetFnWithStrArg[D_co, P],
    dataset_files_or_directories: list[str],
    *dataset_args: P.args,
    **dataset_kwargs: P.kwargs,
) -> bool:
    potential_dirs = cache_dirs_on_same_node_with_dataset_already_prepared(
        root=root,
        dataset_name=dataset_name,
    )

    for potential_dir in potential_dirs:
        all_files_or_dirs_that_should_exist = [
            potential_dir / relative_path_to_file_or_dir
            for relative_path_to_file_or_dir in dataset_files_or_directories
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
                all_files_to_link.update(tree(file_or_dir))
            else:
                all_files_to_link.add(file_or_dir)

        link_paths_to_file_paths = {
            root / file.relative_to(potential_dir): file for file in all_files_to_link
        }
        if all(link_path.exists() for link_path in link_paths_to_file_paths):
            logger.debug(f"Links all already present in {root}!")
        else:
            logger.info(f"Creating hard links in {root} pointing to the files in {potential_dir}.")
            make_links_to_dataset_files(link_paths_to_file_paths)

        root_str = CallDatasetFn(dataset_fn, extract_and_verify_archives=False)(
            root, *dataset_args, **dataset_kwargs
        )
        root = Path(root_str)
        logger.info(f"SUCCESS! Dataset was already prepared in {potential_dir}!")
        # TODO: If calling the dataset constructor doesn't work for some reason, perhaps we
        # should remove all the hard links we just created?
        return True

    logger.info("Unable to find an already prepared version of this dataset on this node.")
    return False


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


def get_other_slurm_tmpdirs(root_dir_in_this_job: Path | None = None) -> list[Path]:
    # TODO: This might vary by cluster. Assumes that SLURM_TMPDIR is in a dir with other
    # SLURM_TMPDIR's
    root_dir_in_this_job = root_dir_in_this_job or get_slurm_tmpdir()
    return list(
        d
        for d in root_dir_in_this_job.parent.iterdir()
        if d.is_dir() and d != root_dir_in_this_job and "slurm" in d.name
    )


def cache_dirs_on_same_node_with_dataset_already_prepared(
    root: Path,
    dataset_name: str,
) -> list[Path]:
    root = Path(root)
    # TODO: `cache` is currently only created when using HuggingFace datasets.
    slurm_tmpdir = get_slurm_tmpdir()
    other_slurm_tmpdirs = get_other_slurm_tmpdirs()

    if not root.is_relative_to(slurm_tmpdir):
        raise RuntimeError(f"Expected root ({root}) to be under SLURM_TMPDIR ({slurm_tmpdir})!")

    relative_path_to_root = root.relative_to(slurm_tmpdir)
    other_root_dirs = [
        other_slurm_tmpdir / relative_path_to_root for other_slurm_tmpdir in other_slurm_tmpdirs
    ]

    def _can_be_used(d: Path) -> bool:
        try:
            return (
                d.exists()
                and d.is_dir()
                and has_permission(d, "x", "others")
                and (d / PREPARED_DATASETS_FILE).exists()
                and has_permission(d / PREPARED_DATASETS_FILE, "r", "others")
                and dataset_name in get_prepared_datasets_from_file(d / PREPARED_DATASETS_FILE)
            )
        except IOError as err:
            logger.debug(f"Unable to read from {d}: {err}")
            return False

    node_name = os.environ.get("SLURMD_NODENAME", "the current node")
    logger.debug(f"Found {len(other_slurm_tmpdirs)} other slurm TMPDIRS on {node_name}.")
    # Look in those to check if any have a `cache` folder and possibly a file that shows which
    # dataset was prepared.
    usable_dirs: list[Path] = []
    for other_root_dir in other_root_dirs:
        if _can_be_used(other_root_dir):
            logger.debug(
                f"Able to read the dataset from {other_root_dir} (owned by "
                f"{other_root_dir.owner()})!"
            )
            usable_dirs.append(other_root_dir)
    return usable_dirs


def get_prepared_datasets_from_file(
    prepared_datasets_file: str | Path,
) -> list[PreparedDatasetEntry]:
    prepared_datasets_file = Path(prepared_datasets_file)
    if not prepared_datasets_file.exists():
        return []
    with open(prepared_datasets_file, "r") as f:
        yaml_contents = yaml.full_load(f)
        return yaml_contents or []


def add_dataset_to_prepared_datasets_file(
    dataset_fn: DatasetFn[D_co, P],
    root: str | Path,
    *dataset_args: P.args,
    **dataset_kwargs: P.kwargs,
) -> None:
    prepared_datasets_file = Path(root) / PREPARED_DATASETS_FILE
    name = dataset_name(dataset_fn)

    # TODO: Do we want to store where the dataset was prepared?
    new_entry = PreparedDatasetEntry(
        # dataset_name=name,
        dataset_fn=dataset_fn,
        dataset_args=list(dataset_args),
        dataset_kwargs=dataset_kwargs.copy(),
        prepared_at=str(root),
    )

    previous_content = yaml.full_load(prepared_datasets_file.read_text()) or []
    previous_entries = [PreparedDatasetEntry(**entry) for entry in previous_content]

    # TODO: Use a there is already a prepared entry at a different `root`, still use it!
    if new_entry in previous_entries:
        logger.debug(
            f"There is already an entry for dataset {name} in the prepared datasets file: "
            f"{new_entry}"
        )
        return

    if previous_entries:
        logger.debug(
            "Other prepared datasets:\n" + "\n".join(str(entry) for entry in previous_entries)
        )

    with open(prepared_datasets_file, "w") as f:
        logger.info(
            f"Adding a new entry in the prepared dataset file ({prepared_datasets_file}):\n"
            f"{new_entry}"
        )
        entries = previous_entries + [new_entry]
        yaml.dump([dataclasses.asdict(entry) for entry in entries], stream=f)


def is_already_prepared(
    dataset_fn: type[D_co] | Callable[P, D_co],
    root: str | Path,
    *dataset_args: P.args,
    **dataset_kwagrs: P.kwargs,
) -> bool:
    prepared_datasets_file = Path(root) / PREPARED_DATASETS_FILE
    name = dataset_name(dataset_fn)
    if prepared_datasets_file.exists() and name in get_prepared_datasets_from_file(
        prepared_datasets_file
    ):
        logger.info(f"Dataset {name} has already been prepared in {root}!")
        return True
    logger.info(f"Dataset {name} wasn't found in the prepared datasets file. Continuing.")

    return False


def _extra_values_based_on_kwargs(
    root: str | Path,
    extra_files_depending_on_kwargs: dict[str, dict[Any, str | list[str]]],
    dataset_fn: type[D_co] | Callable[P, D_co] | None = None,
    *dataset_args: P.args,
    **dataset_kwargs: P.kwargs,
) -> list[str]:
    """Adds arguments based on the values argument passed to the dataset function.

    For example, if a dataset function takes a `split` argument, and the value of that argument is
    the list of directories to add for each split, then this function will add those directories to
    a list and return it.
    """
    root = Path(root)
    extra_values: list[str] = []

    bound_args = None
    if dataset_fn is not None:
        try:
            bound_args = inspect.signature(dataset_fn).bind_partial(
                *dataset_args, **dataset_kwargs
            )
            bound_args.apply_defaults()
        except TypeError:
            pass

    def _get_arg_value(arg_name: str) -> Any:
        if bound_args is None:
            return dataset_kwargs.get(arg_name)
        return bound_args.arguments.get(arg_name)

    for kwarg, value_to_extra_files_or_dirs in extra_files_depending_on_kwargs.items():
        # NOTE: Using .get here so we also allow a default value to use if the kwarg isn't
        # passed. For example:
        # {"split": {"train": "train_dir", "val": "val_dir", None: "train_dir"}}
        argument_value = _get_arg_value(kwarg)

        if argument_value in value_to_extra_files_or_dirs:
            extra_path_or_paths = value_to_extra_files_or_dirs[argument_value]
            extra_paths = (
                [extra_path_or_paths]
                if isinstance(extra_path_or_paths, (str, Path))
                else list(extra_path_or_paths)
            )
            logger.info(
                f"Also sharing {[str(root / p) for p in extra_paths]} because "
                f"{kwarg}={argument_value} was used."
            )
            extra_values.extend(p for p in extra_paths)
    return extra_values


def make_prepared_dataset_usable_by_others_on_same_node(
    root: str | Path, files_or_dirs_in_root: list[str] | None
) -> None:
    root = Path(root)

    dataset_files: list[Path] = []
    if files_or_dirs_in_root is None:
        dataset_files = list(tree(root))
    else:
        for relative_path_to_file_or_dir in files_or_dirs_in_root:
            file_or_dir = root / relative_path_to_file_or_dir
            if file_or_dir.is_dir():
                dataset_files.extend(tree(file_or_dir))
            else:
                dataset_files.append(file_or_dir)

    assert all(p.exists() for p in dataset_files), [p for p in dataset_files if not p.exists()]

    logger.debug("Checking permissions on dataset files...")
    files_to_make_readonly_to_others = [
        file for file in dataset_files if not has_permission(file, "r", "others")
    ]
    if not files_to_make_readonly_to_others:
        logger.info("All dataset files are already marked as readable by others.")
        return

    if len(files_to_make_readonly_to_others) < len(dataset_files):
        _total = len(dataset_files)
        _n_with_permissions_already = _total - len(files_to_make_readonly_to_others)
        logger.debug(
            f"There are already {_n_with_permissions_already}/{_total} files with read "
            f"permissions for others."
        )

    user = root.owner()
    logger.info(
        f"Making {len(files_to_make_readonly_to_others)} dataset files in {root} "
        f"readable by others on the same node."
    )

    parent_directories: list[Path] = []
    # Make all the dirs from / to `root` executable, so that other users can access the dataset
    # files but can't read the files in the intermediate directories (e.g. SLURM_TMPDIR)
    for parent_dir in [p for p in root.parents if p.owner() == user]:
        logger.debug(f"Marking {parent_dir} as executable by others on the same node.")
        parent_dir.chmod(parent_dir.stat().st_mode | 0b000_001_001)
        parent_directories.append(parent_dir)

    logger.info(f"Making the {root} directory readable by others on the same node.")
    logger.info(
        "NOTE: Other users won't be able to read files from "
        + (
            f"any of the parent directories ({','.join(str(p) for p in parent_directories[:-1])} "
            f"or {parent_directories[-1]})."
            if len(parent_directories) > 1
            else str(root.parent)
        )
    )
    root.chmod(root.stat().st_mode | 0b000_101_101)

    for file in rich_pbar(files_to_make_readonly_to_others, desc="Making files readable..."):
        file.chmod(file.stat().st_mode | 0b000_100_100)
