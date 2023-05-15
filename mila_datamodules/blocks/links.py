from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
from typing import Mapping

from mila_datamodules.blocks.path_utils import all_files_in_dir
from mila_datamodules.blocks.types import PrepareDatasetFn
from mila_datamodules.cli.utils import runs_on_local_main_process_first
from mila_datamodules.types import D, P

logger = get_logger(__name__)


class MakeSymlinksToDatasetFiles(PrepareDatasetFn[D, P]):
    """Creates symlinks to the datasets' files in the `root` directory."""

    dataset_fn = None

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
