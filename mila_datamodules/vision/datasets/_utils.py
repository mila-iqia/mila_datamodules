from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, Sequence

ARCHIVE_FORMATS = ["*.zip", "*.gzip", "*.tar", "*.tar.gz", "*.hdf5"]
METADATA_FORMATS = ["*.txt", "*.csv", "*.json"]


def metadata_files_in_dir(
    root: str | Path, recurse: bool = True, patterns: Sequence[str] = METADATA_FORMATS
) -> Iterable[Path]:
    """Returns paths to all metadata files in the given directory (and all subdirs if
    `recurse`)."""
    root = Path(root)
    return rglob_any(root, patterns) if recurse else glob_any(root, patterns)


def archives_in_dir(
    root: str | Path, recurse: bool = True, patterns: Sequence[str] = ARCHIVE_FORMATS
) -> Iterable[Path]:
    """Returns paths to all archives in the given directory (and all subdirs if `recurse`)."""
    root = Path(root)
    fn = rglob_any if recurse else glob_any
    paths = fn(root, patterns)
    return paths
    return (path.relative_to(root) for path in paths)


def glob_patterns_in_each_dir(
    base_dirs: Sequence[Path], patterns: Sequence[str]
) -> Iterable[Path]:
    """yields files matching any of the given patterns in any of the base dirs."""
    return itertools.chain(*(path.glob(pattern) for pattern in patterns for path in base_dirs))


def rglob_patterns_in_each_dir(
    base_dirs: Sequence[Path], patterns: Sequence[str]
) -> Iterable[Path]:
    """yields files matching any of the patterns in any of the base dirs or their subdirs."""
    return itertools.chain(*(path.rglob(pattern) for pattern in patterns for path in base_dirs))


def glob_any(path: Path, patterns: Sequence[str]) -> Iterable[Path]:
    """yields files matching any of the given patterns."""
    return itertools.chain(*(path.glob(pattern) for pattern in patterns))


def rglob_any(path: Path, patterns: Sequence[str]) -> Iterable[Path]:
    """yields files matching any of the given patterns with `path.rglob(pattern)`."""
    return itertools.chain(*(path.rglob(pattern) for pattern in patterns))
