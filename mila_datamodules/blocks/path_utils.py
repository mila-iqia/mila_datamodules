from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal


def has_permission(
    path: str | Path,
    flag: Literal["r", "w", "x", "read", "write", "execute"],
    who: Literal["u", "g", "o", "user", "group", "others"],
) -> bool:
    mask = 0b100 if flag.startswith("r") else 0b010 if flag.startswith("w") else 0b001
    mask << 6 if who.startswith("u") else mask << 3 if who.startswith("g") else mask
    return Path(path).stat().st_mode & mask == mask


def set_permission(
    path: str | Path,
    flag: Literal["r", "w", "x", "read", "write", "execute"],
    who: Literal["u", "g", "o", "user", "group", "others"],
    value: bool,
) -> bool:
    """Sets the selected bit to a value of `1` or `0` for the given file or directory."""
    raise NotImplementedError("todo")


def tree(root: str | Path, ignore_prefix: tuple[str, ...] = (".",)) -> Iterable[Path]:
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
            yield from tree(entry, ignore_prefix=ignore_prefix)


def all_files_in_dir(
    source: str | Path, ignore_prefixes=(".", "scripts", "README")
) -> dict[str, Path]:
    source = Path(source).expanduser().resolve()
    return {
        str(file.relative_to(source)): file
        for file in tree(Path(source), ignore_prefix=ignore_prefixes)
    }
