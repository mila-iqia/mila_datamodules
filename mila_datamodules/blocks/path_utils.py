from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal


def has_permission(
    path: str | Path,
    flag: Literal["r", "w", "x", "read", "write", "execute"],
    who: Literal["u", "g", "o", "user", "group", "others"],
) -> bool:
    mask = 0b100 if flag.startswith("r") else 0b010 if flag.startswith("w") else 0b001
    mask <<= 6 if who.startswith("u") else 3 if who.startswith("g") else 0
    return Path(path).stat().st_mode & mask == mask


def set_permission(
    path: str | Path,
    flag: Literal["r", "w", "x", "read", "write", "execute"],
    who: Literal["u", "g", "o", "user", "group", "others"],
    value: bool,
) -> None:
    """Sets the selected bit to a value of `1` or `0` for the given file or directory."""
    path = Path(path)
    mask = 0b100 if flag.startswith("r") else 0b010 if flag.startswith("w") else 0b001
    mask <<= 6 if who.startswith("u") else 3 if who.startswith("g") else 0
    mode = path.stat().st_mode
    if value:
        raise NotImplementedError("Test this.")
        # We now have a mask with a 1 at the right position and zeroes everywhere else.
        path.chmod(mode | mask)
    else:
        # bitwise and the current mode with a mask with 1's everywhere except at the right
        # position
        raise NotImplementedError("todo")
        # path.chmod(mode & ~mask) # note, this isn't correct.


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
