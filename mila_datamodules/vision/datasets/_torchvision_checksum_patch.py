from __future__ import annotations

import os
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any

import torchvision.datasets.utils
from torchvision.datasets.utils import calculate_md5

logger = get_logger(__name__)


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    actual_md5 = calculate_md5(fpath, **kwargs)
    expected = md5
    if actual_md5 == expected:
        logger.debug(f"MD5 checksum of {fpath} matches expected value: {expected}")
        return True
    else:
        logger.debug(f"MD5 checksum of {fpath} does not match expected value: {expected}!")
        return False


def check_integrity(fpath: str, md5: str | None = None) -> bool:
    logger.debug(f"Using our patched version of `check_integrity` on path {fpath}")
    path = Path(fpath).resolve()
    while path.is_symlink():
        logger.debug(f"Following symlink for {path} instead of redownloading dataset!")
        path = path.readlink()
        logger.debug(f"Resolved path: {path}")
    if not os.path.isfile(fpath):
        logger.debug(f"{fpath} is still not real path?!")
        return False
    if md5 is None:
        logger.debug(f"no md5 check for {fpath}!")
        return True
    return check_md5(fpath, md5)


def apply_patch() -> None:
    setattr(torchvision.datasets.utils, "check_integrity", check_integrity)
    setattr(torchvision.datasets.utils, "check_md5", check_md5)
