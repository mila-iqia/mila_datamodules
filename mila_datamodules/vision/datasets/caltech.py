"""'Patch' for the Caltech101 dataset, which has an unused folder missing on the Mila cluster."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from torchvision.datasets import Caltech101 as _Caltech101
from torchvision.datasets import Caltech256 as _Caltech256


class Caltech101(_Caltech101):
    def __init__(
        self,
        root: str,
        target_type: list[str] | str = "category",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        # --> These lines in the super().__init__ cause an error, because the `"BACKGROUND_Google"`
        # isn't in `categories`.
        # self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        # self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # NOTE: Here we just 'patch' this by creating an empty folder at that location before
        # calling the actual constructor.
        background_google_folder = (
            Path(root) / "caltech101" / "101_ObjectCategories" / "BACKGROUND_Google"
        )
        if not background_google_folder.exists():
            background_google_folder.mkdir(parents=True, exist_ok=False)
        super().__init__(root, target_type, transform, target_transform, download)


# TODO: This dataset doesn't work: Indexing into the dataset looks for an unexisting file
# (014.blimp/001_0001.jpg).
# I redownloaded it into SCRATCH, and it still has the exact same structure, and raises the same
# error!


class Caltech256(_Caltech256):
    # TODO: Apply some kind of patch to fix the problem with this dataset.
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform, download)
