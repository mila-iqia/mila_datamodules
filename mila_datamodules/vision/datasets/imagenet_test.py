from typing import Any

import pytest
import torchvision.datasets as tvd

from .datasets_test import LoadsFromArchives, Required, _FixtureRequest


class TestImageNet(LoadsFromArchives[tvd.ImageNet], Required):
    @pytest.fixture(params=["train", "val"], scope="class")
    def split(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture(scope="class")
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(
            split=split,
        )
