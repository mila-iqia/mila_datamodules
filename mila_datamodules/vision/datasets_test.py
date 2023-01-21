"""Tests for the `mila_datamodules.datasets` module.

Checks that the 'optimized' constructors work on the current cluster.
"""
from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar, cast, get_args

import pytest
import torchvision.datasets as tvd
from torchvision.datasets import VisionDataset

import mila_datamodules.vision.datasets
from mila_datamodules.clusters import CURRENT_CLUSTER, Cluster
from mila_datamodules.registry import (
    dataset_files,
    is_stored_on_cluster,
    locate_dataset_root_on_cluster,
)
from mila_datamodules.registry_test import check_dataset_creation_works_without_download
from mila_datamodules.vision.coco_test import coco_required

datasets = {
    k: v
    for k, v in vars(mila_datamodules.vision.datasets).items()
    if inspect.isclass(v) and issubclass(v, VisionDataset)
}

# TODO: Need to stop doing this kind of hard-coded fixing and listing of stuff.
datasets["EMNIST"] = partial(datasets.pop("EMNIST"), split="mnist")
datasets["BinaryEMNIST"] = partial(datasets.pop("BinaryEMNIST"), split="mnist")

# Dataset takes a bit longer to copy.
dataset_names = list(datasets.keys())
dataset_names = [
    pytest.param(
        dataset_name,
        # Put the tests for a given dataset in the same group, so that they (eventually) run on the
        # same node (same process for now, until we figure out how to distribute the tests).
        marks=[pytest.mark.xdist_group(name=dataset_name)]
        + ([pytest.mark.timeout(120)] if dataset_name == "CelebA" else []),
    )
    for dataset_name in dataset_names
]


# TODO: Make this quicker to test. Each test currently copies the entire dataset to SLURM_TMPDIR.
@pytest.mark.timeout(30)
@pytest.mark.parametrize("dataset_name", dataset_names)
def test_optimized_dataset_creation(dataset_name: str, tmp_path: Path):
    """Test that the dataset can be created, with the optimizations (copies/etc)."""
    dataset_cls = datasets[dataset_name]
    bad_root = str(tmp_path / "fake_path")
    dataset = check_dataset_creation_works_without_download(dataset_cls, root=bad_root)
    assert dataset.root != bad_root
    assert not dataset.root.startswith(bad_root)
    assert not Path(bad_root).exists()
    # NOTE: Doesn't always hold. For example, for datasets like fashion-mnist, the data doesn't
    # really need to be moved to SLURM_TMPDIR, since we're loading it into memory anyway...
    # assert dataset.root == str(Cluster.current().slurm_tmpdir / "data")


def _unsupported_variant(version: str, cluster: Cluster | Sequence[Cluster]):
    if isinstance(cluster, Cluster):
        condition = CURRENT_CLUSTER is cluster
        reason = f"This variant isn't stored on the {cluster.name} cluster."
    else:
        condition = CURRENT_CLUSTER in cluster
        if CURRENT_CLUSTER is None:
            reason = "Don't know if this variant is stored on the current machine (not a cluster)."
        else:
            reason = f"This variant isn't stored on the {CURRENT_CLUSTER} cluster."

    return pytest.param(version, marks=pytest.mark.xfail(condition=condition, reason=reason))


DatasetType = TypeVar("DatasetType", bound=tvd.VisionDataset)


class PreStoredDatasetTests(Generic[DatasetType]):
    """Tests for the datasets that we know are stored on the clusters (that are in the
    registry)."""

    @pytest.fixture()
    def dataset_cls(self) -> type[DatasetType]:
        """Retrieves the dataset class under test from the class definition (without having to set
        the `dataset_cls` attribute."""

        class_under_test = get_args(type(self).__orig_bases__[0])[0]  # type: ignore
        assert issubclass(class_under_test, tvd.VisionDataset)
        return cast(type[DatasetType], class_under_test)

    @pytest.fixture()
    def dataset_kwargs(self) -> dict[str, Any]:
        """Fixture that returns the kwargs that should be passed to the dataset constructor.

        You can override this fixture in your test class and add dependencies to it, so that each
        variant is tested.
        """
        return dict()

    def test_we_know_what_files_are_required(self, dataset_cls: type[DatasetType]):
        """Test that we know which files are required in order to load this dataset."""
        assert dataset_cls in dataset_files

    def test_creation_without_download(
        self, dataset_cls: type[DatasetType], dataset_kwargs: dict[str, Any]
    ):
        """Test that the dataset can be created without downloading it if the known location for
        that dataset on the current cluster is passed as the `root` argument."""
        if not is_stored_on_cluster(dataset_cls):
            pytest.skip(f"Dataset isn't stored on {CURRENT_CLUSTER} cluster")

        kwargs = dataset_kwargs.copy()
        kwargs.setdefault("root", locate_dataset_root_on_cluster(dataset_cls))
        check_dataset_creation_works_without_download(dataset_cls, **kwargs)

    def test_we_have_adapted_version_of_dataset(self, dataset_cls: type[DatasetType]):
        import mila_datamodules.vision.datasets

        assert hasattr(mila_datamodules.vision.datasets, dataset_cls.__name__)
        adapted_dataset_cls = getattr(mila_datamodules.vision.datasets, dataset_cls.__name__)
        assert issubclass(adapted_dataset_cls, dataset_cls)

    def test_we_can_avoid_download(
        self, tmp_path: Path, dataset_cls: type[DatasetType], dataset_kwargs: dict[str, Any]
    ):
        """TODO: Clarify exactly what we want to test here."""

        if "download" not in inspect.signature(dataset_cls).parameters:
            pytest.skip("Dataset doesn't have a 'download' parameter.")

        import mila_datamodules.vision.datasets

        wrapped_dataset_cls: type[DatasetType] = getattr(
            mila_datamodules.vision.datasets, dataset_cls.__name__
        )
        bad_path = tmp_path / "bad_path"
        bad_path.mkdir()
        kwargs = dataset_kwargs.copy()
        kwargs["download"] = True
        dataset = wrapped_dataset_cls(root=str(bad_path), **kwargs)

        assert dataset.root not in {bad_path, str(bad_path)}
        assert list(bad_path.iterdir()) == []


class TestCityScapes(PreStoredDatasetTests[tvd.Cityscapes]):
    @pytest.fixture(params=["fine", "coarse"])
    def mode(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture(params=["instance", "semantic", "polygon", "color"])
    def target_type(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture()
    def dataset_kwargs(self, mode: str, target_type: str) -> dict[str, Any]:
        return dict(mode=mode, target_type=target_type)


# TODO: Need a kind of "skip if not stored on current cluster" mark.


class TestINaturalist(PreStoredDatasetTests[tvd.INaturalist]):
    @pytest.fixture(
        params=[
            _unsupported_variant("2017", Cluster.Mila),
            _unsupported_variant("2018", Cluster.Mila),
            _unsupported_variant("2019", Cluster.Mila),
            "2021_train",
            "2021_train_mini",
            "2021_valid",
        ]
    )
    def version(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture()
    def dataset_kwargs(self, version: str) -> dict[str, Any]:
        return dict(version=version)


class TestPlaces365(PreStoredDatasetTests[tvd.Places365]):
    @pytest.fixture(
        params=["train-standard", _unsupported_variant("train-challenge", Cluster.Mila), "val"]
    )
    def split(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(split=split)


class TestSTL10(PreStoredDatasetTests[tvd.STL10]):
    @pytest.fixture(params=["train", "test", "unlabeled", "train+unlabeled"])
    def split(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(split=split)


@coco_required
class TestCocoDetection(PreStoredDatasetTests[tvd.CocoDetection]):
    @pytest.fixture(params=["train", "val"])
    def split(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(
            split=split,
            root=f"{locate_dataset_root_on_cluster(tvd.CocoDetection)}/{split}2017",
            annFile=str(
                Path(locate_dataset_root_on_cluster(tvd.CocoDetection))
                / f"annotations/instances_{split}2017.json"
            ),
        )


@coco_required
class TestCocoCaptions(PreStoredDatasetTests[tvd.CocoDetection]):
    @pytest.fixture(params=["train", "val"])
    def split(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(
            split=split,
            root=f"{locate_dataset_root_on_cluster(tvd.CocoCaptions)}/{split}2017",
            annFile=str(
                Path(locate_dataset_root_on_cluster(tvd.CocoCaptions))
                / f"annotations/captions_{split}2017.json"
            ),
        )
