"""Tests that ensure that the information in the registry is correct.

1. Make sure that the files for each dataset are available in the clusters.
2. Checks that these files are sufficient to instantiate the datasets.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, TypeVar

import pytest
import torchvision.datasets
from torch.utils.data import Dataset
from typing_extensions import ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER, Cluster

from .registry import (
    dataset_files,
    dataset_roots_per_cluster,
    get_dataset_root,
    is_stored_on_cluster,
)
from .vision.coco_test import coco_required

P = ParamSpec("P")
D = TypeVar("D", bound=Dataset)


def check_dataset_creation_works(
    dataset_type: Callable[P, D], *args: P.args, **kwargs: P.kwargs
) -> D:
    """Utility function that creates the dataset with the given args and checks that it 'works'."""
    dataset = dataset_type(*args, **kwargs)
    length = len(dataset)  # type: ignore
    assert length > 0
    _ = dataset[0]
    _ = dataset[length // 2]
    _ = dataset[length - 1]
    return dataset


@pytest.mark.parametrize("dataset", dataset_roots_per_cluster.keys())
def test_datasets_in_registry_are_actually_there(dataset: type):
    """Test that the files associated with the dataset class are actually present in the `root` of
    that dataset, if supported on the current cluster."""
    if not is_stored_on_cluster(dataset):
        pytest.skip(f"Dataset isn't stored on cluster {CURRENT_CLUSTER}")

    # Cluster has this dataset (or so it says). Check that all the required files are there.
    root = get_dataset_root(dataset)
    required_files = dataset_files[dataset]  # type: ignore
    for file in (Path(root) / file for file in required_files):
        assert file.exists()


def unsupported_param(
    param,
    cluster: Cluster | None = None,
    reason: str = f"Unsupported on cluster {CURRENT_CLUSTER}",
):
    if cluster is None or cluster is CURRENT_CLUSTER:
        return pytest.param(param, marks=pytest.mark.xfail(reason=reason))
    # Not supposed to fail in the current cluster.
    return param


# Datasets that only have `root` as a required parameter.
easy_to_use_datasets = [
    dataset
    for dataset in vars(torchvision.datasets).values()
    if inspect.isclass(dataset)
    and dataset is not torchvision.datasets.VisionDataset
    and not any(
        n != "root" and p.default is p.empty
        for n, p in inspect.signature(dataset).parameters.items()
    )
]

easy_to_use_datasets = [
    dataset
    if is_stored_on_cluster(dataset)
    else unsupported_param(dataset, reason=f"Dataset isn't stored on {CURRENT_CLUSTER} cluster")
    for dataset in easy_to_use_datasets
]


@pytest.mark.parametrize("dataset", easy_to_use_datasets)
def test_dataset_creation(dataset: type[Dataset]):
    """Test creating the torchvision datasets that don't have any other required arguments besides
    'root', using the root that we get from `get_dataset_root`."""
    check_dataset_creation_works(
        dataset, root=get_dataset_root(dataset, default="/network/datasets/torchvision")
    )


def _unsupported_variant(version: str, cluster: Cluster):
    return unsupported_param(
        version,
        cluster,
        reason=f"This variant isn't stored on the {cluster.name} cluster.",
    )


@pytest.mark.parametrize("mode", ["fine", "coarse"])
@pytest.mark.parametrize("target_type", ["instance", "semantic", "polygon", "color"])
def test_cityscapes(mode: str, target_type: str):
    from torchvision.datasets import Cityscapes

    check_dataset_creation_works(
        Cityscapes, root=get_dataset_root(Cityscapes), mode=mode, target_type=target_type
    )


@pytest.mark.parametrize(
    "version",
    [
        _unsupported_variant("2017", Cluster.Mila),
        _unsupported_variant("2018", Cluster.Mila),
        _unsupported_variant("2019", Cluster.Mila),
        "2021_train",
        "2021_train_mini",
        "2021_valid",
    ],
)
def test_inaturalist(version: str):
    from torchvision.datasets import INaturalist

    check_dataset_creation_works(INaturalist, root=get_dataset_root(INaturalist), version=version)


@pytest.mark.parametrize(
    "split",
    [
        "train-standard",
        _unsupported_variant("train-challenge", Cluster.Mila),
        "val",
    ],
)
def test_places365(split: str):
    from torchvision.datasets import Places365

    check_dataset_creation_works(Places365, root=get_dataset_root(Places365), split=split)


@pytest.mark.parametrize("split", ["train", "test", "unlabeled", "train+unlabeled"])
def test_stl10(split: str):
    from torchvision.datasets import STL10

    check_dataset_creation_works(STL10, root=get_dataset_root(STL10), split=split)


@coco_required
@pytest.mark.parametrize("split", ["train", "val"])
def test_coco_detection(split: str):
    from torchvision.datasets import CocoDetection

    check_dataset_creation_works(
        CocoDetection,
        root=f"{get_dataset_root(CocoDetection)}/{split}2017",
        annFile=str(
            Path(get_dataset_root(CocoDetection)) / f"annotations/instances_{split}2017.json"
        ),
    )


@coco_required
@pytest.mark.parametrize("split", ["train", "val"])
def test_coco_captions(split: str):
    from torchvision.datasets import CocoCaptions

    check_dataset_creation_works(
        CocoCaptions,
        root=f"{get_dataset_root(CocoCaptions)}/{split}2017",
        annFile=str(
            Path(get_dataset_root(CocoCaptions)) / f"annotations/captions_{split}2017.json"
        ),
    )
