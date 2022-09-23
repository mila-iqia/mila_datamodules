from __future__ import annotations

from pathlib import Path

import pytest

from mila_datamodules.clusters import Cluster

from .registry import dataset_files, dataset_roots_per_cluster, get_dataset_root


@pytest.mark.parametrize("dataset", dataset_roots_per_cluster)
def test_datasets_in_registry_are_actually_there(dataset: type):
    """Test that all files associated with the dataset class are actually present in the `root`
    associated with them in the registry's dictionary."""
    root = get_dataset_root(dataset)
    assert dataset in dataset_files
    required_files = dataset_files[dataset]  # type: ignore
    for file in (Path(root) / file for file in required_files):
        assert file.exists()


def unsupported_param(
    param,
    cluster: Cluster | None = None,
    reason: str = f"Unsupported on cluster {Cluster.current().normal_name}",
):
    if cluster is None or cluster is Cluster.current():
        return pytest.param(param, marks=pytest.mark.xfail(reason=reason))
    # Not supposed to fail in the current cluster.
    return param


def _unsupported_variant(version: str, cluster: Cluster):
    return unsupported_param(
        version,
        cluster,
        reason=f"This variant isn't stored on the {cluster.normal_name} cluster.",
    )


@pytest.mark.parametrize("mode", ["fine", "coarse"])
@pytest.mark.parametrize("target_type", ["instance", "semantic", "polygon", "color"])
def test_cityscapes(mode: str, target_type: str):
    from torchvision.datasets import Cityscapes

    dataset = Cityscapes("/network/datasets/torchvision", mode=mode, target_type=target_type)
    assert len(dataset) > 0
    thing = dataset[0]


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

    dataset = INaturalist(root=get_dataset_root(INaturalist), version=version)
    assert len(dataset) > 0
    thing = dataset[0]


@pytest.mark.parametrize(
    "split",
    [
        "train-standard",
        _unsupported_variant("train-challenge", Cluster.Mila),
        "val",
    ],
)
def test_places365(split: str, root: str):
    from torchvision.datasets import Places365

    dataset = Places365(root=get_dataset_root(Places365), split=split)
    assert len(dataset) > 0
    thing = dataset[0]
