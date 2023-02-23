"""Tests for the `mila_datamodules.datasets` module.

Checks that the 'optimized' constructors work on the current cluster.

TODO: Different types of datasets:
1. Datasets that are loaded into memory from a binary/numpy/other file (MNIST, CIFAR, etc.)
2. Datasets that are read from an extracted archive.
    2.1  Datasets where the archive extraction is performed automatically by torchvision.
    2.2. Datasets where the archive extraction needs to be performed manually.

Also, there are different characteristics of the datasets:
- Can the datasets be downloaded with the `download` kwarg?
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, TypeVar

import pytest
import torchvision.datasets as tvd

from mila_datamodules.clusters import Cluster
from mila_datamodules.registry import (
    is_stored_on_cluster,
    is_supported_dataset,
    locate_dataset_root_on_cluster,
)
from mila_datamodules.vision.coco_test import coco_required
from mila_datamodules.vision.datasets import _binary_mnist, _mnist

from .base_test import (
    DownloadableDatasetTests,
    LoadsFromArchives,
    ReadFromRoot,
    Required,
    VisionDatasetTests,
    only_runs_on_clusters,
    unsupported_variant,
)

T = TypeVar("T")


class _FixtureRequest(pytest.FixtureRequest, Generic[T]):
    param: T


# ------------------ Dataset Tests ------------------


class TestCIFAR10(ReadFromRoot[tvd.CIFAR10], DownloadableDatasetTests, Required):
    @only_runs_on_clusters()
    def test_always_stored(self):
        assert is_supported_dataset(self.dataset_cls)
        assert is_stored_on_cluster(self.dataset_cls)
        assert Path(locate_dataset_root_on_cluster(self.dataset_cls)).exists()


class TestCIFAR100(ReadFromRoot[tvd.CIFAR100], DownloadableDatasetTests, Required):
    pass


class TestCityscapes(LoadsFromArchives[tvd.Cityscapes], Required):
    @pytest.fixture(params=["fine", "coarse"])
    def mode(self, request) -> str:
        return request.param

    @pytest.fixture(params=["instance", "semantic", "polygon", "color"])
    def target_type(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture()
    def dataset_kwargs(self, mode: str, target_type: str) -> dict[str, Any]:
        return dict(mode=mode, target_type=target_type)


class TestINaturalist(
    LoadsFromArchives[tvd.INaturalist],
    DownloadableDatasetTests,
):
    @pytest.fixture(
        params=[
            unsupported_variant("2017", Cluster.Mila),
            unsupported_variant("2018", Cluster.Mila),
            unsupported_variant("2019", Cluster.Mila),
            "2021_train",
            "2021_train_mini",
            "2021_valid",
        ]
    )
    def version(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture()
    def dataset_kwargs(self, version: str) -> dict[str, Any]:
        return dict(version=version)


class TestPlaces365(LoadsFromArchives[tvd.Places365], DownloadableDatasetTests):
    @pytest.fixture(
        params=["train-standard", unsupported_variant("train-challenge", Cluster.Mila), "val"]
    )
    def split(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(split=split)


class TestSTL10(VisionDatasetTests[tvd.STL10]):
    @pytest.fixture(params=["train", "test", "unlabeled", "train+unlabeled"])
    def split(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(split=split)


@coco_required
class TestCocoDetection(VisionDatasetTests[tvd.CocoDetection]):
    @pytest.fixture(params=["train", "val"])
    def split(self, request: _FixtureRequest[str]) -> str:
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
class TestCocoCaptions(VisionDatasetTests[tvd.CocoCaptions]):
    @pytest.fixture(params=["train", "val"])
    def split(self, request: _FixtureRequest[str]) -> str:
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


# NOTE: Here the 'original class' is already in `mila_datamodules.datasets.binary_mnist` because
# we include fixes for bugs in the base class (nothing to do with the clusters though)
class TestBinaryMNIST(ReadFromRoot[_binary_mnist._PatchedBinaryMNIST]):
    pass


class TestBinaryEMNIST(ReadFromRoot[_binary_mnist._PatchedBinaryEMNIST]):
    @pytest.fixture(params=["byclass", "bymerge"])
    def split(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(split=split)


# TODO: Same as for binarymnist (original class is actually our "patched" version of the class) but
# here the fix is specific to one cluster. This isn't pretty.
class TestCaltech101(LoadsFromArchives[tvd.Caltech101]):
    def test_pre_init_places_archives_in_slurm_tmpdir(self):
        "101_ObjectCategories.tar.gz"
        "Annotations.tar"
        raise NotImplementedError


class TestCaltech256(VisionDatasetTests[tvd.Caltech256]):
    pass


class TestCelebA(LoadsFromArchives[tvd.CelebA], DownloadableDatasetTests):
    pass


# Same here for `MNIST`, we have a 'patch' that fixes an issue with dataset folder names on Beluga.
class TestMNIST(ReadFromRoot[_mnist.MNIST], DownloadableDatasetTests, Required):
    pass


class TestFashionMNIST(ReadFromRoot[tvd.FashionMNIST], DownloadableDatasetTests, Required):
    pass


class TestEMNIST(ReadFromRoot[tvd.EMNIST], DownloadableDatasetTests, Required):
    pass


class TestSVHN(ReadFromRoot[tvd.SVHN]):
    pass


class TestFlyingChairs(VisionDatasetTests[tvd.FlyingChairs]):
    pass


class TestKittiFlow(VisionDatasetTests[tvd.KittiFlow]):
    pass


class TestSintel(VisionDatasetTests[tvd.Sintel]):
    pass


class TestFlyingThings3D(VisionDatasetTests[tvd.FlyingThings3D]):
    pass


class TestHD1K(VisionDatasetTests[tvd.HD1K]):
    pass


class TestCLEVRClassification(VisionDatasetTests[tvd.CLEVRClassification]):
    pass


class TestCountry211(VisionDatasetTests[tvd.Country211]):
    pass


class TestDTD(VisionDatasetTests[tvd.DTD]):
    pass


class TestEuroSAT(VisionDatasetTests[tvd.EuroSAT]):
    pass


class TestFER2013(VisionDatasetTests[tvd.FER2013]):
    pass


class TestFGVCAircraft(VisionDatasetTests[tvd.FGVCAircraft]):
    pass


@pytest.mark.xfail(reason="http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html is down.")
class TestFlickr8k(VisionDatasetTests[tvd.Flickr8k]):
    pass


@pytest.mark.xfail(reason="Isn't freely available for download.")
class TestFlickr30k(VisionDatasetTests[tvd.Flickr30k]):
    pass


class TestFlowers102(VisionDatasetTests[tvd.Flowers102]):
    pass


class TestFood101(VisionDatasetTests[tvd.Food101]):
    pass


class TestGTSRB(VisionDatasetTests[tvd.GTSRB]):
    pass


class TestHMDB51(VisionDatasetTests[tvd.HMDB51]):
    pass


if hasattr(tvd, "Kinetics400"):

    class TestKinetics400(VisionDatasetTests[tvd.Kinetics400]):  # type: ignore[attr-defined]
        pass


class TestKinetics(VisionDatasetTests[tvd.Kinetics]):
    pass


class TestKitti(VisionDatasetTests[tvd.Kitti]):
    pass


class TestLFWPeople(VisionDatasetTests[tvd.LFWPeople]):
    pass


class TestLFWPairs(VisionDatasetTests[tvd.LFWPairs]):
    pass


class TestLSUN(VisionDatasetTests[tvd.LSUN]):
    pass


class TestLSUNClass(VisionDatasetTests[tvd.LSUNClass]):
    pass


class TestKMNIST(VisionDatasetTests[tvd.KMNIST]):
    pass


class TestQMNIST(VisionDatasetTests[tvd.QMNIST]):
    pass


class TestOmniglot(VisionDatasetTests[tvd.Omniglot]):
    pass


class TestOxfordIIITPet(VisionDatasetTests[tvd.OxfordIIITPet]):
    pass


class TestPCAM(VisionDatasetTests[tvd.PCAM]):
    pass


class TestPhotoTour(VisionDatasetTests[tvd.PhotoTour]):
    pass


class TestRenderedSST2(VisionDatasetTests[tvd.RenderedSST2]):
    pass


class TestSBDataset(VisionDatasetTests[tvd.SBDataset]):
    pass


class TestSBU(VisionDatasetTests[tvd.SBU]):
    pass


class TestSEMEION(VisionDatasetTests[tvd.SEMEION]):
    pass


class TestStanfordCars(VisionDatasetTests[tvd.StanfordCars]):
    pass


class TestSUN397(VisionDatasetTests[tvd.SUN397]):
    pass


class TestUCF101(VisionDatasetTests[tvd.UCF101]):
    pass


class TestUSPS(VisionDatasetTests[tvd.USPS]):
    pass


class TestVisionDataset(VisionDatasetTests[tvd.VisionDataset]):
    pass


class TestVOCSegmentation(VisionDatasetTests[tvd.VOCSegmentation]):
    pass


class TestVOCDetection(VisionDatasetTests[tvd.VOCDetection]):
    pass


class TestWIDERFace(VisionDatasetTests[tvd.WIDERFace]):
    pass


# New torchvision version:


class TestCarlaStereo(VisionDatasetTests[tvd.CarlaStereo]):
    pass


class TestCREStereo(VisionDatasetTests[tvd.CREStereo]):
    pass


class TestETH3DStereo(VisionDatasetTests[tvd.ETH3DStereo]):
    pass


class TestFallingThingsStereo(VisionDatasetTests[tvd.FallingThingsStereo]):
    pass


class TestInStereo2k(VisionDatasetTests[tvd.InStereo2k]):
    pass


class TestKitti2012Stereo(VisionDatasetTests[tvd.Kitti2012Stereo]):
    pass


class TestKitti2015Stereo(VisionDatasetTests[tvd.Kitti2015Stereo]):
    pass


class TestMiddlebury2014Stereo(VisionDatasetTests[tvd.Middlebury2014Stereo]):
    pass


class TestSceneFlowStereo(VisionDatasetTests[tvd.SceneFlowStereo]):
    pass


class TestSintelStereo(VisionDatasetTests[tvd.SintelStereo]):
    pass
