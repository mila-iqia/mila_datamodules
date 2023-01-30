"""Tests for the `mila_datamodules.datasets` module.

Checks that the 'optimized' constructors work on the current cluster.
"""
from __future__ import annotations

import inspect
from abc import ABC
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Sequence,
    TypeVar,
    cast,
    get_args,
)

import pl_bolts
import pytest
import torchvision.datasets
import torchvision.datasets as tvd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import ParamSpec

import mila_datamodules.vision.datasets
from mila_datamodules.clusters import CURRENT_CLUSTER, Cluster
from mila_datamodules.registry import (
    archives_required_for,
    files_required_for,
    is_stored_on_cluster,
    locate_dataset_root_on_cluster,
)
from mila_datamodules.registry_test import check_dataset_creation_works_without_download
from mila_datamodules.testutils import (
    only_runs_on_slurm_clusters,
    only_runs_outside_slurm_cluster,
)
from mila_datamodules.vision.coco_test import coco_required

datasets = {
    k: v
    for k, v in vars(mila_datamodules.vision.datasets).items()
    if inspect.isclass(v) and issubclass(v, VisionDataset)
}

P = ParamSpec("P")
T = TypeVar("T")
DatasetType = TypeVar("DatasetType", bound=Dataset)
VisionDatasetType = TypeVar("VisionDatasetType", bound=tvd.VisionDataset)


if TYPE_CHECKING:

    class _FixtureRequest(pytest.FixtureRequest, Generic[T]):
        param: T


torchvision_dataset_classes = {
    name: dataset_cls
    for name, dataset_cls in vars(tvd).items()
    if inspect.isclass(dataset_cls) and issubclass(dataset_cls, torchvision.datasets.VisionDataset)
}


@pytest.mark.parametrize("torchvision_dataset_class", torchvision_dataset_classes.values())
def test_all_torchvision_datasets_have_a_test_class(torchvision_dataset_class: type[Dataset]):
    dataset_name = torchvision_dataset_class.__name__
    # Check that there is a subclass of the base test class for this dataset.
    test_classes = {v.__qualname__: v for v in VisionDatasetTests.__subclasses__()}
    assert f"Test{dataset_name}" in test_classes, f"Missing test class for {dataset_name}."
    test_class = globals()[f"Test{dataset_name}"]
    assert issubclass(test_class, VisionDatasetTests)

    # Check that the test class is indeed a VisionDatasetTests[dataset_cls].
    dataset_class_under_test = test_class._dataset_cls()
    # dataset_class_under_test = get_args(type(test_class).__orig_bases__[0])[0]  # type: ignore
    assert issubclass(dataset_class_under_test, tvd.VisionDataset)
    if dataset_class_under_test is not torchvision_dataset_class:
        assert issubclass(torchvision_dataset_class, dataset_class_under_test)


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


class VisionDatasetTests(Generic[VisionDatasetType], ABC):
    """Suite of basic unit tests for any dataset class from `torchvision.datasets`."""

    @property
    def dataset_cls(self) -> type[VisionDatasetType]:
        """The original dataset class from torchvision.datasets that is being tested."""
        # TODO: Perhaps we could add some skips / xfails here if we can tell that the dataset isn't
        # supported yet, or isn't stored on the current cluster?
        # This way, any test that accesses `self.dataset_cls` would be skipped/xfailed
        # appropriately?
        dataset_cls = self._dataset_cls()
        return dataset_cls

    @property
    def adapted_dataset_cls(self) -> type[VisionDatasetType]:
        """The adapted dataset class from mila_datamodules.vision.datasets."""
        # TODO: Perhaps we could add some skips / xfails here if we can tell that the dataset isn't
        # supported yet, or isn't stored on the current cluster?
        # This way, any test that accesses `self.dataset_cls` would be skipped/xfailed
        # appropriately?
        dataset_cls = self._adapted_dataset_cls()
        return dataset_cls

    @pytest.fixture()
    def dataset_kwargs(self) -> dict[str, Any]:
        """Fixture that returns the kwargs that should be passed to the dataset constructor.

        You can override this fixture in your test class and add dependencies to it, so that each
        variant is tested.
        """
        return dict()

    @classmethod
    def _dataset_cls(cls) -> type[VisionDatasetType]:
        """Retrieves the dataset class under test from the class definition (without having to set
        the `dataset_cls` attribute."""
        class_under_test = get_args(cls.__orig_bases__[0])[0]  # type: ignore
        if not (
            inspect.isclass(class_under_test) and issubclass(class_under_test, tvd.VisionDataset)
        ):
            raise RuntimeError(
                "Your test class needs to pass the class under test to the generic base class!\n"
                "for example: `class TestMyDataset(DatasetTests[MyDataset]):`\n"
                f"Got {class_under_test}"
            )
        return cast(type[VisionDatasetType], class_under_test)

    @classmethod
    def _adapted_dataset_cls(cls) -> type[VisionDatasetType]:
        dataset_class = cls._dataset_cls()
        return getattr(mila_datamodules.vision.datasets, dataset_class.__name__)

    def make_dataset(
        self: VisionDatasetTests[VisionDatasetType],
        dataset_cls: Callable[P, VisionDatasetType] | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> VisionDatasetType:
        dataset_cls = dataset_cls or self.dataset_cls  # type: ignore
        assert dataset_cls
        return dataset_cls(*args, **kwargs)

    @only_runs_outside_slurm_cluster()
    def test_no_change_when_outside_cluster(self):
        class_name = self.dataset_cls.__name__
        assert Cluster.current() is None
        class_from_mila_datamodules_datasets = self._adapted_dataset_cls()
        class_from_torchvision_datasets = getattr(tvd, class_name)
        assert class_from_mila_datamodules_datasets is class_from_torchvision_datasets
        # NOTE: Redundant in this case. However we'd like to be able to test that we didn't modify
        # the torchvision package in-place somehow too!
        # assert inspect.signature(class_from_mila_datamodules_datasets) == inspect.signature(
        #     class_from_torchvision_datasets
        # )

    @only_runs_on_slurm_clusters()
    def test_we_have_adapted_version_of_dataset(self):
        dataset_cls: type[VisionDatasetType] = self._dataset_cls()
        assert hasattr(mila_datamodules.vision.datasets, dataset_cls.__name__)
        adapted_dataset_cls = getattr(mila_datamodules.vision.datasets, dataset_cls.__name__)
        assert issubclass(adapted_dataset_cls, dataset_cls)

    def test_we_know_what_files_are_required(self):
        """Test that we know which files are required in order to load this dataset."""
        files = files_required_for(self._dataset_cls())
        assert files

    def test_required_files_exist(self):
        """Test that if the registry says that we have the files required to load this dataset on
        the current cluster, then they actually exist."""
        dataset_cls = self._dataset_cls()
        files = files_required_for(dataset_cls)
        assert files
        if not is_stored_on_cluster(dataset_cls):
            pytest.skip(reason="Dataset isn't stored on the current cluster.")

        # Also check that the files exist on the current cluster.
        dataset_root = locate_dataset_root_on_cluster(self._dataset_cls())
        for file in files:
            path = dataset_root / file
            assert path.exists()

    def test_root_becomes_optional_arg(self):
        """Checks that the `root` argument becomes optional in the adapted dataset class."""
        dataset_cls = self._dataset_cls()
        constructor_root_parameter = inspect.signature(dataset_cls).parameters["root"]

        if is_stored_on_cluster(dataset_cls):
            # Check that we did indeed change the signature of the constructor to have the 'root'
            # set to the value we want.
            expected_default_root = locate_dataset_root_on_cluster(dataset_cls)
            assert constructor_root_parameter.default == expected_default_root
        else:
            # We don't have the dataset stored on this cluster, so expect root to be required.
            assert constructor_root_parameter.default == inspect.Parameter.empty

    @pytest.mark.disable_socket
    def test_creation_without_download(self, dataset_kwargs: dict[str, Any]):
        dataset_cls = self._dataset_cls()
        """Test that the dataset can be created without downloading it if the known location for
        that dataset on the current cluster is passed as the `root` argument."""
        if not is_stored_on_cluster(dataset_cls):
            pytest.skip(f"Dataset isn't stored on {CURRENT_CLUSTER} cluster")

        kwargs = dataset_kwargs.copy()
        kwargs.setdefault("root", locate_dataset_root_on_cluster(dataset_cls))
        check_dataset_creation_works_without_download(dataset_cls, **kwargs)

    @pytest.mark.disable_socket
    def test_we_can_avoid_download(self, tmp_path: Path, dataset_kwargs: dict[str, Any]):
        """TODO: Clarify exactly what we want to test here."""
        dataset_cls = self.dataset_cls
        adapted_dataset_cls = self.adapted_dataset_cls
        # IDEA: Move this to the `adapted_dataset_cls` property?
        if not is_stored_on_cluster(dataset_cls):
            pytest.skip(reason="Can only avoid a download when the dataset isn't stored.")
        bad_path = tmp_path / "bad_path"
        bad_path.mkdir()

        kwargs = dataset_kwargs.copy()
        kwargs["download"] = True
        with pytest.warns(UserWarning, match=f"Ignoring path {str(bad_path)}"):
            dataset = adapted_dataset_cls(root=str(bad_path), **kwargs)

        assert dataset.root not in {bad_path, str(bad_path)}
        assert list(bad_path.iterdir()) == []


class FitsInMemoryTests(Generic[VisionDatasetType], ABC):
    """Tests for datasets that fit in RAM or are loaded into RAM anyway, e.g. mnist/cifar10/etc.

    - we should just read them from /network/datasets/torchvision, and not bother copying them to SLURM_TMPDIR.
    """


class LoadFromArchivesTests(VisionDatasetTests[VisionDatasetType]):
    """For datasets that don't fit in RAM (e.g. ImageNet), extract the archive directly to.

    $SLURM_TMPDIR.
    NOTE: Might need to also create a symlink of the archive in $SLURM_TMPDIR so that the tvd Dataset
    constructor doesn't re-download it to SLURM_TMPDIR.
    - NOTE: No speedup reading from $SCRATCH or /network/datasets. Same filesystem

    For ComputeCanada:
    - Extract the archive from the datasets folder to $SLURM_TMPDIR without copying.

    In general, for datasets that don't fit in SLURM_TMPDIR, we should use $SCRATCH as the
    "SLURM_TMPDIR".
    NOTE: setting --tmp=800G is a good idea if you're going to move a 600gb dataset to SLURM_TMPDIR.
    """

    def test_we_know_what_archives_are_required(self):
        """Test that we know which archives are required in order to load this dataset on the
        current cluster."""
        files = archives_required_for(self.dataset_cls)
        assert files

    def test_we_have_required_archives_(self):
        """Test that if the registry says that we have the files required to load this dataset on
        the current cluster, then they actually exist."""
        dataset_cls = self.dataset_cls
        files = files_required_for(dataset_cls)
        assert files
        if not is_stored_on_cluster(dataset_cls):
            pytest.skip(reason="Dataset isn't stored on the current cluster.")

        # Also check that the files exist on the current cluster.
        dataset_root = locate_dataset_root_on_cluster(dataset_cls)
        for file in files:
            path = dataset_root / file
            assert path.exists()


class TestCityscapes(VisionDatasetTests[tvd.Cityscapes]):
    @pytest.fixture(params=["fine", "coarse"])
    def mode(self, request) -> str:
        return request.param

    @pytest.fixture(params=["instance", "semantic", "polygon", "color"])
    def target_type(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture()
    def dataset_kwargs(self, mode: str, target_type: str) -> dict[str, Any]:
        return dict(mode=mode, target_type=target_type)


# TODO: Need a kind of "skip if not stored on current cluster" mark.


class TestINaturalist(VisionDatasetTests[tvd.INaturalist]):
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
    def version(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture()
    def dataset_kwargs(self, version: str) -> dict[str, Any]:
        return dict(version=version)


class TestPlaces365(VisionDatasetTests[tvd.Places365]):
    @pytest.fixture(
        params=["train-standard", _unsupported_variant("train-challenge", Cluster.Mila), "val"]
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
class TestCocoCaptions(VisionDatasetTests[tvd.CocoDetection]):
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


class TestCIFAR10(VisionDatasetTests[tvd.CIFAR10], FitsInMemoryTests):
    pass


class TestCIFAR100(VisionDatasetTests[tvd.CIFAR100], FitsInMemoryTests):
    pass


class TestBinaryMNIST(VisionDatasetTests[pl_bolts.datasets.BinaryMNIST], FitsInMemoryTests):
    pass


class TestBinaryEMNIST(VisionDatasetTests[pl_bolts.datasets.BinaryEMNIST], FitsInMemoryTests):
    @pytest.fixture(params=["byclass", "bymerge"])
    def split(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(split=split)


class TestCaltech101(VisionDatasetTests[tvd.Caltech101]):
    pass


class TestCaltech256(VisionDatasetTests[tvd.Caltech256]):
    pass


class TestCelebA(VisionDatasetTests[tvd.CelebA]):
    pass


class TestMNIST(VisionDatasetTests[tvd.MNIST], FitsInMemoryTests):
    pass


class TestFashionMNIST(VisionDatasetTests[tvd.FashionMNIST], FitsInMemoryTests):
    pass


class TestEMNIST(VisionDatasetTests[tvd.EMNIST], FitsInMemoryTests):
    pass


class TestSVHN(VisionDatasetTests[tvd.SVHN], FitsInMemoryTests):
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


class TestFakeData(VisionDatasetTests[tvd.FakeData]):
    pass


class TestFER2013(VisionDatasetTests[tvd.FER2013]):
    pass


class TestFGVCAircraft(VisionDatasetTests[tvd.FGVCAircraft]):
    pass


class TestFlickr8k(VisionDatasetTests[tvd.Flickr8k]):
    pass


class TestFlickr30k(VisionDatasetTests[tvd.Flickr30k]):
    pass


class TestFlowers102(VisionDatasetTests[tvd.Flowers102]):
    pass


class TestImageFolder(VisionDatasetTests[tvd.ImageFolder]):
    pass


class TestDatasetFolder(VisionDatasetTests[tvd.DatasetFolder]):
    pass


class TestFood101(VisionDatasetTests[tvd.Food101]):
    pass


class TestGTSRB(VisionDatasetTests[tvd.GTSRB]):
    pass


class TestHMDB51(VisionDatasetTests[tvd.HMDB51]):
    pass


class TestImageNet(VisionDatasetTests[tvd.ImageNet]):
    pass


class TestKinetics400(VisionDatasetTests[tvd.Kinetics400]):
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
