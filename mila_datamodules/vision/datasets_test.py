"""Tests for the `mila_datamodules.datasets` module.

Checks that the 'optimized' constructors work on the current cluster.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Sequence,
    TypeVar,
    get_args,
)

import filelock
import pytest
import torchvision.datasets
import torchvision.datasets as tvd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import ParamSpec

import mila_datamodules.vision.datasets
from mila_datamodules.clusters import CURRENT_CLUSTER, Cluster
from mila_datamodules.clusters.utils import get_scratch_dir
from mila_datamodules.errors import UnsupportedDatasetError
from mila_datamodules.registry import (
    _archives_required_for,
    files_required_for,
    is_stored_on_cluster,
    is_supported_dataset,
    locate_dataset_root_on_cluster,
)
from mila_datamodules.registry_test import check_dataset_creation_works_without_download
from mila_datamodules.testutils import (
    only_runs_on_clusters,
    only_runs_when_not_on_a_slurm_cluster,
)
from mila_datamodules.vision.coco_test import coco_required
from mila_datamodules.vision.datasets import binary_mnist, caltech101, mnist
from mila_datamodules.vision.datasets.adapted_datasets import AdaptedDataset

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
    if inspect.isclass(dataset_cls)
    and issubclass(dataset_cls, torchvision.datasets.VisionDataset)
    and dataset_cls not in {tvd.FakeData, tvd.VisionDataset, tvd.ImageFolder, tvd.DatasetFolder}
}


@pytest.mark.parametrize("torchvision_dataset_class", torchvision_dataset_classes.values())
def test_all_torchvision_datasets_have_a_test_class(torchvision_dataset_class: type[Dataset]):
    dataset_name = torchvision_dataset_class.__name__
    # Check that there is a subclass of the base test class for this dataset.
    test_classes = {
        k: v
        for k, v in globals().items()
        if inspect.isclass(v) and issubclass(v, VisionDatasetTests)
    }
    assert f"Test{dataset_name}" in test_classes, f"Missing test class for {dataset_name}."
    test_class = globals()[f"Test{dataset_name}"]
    assert issubclass(test_class, VisionDatasetTests)

    # Check that the test class is indeed a VisionDatasetTests[dataset_cls].
    dataset_class_under_test = test_class._dataset_cls()
    # dataset_class_under_test = get_args(type(test_class).__orig_bases__[0])[0]  # type: ignore
    assert issubclass(dataset_class_under_test, tvd.VisionDataset)
    if dataset_class_under_test is not torchvision_dataset_class:
        assert issubclass(dataset_class_under_test, torchvision_dataset_class)


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


class DatasetTests(Generic[DatasetType]):
    _bound: ClassVar[type[Dataset]] = Dataset

    required_on_all_clusters: ClassVar[bool] = False

    @pytest.fixture()
    def dataset_kwargs(self) -> dict[str, Any]:
        """Fixture that returns the kwargs that should be passed to the dataset constructor.

        You can override this fixture in your test class and add dependencies to it, so that each
        variant is tested.
        """
        return dict()

    @property
    def dataset_cls(self) -> type[DatasetType]:
        """The original dataset class from torchvision.datasets that is being tested."""
        # TODO: Perhaps we could add some skips / xfails here if we can tell that the dataset isn't
        # supported yet, or isn't stored on the current cluster?
        # This way, any test that accesses `self.dataset_cls` would be skipped/xfailed
        # appropriately?
        dataset_cls = self._dataset_cls()
        return dataset_cls

    @property
    def dataset_name(self) -> str:
        return self.dataset_cls.__name__

    @property
    def is_stored_on_cluster(self) -> bool:
        return is_stored_on_cluster(self.dataset_cls, cluster=Cluster.current())

    @classmethod
    def _dataset_cls(cls) -> type[DatasetType]:
        """Retrieves the dataset class under test from the class definition (without having to set
        the `dataset_cls` attribute."""
        class_under_test = get_args(cls.__orig_bases__[0])[0]  # type: ignore
        # TODO: Get the bound programmatically to avoid hardcoding `Dataset` here.
        if not (inspect.isclass(class_under_test) and issubclass(class_under_test, cls._bound)):
            raise RuntimeError(
                "Your test class needs to pass the class under test to the generic base class.\n"
                "for example: `class TestMyDataset(DatasetTests[MyDataset]):`\n"
                f"(Got {class_under_test})"
            )
        return class_under_test  # type: ignore

    def make_dataset(
        self,
        dataset_cls: Callable[P, DatasetType] | type[DatasetType] | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DatasetType:
        dataset_cls = dataset_cls or self.dataset_cls
        assert dataset_cls
        return dataset_cls(*args, **kwargs)


class Required(Generic[DatasetType]):
    """Tests that raise an error if the dataset isn't stored.

    By default, tests will be skipped if the dataset isn't stored on the current cluster.
    """

    required_on_all_clusters: ClassVar[bool] = True


class VisionDatasetTests(DatasetTests[VisionDatasetType]):
    """Suite of basic unit tests for any dataset class from `torchvision.datasets`."""

    _bound: ClassVar[type[Dataset]] = tvd.VisionDataset

    @property
    def adapted_dataset_cls(
        self,
    ) -> type[VisionDatasetType]:
        """The adapted dataset class from mila_datamodules.vision.datasets."""
        # TODO: Perhaps we could add some skips / xfails here if we can tell that the dataset isn't
        # supported yet, or isn't stored on the current cluster?
        # This way, any test that accesses `self.dataset_cls` would be skipped/xfailed
        # appropriately?
        return self._adapted_dataset_cls()

    @classmethod
    def _adapted_dataset_cls(cls) -> type[VisionDatasetType]:
        original_dataset_class = cls._dataset_cls()
        if hasattr(mila_datamodules.vision.datasets, original_dataset_class.__name__):
            return getattr(mila_datamodules.vision.datasets, original_dataset_class.__name__)
        # Try to create the adapter dynamically.
        # NOTE: might want to turn this off so we're more clear on which ones we support and don't.
        return getattr(
            mila_datamodules.vision.datasets.adapted_datasets, original_dataset_class.__name__
        )

    @only_runs_when_not_on_a_slurm_cluster()
    def test_import_dataset_when_not_on_cluster(self):
        """Importing a dataset from mila_datamodules.vision.datasets gives back the original class
        from `torchvision.datasets` when not on a SLURM cluster."""
        assert Cluster.current() is None
        class_from_torchvision = self.dataset_cls
        class_from_mila_datamodules = getattr(
            mila_datamodules.vision.datasets, self.dataset_cls.__name__
        )

        assert class_from_mila_datamodules is class_from_torchvision

    @only_runs_on_clusters()
    def test_import_from_package_gives_adapted_dataset_class(self):
        """We should have an adapted dataset class for explicitly supported datasets in
        `mila_datamodules.vision.datasets`.

        NOTE: For 'experimental' datasets which we don't explicitly support, we could get them with
        `from mila_datamodules.vision.datasets.adapted_datasets import SomeFancyNewDataset`, but it
        should at the very least raise a Warning. (this uses the module-level __getattr__ which is
        very cool and hacky!)
        """
        original_cls = self.dataset_cls
        adapted_cls = getattr(mila_datamodules.vision.datasets, original_cls.__name__)

        if not is_stored_on_cluster(original_cls):
            if self.required_on_all_clusters:
                pytest.fail(
                    reason=f"Dataset {self.dataset_cls.__name__} is required and is not stored on this cluster!"
                )

            # No change when the dataset isn't stored on the current SLURM cluster:
            # the "adapted" class is the original class.
            assert adapted_cls is original_cls
            # NOTE: unnecessary, but just to make sure we didn't change the constructor at all.
            assert adapted_cls.__init__ is original_cls.__init__
        else:
            assert adapted_cls is not original_cls
            assert issubclass(adapted_cls, original_cls)
            # TODO: Not sure if we can successfully check this with `inspect`:
            # Check that we did indeed change the signature of the constructor to have the 'root'
            # set to the right default value.
            # expected_default_root = locate_dataset_root_on_cluster(original_cls)
            # assert adapted_constructor_root_parameter.default == expected_default_root

    def test_we_know_what_files_are_required_for_dataset(self):
        """Test that we know which files are required in order to load this dataset."""
        try:
            required_files = files_required_for(self.dataset_cls)
            assert required_files
        except UnsupportedDatasetError as e:
            if self.required_on_all_clusters:
                raise AssertionError(
                    f"Dataset {self.dataset_cls.__name__} is required and is not stored on this "
                    f"cluster!"
                    f"The required files or archives should be added in the registry. Here's some "
                    f"context that might be helpful: \n"
                    f"{inspect.getabsfile(self.dataset_cls)}: \n"
                    f"{self.dataset_cls.__doc__}"
                ) from e
            pytest.xfail(reason=f"The dataset {self.dataset_name} isn't supported yet.")
        except AssertionError as e:
            # Huh? The dataset doesn't require any files?
            raise AssertionError(f"Huh? dataset {self.dataset_name} doesn't require any files?")

    def test_required_files_exist(self):
        """Test that if the registry says that we have the files required to load this dataset on
        the current cluster, then they actually exist."""
        dataset_cls = self.dataset_cls
        files = files_required_for(dataset_cls)

        assert files  # note: this is a bit redundant with the test above.

        if not is_stored_on_cluster(dataset_cls):
            if self.required_on_all_clusters:
                pytest.xfail()

            pytest.skip(reason="Dataset isn't stored on the current cluster.")

        # Also check that the files exist on the current cluster.
        dataset_root = locate_dataset_root_on_cluster(self.dataset_cls)
        assert dataset_root is not None
        dataset_root = Path(dataset_root)
        for file in files:
            path = dataset_root / file
            assert path.exists()

    @pytest.mark.xfail(
        reason="TODO: The check with `inspect.signature` might not actually be able to pickup the fact that we changed the default value for the `root` argument."
    )
    def test_root_becomes_optional_arg_if_stored(self):
        """Checks that the `root` argument becomes optional in the adapted dataset class."""
        dataset_cls = self.dataset_cls
        adapted_cls = self.adapted_dataset_cls
        default_constructor_root_parameter = inspect.signature(dataset_cls).parameters["root"]
        adapted_constructor_root_parameter = inspect.signature(adapted_cls).parameters["root"]

        if is_stored_on_cluster(dataset_cls):
            # Check that we did indeed change the signature of the constructor to have the 'root'
            # set to the value we want.
            expected_default_root = locate_dataset_root_on_cluster(dataset_cls)
            assert adapted_constructor_root_parameter.default == expected_default_root
        else:
            # We don't have the dataset stored on this cluster, so expect root to be required.
            assert (
                adapted_constructor_root_parameter.default
                == default_constructor_root_parameter.default
            )

    def _assert_downloaded_if_required_else_skip(self):
        """Asserts that the dataset is present on the current cluster if it is required.

        Otherwise skips the current test.
        """
        cluster = Cluster.current()
        if not is_stored_on_cluster(self.dataset_cls, cluster=cluster):
            if self.required_on_all_clusters:
                pytest.fail(
                    f"Dataset {self.dataset_cls.__name__} is required and is not stored on {cluster} cluster!"
                )
            pytest.skip(f"Dataset isn't stored on {cluster} cluster")

    @pytest.mark.disable_socket
    def test_can_read_dataset_from_root_dir(self, dataset_kwargs: dict[str, Any]):
        """Test that the dataset can be created without downloading it the known location for that
        dataset on the current cluster is passed as the `root` argument.

        NOTE: This test isn't applicable to datasets like ImageNet!
        """
        dataset_cls = self.dataset_cls
        self._assert_downloaded_if_required_else_skip()

        # TODO: Is it ok for us to (e.g.) read a large dataset from the /network/datasets folder
        # during unit tests?
        # I think it is.
        # if issubclass(dataset_cls, tvd.ImageFolder):
        #     pytest.skip(reason=f"Avoiding loading the dataset from network dataset storage.")

        dataset_location = locate_dataset_root_on_cluster(dataset_cls)

        kwargs = dataset_kwargs.copy()
        kwargs.setdefault("root", dataset_location)
        check_dataset_creation_works_without_download(dataset_cls, **kwargs)

    # TODO: Some datasets take a lot longer to extract than others. Might want to customize this
    # timeout value on a per-dataset basis.
    @pytest.mark.timeout(300)
    @pytest.mark.disable_socket
    def test_doesnt_download_even_if_user_asks(
        self, tmp_path: Path, dataset_kwargs: dict[str, Any]
    ):
        """Test that even if `download=True` is passed to the constructor, the dataset is not
        downloaded.

        TODO: It might be a good idea to let the user override this behaviour in some cases. For
        example, if there is some issue with the stored dataset files on the cluster, or a new
        version of the dataset was published. Perhaps by setting some global configuration variable
        the user could say "No, really do download this please?" Or a new argument to one of the
        functions? or some context manager?
        """
        dataset_cls = self.dataset_cls
        adapted_dataset_cls = self.adapted_dataset_cls
        assert issubclass(adapted_dataset_cls, AdaptedDataset)

        self._assert_downloaded_if_required_else_skip()

        if "download" not in inspect.signature(dataset_cls).parameters:
            pytest.skip(reason="Can't donload dataset (no download constructor argument).")
        kwargs = dataset_kwargs.copy()
        kwargs["download"] = True

        bad_path = tmp_path / "bad_path"
        bad_path.mkdir()

        dataset_location = locate_dataset_root_on_cluster(dataset_cls)

        with pytest.warns(
            RuntimeWarning, match=f"Ignoring passed 'root' argument: {str(bad_path)}"
        ):
            dataset = adapted_dataset_cls(root=str(bad_path), **kwargs)
        assert dataset.root not in {bad_path, str(bad_path)}
        assert list(bad_path.iterdir()) == []

        with pytest.warns(
            UserWarning,
            match=(
                f"Not downloading the {self.dataset_name} dataset, since it is "
                f"already stored on the cluster at {dataset_location}"
            ),
        ):
            dataset = adapted_dataset_cls(root=str(bad_path), **kwargs)
        assert dataset.root not in {bad_path, str(bad_path)}
        assert list(bad_path.iterdir()) == []


class FitsInMemoryTests(DatasetTests[VisionDatasetType]):
    """Tests for datasets that fit in RAM or are loaded into RAM anyway, e.g. mnist/cifar10/etc.

    - we should just read them from /network/datasets/torchvision, and not bother copying them to SLURM_TMPDIR.
    """

    # TODO: Add tests that check specifically for this behaviour.


def fake_slurm_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a fake SLURM_TMPDIR in the temporary directory and monkeypatch the environment
    variables to point to it."""
    slurm_tmpdir = tmp_path / "slurm_tmpdir"
    slurm_tmpdir.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(slurm_tmpdir))
    monkeypatch.setenv("FAKE_SLURM_TMPDIR", str(slurm_tmpdir))
    return slurm_tmpdir


class LoadsFromArchives(VisionDatasetTests[VisionDatasetType]):
    """For datasets that don't fit in RAM (e.g. ImageNet), extract the archive directly to.

    $SLURM_TMPDIR.

    NOTE: Might need to also create a symlink of the archive in $SLURM_TMPDIR so that the dataset
    constructor doesn't re-download it to SLURM_TMPDIR.
    - NOTE: No speedup reading from $SCRATCH or /network/datasets. Same filesystem

    For ComputeCanada:
    - Extract the archive from the datasets folder to $SLURM_TMPDIR (also without copying the
    archive if possible)

    In general, for datasets that don't fit in SLURM_TMPDIR, we should use $SCRATCH as the
    "SLURM_TMPDIR".
    NOTE: setting --tmp=800G is a good idea if you're going to move a 600gb dataset to SLURM_TMPDIR.
    """

    def test_pre_init_creates_symlinks_to_archives(self, fake_slurm_tmpdir: Path):
        """Test that the 'pre-init' portion of the adapted constructor creates symlinks to the
        archives that are to be used later."""
        adapted_dataset_class = self.adapted_dataset_cls
        new_root = adapted_dataset_class._pre_init_(root="bad_root")
        required_archives = _archives_required_for(self.dataset_cls)
        assert all((fake_slurm_tmpdir / archive).exists() for archive in required_archives)

    # TODO: Add tests that check specifically that the dataset is loaded from the archives as
    # described in the doc above.


class DownloadForMockTests(DatasetTests[VisionDatasetType]):
    _bound: ClassVar[type[Dataset]] = tvd.VisionDataset

    @pytest.fixture(scope="session", autouse=True)
    def download_to_fake_scratch_before_tests(
        self, worker_id: str, tmp_path_factory: pytest.TempPathFactory
    ):
        cluster = Cluster.current()
        if cluster is not Cluster._mock:
            # Don't download the datasets.
            return
        assert "FAKE_SCRATCH" in os.environ
        fake_scratch_dir = get_scratch_dir()
        assert fake_scratch_dir is not None
        fake_scratch_dir.mkdir(exist_ok=True, parents=True)

        assert "download" in inspect.signature(self.dataset_cls.__init__).parameters

        # get the temp directory shared by all workers
        temp_dir = tmp_path_factory.getbasetemp().parent

        # Make this only download the dataset on one worker, using filelocks.
        with filelock.FileLock(temp_dir / f"{self.dataset_cls.__name__}.lock"):
            self.dataset_cls(root=fake_scratch_dir, download=worker_id == "master")


# ------------------ Dataset Tests ------------------


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


class TestPlaces365(LoadsFromArchives[tvd.Places365]):
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


class TestCIFAR10(
    VisionDatasetTests[tvd.CIFAR10], FitsInMemoryTests, DownloadForMockTests, Required
):
    @only_runs_on_clusters()
    def test_always_stored(self):
        assert is_supported_dataset(self.dataset_cls)
        assert is_stored_on_cluster(self.dataset_cls)
        assert Path(locate_dataset_root_on_cluster(self.dataset_cls)).exists()


class TestCIFAR100(
    VisionDatasetTests[tvd.CIFAR100], FitsInMemoryTests, DownloadForMockTests, Required
):
    pass


# NOTE: Here the 'original class' is already in `mila_datamodules.datasets.binary_mnist` because
# we include fixes for bugs in the base class (nothing to do with the clusters though)
class TestBinaryMNIST(VisionDatasetTests[binary_mnist.BinaryMNIST], FitsInMemoryTests):
    pass


class TestBinaryEMNIST(VisionDatasetTests[binary_mnist.BinaryEMNIST], FitsInMemoryTests):
    @pytest.fixture(params=["byclass", "bymerge"])
    def split(self, request: _FixtureRequest[str]) -> str:
        return request.param

    @pytest.fixture
    def dataset_kwargs(self, split: str) -> dict[str, Any]:
        return dict(split=split)


# TODO: Same as for binarymnist (original class is actually our "patched" version of the class) but
# here the fix is specific to one cluster. This isn't pretty.
class TestCaltech101(VisionDatasetTests[caltech101.Caltech101]):
    def test_archives_are_placed_in_slurm_tmpdir(self):
        "101_ObjectCategories.tar.gz"
        "Annotations.tar"
        raise NotImplementedError


class TestCaltech256(VisionDatasetTests[tvd.Caltech256]):
    pass


class TestCelebA(VisionDatasetTests[tvd.CelebA]):
    pass


# Same here for `MNIST`, we have a 'patch' that fixes an issue with dataset folder names on Beluga.
class TestMNIST(VisionDatasetTests[mnist.MNIST], FitsInMemoryTests):
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


class TestImageNet(VisionDatasetTests[tvd.ImageNet]):
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
