from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Sequence, TypeVar, Union, get_args

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
    files_required_for,
    files_to_copy_for_dataset,
    is_stored_on_cluster,
    locate_dataset_root_on_cluster,
)
from mila_datamodules.registry_test import check_dataset_creation_works_without_download
from mila_datamodules.testutils import (
    only_runs_on_clusters,
    only_runs_when_not_on_a_slurm_cluster,
)
from mila_datamodules.vision.datasets.adapted_datasets import (
    AdaptedDataset,
    prepare_dataset,
)

from ._utils import DatasetType, DownloadableDataset, VisionDatasetType

datasets = {
    k: v
    for k, v in vars(mila_datamodules.vision.datasets).items()
    if inspect.isclass(v) and issubclass(v, VisionDataset)
}

P = ParamSpec("P")
T = TypeVar("T")
DownloadableDatasetT = TypeVar(
    "DownloadableDatasetT", bound=Union[VisionDataset, DownloadableDataset]
)

torchvision_dataset_classes = {
    name: dataset_cls
    for name, dataset_cls in vars(tvd).items()
    if inspect.isclass(dataset_cls)
    and issubclass(dataset_cls, torchvision.datasets.VisionDataset)
    and dataset_cls not in {tvd.FakeData, tvd.VisionDataset, tvd.ImageFolder, tvd.DatasetFolder}
}


@pytest.fixture()
def fake_slurm_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a fake SLURM_TMPDIR in the temporary directory and monkeypatch the environment
    variables to point to it."""
    slurm_tmpdir = tmp_path / "slurm_tmpdir"
    slurm_tmpdir.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(slurm_tmpdir))
    monkeypatch.setenv("FAKE_SLURM_TMPDIR", str(slurm_tmpdir))
    return slurm_tmpdir


@pytest.mark.parametrize("torchvision_dataset_class", torchvision_dataset_classes.values())
def test_all_torchvision_datasets_have_a_test_class(torchvision_dataset_class: type[Dataset]):
    """TODO: This won't work if we split the tests into different files, which is a good idea."""
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


class VisionDatasetTests(DatasetTests[VisionDatasetType]):
    """Suite of basic unit tests for any dataset class from `torchvision.datasets`."""

    _bound: ClassVar[type[Dataset]] = tvd.VisionDataset

    @property
    def adapted_dataset_cls(self) -> type[AdaptedDataset[VisionDatasetType]]:
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
                pytest.fail(
                    "Dataset is required, and `is_stored_on_cluster` says it isn't stored."
                )
            pytest.skip(reason="Dataset isn't stored on the current cluster.")

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


class ReadsFromRoot(VisionDatasetTests[VisionDatasetType]):
    """Tests for datasets that are read from a 'root' directory on the cluster."""

    @only_runs_on_clusters()
    def test_pre_init_returns_existing_root_on_cluster(self):
        location_on_cluster = locate_dataset_root_on_cluster(
            self.dataset_cls, cluster=Cluster.current_or_error()
        )
        assert prepare_dataset(self.dataset_cls) == location_on_cluster

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


class DownloadableDatasetTests(VisionDatasetTests[DownloadableDatasetT]):
    """Tests for dataset that can be downloaded."""

    ...


class Required(VisionDatasetTests[VisionDatasetType]):
    """Tests that raise an error if the dataset isn't stored.

    By default, tests will be skipped if the dataset isn't stored on the current cluster.
    """

    required_on_all_clusters: ClassVar[bool] = True

    @pytest.fixture(scope="session", autouse=True)
    def download_before_tests_on_mock_cluster(
        self, worker_id: str, tmp_path_factory: pytest.TempPathFactory
    ):
        cluster = Cluster.current()
        if cluster is not Cluster._mock:
            # Don't download the dataset, it should already be stored.
            return

        assert "FAKE_SCRATCH" in os.environ
        fake_scratch_dir = get_scratch_dir()
        assert fake_scratch_dir is not None
        fake_scratch_dir.mkdir(exist_ok=True, parents=True)

        if "download" not in inspect.signature(self.dataset_cls.__init__).parameters:
            # TODO: The dataset should be downloaded before the tests are run.
            pytest.fail(
                reason="Dataset needs to be downloaded manually on this machine for these tests to run."
            )
            return

        # get the temp directory shared by all workers.
        # note: we only use this for the filelock, not for downloading the dataset.
        temp_dir = tmp_path_factory.getbasetemp().parent
        # Make this only download the dataset on one worker, using filelocks.
        with filelock.FileLock(temp_dir / f"{self.dataset_cls.__name__}.lock"):
            self.dataset_cls(root=fake_scratch_dir, download=worker_id == "master")  # type: ignore


class FitsInMemoryTests(VisionDatasetTests[VisionDatasetType]):
    """Tests for datasets that fit in RAM or are loaded into RAM anyway, e.g. mnist/cifar10/etc.

    - we should just read them from /network/datasets/torchvision, and not bother copying them to SLURM_TMPDIR.
    """

    # TODO: Add tests that check specifically for this behaviour.


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

    def test_is_stored_on_cluster(self):
        assert is_stored_on_cluster(self.dataset_cls, cluster=Cluster.current_or_error())

    @pytest.mark.disable_socket
    def test_create_with_download_true(self):
        raise NotImplementedError

    def test_pre_init_creates_symlinks_to_archives(self, fake_slurm_tmpdir: Path, tmp_path: Path):
        """Test that the 'pre-init' portion of the adapted constructor creates symlinks to the
        archives that are to be used later."""
        adapted_dataset_class = self.adapted_dataset_cls
        bad_root = tmp_path / "bad_root"
        files_to_copy = files_to_copy_for_dataset(
            adapted_dataset_class.original_class,
            cluster=Cluster.current_or_error(),
            root=str(bad_root),
        )
        assert files_to_copy
        assert not list(fake_slurm_tmpdir.iterdir())

        new_root = prepare_dataset(adapted_dataset_class, root=str(bad_root))

        assert new_root.startswith(str(fake_slurm_tmpdir))

        for file_relative_to_root, file_on_cluster in files_to_copy.items():
            # TODO: Change this depending on if we preserve the structure or if we flatten stuff.
            new_file = Path(new_root) / file_on_cluster.name
            assert new_file.exists() and new_file.is_symlink()
            # TODO: Doesn't work?
            # assert new_file.readlink() == file_on_cluster
            # assert str(new_file.readlink()) == str(file_on_cluster)
            assert new_file.readlink().name == file_on_cluster.name
