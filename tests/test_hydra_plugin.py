# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule

import mila_datamodules
import mila_datamodules.configs
from mila_datamodules.vision.coco import CocoCaptionsDataModule

TEST_DIR = Path(__file__).parent
config_path = "dummy_conf"


def test_discovery() -> None:
    # from hydra_plugins.mila_datamodules_plugin import MilaDatamodulesSearchPathPlugin
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    # NOTE: Using just the string to make sure that it's discovered even when not imported!
    assert "MilaDatamodulesSearchPathPlugin" in [
        x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)
    ]


datamodules = [
    v
    for v in vars(mila_datamodules).values()
    if inspect.isclass(v) and issubclass(v, LightningDataModule)
]


@pytest.mark.parametrize("datamodule_cls", datamodules)
def test_every_datamodule_has_a_config(datamodule_cls: type[LightningDataModule]):
    assert hasattr(mila_datamodules.configs, datamodule_cls.__name__ + "Config")


def test_dynamically_created_datamodule_config():
    """When importing a config class that doesn't explicitly exist yet, we get a dynamically
    created config dataclass."""
    # TODO: Ideally, we'd create a new datamodule class, so that this test stays
    from mila_datamodules.configs.datamodule import CocoCaptionsDataModuleConfig

    config = CocoCaptionsDataModuleConfig()
    assert (
        config._target_
        == CocoCaptionsDataModule.__module__ + "." + CocoCaptionsDataModule.__qualname__
    )


def test_configs_from_plugin_are_available() -> None:
    with initialize(version_base=None, config_path=config_path):
        config_loader = GlobalHydra.instance().config_loader()
        available_datamodules = config_loader.get_group_options("datamodule")
        assert "imagenet" in available_datamodules


def test_user_configs_also_available() -> None:
    with initialize(version_base=None, config_path=config_path):
        config_loader = GlobalHydra.instance().config_loader()
        available_datamodules = config_loader.get_group_options("datamodule")
        assert "becky" in available_datamodules


def test_use_plugin_config_in_defaults() -> None:
    """Test that user code can use a config value from the plugin as a default value.

    NOTE: This assumes that `datamodule=cifar10` is set in the defaults list of conf/config.yaml.
    """

    with initialize(version_base=None, config_path=config_path):
        config = compose(config_name="config")
        datamodule_config = OmegaConf.to_object(config.datamodule)

        from mila_datamodules.configs import CIFAR10DataModuleConfig

        assert isinstance(datamodule_config, CIFAR10DataModuleConfig)
        assert datamodule_config == CIFAR10DataModuleConfig()


def test_customizing_config_from_plugin() -> None:
    """Test that user code can use a .yaml file to overwrite values of the datamodule arguments."""
    with initialize(version_base=None, config_path=config_path):
        config = compose(config_name="config", overrides=["datamodule=custom_mnist"])
        config = OmegaConf.to_object(config.datamodule)
        from mila_datamodules.configs import MNISTDataModuleConfig

        assert config == MNISTDataModuleConfig(batch_size=111)


def test_instantiate() -> None:
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        config = compose(config_name="config", overrides=["datamodule=cifar10"])
        # TODO: Test that this `cifar10` isn't just a string!
        # TODO: Try to instantiate the datamodule.
        datamodule = instantiate(config.datamodule)

        from mila_datamodules.vision.cifar10 import CIFAR10DataModule

        assert isinstance(datamodule, CIFAR10DataModule)


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def my_app(config: DictConfig) -> None:
    print(config)
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    my_app()
