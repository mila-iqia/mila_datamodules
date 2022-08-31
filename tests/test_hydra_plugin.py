# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hydra
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.mila_datamodules_plugin import MilaDatamodulesSearchPathPlugin


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert MilaDatamodulesSearchPathPlugin.__name__ in [
        x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)
    ]


def test_config_installed() -> None:
    with initialize(version_base=None):
        config_loader = GlobalHydra.instance().config_loader()
        assert "imagenet" in config_loader.get_group_options("datamodule")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


# from hydra import initialize, compose

# # 1. initialize will add config_path the config search path within the context
# # 2. The module with your configs should be importable.
# #    it needs to have a __init__.py (can be empty).
# # 3. THe config path is relative to the file calling initialize (this file)
# def test_with_initialize() -> None:
#     with initialize(config_path="./conf"):
#         # config is relative to a module
#         cfg = compose(config_name="config", overrides=["datamodule=imagenet"])
#         assert cfg == {
#             "app": {"user": "test_user", "num1": 10, "num2": 20},
#             "db": {"host": "localhost", "port": 3306},
#         }


if __name__ == "__main__":
    my_app()
