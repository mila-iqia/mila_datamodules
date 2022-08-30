"""Auto-generated Configuration dataclasses for use with Hydra or SimpleParsing."""

from .datamodule import (
    CIFAR10DataModuleConfig,
    CityscapesDataModuleConfig,
    ImagenetDataModuleConfig,
    ImagenetFfcvDataModuleConfig,
    MNISTDataModuleConfig,
)

# TODO: Add to lightning-hydra-template repo:
# from hydra.core.config_store import ConfigStore
# from hydra_configs.pytorch_lightning.trainer import TrainerConf

# cs = ConfigStore.instance()
# cs.store(group="trainer", name="default", node=TrainerConf)
