"""Auto-generated Configuration dataclasses for use with Hydra or SimpleParsing."""

from hydra.core.config_store import ConfigStore
from hydra_zen import make_custom_builds_fn

from mila_datamodules.vision import (
    CIFAR10DataModule,
    CityscapesDataModule,
    FashionMNISTDataModule,
    ImagenetDataModule,
    ImagenetFfcvDataModule,
    MNISTDataModule,
)

# note: this is just so we don't have to pass the `populate_full_signature` to all calls below.
builds = make_custom_builds_fn(populate_full_signature=True)
MNISTDataModuleConfig = builds(MNISTDataModule)
FashionMNISTDataModuleConfig = builds(FashionMNISTDataModule)
CIFAR10DataModuleConfig = builds(CIFAR10DataModule)
ImagenetDataModuleConfig = builds(ImagenetDataModule)
ImagenetFfcvDataModuleConfig = builds(ImagenetFfcvDataModule)
CityscapesDataModuleConfig = builds(CityscapesDataModule)

_cs = ConfigStore.instance()
_cs.store(group="datamodule", name="mnist", node=MNISTDataModuleConfig)
_cs.store(group="datamodule", name="fashionmnist", node=FashionMNISTDataModuleConfig)
_cs.store(group="datamodule", name="cifar10", node=CIFAR10DataModuleConfig)
_cs.store(group="datamodule", name="imagenet", node=ImagenetDataModuleConfig)
_cs.store(group="datamodule", name="imagenet_ffcv", node=ImagenetFfcvDataModuleConfig)
_cs.store(group="datamodule", name="cityscapes", node=CityscapesDataModuleConfig)

# TODO: Add to lightning-hydra-template repo:
# from hydra.core.config_store import ConfigStore
# from hydra_configs.pytorch_lightning.trainer import TrainerConf

# cs = ConfigStore.instance()
# cs.store(group="trainer", name="default", node=TrainerConf)
