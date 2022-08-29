from hydra.core.config_store import ConfigStore
from hydra_zen import builds

from mila_datamodules.vision import (
    CIFAR10DataModule,
    CityscapesDataModule,
    ImagenetDataModule,
    ImagenetFfcvDataModule,
    MNISTDataModule,
)

MNISTDataModuleConfig = builds(MNISTDataModule, populate_full_signature=True)
CIFAR10DataModuleConfig = builds(CIFAR10DataModule, populate_full_signature=True)
ImagenetDataModuleConfig = builds(ImagenetDataModule, populate_full_signature=True)
ImagenetFfcvDataModuleConfig = builds(ImagenetFfcvDataModule, populate_full_signature=True)
CityscapesDataModuleConfig = builds(CityscapesDataModule, populate_full_signature=True)

cs = ConfigStore.instance()
cs.store(group="datamodule", name="mnist", node=MNISTDataModuleConfig)
cs.store(group="datamodule", name="cifar10", node=CIFAR10DataModuleConfig)
cs.store(group="datamodule", name="imagenet", node=ImagenetDataModuleConfig)
cs.store(group="datamodule", name="imagenet_ffcv", node=ImagenetFfcvDataModuleConfig)
cs.store(group="datamodule", name="cityscapes", node=CityscapesDataModuleConfig)
