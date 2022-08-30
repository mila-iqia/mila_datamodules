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

cs = ConfigStore.instance()
cs.store(group="datamodule", name="mnist", node=MNISTDataModuleConfig)
cs.store(group="datamodule", name="fashionmnist", node=FashionMNISTDataModuleConfig)
cs.store(group="datamodule", name="cifar10", node=CIFAR10DataModuleConfig)
cs.store(group="datamodule", name="imagenet", node=ImagenetDataModuleConfig)
cs.store(group="datamodule", name="imagenet_ffcv", node=ImagenetFfcvDataModuleConfig)
cs.store(group="datamodule", name="cityscapes", node=CityscapesDataModuleConfig)
