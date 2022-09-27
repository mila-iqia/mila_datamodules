import mila_datamodules.vision as _mdv

from ._utils import _builds

# TODO: Figure out a nicer way of selectively turning the registration on/off?
REGISTER_CONFIGS = True
# Create and declare the config classes for each datmodule.
# NOTE: These are really nicely typed, thanks to `hydra_zen`!
# NOTE: For any other datamodule that isn't found, we generate a config dynamically.

BinaryEMNISTDataModuleConfig = _builds(_mdv.BinaryEMNISTDataModule, register=REGISTER_CONFIGS)
BinaryMNISTDataModuleConfig = _builds(_mdv.BinaryMNISTDataModule, register=REGISTER_CONFIGS)
CIFAR10DataModuleConfig = _builds(_mdv.CIFAR10DataModule, register=REGISTER_CONFIGS)
CityscapesDataModuleConfig = _builds(_mdv.CityscapesDataModule, register=REGISTER_CONFIGS)
EMNISTDataModuleConfig = _builds(_mdv.EMNISTDataModule, register=REGISTER_CONFIGS)
FashionMNISTDataModuleConfig = _builds(_mdv.FashionMNISTDataModule, register=REGISTER_CONFIGS)
ImagenetDataModuleConfig = _builds(_mdv.ImagenetDataModule, register=REGISTER_CONFIGS)
ImagenetFfcvDataModuleConfig = _builds(_mdv.ImagenetFfcvDataModule, register=REGISTER_CONFIGS)
MNISTDataModuleConfig = _builds(_mdv.MNISTDataModule, register=REGISTER_CONFIGS)
STL10DataModuleConfig = _builds(_mdv.STL10DataModule, register=REGISTER_CONFIGS)
