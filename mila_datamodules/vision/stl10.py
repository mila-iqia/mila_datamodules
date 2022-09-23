import torchvision.datasets as tvd
from pl_bolts.datamodules import STL10DataModule as _STL10DataModule

from mila_datamodules.registry import get_dataset_root
from mila_datamodules.utils import replace_arg_defaults

# Get the data directory, if possible.
default_data_dir = get_dataset_root(tvd.STL10)


# NOTE: This prevents a useless download by setting the `data_dir` to a good default value.
class STL10DataModule(_STL10DataModule):
    # NOTE: This doesn't subclass VisionDataModule, or set a `dataset_cls` attribute, so it's a
    # little bit harder to optimize this datamodule. However, it's probably fine in this case to
    # just load the data from the dataset dir, since it's in-memory numpy arrays anyway.
    __init__ = replace_arg_defaults(
        _STL10DataModule.__init__,
        data_dir=default_data_dir,
    )
