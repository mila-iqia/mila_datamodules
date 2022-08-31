"""Script to check which LightningDataModules work out-of-the-box on the current cluster."""
import json
from collections import defaultdict

import cv2  # noqa
from pl_bolts.datamodules import *  # noqa
from pytorch_lightning import LightningDataModule

from mila_datamodules.clusters import CURRENT_CLUSTER

successes = []
failures = []
successes = []
failures = defaultdict(list)
for datamodule_class in LightningDataModule.__subclasses__():
    k = datamodule_class.__qualname__
    print(k)
    try:
        datamodule = datamodule_class(str(CURRENT_CLUSTER.torchvision_dir))
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        for batch in loader:
            break
        print(len(loader))
    except Exception as err:
        exception_str = f"{type(err).__name__}('{err}')"
        failures[exception_str].append(datamodule_class)
    else:
        print(f"Success for {k}")
        successes.append(datamodule_class)

print(successes)
print(
    json.dumps(
        {k: [v.__qualname__ for v in vs] for k, vs in failures.items()},
        indent="\t",
    )
)
# for failure in failures:
#     class Foo(failure):
#         dataset_cls = _adapt_dataset(failure.dataset_cls)
#     datamodule = Foo("/network/datasets/torchvision")
#     datamodule.prepare_data()
