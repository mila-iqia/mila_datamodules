import json
from collections import defaultdict

from torchvision.datasets import VisionDataset

from mila_datamodules.clusters import CURRENT_CLUSTER

successes = []
failures = defaultdict(list)
for v in VisionDataset.__subclasses__():
    k = v.__qualname__
    print(k)
    try:
        dataset = v(str(CURRENT_CLUSTER.torchvision_dir))
    except Exception as err:
        failures[f"{type(err).__name__}('{err}')"].append(v)
    else:
        print(f"Success for {k}")
        successes.append(v)

print("Successes:", successes)

print("Failures:")
print(
    json.dumps(
        {k: [v.__qualname__ for v in vs] for k, vs in failures.items()},
        indent="\t",
    )
)

FAILURES_MILA = """{
    TypeError("Can't instantiate abstract class FlowDataset with abstract method _read_flow"): [
        "FlowDataset"
    ],
    RuntimeError("Dataset not found or corrupted. You can use download=True to download it"): [
        "CLEVRClassification",
        "Flowers102",
        "Omniglot",
        "_VOCBase",
    ],
    TypeError("__init__() missing 1 required positional argument: 'annFile'"): [tvd.CocoDetection],
    TypeError("__init__() missing 1 required positional argument: 'loader'"): ["DatasetFolder"],
    RuntimeError("Dataset not found. You can use download=True to download it"): [
        "DTD",
        "FGVCAircraft",
        "Food101",
        "GTSRB",
        "OxfordIIITPet",
        "RenderedSST2",
        "StanfordCars",
        "SUN397",
    ],
    RuntimeError(
        "train.csv not found in /network/datasets/torchvision/fer2013 or corrupted. You can download it from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
    ): ["FER2013"],
    TypeError("__init__() missing 1 required positional argument: 'ann_file'"): [
        "Flickr8k",
        "Flickr30k",
    ],
    TypeError(
        "__init__() missing 2 required positional arguments: 'annotation_path' and 'frames_per_clip'"
    ): [
        "HMDB51",
        "UCF101",
    ],
    TypeError("__init__() missing 1 required positional argument: 'frames_per_clip'"): [
        "Kinetics"
    ],
    RuntimeError("Dataset not found. You may use download=True to download it."): ["Kitti"],
    TypeError(
        "__init__() missing 3 required positional arguments: 'split', 'image_set', and 'view'"
    ): ["_LFW"],
    ModuleNotFoundError("No module named 'lmdb'"): ["LSUNClass", "LSUN"],
    RuntimeError(
        "h5py is not found. This dataset needs to have h5py installed: please run pip install h5py"
    ): ["PCAM"],
    TypeError("__init__() missing 1 required positional argument: 'name'"): ["PhotoTour"],
    FileNotFoundError(
        "[Errno 2] No such file or directory: '/network/datasets/torchvision/train.txt'"
    ): ["SBDataset"],
    HTTPError("HTTP Error 403: Forbidden"): ["SBU"],
    OSError("[Errno 30] Read-only file system: '/network/datasets/torchvision/semeion.data'"): [
        "SEMEION"
    ],
    FileNotFoundError(
        "[Errno 2] No such file or directory: '/network/datasets/torchvision/usps.bz2'"
    ): ["USPS"],
    RuntimeError(
        "Dataset not found or corrupted. You can use download=True to download and prepare it"
    ): ["WIDERFace"],
}
"""
