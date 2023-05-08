"""Downloads stuff from HuggingFace/torchvision/etc that many researchers might want to use.

If you want to add an entry to this file, please send a message to the IDT team. (or specifically
to Fabrice Normandin)
"""
from __future__ import annotations

import os

SCRATCH = os.environ["SCRATCH"]

shared_cache_dir = "/network/datasets/.weights/.shared_cache"
shared_cache_dir = f"{SCRATCH}/shared_cache"

os.environ["TORCH_HOME"] = f"{shared_cache_dir}/torch"
os.environ["HF_HOME"] = f"{shared_cache_dir}/huggingface"
os.environ.pop("TRANSFORMERS_CACHE", "")
os.environ.pop("HUGGINGFACE_HUB_CACHE", "")
os.environ.pop("HF_DATASETS_CACHE", "")

import torch  # noqa
from datasets import load_dataset  # noqa
from transformers import AutoModel  # noqa

models_to_download = [
    # "bigscience/bloom",
    # "facebook/opt-13b",
    # "facebook/opt-30b",
    # "facebook/opt-66b",
    "t5-large",
    "google/flan-t5-xxl",
]

for model in models_to_download:
    print(f"Downloading {model}")
    # TODO: This crashes because it instantiates the whole model in memory.
    AutoModel.from_pretrained(
        model, resume_download=True, device_map="auto", offload_folder=f"{shared_cache_dir}/.temp"
    )

# TODO: Also download big datasets
datasets_to_download: list[tuple[str, list[str]]] = [
    ("c4", ["en", "realnewslike"]),
    ("gsm8k", ["main", "socratic"]),
    ("wikitext", ["wikitext-103-v1", "wikitext-2-v1", "wikitext-103-raw-v1", "wikitext-2-raw-v1"]),
]
for dataset, configs in datasets_to_download:
    for config in configs:
        print(f"Downloading {dataset} {config}")
        load_dataset(dataset, config, num_proc=16)


tv_models_to_download = [
    "resnet18",
    "resnet50",
    "resnet101",
]
for model in tv_models_to_download:
    print(f"Downloading {model}")
    torch.hub.load("pytorch/vision", model, weights="DEFAULT")
