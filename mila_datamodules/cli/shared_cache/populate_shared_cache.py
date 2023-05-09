#!/home/mila/n/normandf/.conda/envs/datamodules/bin/python
"""Downloads stuff from HuggingFace/torchvision/etc that many researchers might want to use.

If you want to add an entry to this file, please send a message to the IDT team. (or specifically
to Fabrice Normandin or Satya Ortiz-Gagné)
"""
from __future__ import annotations

import argparse
import os

default_shared_cache_dir = "/network/datasets/.weights/shared_cache"


hf_models_to_download = [
    "gpt2",
    "bigscience/bloom",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",
    "t5-large",
    "google/flan-t5-xxl",
]

hf_datasets_to_download: list[tuple[str, list[str] | None]] = [
    ("c4", ["en", "realnewslike", "en.noclean", "en.noblocklist"]),
    ("gsm8k", ["main", "socratic"]),
    ("wikitext", ["wikitext-103-v1", "wikitext-2-v1", "wikitext-103-raw-v1", "wikitext-2-raw-v1"]),
    ("EleutherAI/pile", ["all"]),
    ("togethercomputer/RedPajama-Data-1T", None),
]

torchvision_models_to_download = [
    "resnet18",
    "resnet50",
    "resnet101",
]
_offload_folder = "/tmp/offload"
_num_procs = 16


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared_cache_dir", type=str, default=default_shared_cache_dir)
    args = parser.parse_args(argv)

    shared_cache_dir: str = args.shared_cache_dir

    os.environ["TORCH_HOME"] = f"{shared_cache_dir}/torch"
    os.environ["HF_HOME"] = f"{shared_cache_dir}/huggingface"
    os.environ.pop("TRANSFORMERS_CACHE", "")
    os.environ.pop("HUGGINGFACE_HUB_CACHE", "")
    os.environ.pop("HF_DATASETS_CACHE", "")

    import torch  # noqa
    from datasets import load_dataset  # noqa
    from transformers import AutoModel, AutoConfig  # noqa

    for dataset, configs in hf_datasets_to_download:
        configs = configs or [None]
        for config in configs:
            print(f"Downloading {dataset} {config}")
            load_dataset(dataset, config, num_proc=_num_procs)

    for model in hf_models_to_download:
        print(f"Downloading {model}")

        # TODO: This instantiates the model, which we don't need to be doing!
        # Need to checkout how to download the model without instantiating it.
        # from huggingface_hub import cached_download, hf_hub_download
        # from transformers.utils.hub import cached_file
        # NOTE: also, this saves the weights in the offload directory (which is currently in /tmp).
        AutoModel.from_pretrained(
            model,
            resume_download=True,
            device_map="auto",
            offload_folder=_offload_folder,
        )

    for model in torchvision_models_to_download:
        print(f"Downloading {model}")
        torch.hub.load("pytorch/vision", model, weights="DEFAULT")


if __name__ == "__main__":
    main()
