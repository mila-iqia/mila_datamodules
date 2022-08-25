""" GLUEDataModule from the PL + HuggingFace example.

https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
"""
from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional

import datasets
import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets.load import load_dataset

Task = Literal[
    "cola",
    "sst2",
    "mrpc",
    "qqp",
    "stsb",
    "mnli",
    "qnli",
    "rte",
    "wnli",
    "ax",
]


class GlueDataModule(LightningDataModule):

    task_text_field_map: dict[Task, list[str]] = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels: dict[Task, int] = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: Task = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer: AutoTokenizer | None = None
        self.eval_splits: list[str]
        self.columns: list[str]
        self.dataset: datasets.dataset_dict.DatasetDict
        self.save_hyperparameters()

    def prepare_data(self):
        load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str | None = None) -> None:
        dataset = load_dataset("glue", self.task_name)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, use_fast=True
            )
        assert isinstance(dataset, DatasetDict)
        self.dataset = dataset

        for split in self.dataset.keys():
            # IDEA (@lebrice): What about putting this on the model instead? Since the choice of
            # tokenizer is closely linked to the choice of model, this kind of implicit coupling
            # isn't great.
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"], batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"], batch_size=self.eval_batch_size
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
