"""
"""
from __future__ import annotations

import datetime
import itertools

import torch
import tqdm
from pytorch_lightning import Trainer
from typing_extensions import ParamSpec
from torchvision.models import resnet50
from mila_datamodules import ImagenetDataModule, ImagenetFfcvDataModule
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam

P = ParamSpec("P")
MAX_BATCHES = 50


class DummyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        # Check that the memory address is the same as the source tensor, e.g. that no extra
        # allocation or move was made between the loader and the model.
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


def _trainer():
    return Trainer(
        accelerator="gpu",
        devices=1,
        strategy=None,
        limit_val_batches=MAX_BATCHES,
        limit_train_batches=MAX_BATCHES,
        max_epochs=1,
        enable_checkpointing=False,
    )


def main():

    datamodule = ImagenetDataModule(batch_size=512, num_workers=4)
    datamodule.prepare_data()

    print(f"Pure for loops with 100 batches:")
    pure_loop("PyTorch", datamodule, max_batches=100)

    datamodule_ffcv = ImagenetFfcvDataModule(batch_size=512, num_workers=4)
    datamodule_ffcv.prepare_data()
    pure_loop("FFCV", datamodule_ffcv)
    # pure_loop(
    #     "FFCV + torchvision",
    #     HackyImagenetFfcvDataModule(batch_size=512, num_workers=4),
    # )

    # print(f"Training with {MAX_BATCHES} batches:")
    # TODO: Check if order of operations doesn't have an impact on the results?
    # TODO: Roll everything twice, and only take the second value.
    # (to avoid caching having an impact)
    # print(f"Hacky FFCV: {debug()}")
    # print("FFCV (manual eval loop):", ffcv_manual())
    # print("PL + FFCV:", pl_ffcv())
    # print("PL + DataLoaders:", pl())
    # print("PL + DataLoaders (no optimizations):", pl_without_dataloader_optimizations())


def pure_loop(desc: str, datamodule: ImagenetDataModule, max_batches=1000):
    loader = datamodule.train_dataloader()
    start_time = datetime.datetime.now()
    for _ in tqdm.tqdm(
        itertools.islice(loader, max_batches),
        desc=desc,
        leave=True,
        total=max_batches,
    ):
        pass
    # trainer.fit(model, train_dataloaders=loader)
    return datetime.datetime.now() - start_time


def debug():
    datamodule = HackyImagenetFfcvDataModule(batch_size=512, num_workers=16)
    trainer = _trainer()
    model = Model(
        image_dims=datamodule.dims,
        num_classes=datamodule.num_classes,
        extra_preprocessing=getattr(datamodule, "extra_preprocessing", None),
    )
    loader = datamodule.train_dataloader()

    start_time = datetime.datetime.now()
    for _ in tqdm.tqdm(itertools.islice(loader, MAX_BATCHES)):
        pass
    # trainer.fit(model, train_dataloaders=loader)
    return datetime.datetime.now() - start_time


def pl():
    datamodule = ImagenetDataModule(batch_size=512, num_workers=16)
    trainer = _trainer()
    model = Model(image_dims=datamodule.dims, num_classes=datamodule.num_classes)
    loader = datamodule.train_dataloader()
    start_time = datetime.datetime.now()
    trainer.fit(model, train_dataloaders=loader)
    return datetime.datetime.now() - start_time


def pl_without_dataloader_optimizations():
    datamodule = ImagenetDataModule(batch_size=512, num_workers=16)
    trainer = _trainer()
    model = Model(image_dims=datamodule.dims, num_classes=datamodule.num_classes)
    loader = datamodule.train_dataloader()
    start_time = datetime.datetime.now()
    # NOTE: By passing an islice, the idea is that now PL doesn't get a DataLoader,
    # and therefore won't apply the optimizations that it might normally apply.
    trainer.fit(model, train_dataloaders=itertools.islice(loader, MAX_BATCHES))
    return datetime.datetime.now() - start_time


def pl_ffcv():
    datamodule = ImagenetFfcvDataModule(batch_size=512, num_workers=16)
    trainer = _trainer()
    model = Model(
        image_dims=datamodule.dims,
        num_classes=datamodule.num_classes,
    )
    loader_ffcv = datamodule.train_dataloader()
    start_time = datetime.datetime.now()
    trainer.fit(model, train_dataloaders=loader_ffcv)
    return datetime.datetime.now() - start_time


def ffcv_manual():
    datamodule = ImagenetFfcvDataModule(batch_size=512, num_workers=16)
    model = Model(image_dims=datamodule.dims, num_classes=datamodule.num_classes)
    model.cuda()
    model.eval()
    loader_ffcv = datamodule.train_dataloader()
    start_time = datetime.datetime.now()
    with torch.set_grad_enabled(False):
        for batch in tqdm.tqdm(itertools.islice(loader_ffcv, MAX_BATCHES)):
            assert all(v.device.type == "cuda" for v in batch)
            _ = model(batch[0])
            # _ = model.validation_step(batch, batch_idx=batch_idx)
            # loss = model.training_step(batch, batch_idx=batch_idx)

    return datetime.datetime.now() - start_time


if __name__ == "__main__":
    main()
