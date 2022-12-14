""""""
from __future__ import annotations

import datetime
import itertools

# NOTE: Need to import cv2 to prevent a loading error for GLIBCXX with ffcv.
import cv2  # noqa
import tqdm
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from typing_extensions import ParamSpec

from mila_datamodules.vision import ImagenetDataModule

# logging.basicConfig(level=logging.DEBUG)
P = ParamSpec("P")
MAX_BATCHES = 50


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.encoder(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore
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
        limit_val_batches=0,
        limit_train_batches=MAX_BATCHES,
        max_epochs=1,
        enable_checkpointing=False,
        # log_every_n_steps=0,
        logger=False,
    )


def main():
    num_workers = 6
    batch_size = 128
    # Debugging those:
    # datamodule_torch = CityscapesDataModule(batch_size=batch_size, num_workers=num_workers)
    # datamodule_torch = CIFAR10DataModule(batch_size=batch_size, num_workers=num_workers)

    # datamodule_torch = ImagenetDataModule(batch_size=batch_size, num_workers=num_workers)
    # datamodule_torch.prepare_data()
    # datamodule_torch.setup()

    # datamodule_ffcv = ImagenetFfcvDataModule(batch_size=batch_size, num_workers=num_workers)
    # datamodule_ffcv.prepare_data()

    # datamodule_ffcv_smaller_res = ImagenetFfcvDataModule(
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     img_resolution_config=ImageResolutionConfig(max_res=160),
    # )
    # datamodule_ffcv_smaller_res.prepare_data()

    # print("Pure for loops over 200 batches:")
    # print("PyTorch:\n", for_loop(datamodule_torch, max_batches=200))
    # print("FFCV:\n", for_loop(datamodule_ffcv, max_batches=200))
    # print("FFCV (smaller res):\n", for_loop(datamodule_ffcv_smaller_res, max_batches=200))

    # print(f"Training on {MAX_BATCHES} batches:")
    # TODO: Maybe caching has an impact? Roll everything twice, and only take the second value.
    # print("Manual train loop (Pytorch):", manual_loop(datamodule_torch))
    # print("Manual train loop (FFCV):", manual_loop(datamodule_ffcv))
    # print("Manual train loop (FFCV, smaller res):", manual_loop(datamodule_ffcv_smaller_res))
    # print("PL + DataLoaders:", pl_train_time(datamodule_torch))
    # print("PL + FFCV:", pl_train_time(datamodule_ffcv))
    # print("PL + FFCV (smaller res):", pl_train_time(datamodule_ffcv_smaller_res))
    # print("PL + DataLoaders (hidden):", pl_train_time(datamodule_torch, obfuscate=True))


def for_loop(datamodule: LightningDataModule, max_batches=1000):
    loader = datamodule.train_dataloader()
    start_time = datetime.datetime.now()
    for _ in tqdm.tqdm(
        itertools.islice(loader, max_batches),
        leave=True,
        total=max_batches,
    ):
        pass
    # trainer.fit(model, train_dataloaders=loader)
    return datetime.datetime.now() - start_time


def pl_train_time(datamodule: ImagenetDataModule, obfuscate=False):
    loader = datamodule.train_dataloader()
    trainer = _trainer()
    model = Model()

    def _obfuscate(loader):
        def _inner_hidden_fn():
            yield from loader

        return _inner_hidden_fn()

    if obfuscate:
        loader = _obfuscate(loader)
    start_time = datetime.datetime.now()
    # NOTE: By passing an islice, the idea is that now PL doesn't get a DataLoader,
    # and therefore won't apply the optimizations that it might normally apply.
    trainer.fit(model, train_dataloaders=loader)  # type: ignore
    return datetime.datetime.now() - start_time


def manual_loop(datamodule: ImagenetDataModule):
    model = Model()
    model.cuda()
    model.eval()
    loader = datamodule.train_dataloader()
    optimizer = model.configure_optimizers()
    start_time = datetime.datetime.now()
    for i, batch in enumerate(tqdm.tqdm(itertools.islice(loader, MAX_BATCHES))):
        if isinstance(loader, DataLoader):
            batch = tuple(v.cuda() for v in batch)
        assert all(v.device.type == "cuda" for v in batch)
        # _ = model.validation_step(batch, batch_idx=batch_idx)
        optimizer.zero_grad()
        loss = model.training_step(batch, batch_idx=i)
        loss.backward()
        optimizer.step()

    return datetime.datetime.now() - start_time


if __name__ == "__main__":
    main()
