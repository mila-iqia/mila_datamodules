from __future__ import annotations

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim.adam import Adam
from torchvision.models import resnet18


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


@pytest.mark.parametrize("accelerator", ["gpu", "auto"])
@pytest.mark.parametrize("devices", [1, "auto"])
@pytest.mark.parametrize("strategy", [None, "dp", "ddp", "fsdp"])
def test_no_extra_copy(strategy: str | None, devices: int | str, accelerator: str):
    """Test that no extra move or copy is made by PyTorch-Lightning when the DataLoader gives
    tensors that are already on the GPU."""
    assert torch.cuda.is_available()
    num_batches = 5

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=1,
        limit_train_batches=num_batches,
        enable_checkpointing=False,
    )
    batch_size = 4
    # Note: Uncomment to test with the same thing.
    # datamodule = ImagenetFfcvDataModule(batch_size=batch_size, num_workers=4)
    # datamodule.prepare_data()
    # batches = list(itertools.islice(datamodule.train_dataloader(), num_batches))
    batches = [
        (
            torch.rand([batch_size, 128], dtype=torch.float32, device="cuda"),
            torch.randint(1000, [batch_size], device="cuda"),
        )
        for _ in range(num_batches)
    ]

    class DummyModel(Model):
        def __init__(self, num_classes: int = 1000):
            super().__init__(num_classes=num_classes)
            self.batches_seen = 0

        def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
            x, y = batch
            # Check that the memory address is the same as the source tensor, e.g. that no extra
            # allocation or move was made between the loader and the model.

            ref_x, ref_y = batches[batch_idx]
            # TODO: Doesn't work! Why?
            # assert x is ref_x
            # assert y is ref_y
            assert x.dtype == ref_x.dtype
            assert y.dtype == ref_y.dtype
            assert x.shape == ref_x.shape
            assert y.shape == ref_y.shape
            assert (x == ref_x).all()
            assert (y == ref_y).all()
            assert x.is_contiguous() == ref_x.is_contiguous() is True
            assert y.is_contiguous() == ref_y.is_contiguous() is True
            assert x.data_ptr() == ref_x.data_ptr()
            assert y.data_ptr() == ref_y.data_ptr()
            return super().training_step((x, y), batch_idx)

    model = DummyModel()
    model.cuda()
    # Initialize the nn.LazyLinear by doing a forward pass with a batch of data.
    model(batches[0][0])

    # Pretend that this is a dataloader-ish, and fit the model with it.
    train_dataloader = iter(batches)
    trainer.fit(model, train_dataloaders=train_dataloader)
    assert model.batches_seen == num_batches
