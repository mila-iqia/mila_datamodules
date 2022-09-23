import torch
from pl_bolts.datamodules.sr_datamodule import TVTDataModule
from torch.utils.data import random_split

from mila_datamodules.clusters import SLURM_TMPDIR, Cluster

from .datasets import CocoCaptions

captions_train_annFile_location = {
    Cluster.Mila: "/network/datasets/torchvision/annotations/captions_train2017.json",
    # TODO:
    # ClusterType.BELUGA: "?",
}

captions_test_annFile_location = {
    Cluster.Mila: "/network/datasets/torchvision/annotations/captions_val2017.json",
    # TODO:
    # ClusterType.BELUGA: "?",
}


class CocoCaptionsDataModule(TVTDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        val_split: float = 0.1,
        val_split_seed: int = 42,
    ) -> None:
        self.val_split = val_split
        self.val_split_seed = val_split_seed

        train_annFile = captions_train_annFile_location[Cluster.current()]
        test_annFile = captions_test_annFile_location[Cluster.current()]

        # NOTE: Root gets ignored anyway.
        dataset_trainval = CocoCaptions(root=str(SLURM_TMPDIR), annFile=train_annFile)
        dataset_test = CocoCaptions(root=str(SLURM_TMPDIR), annFile=test_annFile)
        val_len = int(len(dataset_trainval) * val_split)
        dataset_train, dataset_val = random_split(
            dataset_trainval,
            [len(dataset_trainval) - val_len, val_len],
            generator=torch.random.manual_seed(val_split_seed),
        )

        super().__init__(
            dataset_train,
            dataset_val,
            dataset_test,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
            drop_last,
        )
