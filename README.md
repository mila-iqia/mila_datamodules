# (WIP) mila_datamodules

Efficient Datamodules customized for the Mila/CC clusters

## Installation:

This can be installed as usual with `pip`:

```console
pip install "mila_datamodules @ git+https://www.github.com/lebrice/mila_datamodules.git"
```

`ffcv` is also an optional dependency used in the `ImagenetFfcvDataModule`.
Installing FFCV can be a bit of a pain at the moment. But we're working on it.
For now, your best bet is to use conda with the provided env.yaml:

```console
$ module load miniconda/3
$ conda env create -n ffcv -f env.yaml
$ conda activate ffcv
$ pip install git+https://www.github.com/lebrice/mila_datamodules
```

## TODOs:

- [x] Add all the datamodules from pl_bolts for the Mila Cluster
- [x] Add all the datasets from torchvision for the Mila Cluster
- [ ] Add datamodules for the existing datasets of the Mila cluster at https://datasets.server.mila.quebec/
  - [ ] Figure out which datasets to tackle first!
  - [ ] Figure out how to instantiate each dataset in code (@satyaog ?)
  - [ ] Implement a datamodule for each dataset

## Supported Datasets and clusters

| Symbol | Meaning                               |
| ------ | ------------------------------------- |
| ✅      | Supported, tested.                    |
| ✓      | Supported, not tested.                |
| ?      | Don't know if the cluster has it yet. |
| TODO   | Cluster has it, not yet added         |
| ❌      | Not available in that cluster         |

### Vision

| Dataset         | Type                 | Mila | Beluga | Cedar | Graham | Narval |
| --------------- | -------------------- | ---- | ------ | ----- | ------ | ------ |
| Imagenet        | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| Imagenet (ffcv) | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| Cityscapes      | Image Segmentation?  | ✅    | ?      | ?     | ?      | ?      |
| CIFAR10         | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| CIFAR100        | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| STL10           | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| MNIST           | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| FashionMNIST    | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| EMNIST          | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| BinaryMNIST     | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| BinaryEMNIST    | Image Classification | ✅    | ?      | ?     | ?      | ?      |
| FFHQ            | Images (faces)       | TODO | ?      | ?     | ?      | ?      |
| GLUE            | Text                 | ✅    | ✓      | ✓     | ✓      | ✓      |
| COCO_captions   | Image + Text         | ✓    | ?      | ?     | ?      | ?      |
