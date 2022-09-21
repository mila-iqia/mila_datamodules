# (WIP) Cluster-Aware Datamodules

Efficient Datamodules customized for the Mila/ComputeCanada clusters

## Installation:

This can be installed with `pip`:

```console
pip install "mila_datamodules @ git+https://www.github.com/lebrice/mila_datamodules.git"
```

`ffcv` is an optional dependency, only currently used in the `ImagenetFfcvDataModule`.
Installing FFCV can be a bit of a pain at the moment. But we're working on it.
For now, your best bet is to use conda with the provided env.yaml:

```console
$ module load miniconda/3
$ conda env create -n ffcv -f env.yaml
$ conda activate ffcv
$ pip install git+https://www.github.com/lebrice/mila_datamodules
```

## Usage:

DataModules are a great way to organize your data loading code.
The datamodules from `mila_datamodules` are opinionated in the way they handle where/how to
download/store the data, so you don't have to think about it.

```python

from mila_datamodules.vision import ImagenetDataModule
imagenet = ImagenetDataModule(batch_size=512)

# With PyTorch Lightning:
trainer = Trainer(gpus=8, accelerator="ddp")
trainer.fit(model, imagenet)

# Without PyTorch Lightning:
imagenet.prepare_data()
imagenet.setup()
train_dataloader = imagenet.train_dataloader()
val_dataloader = imagenet.val_dataloader()
test_dataloader = imagenet.test_dataloader()
```

## TODOs:

- [x] Add all the datamodules from pl_bolts for the Mila Cluster
- [x] Add all the datasets from torchvision for the Mila Cluster
- [ ] Add datamodules for the existing datasets of the Mila cluster at https://datasets.server.mila.quebec/
  - [ ] Figure out which datasets to tackle first!
  - [ ] Figure out how to instantiate each dataset in code (@satyaog ?)
  - [ ] Implement a datamodule for each dataset
- [ ] Do the same for the CC clusters, one at a time!

## Supported Datasets

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
| Imagenet        | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| Imagenet (ffcv) | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| CIFAR10         | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| CIFAR100        | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| STL10           | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| MNIST           | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| FashionMNIST    | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| EMNIST          | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| BinaryMNIST     | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| BinaryEMNIST    | Image Classification | ✅    | TODO   | ?     | ?      | ?      |
| Cityscapes      | Image Segmentation?  | ✅    | TODO   | ?     | ?      | ?      |
| FFHQ            | Images (faces)       | TODO | ?      | ?     | ?      | ?      |
| COCO_captions   | Image + Text         | ✓    | ?      | ?     | ?      | ?      |

### NLP

| Dataset            | Type | Mila | Beluga | Cedar | Graham | Narval |
| ------------------ | ---- | ---- | ------ | ----- | ------ | ------ |
| GLUE (\*downloads) | Text | ✅    | ✓      | ✓     | ✓      | ✓      |
| C4                 | Text | TODO | ?      | ?     | ?      | ?      |
| wikitext           | Text | TODO | ?      | ?     | ?      | ?      |
| Wikipedia          | Text | ?    | ?      | ?     | ?      | ?      |

______________________________________________________________________

(wip) Note: Datasets on the Mila cluster (/network/datasets)

- [ ] ami
- [ ] c4
- [ ] caltech101
- [ ] caltech256
- [ ] celeba
- [ ] cifar10
- [ ] cifar100
- [ ] .cifar100.git.bak
- [ ] .cifar10.git.bak
- [x] cityscapes
- [ ] climatenet
- [x] coco
- [ ] commonvoice
- [ ] conceptualcaptions
- [ ] convai2
- [ ] covid-19
- [ ] cub200
- [ ] dcase2020
- [ ] describable-textures
- [ ] dns-challenge
- [ ] domainnet
- [ ] dtd
- [x] fashionmnist
- [ ] ffhq
- [ ] fgvcaircraft
- [ ] fgvcxfungi
- [ ] flowers102
- [ ] gaia
- [ ] geolifeclef
- [ ] gtsrb
- [ ] hotels50K
- [ ] icentia11k
- [x] imagenet
- [ ] inat
- [ ] kitti
- [x] kmnist
- [ ] LDC
- [ ] librispeech
- [ ] lincs_l1000
- [ ] .log
- [ ] metadataset
- [ ] mimiciii
- [ ] mimii
- [x] mnist
- [ ] modelnet40
- [ ] msd
- [ ] multiobjectdatasets
- [ ] naturalquestions
- [ ] nlvr2
- [ ] nuscenes
- [ ] nwm
- [ ] oc20
- [ ] omniglot
- [ ] open_images
- [ ] parlai
- [ ] partnet
- [ ] personachat
- [ ] perturbseq
- [ ] places365
- [ ] playing_for_data
- [ ] qmnist
- [ ] quickdraw
- [ ] robotcar
- [ ] shapenet
- [ ] songlyrics
- [x] stl10
- [ ] svhn
- [ ] tensorflow
- [ ] timit
- [ ] tinyimagenet
- [x] torchvision
- [ ] toyadmos
- [ ] twitter
- [ ] ubuntu
- [ ] waymoperception
- [ ] wikitext
