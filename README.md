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

## Supported Datasets

| Symbol | Meaning                                                      |
| ------ | ------------------------------------------------------------ |
| ✅      | Supported, tested.                                           |
| ✓      | Supported, not tested.                                       |
| TODO   | Dataset is available on the Cluster, but isn't supported yet |
| ?      | Don't know if the dataset is available on that cluster       |
| ❌      | Dataset isn't available on that cluster                      |

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

## TODOs / Roadmap / ideas

- [x] Add all the datamodules from pl_bolts for the Mila Cluster
- [x] Add all the datasets from torchvision for the Mila Cluster
- [ ] Add datamodules for the existing datasets of the Mila cluster at https://datasets.server.mila.quebec/
- [ ] Figure out which datasets to tackle first!
- [ ] Figure out how to instantiate each dataset in code (@satyaog ?)
- [ ] Implement a datamodule for each dataset
- [ ] Do the same for the CC clusters, one at a time!

<!-- DATASET SUPPORT TABLE START -->

| Dataset                                                             | mila | beluga | cedar | graham | narval |
| ------------------------------------------------------------------- | ---- | ------ | ----- | ------ | ------ |
| LDC/ontonotes                                                       | TODO | ?      | ?     | ?      | ?      |
| ami                                                                 | TODO | ?      | ?     | ?      | ?      |
| ami.var/ami_zip                                                     | TODO | ?      | ?     | ?      | ?      |
| ava-kinetics                                                        | TODO | ?      | ?     | ?      | ?      |
| c4                                                                  | TODO | ?      | ?     | ?      | ?      |
| caltech101                                                          | TODO | ?      | ?     | ?      | ?      |
| caltech101.var/caltech101_torchvision                               | TODO | ?      | ?     | ?      | ?      |
| caltech256                                                          | TODO | ?      | ?     | ?      | ?      |
| caltech256.var/caltech256_torchvision                               | TODO | ?      | ?     | ?      | ?      |
| celeba                                                              | TODO | ?      | ?     | ?      | ?      |
| celeba.var/celeba_torchvision                                       | TODO | ?      | ?     | ?      | ?      |
| cifar10                                                             | ✅    | TODO   | ?     | ?      | ?      |
| cifar10.var/cifar10_torchvision                                     | TODO | ?      | ?     | ?      | ?      |
| cifar100                                                            | ✅    | TODO   | ?     | ?      | ?      |
| cifar100.var/cifar100_torchvision                                   | TODO | ?      | ?     | ?      | ?      |
| cityscapes                                                          | ✅    | TODO   | ?     | ?      | ?      |
| cityscapes.var/cityscapes_torchvision                               | TODO | ?      | ?     | ?      | ?      |
| climatenet                                                          | TODO | ?      | ?     | ?      | ?      |
| coco                                                                | ✅    | TODO   | ?     | ?      | ?      |
| coco.var/coco_bcachefs                                              | TODO | ?      | ?     | ?      | ?      |
| coco.var/coco_torchvision                                           | TODO | ?      | ?     | ?      | ?      |
| commonvoice                                                         | TODO | ?      | ?     | ?      | ?      |
| conceptualcaptions                                                  | TODO | ?      | ?     | ?      | ?      |
| convai2                                                             | TODO | TODO   | ?     | ?      | ?      |
| convai2.var/convai2_parlai                                          | TODO | ?      | ?     | ?      | ?      |
| covid-19/cord-19                                                    | TODO | TODO   | ?     | ?      | ?      |
| covid-19/cord-19.var/cord-19_extract                                | TODO | ?      | ?     | ?      | ?      |
| covid-19/ecdc_covid-19                                              | TODO | TODO   | ?     | ?      | ?      |
| cub200                                                              | TODO | ?      | ?     | ?      | ?      |
| dcase2020                                                           | TODO | ?      | ?     | ?      | ?      |
| dcase2020.var/dcase2020_extract                                     | TODO | ?      | ?     | ?      | ?      |
| describable-textures                                                | TODO | ?      | ?     | ?      | ?      |
| dns-challenge                                                       | TODO | ?      | ?     | ?      | ?      |
| dns-challenge.var/dns-challenge_extract                             | TODO | ?      | ?     | ?      | ?      |
| domainnet                                                           | TODO | ?      | ?     | ?      | ?      |
| dtd                                                                 | TODO | ?      | ?     | ?      | ?      |
| fashionmnist                                                        | ✅    | ?      | ?     | ?      | ?      |
| fashionmnist.var/fashionmnist_torchvision                           | TODO | ?      | ?     | ?      | ?      |
| ffhq                                                                | TODO | ?      | ?     | ?      | ?      |
| fgvcaircraft                                                        | TODO | ?      | ?     | ?      | ?      |
| fgvcxfungi                                                          | TODO | ?      | ?     | ?      | ?      |
| flowers102                                                          | TODO | ?      | ?     | ?      | ?      |
| gaia                                                                | TODO | ?      | ?     | ?      | ?      |
| geolifeclef                                                         | TODO | ?      | ?     | ?      | ?      |
| gtsrb                                                               | TODO | ?      | ?     | ?      | ?      |
| hotels50K                                                           | TODO | ?      | ?     | ?      | ?      |
| hotels50K.var/hotels50K_extract                                     | TODO | ?      | ?     | ?      | ?      |
| icentia11k                                                          | TODO | TODO   | ?     | ?      | ?      |
| imagenet                                                            | ✅    | TODO   | ?     | ?      | ?      |
| imagenet.var/imagenet_benzina                                       | TODO | ?      | ?     | ?      | ?      |
| imagenet.var/imagenet_hdf5                                          | TODO | ?      | ?     | ?      | ?      |
| imagenet.var/imagenet_tensorflow                                    | TODO | ?      | ?     | ?      | ?      |
| imagenet.var/imagenet_torchvision                                   | TODO | ?      | ?     | ?      | ?      |
| inat                                                                | TODO | ?      | ?     | ?      | ?      |
| inat.var/inat_torchvision                                           | TODO | ?      | ?     | ?      | ?      |
| kitti                                                               | TODO | ?      | ?     | ?      | ?      |
| kmnist                                                              | TODO | ?      | ?     | ?      | ?      |
| kmnist.var/kmnist_torchvision                                       | TODO | ?      | ?     | ?      | ?      |
| librispeech                                                         | TODO | ?      | ?     | ?      | ?      |
| librispeech.var/librispeech_extract                                 | TODO | ?      | ?     | ?      | ?      |
| lincs_l1000/lincs_l1000_phase_i                                     | TODO | TODO   | ?     | ?      | ?      |
| lincs_l1000/lincs_l1000_phase_ii                                    | TODO | TODO   | ?     | ?      | ?      |
| lincs_l1000/lincs_l1000_phase_ii.var/lincs_l1000_phase_ii_extract   | TODO | ?      | ?     | ?      | ?      |
| metadataset                                                         | TODO | ?      | ?     | ?      | ?      |
| mimiciii                                                            | TODO | ?      | ?     | ?      | ?      |
| mimii                                                               | TODO | TODO   | ?     | ?      | ?      |
| mimii.var/mimii_extract                                             | TODO | ?      | ?     | ?      | ?      |
| mnist                                                               | ✅    | TODO   | ?     | ?      | ?      |
| mnist.var/mnist_torchvision                                         | TODO | ?      | ?     | ?      | ?      |
| modelnet40                                                          | TODO | ?      | ?     | ?      | ?      |
| msd                                                                 | TODO | ?      | ?     | ?      | ?      |
| msd.var/msd_top50tags                                               | TODO | ?      | ?     | ?      | ?      |
| multiobjectdatasets                                                 | TODO | ?      | ?     | ?      | ?      |
| naturalquestions                                                    | TODO | ?      | ?     | ?      | ?      |
| nlvr2                                                               | TODO | ?      | ?     | ?      | ?      |
| nuscenes                                                            | TODO | ?      | ?     | ?      | ?      |
| nwm                                                                 | TODO | TODO   | ?     | ?      | ?      |
| oc20                                                                | TODO | ?      | ?     | ?      | ?      |
| omniglot                                                            | TODO | ?      | ?     | ?      | ?      |
| open_images                                                         | TODO | ?      | ?     | ?      | ?      |
| openwebtext                                                         | TODO | ?      | ?     | ?      | ?      |
| partnet                                                             | TODO | ?      | ?     | ?      | ?      |
| personachat                                                         | TODO | TODO   | ?     | ?      | ?      |
| personachat.var/personachat_parlai                                  | TODO | ?      | ?     | ?      | ?      |
| perturbseq/perturbseq_2016                                          | TODO | TODO   | ?     | ?      | ?      |
| perturbseq/perturbseq_2016.var/perturbseq_2016_extract              | TODO | ?      | ?     | ?      | ?      |
| perturbseq/perturbseq_2019                                          | TODO | TODO   | ?     | ?      | ?      |
| perturbseq/perturbseq_2019.var/perturbseq_2019_extract              | TODO | ?      | ?     | ?      | ?      |
| places365                                                           | TODO | ?      | ?     | ?      | ?      |
| places365.var/places365_challenge                                   | TODO | ?      | ?     | ?      | ?      |
| places365.var/places365_torchvision                                 | TODO | ?      | ?     | ?      | ?      |
| places365.var/places365_torchvision_256                             | TODO | ?      | ?     | ?      | ?      |
| playing_for_data                                                    | TODO | ?      | ?     | ?      | ?      |
| qmnist                                                              | TODO | ?      | ?     | ?      | ?      |
| qmnist.var/qmnist_torchvision                                       | TODO | ?      | ?     | ?      | ?      |
| quickdraw                                                           | TODO | ?      | ?     | ?      | ?      |
| restricted/2d3dsemantics_users/2d3dsemantics                        | TODO | ?      | ?     | ?      | ?      |
| restricted/chime5_users/chime5                                      | TODO | ?      | ?     | ?      | ?      |
| restricted/icmlexvo2022_users/icmlexvo2022                          | TODO | ?      | ?     | ?      | ?      |
| restricted/icmlexvo2022_users/icmlexvo2022.var/icmlexvo2022_extract | TODO | ?      | ?     | ?      | ?      |
| restricted/inat_users/inat                                          | TODO | ?      | ?     | ?      | ?      |
| restricted/mimiciii_users/mimiciii                                  | TODO | ?      | ?     | ?      | ?      |
| restricted/mimiciii_users/mimiciii.var/mimiciii_postgres            | TODO | ?      | ?     | ?      | ?      |
| restricted/objects365_users/objects365                              | TODO | ?      | ?     | ?      | ?      |
| restricted/recursionmilacausal_users/recursionmilacausal            | TODO | ?      | ?     | ?      | ?      |
| restricted/scannet_users/scannet                                    | TODO | TODO   | ?     | ?      | ?      |
| restricted/scannet_users/scannet.var/scannet_extract                | TODO | ?      | ?     | ?      | ?      |
| restricted/thirddihard_users/third_dihard_challenge                 | TODO | ?      | ?     | ?      | ?      |
| restricted/voxceleb_users/voxceleb/voxceleb1                        | TODO | TODO   | ?     | ?      | ?      |
| restricted/voxceleb_users/voxceleb/voxceleb2                        | TODO | TODO   | ?     | ?      | ?      |
| robotcar                                                            | TODO | ?      | ?     | ?      | ?      |
| shapenet                                                            | TODO | ?      | ?     | ?      | ?      |
| songlyrics/songlyrics                                               | TODO | TODO   | ?     | ?      | ?      |
| songlyrics/songlyrics_artimous                                      | TODO | TODO   | ?     | ?      | ?      |
| songlyrics/songlyrics_gyani95                                       | TODO | TODO   | ?     | ?      | ?      |
| songlyrics/songlyrics_mousehead                                     | TODO | TODO   | ?     | ?      | ?      |
| stl10                                                               | ✅    | ?      | ?     | ?      | ?      |
| stl10.var/stl10_torchvision                                         | TODO | ?      | ?     | ?      | ?      |
| svhn                                                                | TODO | ?      | ?     | ?      | ?      |
| svhn.var/svhn_torchvision                                           | TODO | ?      | ?     | ?      | ?      |
| timit                                                               | TODO | TODO   | ?     | ?      | ?      |
| tinyimagenet                                                        | TODO | ?      | ?     | ?      | ?      |
| tinyimagenet.var/tinyimagenet_benzina                               | TODO | ?      | ?     | ?      | ?      |
| toyadmos                                                            | TODO | TODO   | ?     | ?      | ?      |
| toyadmos.var/toyadmos_extract                                       | TODO | ?      | ?     | ?      | ?      |
| twitter                                                             | TODO | ?      | ?     | ?      | ?      |
| twitter.var/twitter_bpe                                             | TODO | ?      | ?     | ?      | ?      |
| ubuntu                                                              | TODO | ?      | ?     | ?      | ?      |
| ubuntu.var/ubuntu_bpe                                               | TODO | ?      | ?     | ?      | ?      |
| ubuntu.var/ubuntu_v1                                                | TODO | ?      | ?     | ?      | ?      |
| waymoperception                                                     | TODO | ?      | ?     | ?      | ?      |
| webtext                                                             | TODO | ?      | ?     | ?      | ?      |
| wikitext                                                            | TODO | ?      | ?     | ?      | ?      |

<!-- DATASET SUPPORT TABLE END -->
