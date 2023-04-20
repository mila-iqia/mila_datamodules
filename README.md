# (WIP) Cluster-Aware Datamodules

Efficient Datamodules customized for the Mila/ComputeCanada clusters.

NOTE: This is a work in progress. Please let us know what you think, what features you'd like to see, what datasets we're forgetting, etc, using the issues tab.

## Installation:

This can be installed with `pip`:

```console
pip install "mila_datamodules @ git+https://www.github.com/mila-iqia/mila_datamodules.git"
```

## Usage:

TODO once the CLI is finalized


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

| Dataset                          | mila | beluga | cedar | graham | narval |
| -------------------------------- | ---- | ------ | ----- | ------ | ------ |
| LDC/ontonotes                    | TODO | ?      | ?     | ?      | TODO   |
| ami                              | TODO | ?      | ?     | ?      | ?      |
| ava-kinetics                     | TODO | ?      | ?     | ?      | ?      |
| c4                               | TODO | ?      | ?     | ?      | ?      |
| caltech101                       | TODO | ?      | ?     | ?      | TODO   |
| caltech256                       | TODO | ?      | ?     | ?      | TODO   |
| celeba                           | TODO | ?      | ?     | ?      | TODO   |
| cifar10                          | ✅    | TODO   | ?     | ?      | TODO   |
| cifar100                         | ✅    | TODO   | ?     | ?      | TODO   |
| cityscapes                       | ✅    | TODO   | ?     | ?      | TODO   |
| climatenet                       | TODO | ?      | ?     | ?      | ?      |
| coco                             | ✓    | TODO   | ?     | ?      | TODO   |
| commonvoice                      | TODO | ?      | ?     | ?      | TODO   |
| conceptualcaptions               | TODO | ?      | ?     | ?      | ?      |
| convai2                          | TODO | TODO   | ?     | ?      | TODO   |
| covid-19/cord-19                 | TODO | TODO   | ?     | ?      | TODO   |
| covid-19/ecdc_covid-19           | TODO | TODO   | ?     | ?      | TODO   |
| cub200                           | TODO | ?      | ?     | ?      | TODO   |
| dcase2020                        | TODO | ?      | ?     | ?      | ?      |
| describable-textures             | TODO | ?      | ?     | ?      | TODO   |
| dns-challenge                    | TODO | ?      | ?     | ?      | ?      |
| domainnet                        | TODO | ?      | ?     | ?      | TODO   |
| dtd                              | TODO | ?      | ?     | ?      | TODO   |
| fashionmnist                     | ✅    | ?      | ?     | ?      | TODO   |
| ffhq                             | TODO | ?      | ?     | ?      | TODO   |
| fgvcaircraft                     | TODO | ?      | ?     | ?      | TODO   |
| fgvcxfungi                       | TODO | ?      | ?     | ?      | TODO   |
| flowers102                       | TODO | ?      | ?     | ?      | TODO   |
| gaia                             | TODO | ?      | ?     | ?      | ?      |
| geolifeclef                      | TODO | ?      | ?     | ?      | TODO   |
| gtsrb                            | TODO | ?      | ?     | ?      | TODO   |
| hotels50K                        | TODO | ?      | ?     | ?      | TODO   |
| icentia11k                       | TODO | TODO   | ?     | ?      | ?      |
| imagenet                         | ✅    | TODO   | ?     | ?      | TODO   |
| inat                             | TODO | ?      | ?     | ?      | TODO   |
| kitti                            | TODO | ?      | ?     | ?      | ?      |
| kmnist                           | TODO | ?      | ?     | ?      | TODO   |
| librispeech                      | TODO | ?      | ?     | ?      | TODO   |
| lincs_l1000/lincs_l1000_phase_i  | TODO | TODO   | ?     | ?      | TODO   |
| lincs_l1000/lincs_l1000_phase_ii | TODO | TODO   | ?     | ?      | TODO   |
| metadataset                      | TODO | ?      | ?     | ?      | ?      |
| mimiciii                         | TODO | ?      | ?     | ?      | ?      |
| mimii                            | TODO | TODO   | ?     | ?      | TODO   |
| mnist                            | ✅    | TODO   | ?     | ?      | TODO   |
| modelnet40                       | TODO | ?      | ?     | ?      | TODO   |
| msd                              | TODO | ?      | ?     | ?      | TODO   |
| multiobjectdatasets              | TODO | ?      | ?     | ?      | ?      |
| naturalquestions                 | TODO | ?      | ?     | ?      | ?      |
| nlvr2                            | TODO | ?      | ?     | ?      | TODO   |
| nuscenes                         | TODO | ?      | ?     | ?      | TODO   |
| nwm                              | TODO | TODO   | ?     | ?      | TODO   |
| oc20                             | TODO | ?      | ?     | ?      | TODO   |
| omniglot                         | TODO | ?      | ?     | ?      | TODO   |
| open_images                      | TODO | ?      | ?     | ?      | TODO   |
| openwebtext                      | TODO | ?      | ?     | ?      | ?      |
| partnet                          | TODO | ?      | ?     | ?      | TODO   |
| personachat                      | TODO | TODO   | ?     | ?      | TODO   |
| perturbseq/perturbseq_2016       | TODO | TODO   | ?     | ?      | TODO   |
| perturbseq/perturbseq_2019       | TODO | TODO   | ?     | ?      | TODO   |
| places365                        | TODO | ?      | ?     | ?      | TODO   |
| playing_for_data                 | TODO | ?      | ?     | ?      | TODO   |
| qmnist                           | TODO | ?      | ?     | ?      | TODO   |
| quickdraw                        | TODO | ?      | ?     | ?      | ?      |
| robotcar                         | TODO | ?      | ?     | ?      | TODO   |
| shapenet                         | TODO | ?      | ?     | ?      | TODO   |
| songlyrics/songlyrics            | TODO | TODO   | ?     | ?      | TODO   |
| songlyrics/songlyrics_artimous   | TODO | TODO   | ?     | ?      | TODO   |
| songlyrics/songlyrics_gyani95    | TODO | TODO   | ?     | ?      | TODO   |
| songlyrics/songlyrics_mousehead  | TODO | TODO   | ?     | ?      | TODO   |
| stl10                            | ✅    | ?      | ?     | ?      | TODO   |
| svhn                             | TODO | ?      | ?     | ?      | TODO   |
| timit                            | TODO | TODO   | ?     | ?      | TODO   |
| tinyimagenet                     | TODO | ?      | ?     | ?      | TODO   |
| toyadmos                         | TODO | TODO   | ?     | ?      | TODO   |
| twitter                          | TODO | ?      | ?     | ?      | TODO   |
| ubuntu                           | TODO | ?      | ?     | ?      | TODO   |
| waymoperception                  | TODO | ?      | ?     | ?      | TODO   |
| webtext                          | TODO | ?      | ?     | ?      | ?      |
| wikitext                         | TODO | ?      | ?     | ?      | TODO   |

<!-- DATASET SUPPORT TABLE END -->
