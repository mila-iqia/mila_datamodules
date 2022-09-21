## Imagenet

- ImagenetDataModule: Efficiently copies the dataset to SLURM_TMPDIR, uses right # of workers
- ImagenetFfcvDataModule: Same as above, plus uses FFCV to speed up the train dataloader by >10x

If you have some recipe for creating train / val / tests dataloaders for some dataset in your
particular field, please feel free to submit a PR! :)

## Benchmarking Imagenet FFCV DataModule

# Benchmarking ImageNet (ffcv vs PyTorch vs PyTorch-Lightning)

smaller res: FFCV can load the images at a progressively higher resolution, which makes it a LOT
faster than PyTorch. This is useful to train ResNets or other architectures that don't depend on the image dimensions being exactly fixed.

## Pure for loops over 200 batches

No training, just a for loop over the train dataloader.

| Setup              | Time           |
| ------------------ | -------------- |
| PyTorch            | 0:00:59.851324 |
| FFCV               | 0:00:17.056751 |
| FFCV (smaller res) | 0:00:11.235357 |

## Training on 50 batches:

Obfuscated dataloaders: Trying to hide the fact that we're using regular PyTorch dataloaders by wrapping them in useless generator functions. This was used to check whether PL makes a distinction between receiving DataLoaders or arbitrary iterables. (it doesn't, which is good!)

| Setup                           | Time           |
| ------------------------------- | -------------- |
| Manual loop (Pytorch)           | 0:00:32.682359 |
| Manual loop (FFCV)              | 0:00:47.665201 |
| Manual loop (FFCV, smaller res) | 0:00:12.021310 |
| PL + DataLoaders                | 0:00:33.111235 |
| PL + FFCV                       | 0:00:54.021916 |
| PL + Obfuscated DataLoaders     | 0:00:31.623927 |
| PL + FFCV (smaller res)         | 0:00:33.258705 |

\*: [See this GitHub issue](https://github.com/Lightning-AI/lightning/issues/14189)
