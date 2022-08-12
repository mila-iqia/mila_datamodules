# mila_datamodules
Efficient Datamodules Customized for the Mila cluster

This currently includes the following datamodules:
- ImagenetDataModule: Efficiently copies the dataset to SLURM_TMPDIR, uses right # of workers 
- ImagenetFfcvDataModule: Same as above, plus uses FFCV to speed up the train dataloader by >10x

If you have some recipe for creating train / val / tests dataloaders for some dataset in your
particular field, please feel free to submit a PR! :)

## Installation:

At the time of writing, this currently has ffcv as a core dependency.
Installing FFCV can be a bit of a pain at the moment. But we're working on it.
FFCV will probably be made into an optional dependency eventually, unless we can somehow make it
super easy to install. 

For now, your best bet is to use conda with the provided env.yaml:

```console
$ module load miniconda/3
$ conda env create -n ffcv -f env.yaml
$ conda activate ffcv 
$ pip install git+https://www.github.com/lebrice/mila_datamodules
```


## Benchmarking

| Pure for loops (200 batches) | time           |
| ---------------------------- | -------------- |
| PyTorch                      | 0:00:59.851324 |
| FFCV                         | 0:00:17.056751 |

| Training on 50 batches     | time           |
| -------------------------- | -------------- |
| Manual loop (FFCV)         | 0:00:47.665201 |
| Manual loop (Pytorch)      | 0:00:32.682359 |
| PL + DataLoaders           | 0:00:33.111235 |
| PL + FFCV                  | 0:03:05.346748 |
| PL + Obfuscated DataLoader | 0:00:31.623927 |
