#!/usr/bin/bash
set -o errexit
# set -o xtrace

# NOTE: Now assuming that the entire repo was already copied over properly.
cd mila_datamodules

## ---- With internet access ----
salloc --time=0-1:00:00 --partition=unkillable --gres=gpu:1 --mem=12G --cpus-per-task=4
echo "inside job with id $SLURM_JOBID"

# NOTE: On the Mila cluster, the compute nodes *do* have internet access!
# TODO: Use a conda-based workflow on the Mila cluster.

# module load python/3.9
module load anaconda/3


# # Taken from https://stackoverflow.com/questions/70597896/check-if-conda-env-exists-and-create-if-not-in-bash
# if { conda env list | grep 'mila_datamodules/env'; } >/dev/null 2>&1; then
#     conda activate $HOME/mila_datamodules/env
# else
#     conda create -p $HOME/mila_datamodules/env -y python=3.9
#     conda activate $HOME/mila_datamodules/env
# fi

# NOTE: Actually doing this for now, since the $HOME filesystem is getting HAMMERED right now.
conda create -p $SLURM_TMPDIR/mila_datamodules/env -y python=3.9
conda activate $SLURM_TMPDIR/mila_datamodules/env

# NOTE: Supposedly the "right" way to install ffcv.
# TODO: Getting weird errors atm when trying to install FFCV. It's really a pain in the ass.
# conda install cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision torchaudio cudatoolkit=11.3 numba -c pytorch -c conda-forge
# pip install -e .[all]

pip install -e .[no_ffcv]

pytest -v -x -n 4
