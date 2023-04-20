#!/usr/bin/bash
set -o errexit

# NOTE: On `cedar`, it's not allowed to launch jobs from the `$HOME` directory.
# Assumes that the entire repo was properly copied to the $SCRATCH directory first.
cd $SCRATCH/mila_datamodules
salloc --time=0-1:00:00 --account=rrg-bengioy-ad --mem=12G --cpus-per-task=4
# salloc --time=0-1:00:00 --account=rrg-bengioy-ad_gpu --gres=gpu:1 --mem=12G --cpus-per-task=4
# --- On compute node with internet access ---
echo "inside job with id $SLURM_JOBID"

# Create the virtualenv from scratch.
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install poetry

# Note: No need to cd to $SCRATCH/mila_datamodules, we should already be there.
poetry install --with test
pytest -x -v -n 4
