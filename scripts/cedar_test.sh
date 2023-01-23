#!/usr/bin/bash
set -o errexit

# TODO: On `cedar`, it's not allowed to launch jobs from the `$HOME` directory.
# Assumes that the entire repo was properly copied to the $SCRATCH directory first.
cd $SCRATCH/mila_datamodules
salloc --time=3:0:0 --account=rrg-bengioy-ad_gpu --gres=gpu:1 --mem=12G --cpus-per-task=4
# --- On compute node with internet access ---
echo "inside job with id $SLURM_JOBID"

# Create the virtualenv from scratch.
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Note: No need to cd to $SCRATCH/mila_datamodules, we should already be there.
pip install -e .[no_ffcv]
pytest -x -v -n 4
