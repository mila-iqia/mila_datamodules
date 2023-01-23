#!/usr/bin/bash
set -o errexit
# set -o xtrace

## ---- On login node with internet access ----
# NOTE: assuming that the entire repo was already copied over properly.
cd $HOME/mila_datamodules

module load python/3.9
if [ -d "env" ]
then
    source env/bin/activate
else
    virtualenv --no-download env
    source env/bin/activate
    pip install --no-index --upgrade pip
fi
# note: pip installing the packages on the login node, because the compute nodes don't have internet
pip install -e .[all]
pytest -v -n 4 --collect-only

salloc --time=3:0:0 --account=rrg-bengioy-ad_gpu --gres=gpu:1 --mem=12G --cpus-per-task=4
## ---- On compute node without internet access ----
echo "inside job with id $SLURM_JOBID"
cd ~/mila_datamodules
source env/bin/activate
pytest -v -x -n 4
