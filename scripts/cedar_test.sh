#!/usr/bin/bash
set -o errexit
# set -o xtrace

## ---- With internet access ----
module load python/3.9

# NOTE: Now assuming that the entire repo was already copied over properly.
cd mila_datamodules
# if [ -d "mila_datamodules" ]
# then
#     cd mila_datamodules
#     git checkout ${BRANCH:-master}
#     git pull
# else
#     git clone https://github.com/mila_iqia/mila_datamodules -b ${BRANCH:-master}
#     cd mila_datamodules
# fi

echo $PWD

if [ -d "env" ]
then
    source env/bin/activate
else
    virtualenv --no-download env
    source env/bin/activate
    pip install --no-index --upgrade pip
fi
# note: pip installing the packages on the login node, because the compute nodes don't have
# internet access.
pip install -e .[all]
pytest -v -n 4 --collect-only


## ---- Without internet access ----


salloc --time=3:0:0 --account=rrg-bengioy-ad_gpu --gres=gpu:1 --mem=12G --cpus-per-task=4
echo "inside job with id $SLURM_JOBID"
cd ~/mila_datamodules
source env/bin/activate
pytest -v -x -n 4
