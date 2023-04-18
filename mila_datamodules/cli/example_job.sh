#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=a100:2
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --job-name=llm_training
#SBATCH --output=logs/slurm-%j.out

set -e  # exit on error.

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module --quiet purge

# 'setup_env.sh' should be called before launching the job to create the
# environment and install packages only once in an environment where internet is
# accessible
source setup_env.sh

set -x  # print commands.


## ------------------------------
## Torchvision example (ImageNet)
## ------------------------------


# ---------------------------------
# Preparing the data (on each node)
# ---------------------------------


# Option 1: We ask the user to run this once per node with `srun` themselves:
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_ON_NODE bash -c "mila_datamodules prepare imagenet"

# Option 2: We do the `srun`s ourselves (with all the right flags to use all the combined resources
# of all tasks on each node of the job). This would probably involve spawning some subprocesses.
mila_datamodules prepare imagenet

# ---------------
# Running the job
# ---------------

# Complication: $SLURM_TMPDIR is different on each node, so it needs to be lazily evaluation with
# bash -c

# Option 1: Lazily evaluate $SLURM_TMPDIR with a path that is known in advance (or output from the
# srun above?)
srun --output=logs/slurm-%j_%t.out \
    bash -c "python main.py --data_dir=\$SLURM_TMPDIR/data/imagenet"

# Option 2: Add a command that returns the location to read from (e.g. $SLURM_TMPDIR/data/imagenet in this case)
srun --output=logs/slurm-%j_%t.out \
    bash -c "python main.py --data_dir=$(mila_datamodules get_data_dir imagenet)"

## ----------------------------------
## HuggingFace example (wikitext-103)
## ----------------------------------


# ---------------------------------
# Preparing the data (on each node)
# ---------------------------------


# Option 1: same as before, but we need to pass the dataset name ("path") and the split ("name").
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_ON_NODE bash -c "mila_datamodules prepare wikitext wikitext-103-v1"

# Option 2: same as before, we could have a command that does the sruns internally:
mila_datamodules prepare wikitext wikitext-103-v1

# ---------------
# Running the job
# ---------------

# Complication: HF requires setting some environment variables.
# Complication #2: Compute nodes on ComputeCanada clusters don't have access to the internet.

# Option 1: Ask the user to do it themselves.
srun --output=logs/slurm-%j_%t.out \
    bash -c "
    HF_HOME=$SCRATCH/cache/huggingface \
    HF_DATASETS_CACHE=\$SLURM_TMPDIR/cache/huggingface/datasets \
    HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub \
    python main.py"

# Option 2: Add a command to output the environment variables that need to be set inside the job.
srun --output=logs/slurm-%j_%t.out \
    bash -c "$(mila_datamodules export_env_variables wikitext wikitext-103-v1) python main.py"

# Other complication: We should add an rsync at the end of the jobs to copy the new preprocessed
# dataset files over to $SCRATCH so they aren't re-preprocessed on the next job.
# TODO: After the job is done, use an rsync to copy the new preprocessed datasets to $SCRATCH.
