#!/usr/bin/bash

# IDEA: Runs all the tests of each cluster.

# BRANCH=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)

function copy_repo_folder_to {
    # Adapted from https://stackoverflow.com/questions/13713101/rsync-exclude-according-to-gitignore-hgignore-svnignore-like-filter-c
    local cluster=$1
    local src="."
    local dest=${2:-"mila_datamodules"}
    # Other interesting solution, using git ls-files:
    # BUG: Seems to make the program hang?
    # rsync -ah --delete --verbose --include .git \
    #     --exclude-from=<(git -C $src ls-files --exclude-standard -oi --directory) \
    #     $src $cluster:mila_datamodules

    rsync -h --verbose --recursive -a $src $cluster:$dest \
        --include='**.gitignore' --exclude='/.git' --filter=':- .gitignore' --delete-after
}


function run_in_mila {
    copy_repo_folder_to mila
    ssh mila 'bash -s' < scripts/mila_test.sh
}


# For Beluga / CC, there are steps that need to be done before running the job, since it doesn't
# have internet access on the compute nodes (clone / pull the code from GitHub, activate the
# virtualenv, install the packages, etc.


function run_in_beluga {
    copy_repo_folder_to beluga
    ssh beluga 'bash -s' < scripts/beluga_test.sh
}


function run_in_cedar {
    copy_repo_folder_to cedar scratch/mila_datamodules
    ssh cedar 'bash -s' < scripts/cedar_test.sh
}


# run_in_mila
# run_in_cedar
# run_in_beluga
# wait
# run_on_mila_cluster
# run_on_beluga
