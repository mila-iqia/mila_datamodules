#!/usr/bin/bash

# IDEA: Runs all the tests of each cluster.

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


# For Beluga there are steps that need to be done before running the job, since it doesn't
# have internet access on the compute nodes (clone / pull the code from GitHub, activate the
# virtualenv, install the packages, etc.


function run_in_beluga {
    copy_repo_folder_to beluga
    ssh beluga 'bash -s' < scripts/beluga_test.sh
    # NOTE: since we're also transferring the script over, might as well just run it, instead of
    # piping it through SSH.
    # ssh beluga bash scripts/beluga_test.sh
}


function run_in_cedar {
    copy_repo_folder_to cedar scratch/mila_datamodules
    ssh cedar 'bash -s' < scripts/cedar_test.sh
}


function run_in_graham {
    copy_repo_folder_to graham
    ssh graham 'bash -s' < scripts/cedar_test.sh
}


# TODO: Figure out the proper way to redirect STDOUT and STDERR, and run these in parallel.
run_in_mila
# mkdir -p logs
# run_in_mila > logs/mila.log 2>&1 &
# run_in_cedar > logs/cedar.log 2>&1 &
# run_in_graham > logs/graham.log 2>&1 &
# run_in_beluga > logs/beluga.log 2>&1 &
# wait
echo "All tests finished."
# python merge_cluster_logs mila.log cedar.log --output=merged_logs.out
