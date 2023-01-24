# Script to source from inside the terminal of vscode, to extract the SLURM-related environment
# variables from the current job.
#
# Usage:
#   source ./set_slurm_env_vars.sh
tmp_file="./_set_slurm_env_vars.sh"
srun --pty /bin/bash -c "export | grep SLURM > $tmp_file"
chmod +x $tmp_file
source $tmp_file
rm $tmp_file
