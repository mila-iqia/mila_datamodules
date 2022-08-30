from .cluster_enum import ClusterType

CURRENT_CLUSTER = ClusterType.current()
SLURM_TMPDIR = CURRENT_CLUSTER.slurm_tmpdir
SCRATCH = CURRENT_CLUSTER.scratch
TORCHVISION_DIR = CURRENT_CLUSTER.torchvision_dir
