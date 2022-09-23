from .cluster import Cluster

# TODO: Make this CURRENT_CLUSTER equal to `None`` in the case where we aren't on a SLURM cluster,
# and make sure that everything exported by this package reverts back to the exact class / function
# from the source package.
CURRENT_CLUSTER = Cluster.current()

SLURM_TMPDIR = CURRENT_CLUSTER.slurm_tmpdir
SCRATCH = CURRENT_CLUSTER.scratch
