from .cluster import Cluster
from .utils import get_scratch_dir, get_slurm_tmpdir

# TODO: Make this CURRENT_CLUSTER equal to `None`` in the case where we aren't on a SLURM cluster,
# and make sure that everything exported by this package reverts back to the exact class / function
# from the source package.
CURRENT_CLUSTER = Cluster.current()

__all__ = [
    "get_scratch_dir",
    "get_slurm_tmpdir",
    "Cluster",
    "CURRENT_CLUSTER",
]
