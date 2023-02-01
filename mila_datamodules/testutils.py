from __future__ import annotations

from typing import Sequence

import pytest
from torch.utils.data import Dataset

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import on_slurm_cluster
from mila_datamodules.registry import is_stored_on_cluster


def skip_if_not_stored_on_current_cluster(dataset: type[Dataset]):
    return pytest.mark.skipif(
        not is_stored_on_cluster(dataset, CURRENT_CLUSTER),
        reason=f"Dataset isn't stored on {CURRENT_CLUSTER} cluster.",
    )


def xfail_if_not_stored_on_current_cluster(dataset: type[Dataset]):
    return pytest.mark.xfail(
        condition=not is_stored_on_cluster(dataset, CURRENT_CLUSTER),
        reason=f"Dataset isn't stored on {CURRENT_CLUSTER} cluster.",
    )


def run_only_if_not_stored_on_current_cluster(dataset: type[Dataset]):
    return pytest.mark.skipif(
        is_stored_on_cluster(dataset, CURRENT_CLUSTER),
        reason=f"Dataset is stored on {CURRENT_CLUSTER} cluster.",
    )


def only_runs_on_clusters(*cluster: Cluster):
    """Only runs the test or test param when we're on that cluster or one of these clusters."""
    clusters = list(cluster)
    if clusters:
        condition = (Cluster.current() not in clusters,)
        reason = f"Test only runs on {'|'.join(c.name for c in clusters)} clusters"
    else:
        condition = Cluster.current() is not None
        reason = "Test only runs on SLURM clusters"
    return pytest.mark.skipif(reason=reason, condition=condition)


def param_only_runs_on_clusters(*params, clusters: Sequence[Cluster] = ()):
    """Only runs the test with this parameter when we're on one of the given SLURM clusters."""
    return pytest.param(*params, marks=only_runs_on_clusters(*clusters))


def param_only_runs_on_cluster(*params, cluster: Cluster):
    return param_only_runs_on_clusters(*params, clusters=[cluster])


def param_only_runs_on_slurm_clusters(*params):
    """Only runs the test param when we're on a SLURM cluster."""
    return pytest.param(*params, marks=only_runs_on_clusters())


def only_runs_when_not_on_a_slurm_cluster():
    """Only runs this test if we're not on a SLURM cluster."""
    return pytest.mark.skipif(
        on_slurm_cluster(),
        reason="This test only runs outside of SLURM clusters.",
    )


def param_only_runs_outside_slurm_cluster(*params):
    """Only runs this test parameter if we're not on a SLURM cluster."""
    return pytest.param(*params, marks=only_runs_when_not_on_a_slurm_cluster())
