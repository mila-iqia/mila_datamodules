import os
import socket

from .cluster_enum import ClusterType

current = ClusterType.current()
if current == ClusterType.MILA:
    from .mila import adapt_dataset
