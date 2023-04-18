from __future__ import annotations

import textwrap

from mila_datamodules.clusters.cluster import Cluster

repo_host = "mila-iqia"


def get_github_issue_url(dataset_name: str, cluster_name: str):
    return (
        f"https://github.com/{repo_host}/mila_datamodules/issues/new?"
        f"labels={cluster_name}&template=feature_request.md&"
        f"title=Feature%20request:%20{dataset_name}%20on%20{cluster_name}"
    )


# TODO: Make some error classes with nicely formatted messages with GitHub URLS.


class DatasetNotFoundOnClusterError(NotImplementedError):
    def __init__(
        self, dataset: type, cluster: Cluster | None = Cluster.current(), message: str = ""
    ) -> None:
        message = textwrap.dedent(message)
        cluster_name = cluster.name if cluster is not None else "local"
        dataset_name = dataset.__name__

        github_issue_url = get_github_issue_url(dataset_name, cluster_name)
        message = message or textwrap.dedent(
            f"""\
            No known location for dataset {dataset_name} ({dataset}) on the {cluster_name} cluster!
            If you do know where it can be found on {cluster_name}, 🙏 please 🙏 make an issue on
            our GitHub repository at
            {github_issue_url}
            """
        )
        super().__init__(message)


class UnsupportedDatasetError(NotImplementedError):
    def __init__(
        self, dataset: type, cluster: Cluster | None = Cluster.current(), message: str = ""
    ) -> None:
        message = textwrap.dedent(message)
        cluster_name = cluster.name if cluster is not None else "local"
        dataset_name = dataset.__name__

        github_issue_url = get_github_issue_url(dataset_name, cluster_name)
        message = (
            message or f"We don't know which files are required to load dataset {dataset_name}.\n"
        ) + (
            f"🙏 please 🙏 consider making an issue on our GitHub repository at \n"
            f"{github_issue_url}"
        )
        super().__init__(message)


class NotOnSlurmClusterError(RuntimeError):
    ...
