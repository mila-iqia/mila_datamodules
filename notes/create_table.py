"""Small script to figure out which datasets are common on the different clusters."""
import contextlib
import io
from collections import defaultdict
from pathlib import Path
from typing import Literal

this_folder = Path(__file__).parent
ClusterName = Literal["mila", "beluga", "cedar", "graham", "narval"]

clusters: list[ClusterName] = [
    "mila",
    "beluga",
    "cedar",
    "graham",
    "narval",
]

supported_untested_datasets: dict[ClusterName, list[str]] = defaultdict(list)
supported_untested_datasets["mila"] = [
    "coco",
]


supported_tested_datasets: dict[ClusterName, list[str]] = defaultdict(list)
supported_tested_datasets["mila"] = [
    "imagenet",
    "cifar10",
    "cifar100",
    "stl10",
    "mnist",
    "fashionmnist",
    "emnist",
    "binaryemnist",
    "binarymnist",
    "cityscapes",
]
# NOTE: Just to get rid of some 'duplicates' in the table.
supported_tested_datasets["mila"].extend(
    [f"{dataset}.var/{dataset}_torchvision" for dataset in supported_tested_datasets["mila"]]
)

unavailable_datasets: dict[ClusterName, list[str]] = defaultdict(list)


def create_dataset_support_table() -> str:
    """Generate the table that goes at the bottom of the README."""
    dataset_to_clusters: dict[str, list[ClusterName]] = defaultdict(list)
    for cluster in clusters:
        file = this_folder / f"{cluster}.txt"
        if not file.exists():
            continue
        with open(file) as f:
            cluster_datasets = sorted({line.strip() for line in f.readlines()})

        for dataset in cluster_datasets:
            dataset_to_clusters[dataset].append(cluster)

    def _box_content(
        cluster: ClusterName, dataset: str, clusters_with_this_dataset: list[str]
    ) -> str:
        if dataset in supported_tested_datasets[cluster]:
            return " ✅ "
        if dataset in supported_untested_datasets[cluster]:
            return " ✓ "
        if dataset in unavailable_datasets[cluster]:
            return " ❌ "
        if cluster in clusters_with_this_dataset:
            return " TODO "
        return " ? "

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        print("| Dataset |" + " | ".join(clusters) + " |")
        print("| ------- |" + " | ".join("-----" for c in clusters) + " |")

        for dataset, clusters_that_have_this_dataset in sorted(dataset_to_clusters.items()):
            print(
                f"| {dataset} |"
                + "|".join(
                    _box_content(cluster, dataset, clusters_that_have_this_dataset)
                    for cluster in clusters
                )
                + "|"
            )
    buffer.seek(0)
    return buffer.read()


def update_support_table_in_readme():
    """Updates the table of supported datasets in the README."""
    table_start_flag = "<!-- DATASET SUPPORT TABLE START -->"
    table_end_flag = "<!-- DATASET SUPPORT TABLE END -->"

    readme_path = this_folder.parent / "README.md"
    with open(readme_path) as f:
        lines = f.readlines()
        lines = [line.removesuffix("\n") for line in lines]

        start_line = lines.index(table_start_flag)
        end_line = lines.index(table_end_flag)

        # Insert the new table between these two lines
        lines_before_table = lines[:start_line]
        lines_after_table = lines[end_line + 1 :]

        new_table_lines = create_dataset_support_table().splitlines()

        new_readme_lines = (
            lines_before_table
            + [table_start_flag, "\n"]
            + new_table_lines
            + ["\n", table_end_flag]
            + lines_after_table
        )
    with open(readme_path, "w") as f:
        f.writelines(line + "\n" if not line.endswith("\n") else line for line in new_readme_lines)


if __name__ == "__main__":
    update_support_table_in_readme()
