from __future__ import annotations

import argparse
import logging
from logging import getLogger as get_logger
from typing import Callable

import rich.logging
import simple_parsing
from simple_parsing import ArgumentParser
from typing_extensions import Protocol

from mila_datamodules.clusters.env_variables import run_job_step_to_get_slurm_env_variables
from mila_datamodules.clusters.utils import (
    in_job_but_not_in_job_step_so_no_slurm_env_vars,
)

# Load the SLURM environment variables into the current environment, if we're running inside a job
# but don't have the SLURM variables set.
if in_job_but_not_in_job_step_so_no_slurm_env_vars():
    run_job_step_to_get_slurm_env_variables()

from mila_datamodules.clusters.cluster import Cluster

current_cluster = Cluster.current_or_error()
logger = get_logger(__name__)


class PreparePlugin(Protocol):
    def add_prepare_args(
        self,
        subparsers: argparse._SubParsersAction[simple_parsing.ArgumentParser],
    ):
        ...

    def prepare(
        self,
        args_dict: dict,
    ):
        ...


def add_prepare_arguments(parser: ArgumentParser):
    parser.add_argument("-v", "--verbose", action="count")
    parser.add_argument(
        "-q",
        "--quiet",
        default=False,
        action="store_true",
        help="Don't write anything to stdout/stderr.",
    )
    subparsers = parser.add_subparsers(
        title="dataset/model", description="Which dataset or model to prepare", dest="dataset"
    )

    # Get the prepare_plugin entry points:
    # - https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata
    from importlib_metadata import entry_points

    # TODO: Plugins aren't found when running `poetry run mila_datamodules prepare --help`, but
    # work fine when doing just `mila_datamodules prepare --help`.
    prepare_plugin_entry_points = entry_points(group="mila_datamodules.prepare_plugin")

    # TODO: These logger.debug calls are can't be seen because the logging level hasn't been
    # configured yet.
    plugins: list[PreparePlugin] = []
    for plugin_entry_point in prepare_plugin_entry_points:
        try:
            plugin_class = plugin_entry_point.load()
            plugin = plugin_class()
        except (ImportError, ModuleNotFoundError) as err:
            logger.debug(f"Unable to load the '{plugin_entry_point.name}' dataset plugin: {err}")
        else:
            logger.debug(f"Loaded plugin: {plugin}")
            plugins.append(plugin)

    _previous_parsers = {}
    for plugin in plugins:
        _previous_parsers.update(subparsers.choices)

        # Let the plugin add all its subparsers (datasets/model weights that can be prepared).
        plugin.add_prepare_args(subparsers)

        # Collect the newly added parsers and do the `.setdefaults` on them automatically.
        for dataset_name, dataset_parser in subparsers.choices.items():
            if dataset_name not in _previous_parsers:
                dataset_parser.set_defaults(prepare_fn=plugin.prepare)
                dataset_parser.set_defaults(dataset=dataset_name)


def quiet_logging() -> bool:
    return logging.getLogger("mila_datamodules").disabled


def prepare(args: argparse.Namespace):
    """Prepare a dataset."""
    args_dict = vars(args)

    # assert args_dict.pop("command") in [prepare, None]
    verbose: int = args_dict.pop("verbose") or 0
    quiet: bool = args_dict.pop("quiet")

    def _level(verbose: int) -> int:
        return (
            logging.DEBUG
            if verbose > 1
            else logging.INFO
            if verbose == 1
            else logging.WARNING
            if verbose == 0
            else logging.ERROR
        )

    logger = logging.getLogger("mila_datamodules")
    logger.addHandler(rich.logging.RichHandler(markup=True, tracebacks_width=100))
    logger.setLevel(_level(verbose + 1))
    logger.disabled = quiet

    hf_logger = logging.getLogger("datasets")
    hf_logger.setLevel(_level(verbose - 2))
    hf_logger.addHandler(rich.logging.RichHandler(markup=True, tracebacks_width=100))
    hf_logger.disabled = quiet

    # TODO: Dispatch what to do with `args` (and the output of the function) in a smarter way,
    # based on which module was selected.
    if "prepare_fn" not in args_dict or not callable(args_dict["prepare_fn"]):
        raise RuntimeError(
            "The parsing function isn't configured correctly. It should call "
            "parser.set_defaults(prepare_fn=some_callable) where "
            "some_callable: Callable[[dict], Any]"
        )
    prepare_fn: Callable[[dict], None] = args_dict.pop("prepare_fn")
    prepare_fn(args_dict)


def get_env_variables_to_use():
    """IDEA: Output only the environment variables that need to be set for the current job."""
