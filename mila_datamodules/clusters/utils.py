"""Set of functions for creating torchvision datasets when on the Mila cluster.

IDEA: later on, we could also add some functions for loading torchvision models from a cached
directory.
"""
from __future__ import annotations

import functools
import inspect
import os
import shutil
import socket
import subprocess
import tempfile
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import torchvision.datasets as tvd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import ParamSpec

D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = Callable[P, D]

logger = get_logger(__name__)


def on_login_node() -> bool:
    # IDEA: Detect if we're on a login node somehow.
    return socket.getfqdn().endswith(".server.mila.quebec") and "SLURM_TMPDIR" not in os.environ


def setup_slurm_env_variables(vars_to_ignore: Sequence[str] = ()) -> None:
    """Sets the slurm-related environment variables inside the current shell if they are not set.

    Executes `env | grep SLURM` inside a `srun --pty /bin/bash` sub-command (assuming that no other
    such command is being run). Then, extracts the variables from the outputs and sets them in
    `os.environ`, if not already present.

    if `vars_to_ignore` is provided, those variables are not set.
    """
    if "SLURM_CLUSTER_NAME" in os.environ:
        # SLURM-related environment variables have already been set. Ignoring.
        return
    with tempfile.NamedTemporaryFile() as temp_file:
        try:
            logger.info("Extracting SLURM environment variables... ")
            command = "srun --pty /bin/bash -c 'env | grep SLURM'"
            logger.debug(f"> {command}")
            proc = subprocess.run(
                command,
                shell=True,
                check=True,
                timeout=2,  # max 2 seconds (this is plenty enough as far as I can tell).
                stdout=temp_file,
            )
            lines = Path(temp_file.name).read_text().split()
            logger.info("done!")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Unable to extract SLURM environment variables. Check that there isn't already a "
                "`srun --pty /bin/bash` command running (there can only be one at any given time)."
            )
        else:
            # Read and copy the environment variables from the output of that command.
            for line in lines:
                key, _, value = line.partition("=")
                if key in vars_to_ignore:
                    continue
                logger.debug(f"Setting {line}")
                os.environ.setdefault(key, value)

            # TODO: Using `export` above + `source` here worked at some point, and had the benefit
            # of actually modifying the running shell's env (if I recall correctly).
            # However that might actually have been a fluke or an error on my part, because it
            # seems impossible for Python process to change the env variables in a persistent way.
            # Therefore it doesn't currently work, and I opted for just reading a dump of the env
            # vars instead.
            # temp_file_name.chmod(mode=0o755)
            # command = f"{temp_file_name}"
            # print(f"> {command}")
            # subprocess.run(
            #     command,
            #     shell=True,
            #     executable="/bin/bash",
            #     check=True,
            # )


def all_files_exist(
    required_files: Sequence[str],
    base_dir: Path,
) -> bool:
    return all((base_dir / f).exists() for f in required_files)


def replace_kwargs(dataset_type: Callable[P, D], **fixed_arguments) -> Callable[P, D]:
    """Returns a callable where the given argument values are fixed.

    NOTE: Simply using functools.partial wouldn't work, since passing one of the fixed arguments
    would raise an error.
    """
    init_signature = inspect.signature(dataset_type)

    @functools.wraps(dataset_type)
    def _wrap(*args: P.args, **kwargs: P.kwargs) -> D:
        bound_signature = init_signature.bind_partial(*args, **kwargs)
        for key, value in fixed_arguments.items():
            bound_signature.arguments[key] = value
        args = bound_signature.args  # type: ignore
        kwargs = bound_signature.kwargs  # type: ignore
        return dataset_type(*args, **kwargs)

    return _wrap


def copy_dataset_files(files_to_copy: Sequence[str], source_dir: Path, dest_dir: Path) -> None:
    assert all_files_exist(files_to_copy, base_dir=source_dir)
    for source_file in files_to_copy:
        source_path = source_dir / source_file
        destination_path = dest_dir / source_file
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {source_path} -> {destination_path}")
        if source_path.is_dir():
            # Copy the folder over.
            # TODO: Check that this doesn't overwrite existing files.
            # TODO: Test this out with symlinks?
            shutil.copytree(
                src=source_path,
                dst=destination_path,
                symlinks=False,
                dirs_exist_ok=True,
            )
        elif not destination_path.exists():
            # Copy the file over.
            shutil.copy(src=source_path, dst=destination_path, follow_symlinks=False)
