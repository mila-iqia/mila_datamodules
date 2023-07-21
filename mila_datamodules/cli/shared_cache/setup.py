#!/network/weights/shared_cache/.env/bin/python
"""Sets up the user cache directory with links to pre-downloaded datasets and model weights.

This command makes it possible to use many of the datasets and model weights from HuggingFace
without having to download them yourself, by making symlinks to the files which are already
downloaded in /network/weights/shared_cache.
This can help free up space in your $HOME and $SCRATCH directories.

- To see which datasets and model weights are pre-downloaded in the shared cache, take a look at
  `/network/weights/shared_cache/populate.py`.
- To request a dataset to be dowloaded please fill in [this form.](https://forms.gle/vDVwD2rZBmYHENgZA)
- To request model weights to be downloaded, please fill in [this form.](https://forms.gle/HLeBkJBozjC3jG2D9)

NOTES:
- The user cache directory is set to $SCRATCH/cache by default.
- The shared cache directory is set to /network/weights/shared_cache by default.
- If duplicated files are found in the user cache (for example, a previous download of the WikiText
  dataset from HuggingFace) they are replaced which symlinks to the same files on the shared
  filesystem.
- Other files in the user cache directory that do not have a same-name equivalent in the shared
  cache directory are left as-is.

This command also sets the environment variables via a block in the `$HOME/.bash_aliases` file.
Libraries such as HuggingFace then look in the specified user cache for these files.
"""
from __future__ import annotations

import functools
import glob
import logging
import os
import shlex
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

import rich.logging
import tqdm
import tqdm.rich
from simple_parsing import field
from tqdm.std import TqdmExperimentalWarning

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(rich.logging.RichHandler(rich_tracebacks=True))


SCRATCH = Path(os.environ["SCRATCH"])
DEFAULT_USER_CACHE_DIR = SCRATCH / "cache"
DEFAULT_SHARED_CACHE_DIR = Path("/network/weights/shared_cache")
QUIET: bool = False

IGNORE_DIRS = ("__pycache__", ".env", ".git")
"""Don't create symlinks to files in directories in the shared cache whose name matches any of
these patterns."""

IGNORE_FILES = ("*.lock", ".file_count.txt")
"""Don't create symlinks to files in the shared cache that match any of these patterns."""

COPY_FILES = "*.py"  # copy these files instead of creating symlinks.

T = TypeVar("T")
Predicate = Callable[[T], bool]


@dataclass
class Options:
    """Options for the setup_cache command."""

    user_cache_dir: Path = DEFAULT_USER_CACHE_DIR
    """The user cache directory.

    Should probably be in $SCRATCH (not $HOME!)
    """

    shared_cache_dir: Path = DEFAULT_SHARED_CACHE_DIR
    """The path to the shared cache directory.

    This defaults to the path of the shared cache setup by the IDT team on the Mila cluster.
    """

    subdirectory: str = ""
    """Only create links for files in this subdirectory of the shared cache."""

    verbose: int = field(default=0, action="count", alias="-v")
    """Logging verbosity.

    The default logging level is `INFO`. Use -v to increase the verbosity to `DEBUG`.
    """

    quiet: bool = False
    """Disable all logging output."""


def main(argv: list[str] | None = None):
    global QUIET
    options: Options = _parse_args(argv)
    logger.setLevel(_log_level(options.verbose))
    # logging.basicConfig(
    #     level=_log_level(options.verbose),
    #     format="%(message)s",
    #     handlers=[rich.logging.RichHandler(markup=True)],
    # )
    if options.quiet:
        logger.disabled = True
        QUIET = True
    setup_cache(
        user_cache_dir=options.user_cache_dir,
        shared_cache_dir=options.shared_cache_dir,
        subdirectory=options.subdirectory,
    )


def setup_cache(
    user_cache_dir: Path = DEFAULT_USER_CACHE_DIR,
    shared_cache_dir: Path = DEFAULT_SHARED_CACHE_DIR,
    subdirectory: str = "",
    skip_modify_bash_aliases: bool = False,
) -> None:
    """Set up the user cache directory.

    1. If the user cache directory doesn't exist, creates it.
    2. Sets the optimal striping configuration for the user cache directory so that reading and
       writing large files works optimally.
    3. Removes broken symlinks in the user cache directory if they point to files in
       `shared_cache_dir` that don't exist anymore.
    4. For every file in the shared cache dir, creates a symbolic link to it in the
       user cache dir. (Replaces duplicate downloaded files with symlinks to the same file in the
       shared cache dir).
    5. Adds a block of code to ~/.bash_aliases (creating it if necessary) that sets the relevant
       environment variables so that libraries use those cache directories.
    """

    if not user_cache_dir.exists():
        user_cache_dir.mkdir(parents=True, exist_ok=False)
    if not user_cache_dir.is_dir():
        raise RuntimeError(f"cache_dir is not a directory: {user_cache_dir}")
    if not shared_cache_dir.is_dir():
        raise RuntimeError(
            f"The shared cache directory {shared_cache_dir} doesn't exist, or isn't a directory! "
        )

    set_striping_config_for_dir(user_cache_dir)

    delete_broken_symlinks_to_shared_cache(
        user_cache_dir / subdirectory, shared_cache_dir / subdirectory
    )

    create_links(user_cache_dir / subdirectory, shared_cache_dir / subdirectory)

    if not skip_modify_bash_aliases:
        bash_aliases_file = "~/.bash_aliases"
        bash_aliases_file_changed = set_environment_variables(
            user_cache_dir, bash_aliases_file=bash_aliases_file
        )
        if bash_aliases_file_changed:
            logger.warning(
                f"The {bash_aliases_file} was changed. You may need to restart your shell for the "
                f"changes to take effect.\n"
                f"To set the environment variables in the current shell, source the "
                f"{bash_aliases_file} file like so:\n"
                f"```\n"
                f"source {bash_aliases_file}\n"
                f"```\n"
            )

    if not QUIET:
        print("DONE!")


def set_striping_config_for_dir(dir: Path, num_targets: int = 4, chunksize: str = "512k"):
    """Sets up the data striping configuration for the given directory using beegfs-ctl.

    This is useful when we know that the directory will contain large files, since we can stripe
    the data in chunks across multiple targets, which will improve performance.

    NOTE: This only affects the files that are added in this directory *after* this command is ran.
    """
    # TODO: Find the command to get the striping pattern and only change it if necessary.
    logger.info(f"Setting the striping config for {dir}")
    output = subprocess.check_output(
        shlex.split(
            f"beegfs-ctl --cfgFile=/etc/beegfs/scratch.d/beegfs-client.conf --setpattern "
            f"--numtargets={num_targets} --chunksize={chunksize} {dir}"
        ),
        encoding="utf-8",
    )
    logger.info("Done setting the striping config.")

    logger.debug(output)


def delete_broken_symlinks_to_shared_cache(user_cache_dir: Path, shared_cache_dir: Path):
    """Delete all symlinks in the user cache directory that point to files that don't exist anymore
    in the shared cache directory."""
    logger.info(f"Looking for broken symlinks in {user_cache_dir}")
    for file in _enumerate_all_files_in_dir(
        user_cache_dir,
        desc=f"Looking for broken symlinks in {user_cache_dir} ...",
    ):
        if (
            file.is_symlink()
            and not file.exists()
            and file.readlink().is_relative_to(shared_cache_dir)
        ):
            logger.info(f"Removing broken symlink at {file}")
            file.unlink()


def _skip_file(path_in_shared_cache: Path) -> bool:
    return _matches_pattern(path_in_shared_cache, IGNORE_FILES)


def _skip_dir(path_in_shared_cache: Path) -> bool:
    return _matches_pattern(path_in_shared_cache, IGNORE_DIRS)


def create_links(
    user_cache_dir: Path,
    shared_cache_dir: Path,
    skip_file: Predicate[Path] = _skip_file,
    skip_dir: Predicate[Path] = _skip_dir,
):
    """Create symlinks to the shared cache directory in the user cache directory."""
    # For every file in the shared cache dir, create a symbolic link to it in the user cache dir

    # TODO: Using `shutil.copytree` with a custom `copy_fn` raises a bunch of errors at the end.
    # I'm not sure why. Using this more direct method instead. One disadvantage is that we can't
    # really use `shutil.ignore_dirs` which would be useful to ignore some directories in the
    # shared cache.
    logger.info(f"Creating symlinks in {user_cache_dir} to files in {shared_cache_dir}")
    for path_in_shared_cache in _enumerate_all_files_in_dir(
        shared_cache_dir,
        desc=f"Creating symlinks in {user_cache_dir} ...",
        skip_file=skip_file,
        skip_dir=skip_dir,
    ):
        path_in_user_cache = user_cache_dir / (path_in_shared_cache.relative_to(shared_cache_dir))
        _create_link(
            path_in_user_cache=path_in_user_cache,
            path_in_shared_cache=path_in_shared_cache,
        )


def _enumerate_all_files_in_dir(
    directory: Path,
    write_filecount_txt: bool = True,
    desc: str = "",
    skip_file: Predicate[Path] | None = None,
    skip_dir: Predicate[Path] | None = None,
) -> Iterable[Path]:
    file_iterator = _tree(directory, skip_file=skip_file, skip_dir=skip_dir)

    if tqdm is None:
        yield from file_iterator
        return

    # Using TQDM to make a nice progress bar.

    filecount: int = 0
    expected_filecount: int | None = None

    # Save a 'hint' (or true value if possible) of the number of files in the shared cache
    # directory in a file. This is just used to make a pretty progress bar, and doesn't affect
    # the functionality (If the actual filecount is higher, the progress bar just shows 101/100,
    # 102/100, etc.)
    filecount_file = directory / ".file_count.txt"
    if filecount_file.exists():
        expected_filecount = int(filecount_file.read_text())

    # Catch warning that is raised when instantiating a new `tqdm_rich` progress bar.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
        total = expected_filecount
        pbar_to_use = tqdm.rich.tqdm_rich if total is not None else tqdm.tqdm
        file_iterator = pbar_to_use(
            file_iterator,
            desc=desc or f"Iterating through all the files in {directory}",
            unit="Files",
            total=total,
            disable=QUIET,
        )

    for file in file_iterator:
        yield file
        filecount += 1

    if write_filecount_txt and filecount != expected_filecount:
        try:
            filecount_file.write_text(str(filecount))
            logger.debug(f"Updated the filecount file at {filecount_file} to {filecount}")
        except IOError:
            pass


def _create_link(
    path_in_user_cache: Path,
    path_in_shared_cache: Path,
    copy_files_patterns: str | Sequence[str] = COPY_FILES,
) -> None:
    """Create a symlink in the user cache directory to the file in the shared cache directory.

    NOTE: The relative path from user cache to the file is usually the same as the path from the
    share cache to the file, but perhaps it's best not to rely on that here so we can use this for
    other things later?

    TODO: Could possibly return the saved storage space (in bytes)?
    TODO: Use the `stat` module directly to reduce the number of system calls and make this faster.
    """

    if _is_broken_symlink(path_in_shared_cache):
        logger.warning(f"Ignoring a broken symlink in shared cache at {path_in_shared_cache}!")
        return

    if path_in_shared_cache.is_dir():
        if _is_broken_symlink(path_in_user_cache):
            logger.info(f"Replacing broken symlink at {path_in_user_cache} with a new directory.")
            path_in_user_cache.unlink()
            path_in_user_cache.mkdir(exist_ok=False)
            return

        if path_in_user_cache.is_symlink():
            # TODO: Weird case here. Do we replace the symlink with a directory?
            logger.warning(
                f"Unexpected symlink at {path_in_user_cache} (expected a 'real' directory!)."
            )
            raise NotImplementedError("TODO: What to do in this case?")

        if not path_in_user_cache.exists():
            logger.info(f"Creating directory at {path_in_user_cache}")
            path_in_user_cache.mkdir(exist_ok=True)
            return

        if not path_in_user_cache.is_dir():
            logger.warning(
                f"File in the user cache at {path_in_user_cache} where we expected a "
                f"directory! Replacing it with a new directory."
            )
            path_in_user_cache.unlink()
            path_in_user_cache.mkdir(exist_ok=True)
            return

        logger.debug(f"Directory already exists at {path_in_user_cache}")
        return

    if _is_broken_symlink(path_in_user_cache):
        logger.debug(f"Replacing broken symlink at {path_in_user_cache}")
        path_in_user_cache.unlink(missing_ok=False)
        path_in_user_cache.symlink_to(path_in_shared_cache)
        return

    if not path_in_user_cache.exists():
        # If the file matches the `copy_files_patterns` pattern, then we copy it instead of
        # creating a symlink;
        if _matches_pattern(path_in_shared_cache, copy_files_patterns):
            logger.debug(
                f"Copying file from share cache to {path_in_user_cache} because it matches one of "
                f"the patterns in {copy_files_patterns!r}"
            )
            shutil.copyfile(path_in_shared_cache, path_in_user_cache)
        else:
            logger.debug(f"Creating a new symlink at {path_in_user_cache}")
            path_in_user_cache.symlink_to(path_in_shared_cache)
        return

    if (
        not _matches_pattern(path_in_shared_cache, copy_files_patterns)
        and not path_in_user_cache.is_symlink()
    ):
        logger.info(
            f"Replacing duplicate downloaded file {path_in_user_cache} with a symlink to the "
            f"same file in the shared cache."
        )
        path_in_user_cache.unlink()
        path_in_user_cache.symlink_to(path_in_shared_cache)
        return

    # The file in the user cache is a symlink.
    user_cache_file_target = path_in_user_cache.readlink()

    if user_cache_file_target == path_in_shared_cache:
        # Symlink from a previous run, nothing to do.
        # logger.debug(f"Symlink from a previous run at {path_in_user_cache}")
        return

    # Note: Shouldn't happen, since we already should have removed broken symlinks in the
    # previous step.
    if not user_cache_file_target.exists():
        # broken symlink (perhaps from a previous run?)
        logger.warning(f"Replacing a broken symlink: {path_in_user_cache}")
        path_in_user_cache.unlink()
        path_in_user_cache.symlink_to(path_in_shared_cache)
        return

    # Symlink that points to a different file? Do nothing in this case.
    logger.warning(
        f"Found a Weird symlink at {path_in_user_cache} that doesn't point to "
        f"{path_in_shared_cache}. (points to {user_cache_file_target} instead?) "
        f"Leaving it as-is."
    )


def set_environment_variables(
    user_cache_dir: Path,
    bash_aliases_file: str | Path = "~/.bash_aliases",
    add_block_to_bash_aliases: bool = True,
) -> bool:
    """Adds a block of code to the bash_aliases file (creating it if necessary) that sets some
    relevant environment variables for each library so they start to use the new cache dir.

    Also sets the environment variables in `os.environ` so the current process gets them. Returns
    whether the contents of the bash aliases file was changed.
    """
    logger.info("Setting environment variables.")
    bash_aliases_file = Path(bash_aliases_file).expanduser().resolve()
    file = bash_aliases_file

    env_vars = {
        "HF_HOME": user_cache_dir / "huggingface",
        "HF_DATASETS_CACHE": user_cache_dir / "huggingface" / "datasets",
        "TORCH_HOME": user_cache_dir / "torch",
        # TODO: Possibly unset these variables, in case users already have them set somewhere?
        # OR: Raise a warning if the user has those variables set in their env, because they might
        # prevent the cache from being used correctly.
        # "TRANSFORMERS_CACHE": user_cache_dir / "huggingface" / "transformers",
    }

    for key, value in env_vars.items():
        os.environ[key] = str(value)

    if not add_block_to_bash_aliases:
        return False
    logger.info(f"Looking for the text block with variables in {bash_aliases_file}")

    start_flag = "# >>> cache setup >>>"
    end_flag = "# <<< cache setup <<<"
    lines_to_add = [
        start_flag,
        *(f"export {var_name}={str(var_value)}" for var_name, var_value in env_vars.items()),
        end_flag,
    ]

    if not file.exists():
        file.touch(0o644)
        file.write_text("#!/bin/bash\n")

    with open(file, "r") as f:
        lines = f.readlines()
    block_of_text = "\n".join(lines_to_add) + "\n"
    start_line = start_flag + "\n"
    end_line = end_flag + "\n"

    _update_start_and_end_flags(file, start_line, end_line)

    if start_line not in lines and end_line not in lines:
        logger.info(f"Adding a block of text at the bottom of {file}:")
        with open(file, "a") as f:
            for line in block_of_text.splitlines():
                logger.debug(line.strip())
            f.write("\n\n" + block_of_text + "\n")
        return True

    if start_line in lines and end_line in lines:
        logger.debug(f"Block is already present in {file}.")

        start_index = lines.index(start_line)
        end_index = lines.index(end_line)

        if all(
            line.strip() == lines_to_add[i]
            for i, line in enumerate(lines[start_index : end_index + 1])
        ):
            logger.info("Block already has the right contents. Doing nothing.")
            return False
        else:
            logger.info("Updating the context of the block:")
            new_lines = block_of_text.splitlines(keepends=True)
            lines[start_index : end_index + 1] = new_lines
            with open(file, "w") as f:
                for line in new_lines:
                    logger.debug(line.strip())
                f.writelines(lines)
                if len(lines) == end_index + 1:
                    # Add an empty line at the end.
                    f.write("\n")

            return True

    logger.error(
        f"Weird! The block of text is only partially present in {file}! (unable to find both "
        f"the start and end flags). Doing nothing. \n"
        f"Consider fixing the {file} file manually and re-running the command, or letting IDT "
        f"know."
    )
    return False


def _log_level(verbose: int) -> int:
    return (
        logging.DEBUG
        if verbose > 1
        else logging.INFO
        if verbose == 1
        else logging.WARNING
        if verbose == 0
        else logging.ERROR
    )


def _parse_args(argv: list[str] | None) -> Options:
    try:
        from simple_parsing import ArgumentParser

        parser = ArgumentParser(description=__doc__)
        parser.add_arguments(Options, dest="options")
        args = parser.parse_args(argv)
        options: Options = args.options
    except ImportError:
        from argparse import ArgumentParser

        parser = ArgumentParser(description=__doc__)
        parser.add_argument(
            "--user_cache_dir",
            type=Path,
            default=DEFAULT_USER_CACHE_DIR,
            help="The user cache directory. Should probably be in $SCRATCH (not $HOME!)",
        )
        parser.add_argument(
            "--shared_cache_dir",
            type=Path,
            default=DEFAULT_SHARED_CACHE_DIR,
            help=(
                "The shared cache directory. This defaults to the path of the shared cache setup "
                "by the IDT team on the Mila cluster."
            ),
        )
        parser.add_argument(
            "--subdirectory",
            type=str,
            default="",
            help="Only create links for files in this subdirectory of the shared cache.",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            default=0,
            action="count",
            help="Logging verbosity. The default logging level is `INFO`. Use -v to increase the "
            "verbosity to `DEBUG`.",
        )
        parser.add_argument(
            "-q", "--quiet", default=False, action="store_true", help="Disable logging output."
        )

        args = parser.parse_args(argv)
        options = Options(
            user_cache_dir=args.user_cache_dir,
            shared_cache_dir=args.shared_cache_dir,
            subdirectory=args.subdirectory,
            verbose=args.verbose,
            quiet=args.quiet,
        )
    return options


def _update_start_and_end_flags(file: Path, start_flag: str, end_flag: str) -> bool:
    """Replaces old versions of the start and end flags with the new ones, if they exist.

    Returns whether the file was modified.
    """
    # TODO: Keep a list of the previous block flags if we end up changing either the start or end,
    # so we can identify and replace old blocks too.
    previous_start_flags = []
    previous_end_flags = []
    update_start_and_end_flags = False

    with open(file, "r") as f:
        lines = f.readlines()

    start_line = start_flag + "\n"
    end_line = end_flag + "\n"

    for previous_flag in previous_start_flags:
        if previous_flag + "\n" in lines:
            index = lines.index(previous_flag + "\n")
            lines[index] = start_line
            update_start_and_end_flags = True

    for previous_flag in previous_end_flags:
        if previous_flag + "\n" in lines:
            index = lines.index(previous_flag + "\n")
            lines[index] = end_line
            update_start_and_end_flags = True

    if update_start_and_end_flags:
        with open(file, "w") as f:
            logger.info("Replacing old start and end flags with the new ones.")
            f.writelines(lines)
            return True
    return False


def _matches_pattern(path: str | Path, patterns: str | Sequence[str]) -> bool:
    """Checks if the given path matches any of the given patterns."""
    path = Path(path)
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    return any(
        path in _files_in_dir_matching_pattern(path.parent, pattern) for pattern in patterns
    )


@functools.lru_cache(maxsize=None)
def _files_in_dir_matching_pattern(readonly_dir: Path, pattern: str) -> list[Path]:
    # Reduce redundant calls to dir.glob(pattern) (which needs to list out the dir contents)
    # IDEA: Could add a system audit hook to invalidate the cache if we add files in any of `dirs`?
    return list(readonly_dir.glob(pattern))


def _tree(
    directory: Path,
    skip_file: Predicate[Path] | None = None,
    skip_dir: Predicate[Path] | None = None,
) -> Iterable[Path]:
    for path in directory.iterdir():
        if path.is_dir():
            if not skip_dir or not skip_dir(path):
                yield path
                yield from _tree(path, skip_file=skip_file, skip_dir=skip_dir)
        elif not skip_file or not skip_file(path):
            yield path


def _is_broken_symlink(path: Path) -> bool:
    return path.is_symlink() and not path.exists()


if __name__ == "__main__":
    main()
