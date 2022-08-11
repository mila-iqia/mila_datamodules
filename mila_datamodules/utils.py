import os
from multiprocessing import cpu_count
from pathlib import Path


def get_slurm_tmpdir() -> Path:
    """Returns the SLURM temporary directory. Also works when using `mila code`."""
    if "SLURM_TMPDIR" in os.environ:
        return Path(os.environ["SLURM_TMPDIR"])
    if "SLURM_JOB_ID" in os.environ:
        # Running with `mila code`, so we don't have all the SLURM environment variables.
        # However, we're lucky, because we can reliably predict the SLURM_TMPDIR given the job id.
        job_id = os.environ["SLURM_JOB_ID"]
        return Path(f"/Tmp/slurm.{job_id}.0")
    raise RuntimeError(
        "None of SLURM_TMPDIR or SLURM_JOB_ID are set! "
        "Cannot locate the SLURM temporary directory."
    )


def get_cpus_on_node() -> int:
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return cpu_count()
