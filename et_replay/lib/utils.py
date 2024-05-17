import datetime
import gzip
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Tuple

from .execution_trace import ExecutionTrace


def get_tmp_trace_filename() -> str:
    """
    Generate a temporary filename using the current date, a UUID, and the process ID.

    Returns:
        str: A string representing the temporary trace filename.
    """
    trace_fn = (
        "tmp_"
        + datetime.datetime.today().strftime("%Y%m%d")
        + "_"
        + uuid.uuid4().hex[:7]
        + "_"
        + str(os.getpid())
        + ".json"
    )
    return trace_fn


def trace_handler(prof: Any) -> None:
    """
    Handle the profiling data and export it as a Chrome trace file.

    Args:
        prof (Any): The profiler object with export capabilities.
    """
    fn = get_tmp_trace_filename()
    prof.export_chrome_trace("/tmp/" + fn)
    logging.warning(f"Chrome profile trace written to /tmp/{fn}")


def load_execution_trace_file(et_file_path: str) -> ExecutionTrace:
    """
    Load and parse an Execution Trace from a JSON file.

    Args:
        et_file_path (str): Path to the execution trace JSON file.

    Returns:
        ExecutionTrace: An instance of ExecutionTrace parsed from the file.
    """
    data = read_dictionary_from_json_file(et_file_path)
    return ExecutionTrace(data)


def read_dictionary_from_json_file(file_path: str) -> Dict[Any, Any]:
    """
    Read a JSON file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[Any, Any]: A dictionary representation of the JSON file.
    """
    with (
        gzip.open(file_path, "rb") if file_path.endswith("gz") else open(file_path, "r")
    ) as f:
        return json.load(f)


def write_dictionary_to_json_file(file_path: str, data: Dict[Any, Any]) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        data (Dict[Any, Any]): The data to write to the file.
    """
    if file_path.endswith("gz"):
        with gzip.open(file_path, "wt") as f:
            f.write(json.dumps(data, indent=4))
    else:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)


def get_first_positive_env_value(env_list: List[str], default: int = -1) -> int:
    """
    Retrieves the first non-negative integer value from a list of environment variables.

    Args:
        env_list: List of environment variable names.
        default: Default value to return if all environment variables are not set.

    Returns:
        The first non-negative integer value found in the specified environment variables,
        or the default value if none are set.
    """
    for env_name in env_list:
        value = int(os.environ.get(env_name, -1))
        if value >= 0:
            return value
    return default


def read_mpi_env_vars() -> Tuple[int, int, int, int]:
    """
    Retrieves essential communication environment variables.

    Returns:
        A tuple containing:
        - world_size: Number of tasks/processes involved in the job.
        - local_size: Number of tasks/processes on the local node.
        - global_rank: Rank of the process across all nodes.
        - local_rank: Rank of the process on the local node.
    """
    world_size = get_first_positive_env_value(
        [
            "MV2_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "WORLD_SIZE",
            "SLURM_NTASKS",
        ],
        -1,
    )

    local_size = get_first_positive_env_value(
        [
            "LOCAL_SIZE",
            "MPI_LOCALNRANKS",
            "MV2_COMM_WORLD_LOCAL_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "SLURM_NTASKS_PER_NODE",
        ],
        -1,
    )

    global_rank = get_first_positive_env_value(
        [
            "MV2_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "RANK",
            "SLURM_PROCID",
        ],
        -1,
    )

    local_rank = get_first_positive_env_value(
        [
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
        ],
        -1,
    )

    return world_size, local_size, global_rank, local_rank
