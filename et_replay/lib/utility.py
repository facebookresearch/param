import datetime
import gzip
import json
import logging
import os
import uuid
from typing import Any, Dict, List

from param.execution_trace import ExecutionTrace


def env2int(env_list: List[str], default: int = -1) -> int:
    """
    Takes environment variables list and returns the first value found.

    Args:
        env_list: List of environment variables.
        default: Default value to return if all environment variables are not set.
    Returns:
        val: Returns value located at one of the environment variables, or returns default value if none are set.
    """
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def read_env_vars() -> Dict[str, int]:
    """
    Reads environment variables and record them.

    Args:
        None
    Returns:
        env_vars: Dict containing env var name as key and int for that env var as value.
    """
    world_size = env2int(
        [
            "MV2_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "WORLD_SIZE",
            "SLURM_NTASKS",
        ],
        -1,
    )

    local_size = env2int(
        [
            "LOCAL_SIZE",
            "MPI_LOCALNRANKS",
            "MV2_COMM_WORLD_LOCAL_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "SLURM_NTASKS_PER_NODE",
        ],
        -1,
    )

    global_rank = env2int(
        [
            "MV2_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "RANK",
            "SLURM_PROCID",
        ],
        -1,
    )

    local_rank = env2int(
        [
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
        ],
        -1,
    )

    env_vars = {}
    env_vars["world_size"] = world_size
    env_vars["local_size"] = local_size
    env_vars["global_rank"] = global_rank
    env_vars["local_rank"] = local_rank
    return env_vars


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
    with gzip.open(file_path, "rb") if file_path.endswith("gz") else open(file_path, "r") as f:
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
