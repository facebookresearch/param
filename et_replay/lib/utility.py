import datetime
import gzip
import json
import logging
import os
import uuid
from typing import Any, Dict

from param.execution_trace import ExecutionTrace


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
