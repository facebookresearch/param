import gzip
import json
import logging
from typing import Any, Dict

from param_bench.train.compute.python.tools.execution_graph import ExecutionGraph


def get_tmp_trace_filename():
    import datetime
    import os
    import uuid

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


def trace_handler(prof):
    fn = get_tmp_trace_filename()
    prof.export_chrome_trace("/tmp/" + fn)
    logging.warning(f"Chrome profile trace written to /tmp/{fn}")
    # try:
    #     from param_bench.train.compute.python.tools.internals import upload_trace

    #     upload_trace(fn)
    # except ImportError:
    #     logging.info("FB internals not present")
    # except Exception as e:
    #     logging.info(f"Upload trace error: {e}")
    #     pass


def load_execution_trace_file(et_file_path: str) -> ExecutionGraph:
    """Loads Execution Trace from json file and parses it."""
    data = read_dictionary_from_json_file(et_file_path)
    return ExecutionGraph(data)


def read_dictionary_from_json_file(file_path: str) -> Dict[Any, Any]:
    with gzip.open(file_path, "rb") if file_path.endswith("gz") else open(
        file_path, "r"
    ) as f:
        return json.load(f)
