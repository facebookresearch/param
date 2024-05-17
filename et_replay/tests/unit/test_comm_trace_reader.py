import json

import pytest
from et_replay.comm.comm_trace_reader import CommTraceReader


@pytest.fixture
def comm_trace_reader():
    return CommTraceReader(world_size=4)


@pytest.fixture
def sample_trace_file(tmp_path):
    sample_trace = {
        "nodes": {
            "0": {
                "name": "record_param_comms",
                "inputs": ["0", "1", "2", "3", "4", "5"],
                "input_types": ["GenericList[Tensor(int)]"],
                "outputs": ["0", "1"],
                "output_types": ["GenericList[Tensor(int)]"],
            }
        }
    }
    file_path = tmp_path / "0.json"
    with open(file_path, "w") as f:
        json.dump(sample_trace, f)
    return file_path


def test_read_trace(comm_trace_reader, sample_trace_file):
    pass


def test_read_raw_trace(comm_trace_reader, sample_trace_file):
    pass


def test_parse_execution_trace(comm_trace_reader):
    pass


def test_get_tensor_info_from_pytorch_et_entry(comm_trace_reader):
    pass


def test_create_pg_init_node(comm_trace_reader):
    pass
