import pytest
import torch
from et_replay.comm.backend import MockBackend
from et_replay.comm.backend.coll_args import CollArgs
from torch.distributed import ReduceOp


@pytest.fixture
def backend():
    return MockBackend()


def test_alloc_ones(backend):
    tensor = backend.alloc_ones(
        10, cur_rank_device="cpu", dtype=torch.float32, scale_factor=2.0
    )
    assert torch.equal(tensor, torch.ones(10) * 2.0)


def test_alloc_random(backend):
    tensor = backend.alloc_random(
        10, cur_rank_device="cpu", dtype=torch.float32, scale_factor=2.0
    )
    assert tensor.size(0) == 10
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32
    assert torch.all(tensor == 2.0)


def test_alloc_empty(backend):
    tensor = backend.alloc_empty(10, dtype=torch.float32, cur_rank_device="cpu")
    assert tensor.size(0) == 10
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32


def test_clear_memory(backend):
    coll_args = CollArgs()
    backend.clear_memory(coll_args)  # Just ensure it runs without error


def test_collective_operations(backend):
    coll_args = CollArgs()
    backend.all_gather(coll_args)
    backend.all_reduce(coll_args)
    backend.broadcast(coll_args)
    backend.reduce(coll_args)
    backend.all_to_all(coll_args)
    backend.all_to_allv(coll_args)
    backend.recv(coll_args)
    backend.barrier(coll_args)


def test_get_device(backend):
    device = backend.get_device()
    assert device == "cuda"


def test_get_reduce_op(backend):
    reduce_op = backend.get_reduce_op("SUM")
    assert reduce_op == ReduceOp.SUM


def test_say_hello(backend, capsys):
    backend.say_hello(0, 0, 1, "127.0.0.1")
    captured = capsys.readouterr()
    assert (
        "Hello from global rank 0, local rank 0, world size 1, master IP 127.0.0.1"
        in captured.out
    )


def test_get_local_rank(backend):
    local_rank = backend.get_local_rank()
    assert local_rank == 0


def test_get_global_rank(backend):
    global_rank = backend.get_global_rank()
    assert global_rank == 0


def test_get_world_size(backend):
    world_size = backend.get_world_size()
    assert world_size == 1


def test_get_local_size(backend):
    local_size = backend.get_local_size()
    assert local_size == 1


def test_get_hw_device(backend):
    hw_device = backend.get_hw_device()
    assert hw_device == "cuda:0"


def test_get_default_group(backend):
    group = backend.get_default_group()
    assert group is not None


def test_get_groups(backend):
    groups = backend.get_groups()
    assert isinstance(groups, list)
    assert len(groups) > 0


def test_get_num_pgs(backend):
    num_pgs = backend.get_num_pgs()
    assert num_pgs == 1


def test_initialize_backend(backend):
    backend.initialize_backend("127.0.0.1", "29500", backend="gloo")
    # Just ensure it runs without error


def test_complete_accel_ops(backend):
    coll_args = CollArgs()
    backend.complete_accel_ops(coll_args)
    # Just ensure it runs without error
