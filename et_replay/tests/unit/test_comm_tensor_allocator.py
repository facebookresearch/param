import pytest
import torch
from et_replay.comm import CommTensorAllocator, CommTraceReplayArgs


@pytest.fixture
def backend_mock():
    """
    Mock backend object with necessary methods for tensor allocation.
    """

    class BackendMock:
        def alloc_ones(self, shape, device, dtype, scaleFactor=1):
            return torch.ones(shape, device=device, dtype=dtype) * scaleFactor

        def alloc_random(self, shape, device, dtype, scaleFactor=1):
            return torch.rand(shape, device=device, dtype=dtype) * scaleFactor

        @property
        def collectiveFunc(self):
            return {
                "all_to_all_single": None,
                "all_to_allv": None,
                "all_to_all": None,
                "all_gather": None,
                "gather": None,
                "all_gather_base": None,
                "incast": None,
                "reduce_scatter": None,
                "reduce_scatter_base": None,
                "scatter": None,
                "pt2pt": None,
                "wait": None,
                "barrier": None,
            }

    return BackendMock()


@pytest.fixture
def comm_op_args_mock():
    """
    Mock CommOpArgs object with default values.
    """

    class CommOpArgsMock:
        comms = "all_gather"
        in_msg_size = 1024
        out_msg_size = 1024
        src_ranks = [0, 1, 2, 3]
        out_split = None
        in_split = None

    return CommOpArgsMock()


@pytest.fixture
def comm_trace_replay_args_mock():
    """
    Mock CommTraceReplayArgs object with necessary parameters.
    """

    class Args:
        network_stack = "tcp"
        dtype = torch.float32
        backend = "nccl"
        device = "cuda"
        z = False
        c = False
        use_ext_dist = False
        init_method = "env://"
        enable_local_report = False
        enable_profiler = False
        use_perf_logger = False
        ibv_devices = []
        init_only = False

    return CommTraceReplayArgs(Args())


def test_comm_tensor_allocator_instantiation(backend_mock, comm_trace_replay_args_mock):
    """
    Test the instantiation of CommTensorAllocator.
    """
    allocator = CommTensorAllocator(backend_mock, comm_trace_replay_args_mock)
    assert isinstance(allocator, CommTensorAllocator)


def test_allocate_method(backend_mock, comm_op_args_mock, comm_trace_replay_args_mock):
    """
    Test the allocate method of CommTensorAllocator.
    """
    pass


def test_allocate_all_to_all_single_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_all_to_all_single method of CommTensorAllocator.
    """
    pass


def test_allocate_all_to_allv_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_all_to_allv method of CommTensorAllocator.
    """
    pass


def test_allocate_all_to_all_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_all_to_all method of CommTensorAllocator.
    """
    pass


def test_allocate_all_gather_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_all_gather method of CommTensorAllocator.
    """
    pass


def test_allocate_all_gather_base_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_all_gather_base method of CommTensorAllocator.
    """
    pass


def test_allocate_incast_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_incast method of CommTensorAllocator.
    """
    pass


def test_allocate_reduce_scatter_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_reduce_scatter method of CommTensorAllocator.
    """
    pass


def test_allocate_reduce_scatter_base_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_reduce_scatter_base method of CommTensorAllocator.
    """
    pass


def test_allocate_pt2pt_method(
    backend_mock, comm_op_args_mock, comm_trace_replay_args_mock
):
    """
    Test the _allocate_pt2pt method of CommTensorAllocator.
    """
    pass
