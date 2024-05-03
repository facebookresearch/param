import torch

from param.comm.backend.base_backend import BaseBackend, CollectiveArgsHolder


class MockBackendFunction(BaseBackend):
    """Mock backend for collective communication operations in a distributed environment."""

    def __init__(self) -> None:
        super().__init__()
        self.collective_func = {
            "all_to_all": self.all_to_all,
            "all_to_allv": self.all_to_allv,
            "all_reduce": self.all_reduce,
            "broadcast": self.broadcast,
            "all_gather": self.all_gather,
            "reduce": self.reduce,
            "barrier": self.barrier,
            "recv": self.recv,
            "noop": super().noop,
        }

    def all_gather(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        self.mock_collective(collective_args)

    def all_reduce(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        self.mock_collective(collective_args)

    def broadcast(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        self.mock_collective(collective_args)

    def reduce(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        self.mock_collective(collective_args)

    def all_to_all(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        self.mock_collective(collective_args)

    def all_to_allv(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        self.mock_collective(collective_args)

    def recv(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        self.mock_collective(collective_args)

    def barrier(self, collective_args: CollectiveArgsHolder, name: str = "dummy") -> None:
        self.mock_collective(collective_args)

    def sync_barrier(self, collective_args: CollectiveArgsHolder, desc: str = "world") -> None:
        self.barrier(collective_args, name=desc)

    def mock_collective(self, collective_args: CollectiveArgsHolder) -> CollectiveArgsHolder:
        """Mock implementation to modify collectiveArgs values during testing."""
        return collective_args

    def gemm(self, collective_args: CollectiveArgsHolder) -> None:
        """Simulates a general matrix multiplication operation."""
        pass

    def clear_memory(self, collective_args: CollectiveArgsHolder) -> None:
        """Clears memory associated with the collective operations."""
        pass

    def alloc_random(
        self, size_arr: int, cur_rank_device: str = "cpu", dtype: torch.dtype = torch.int32, scale_factor: float = 1.0
    ) -> torch.Tensor:
        """Allocates a tensor of random values, currently returns a tensor of ones for testing."""
        return self.alloc_ones(size_arr, "cpu", dtype, 1.0)
