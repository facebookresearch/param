from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base_backend import BaseBackend
from .coll_args import CollArgs


class MockBackend(BaseBackend):
    """
    Mock backend for collective communication operations in a distributed
    environment.
    """

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

    def initialize_backend(
        self, master_ip: str, master_port: str, backend: str = "gloo"
    ) -> None:
        """
        Initializes the backend for distributed operations.
        """
        pass

    def say_hello(
        self, global_rank: int, local_rank: int, world_size: int, master_ip: str
    ) -> None:
        """
        Prints a greeting message with rank information.
        """
        print(
            f"Hello from global rank {global_rank}, local rank {local_rank}, "
            f"world size {world_size}, master IP {master_ip}"
        )

    def alloc_ones(
        self,
        size_arr: int,
        cur_rank_device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Allocates a tensor of ones, scaled by a given factor.
        """
        return torch.ones(size_arr, device=cur_rank_device, dtype=dtype) * scale_factor

    def alloc_random(
        self,
        size_arr: int,
        cur_rank_device: str = "cpu",
        dtype: torch.dtype = torch.int32,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Allocates a tensor of random values, currently returns a tensor of ones
        for testing.
        """
        return self.alloc_ones(size_arr, cur_rank_device, dtype, scale_factor)

    def alloc_empty(
        self, size_arr: int, dtype: torch.dtype, cur_rank_device: str = "cuda"
    ) -> torch.Tensor:
        """
        Allocates an uninitialized tensor.
        """
        return torch.empty(size_arr, device=cur_rank_device, dtype=dtype)

    def clear_memory(self, coll_args: CollArgs) -> None:
        """
        Clears memory associated with the collective operations.
        """
        pass

    def all_reduce(self, coll_args: CollArgs, ret_flag: bool = False) -> None:
        self.mock_collective(coll_args)

    def reduce(self, coll_args: CollArgs, ret_flag: bool = False) -> None:
        self.mock_collective(coll_args)

    def all_gather(self, coll_args: CollArgs, ret_flag: bool = False) -> None:
        self.mock_collective(coll_args)

    def all_to_all(self, coll_args: CollArgs, ret_flag: bool = False) -> None:
        self.mock_collective(coll_args)

    def all_to_allv(self, coll_args: CollArgs, ret_flag: bool = False) -> None:
        self.mock_collective(coll_args)

    def broadcast(self, coll_args: CollArgs, ret_flag: bool = False) -> None:
        self.mock_collective(coll_args)

    def recv(self, coll_args: CollArgs, ret_flag: bool = False) -> None:
        self.mock_collective(coll_args)

    def barrier(self, coll_args: CollArgs, name: str = "dummy") -> None:
        self.mock_collective(coll_args)

    def sync_barrier(self, coll_args: CollArgs, desc: str = "world") -> None:
        self.barrier(coll_args, name=desc)

    def mock_collective(self, coll_args: CollArgs) -> CollArgs:
        """
        Mock implementation to modify collectiveArgs values during testing.
        """
        return coll_args

    def get_reduce_op(self, op_name: str) -> dist.ReduceOp:
        """
        Returns the corresponding reduce operation.
        """
        return dist.ReduceOp.SUM

    def get_world_size(self) -> int:
        """
        Returns the total number of processes in the distributed environment.
        """
        return 1

    def get_local_size(self) -> int:
        """
        Returns the number of processes on the local node.
        """
        return 1

    def get_global_rank(self) -> int:
        """
        Returns the global rank of the process.
        """
        return 0

    def get_local_rank(self) -> int:
        """
        Returns the local rank of the process.
        """
        return 0

    def get_device(self) -> str:
        """
        Returns the current device the process is using.
        """
        return "cuda"

    def get_hw_device(self) -> str:
        """
        Returns the hardware device information.
        """
        return "cuda:0"

    def get_default_group(self) -> ProcessGroup:
        """
        Returns the default process group for collective operations.
        """
        return ProcessGroup

    def get_groups(self) -> List[ProcessGroup]:
        """
        Returns all the process groups available.
        """
        return [ProcessGroup]

    def get_num_pgs(self) -> int:
        """
        Returns the number of process groups.
        """
        return 1

    def complete_accel_ops(self, collectiveArgs: CollArgs) -> None:
        """
        Completes any pending accelerator operations.
        """
        pass
