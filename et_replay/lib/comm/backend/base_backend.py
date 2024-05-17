# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .coll_args import CollArgs

SupportedP2pOps = [
    "send",
    "recv",
    "isend",
    "irecv",
]


class BaseBackend(ABC):
    """
    Abstract base class that defines common functionalities for distributed computation backends.

    Attributes:
        tcp_store (Optional[dist.TCPStore]): A TCP store for communication during distributed operations.
        collective_func (Dict[str, Callable]): Mapping of collective function names to their respective callable functions.
    """

    def __init__(self) -> None:
        """
        Initialize common backend functionalities.
        """
        self.tcp_store = None

    @abstractmethod
    def initialize_backend(
        self, master_ip: str, master_port: str, backend: str = "gloo"
    ) -> None:
        """
        Initialize the backend for distributed operations.

        Args:
            master_ip (str): IP address of the master node.
            master_port (str): Port number of the master node.
            backend (str): Backend to be used for initialization.
        """
        pass

    @abstractmethod
    def say_hello(
        self, global_rank: int, local_rank: int, world_size: int, master_ip: str
    ) -> None:
        """
        Print startup information for a specific backend instance.

        Args:
            global_rank (int): Global rank of the process in the distributed setup.
            local_rank (int): Local rank of the process on the node.
            world_size (int): Total number of processes in the distributed setup.
            master_ip (str): IP address of the master node.
        """
        pass

    # Memory management functions
    @abstractmethod
    def alloc_ones(
        self,
        size_arr: int,
        cur_rank_device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Allocate a tensor of ones, scaled by a given factor.

        Args:
            size_arr (int): Number of elements in the tensor.
            cur_rank_device (str): Device to allocate tensor on.
            dtype (torch.dtype): Data type of the tensor.
            scale_factor (float): Factor to scale the tensor values by.

        Returns:
            torch.Tensor: The allocated tensor.
        """
        pass

    @abstractmethod
    def alloc_random(
        self,
        size_arr: int,
        cur_rank_device: str,
        dtype: torch.dtype,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Allocate a tensor with random values.

        Args:
            size_arr (int): Size of the tensor to allocate.
            cur_rank_device (str): Device to allocate tensor on.
            dtype (torch.dtype): Data type of the tensor.
            scale_factor (float): Scaling factor for tensor values.

        Returns:
            torch.Tensor: The allocated tensor.
        """
        pass

    @abstractmethod
    def alloc_empty(
        self, size_arr: int, dtype: torch.dtype, cur_rank_device: str
    ) -> torch.Tensor:
        """
        Allocate an uninitialized tensor.

        Args:
            size_arr (int): Size of the tensor.
            dtype (torch.dtype): Data type of the tensor.
            cur_rank_device (str): Device to allocate tensor on.

        Returns:
            torch.Tensor: The allocated tensor.
        """
        pass

    @abstractmethod
    def clear_memory(self, collective_args: CollArgs) -> None:
        """
        Clear memory allocated for the collective operations.

        Args:
            collective_args (CollArgs): Holder of collective arguments which contains tensors to be cleared.
        """
        pass

    # Collective communication functions
    @abstractmethod
    def all_reduce(self, collective_args: CollArgs, ret_flag: bool = False) -> None:
        """
        Perform an all-reduce operation on the data within the collective_args.

        Args:
            collective_args (CollArgs): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def reduce(self, collective_args: CollArgs, ret_flag: bool = False) -> None:
        """
        Perform a reduce operation on the data within the collective_args.

        Args:
            collective_args (CollArgs): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def all_to_all(self, collective_args: CollArgs, ret_flag: bool = False) -> None:
        """
        Perform an all-to-all operation on the data within the collective_args.

        Args:
            collective_args (CollArgs): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def all_to_allv(self, collective_args: CollArgs, ret_flag: bool = False) -> None:
        """
        Perform an all-to-all variable operation on the data within the collective_args.

        Args:
            collective_args (CollArgs): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def barrier(self, collective_args: CollArgs, name: str = "dummy") -> None:
        """
        Synchronize all processes in the distributed environment.

        Args:
            collective_args (CollArgs): The collective arguments.
            name (str): Name of the barrier for debugging.
        """
        pass

    # Placeholder and utility functions
    def noop(
        self,
        collective_args: CollArgs = None,
        ret_flag: bool = False,
        pair: bool = False,
    ) -> None:
        """
        A no-operation function used as a placeholder.

        Args:
            collective_args (CollArgs, optional): The collective arguments.
            ret_flag (bool, optional): Flag to indicate if the operation should return something.
            pair (bool, optional): Flag to indicate if the operation involves pairs.
        """
        pass

    def sync_barrier(self, collective_args: CollArgs, desc: str = "world") -> None:
        """
        Synchronize all processes in the distributed environment, ensuring all previous operations are completed.

        Args:
            collective_args (CollArgs): The collective arguments.
            desc (str): Description of the sync point for debugging.
        """
        self.barrier(collective_args, name=desc)

    @abstractmethod
    def get_reduce_op(self, op_name: str) -> dist.ReduceOp:
        """
        Get the corresponding reduce operation.

        Args:
            op_name (str): Name of the operation.

        Returns:
            dist.ReduceOp: The reduce operation.
        """
        pass

    @abstractmethod
    def get_world_size(self) -> int:
        """
        Get the total number of processes in the distributed environment.

        Returns:
            int: The world size.
        """
        pass

    @abstractmethod
    def get_local_size(self) -> int:
        """
        Get the number of processes on the local node.

        Returns:
            int: The local size.
        """
        pass

    @abstractmethod
    def get_global_rank(self) -> int:
        """
        Get the global rank of the process.

        Returns:
            int: The global rank.
        """
        pass

    @abstractmethod
    def get_local_rank(self) -> int:
        """
        Get the local rank of the process.

        Returns:
            int: The local rank.
        """
        pass

    @abstractmethod
    def get_device(self) -> str:
        """
        Get the current device the process is using.

        Returns:
            str: The device identifier.
        """
        pass

    @abstractmethod
    def get_hw_device(self) -> str:
        """
        Get the hardware device information.

        Returns:
            str: The hardware device identifier.
        """
        pass

    @abstractmethod
    def get_default_group(self) -> ProcessGroup:
        """
        Get the default process group for collective operations.

        Returns:
            ProcessGroup: The default process group.
        """
        pass

    @abstractmethod
    def get_groups(self) -> List[ProcessGroup]:
        """
        Get all the process groups available.

        Returns:
            list: List of process groups.
        """
        pass

    @abstractmethod
    def get_num_pgs(self) -> int:
        """
        Get the number of process groups.

        Returns:
            int: Number of process groups.
        """
        pass

    @abstractmethod
    def complete_accel_ops(self, collectiveArgs: CollArgs):
        pass
