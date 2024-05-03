# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from param.param_profile import ParamTimer

logger = logging.getLogger(__name__)

SupportedDevices = ["cpu", "cuda", "rocm", "tpu"]
SupportedC10dBackends = ["nccl", "gloo", "mpi", "ucc", "xla", "fairring"]
supportedCollectives = [
    "reduce",
    "all_reduce",
    "all_to_all",
    "all_to_allv",
    "all_gather",
    "all_gather_v",
    "broadcast",
    "reduce_scatter",
    "reduce_scatter_v",
    "reduce_scatter_base",
    "all_gather_base",
    "incast",
    "multicast",
    "gather",
    "scatter",
]
Pt2ptPatterns = [
    "one2one",
    "pairwise",
]
SupportedP2pOps = [
    "send",
    "recv",
    "isend",
    "irecv",
]


class CollectiveArgsHolder:
    """Class holding object for all the parameters related to a collective operation/experiment."""

    def __init__(self) -> None:
        self.group = None
        self.groups = {}  # {pg_id, pg}
        self.num_pgs = 0
        self.device = {}
        self.world_size = 0
        self.data_type = ""

        self.numIters = 0
        self.numWarmupIters = 0
        self.global_rank = -1
        self.backendFuncs = {}
        self.collective = ""
        self.collectiveId = 0
        self.pt2pt = ""
        self.src_rank = -1
        self.dst_rank = -1
        self.p2pOps = []

        self.computeCount = -1

        self.reuseTensors = False

        self.MMdim_dtype = {}
        self.MMdim = -1
        self.MMdtype = ""
        self.MMout = {}
        self.MMin1 = {}
        self.MMin2 = {}
        self.MMin3 = {}
        self.numComputePerIter = 0
        self.numCollPerIter = 0
        self.batch_size = 0

        self.emb = None
        self.embRequests = None
        self.direction = None
        self.emb_dim = 0
        self.num_emb_tables_batched = -1
        self.num_emb_ops = 0
        self.BTBlockSize = {}
        self.LookupOut = {}
        self.grad_output = None

        self.ipTensor_split = []
        self.opTensor_split = []

        self.ipTensor = []
        self.opTensor = []
        self.srcOrDst = -1
        self.asyncOp = -1
        self.dataSize = 0
        self.numElements = 0
        self.waitObj = []
        self.waitObjIds = {}  # mapping of reqID to future of async collectives

        self.ipTensor_split_pair = []
        self.opTensor_split_pair = []

        self.ipTensor_pair = None
        self.opTensor_pair = None
        self.dataSize_pair = 0
        self.numElements_pair = 0

        self.all2all_qcomm = None
        self.reducescatter_allgather_qcomm = None
        self.allreduce_qcomm = 32  # TODO: set it as the bitwidth for now until the quantization kernels be supported
        self.reduce_qcomm = 32
        self.quant_threshold = 0
        self.quant_time = ParamTimer()
        self.dequant_time = ParamTimer()
        self.enable_profiler = False

        self.compute_stream = None
        self.use_ext_dist = False


logger = logging.getLogger(__name__)


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
        self.collective_func = {
            "all_to_all_single": self.all_to_all_single,
            "all_to_all": self.all_to_all,
            "all_to_allv": self.all_to_allv,
            "all_reduce": self.all_reduce,
            "broadcast": self.broadcast,
            "gather": self.gather,
            "all_gather": self.all_gather,
            "all_gather_base": self.all_gather_base,
            "reduce": self.reduce,
            "reduce_scatter": self.reduce_scatter,
            "reduce_scatter_base": self.reduce_scatter_base,
            "scatter": self.scatter,
            "barrier": self.barrier,
            "incast": self.incast,
            "multicast": self.multicast,
            "noop": self.noop,
        }
        self.compute_func = {"gemm": self.gemm}

    @abstractmethod
    def say_hello(self, global_rank: int, local_rank: int, world_size: int, master_ip: str) -> None:
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
        self, size_arr: int, cur_rank_device: str, dtype: torch.dtype, scale_factor: float = 1.0
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
    def alloc_empty(self, size_arr: int, dtype: torch.dtype, cur_rank_device: str) -> torch.Tensor:
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
    def clear_memory(self, collective_args: CollectiveArgsHolder) -> None:
        """
        Clear memory allocated for the collective operations.

        Args:
            collective_args (CollectiveArgsHolder): Holder of collective arguments which contains tensors to be cleared.
        """
        pass

    # Collective communication functions
    @abstractmethod
    def all_reduce(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        """
        Perform an all-reduce operation on the data within the collective_args.

        Args:
            collective_args (CollectiveArgsHolder): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def reduce(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        """
        Perform a reduce operation on the data within the collective_args.

        Args:
            collective_args (CollectiveArgsHolder): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def all_to_all(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        """
        Perform an all-to-all operation on the data within the collective_args.

        Args:
            collective_args (CollectiveArgsHolder): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def all_to_allv(self, collective_args: CollectiveArgsHolder, ret_flag: bool = False) -> None:
        """
        Perform an all-to-all variable operation on the data within the collective_args.

        Args:
            collective_args (CollectiveArgsHolder): The collective arguments.
            ret_flag (bool): Flag to indicate if the operation should return something.
        """
        pass

    @abstractmethod
    def barrier(self, collective_args: CollectiveArgsHolder, name: str = "dummy") -> None:
        """
        Synchronize all processes in the distributed environment.

        Args:
            collective_args (CollectiveArgsHolder): The collective arguments.
            name (str): Name of the barrier for debugging.
        """
        pass

    # Placeholder and utility functions
    def noop(self, collective_args: CollectiveArgsHolder = None, ret_flag: bool = False, pair: bool = False) -> None:
        """
        A no-operation function used as a placeholder.

        Args:
            collective_args (CollectiveArgsHolder, optional): The collective arguments.
            ret_flag (bool, optional): Flag to indicate if the operation should return something.
            pair (bool, optional): Flag to indicate if the operation involves pairs.
        """
        pass

    def sync_barrier(self, collective_args: CollectiveArgsHolder, desc: str = "world") -> None:
        """
        Synchronize all processes in the distributed environment, ensuring all previous operations are completed.

        Args:
            collective_args (CollectiveArgsHolder): The collective arguments.
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
    def gemm(self, collective_args: CollectiveArgsHolder) -> None:
        """
        Perform a general matrix multiplication (GEMM) operation.

        Args:
            collective_args (CollectiveArgsHolder): The collective arguments.
        """
        pass

    # Device and rank information retrieval functions
    @abstractmethod
    def get_local_rank(self) -> int:
        """
        Get the local rank of the process.

        Returns:
            int: The local rank.
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

    # Initialization and setup functions
    @abstractmethod
    def initialize_backend(self, master_ip: str, master_port: str, backend: str = "gloo") -> None:
        """
        Initialize the backend for distributed operations.

        Args:
            master_ip (str): IP address of the master node.
            master_port (str): Port number of the master node.
            backend (str): Backend to be used for initialization.
        """
        pass

    @abstractmethod
    def benchmark_comms(self, bench_time, comms_params) -> None:
        """
        Run benchmarks for the communication operations.

        Args:
            bench_time: The timing function for benchmarks.
            comms_params: Parameters for the communication operations.
        """
        pass

    @abstractmethod
    def complete_accel_ops(self, collectiveArgs: CollectiveArgsHolder):
        pass
