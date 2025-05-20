# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod

import torch

from et_replay.comm.param_profile import paramTimer
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

supportedDevices = ["cpu", "cuda", "rocm", "tpu"]
supportedC10dBackends = ["nccl", "gloo", "mpi", "xla"]
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
    "gather",
    "scatter",
]
pt2ptPatterns = [
    "one2one",
    "pairwise",
]
supportedP2pOps = [
    "send",
    "recv",
    "isend",
    "irecv",
]


class collectiveArgsHolder:
    """Class holding object for all the parameters related to a collective operation/experiment."""

    def __init__(self) -> None:
        self.group = None
        self.groups = {}  # {pg_id, pg}
        self.device = {}
        self.world_size = 0
        self.data_type = ""

        self.numIters = 0
        self.numWarmupIters = 0
        self.global_rank = -1
        self.backendFuncs = {}
        self.collective = ""
        self.wait_obj_key = (0, 0, False)  # (pg_id, req_id, is_p2p)
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
        self.waitObjIds = {}  # mapping of (pg_id, req_id, is_p2p) to future of async collectives

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
        self.quant_time = paramTimer()
        self.dequant_time = paramTimer()
        self.enable_profiler = False

        self.compute_stream = None
        self.use_ext_dist = False


class BaseBackend(ABC):
    """Abstract base class, provides common abstraction for all the backends."""

    def __init__(self) -> None:
        self.tcp_store = None
        self.collectiveFunc = {
            "all_to_all": self.all_to_all,
            "all_to_allv": self.all_to_allv,
            "all_reduce": self.all_reduce,
            "broadcast": self.broadcast,  # pyre-ignore[16]:
            "gather": self.gather,  # pyre-ignore[16]:
            "all_gather": self.all_gather,  # pyre-ignore[16]:
            "all_gather_base": self.all_gather_base,  # pyre-ignore[16]:
            "reduce": self.reduce,
            "reduce_scatter": self.reduce_scatter,  # pyre-ignore[16]:
            "reduce_scatter_base": self.reduce_scatter_base,  # pyre-ignore[16]:
            "scatter": self.scatter,  # pyre-ignore[16]:
            "barrier": self.barrier,
            "noop": self.noop,
        }

        self.computeFunc = {"gemm": self.gemm}

    def alloc_ones(
        self,
        sizeArr: list[int],
        curRankDevice: str = "cuda",
        dtype: torch.dtype = torch.float32,
        scaleFactor: float = 1.0,
    ) -> torch.Tensor:
        """
        Create a tensor filled with 1s of size sizeArr, and return the tensor multiplied by the scaleFactor.

        Args:
            sizeArr: Size of desired tensor.
            curRankDevice: Desired device of returned tensor.
            dtype: Datatype of returned tensor.
            scaleFactor: Factor to scale the returned tensor.
        Returns:
            ipTensor: Tensor filled with 1s.
        """
        ipTensor = torch.ones(sizeArr, device=curRankDevice, dtype=dtype)
        if scaleFactor != 1.0:
            ipTensor = ipTensor * scaleFactor
        return ipTensor

    def noop(
        self,
        collectiveArgs: collectiveArgsHolder,
        retFlag: bool = False,
        pair: bool = False,
    ) -> None:
        """no-op for the case we want to skip comms/compute"""
        pass

    @abstractmethod
    def sayHello(
        self, global_rank: int, local_rank: int, world_size: int, master_ip: str
    ) -> None:
        """Print startup information of the backend."""
        pass

    # Collectives, if you would like more detailed documentation about the behavior of these collectives, visit https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html.
    @abstractmethod
    def all_reduce(self, collectiveArgs: collectiveArgsHolder, retFlag: bool = False):
        pass

    @abstractmethod
    def reduce(self, collectiveArgs: collectiveArgsHolder, retFlag: bool = False):
        pass

    @abstractmethod
    def all_to_all(self, collectiveArgs: collectiveArgsHolder, retFlag: bool = False):
        pass

    @abstractmethod
    def all_to_allv(self, collectiveArgs: collectiveArgsHolder, retFlag: bool = False):
        pass

    @abstractmethod
    def complete_accel_ops(self, collectiveArgs: collectiveArgsHolder):
        pass

    @abstractmethod
    def barrier(self, collectiveArgs: collectiveArgsHolder, name: str = "dummy"):
        pass

    def sync_barrier(self, collectiveArgs: collectiveArgsHolder, desc: str = "world"):
        self.barrier(collectiveArgs, name=desc)

    @abstractmethod
    def get_reduce_op(self, opName: str):
        pass

    # Compute functions
    @abstractmethod
    def gemm(self, collectiveArgs: collectiveArgsHolder) -> None:
        pass

    # Memory related
    @abstractmethod
    def get_mem_size(self, collectiveArgs: collectiveArgsHolder) -> int:
        """Return memory size of current input tensor."""
        pass

    @abstractmethod
    def alloc_random(
        self,
        sizeArr: list[int],
        curRankDevice: str,
        dtype: torch.dtype,
        scaleFactor: float = 1.0,
    ) -> torch.Tensor:
        """Allocate tensor of random values according to parameters."""
        pass

    @abstractmethod
    def alloc_embedding_tables(
        self, n: int, m: int, curRankDevice: str, dtype: torch.dtype
    ):
        """Allocate embedding table based on parameters."""
        pass

    @abstractmethod
    def alloc_empty(
        self, sizeArr: list[int], dtype: torch.dtype, curRankDevice: str
    ) -> torch.Tensor:
        """Allocate tensor with uninitialized data based on parameters."""
        pass

    @abstractmethod
    def clear_memory(self, collectiveArgs: collectiveArgsHolder):
        """Clear memory in use by backend function."""
        pass

    # Getting world-size and other information.
    @abstractmethod
    def get_local_rank(self) -> int:
        pass

    @abstractmethod
    def get_global_rank(self) -> int:
        pass

    @abstractmethod
    def get_world_size(self) -> int:
        pass

    @abstractmethod
    def get_local_size(self) -> int:
        pass

    @abstractmethod
    def get_device(self) -> str:
        pass

    @abstractmethod
    def get_hw_device(self) -> str:
        pass

    @abstractmethod
    def get_default_group(self) -> ProcessGroup:
        pass

    @abstractmethod
    def get_groups(self) -> list[ProcessGroup]:
        pass

    # Init functions
    @abstractmethod
    def initialize_backend(
        self, master_ip: str, master_port: str, backend: str = "gloo"
    ) -> None:
        pass

    @abstractmethod
    def benchmark_comms(self, benchTime, commsParams) -> None:
        pass


customized_backend: dict[str, BaseBackend] = {}


def register_customized_backend(
    name: str,
    func: BaseBackend,
    device: str | None = None,
) -> None:
    global customized_backend
    customized_backend[name] = func
    if device is not None:
        global supportedDevices
        supportedDevices.append(device)
    logger.info(f"Registered custom backend {name} with function {func.__name__}")
