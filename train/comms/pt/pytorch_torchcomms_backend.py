# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from itertools import cycle

import numpy as np
import torch
from param_bench.train.comms.pt.pytorch_backend_utils import (
    backendFunctions,
    collectiveArgsHolder,
)
from torchcomms import new_comm, objcol, ReduceOp

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ["nccl", "ncclx", "rcclx"]


class PyTorchTorchcommsBackend(backendFunctions):
    def __init__(self, bootstrap_info, commsParams):
        super().__init__()

        self.bootstrap_info = bootstrap_info
        self.commsParams = commsParams
        self.torchcomm = None
        self.groupRanks = {}
        self.groups = {}

        # Map of op names to ReduceOp enums
        self.reduce_op_map = {
            "sum": ReduceOp.SUM,
            "max": ReduceOp.MAX,
            "min": ReduceOp.MIN,
            "product": ReduceOp.PRODUCT,
            "avg": ReduceOp.AVG,
        }

    def sayHello(self):
        myhost = os.uname()[1]
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        device = self.get_device()

        hello_msg = f"[Rank {global_rank:3}] host {myhost}, device: {device}, local_rank: {local_rank} world_size: {world_size}"

        # Use torchcomms communicator to sync hello messages
        if global_rank == 0:
            for rank in range(world_size):
                if rank == 0:
                    print(f"Hello from Rank {rank}: {hello_msg}")
                else:
                    # For other ranks, we'll just print a placeholder
                    print(f"Hello from Rank {rank}: [Rank {rank:3}] ...")

    def store_get(self, key):
        # For torchcomms, we don't have a built-in key-value store
        # This is a placeholder implementation
        return b"dummy_value"

    def store_set(self, key, val):
        # For torchcomms, we don't have a built-in key-value store
        # This is a placeholder implementation
        pass

    # Collectives
    def all_reduce(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        op = self.get_reduce_op(
            collectiveArgs.op if hasattr(collectiveArgs, "op") else "sum"
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.all_reduce(tensor, op, async_op=True)
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.all_reduce(tensor, op, async_op=False)
            if retFlag:
                return work

    def reduce(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        op = self.get_reduce_op(
            collectiveArgs.op if hasattr(collectiveArgs, "op") else "sum"
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.reduce(
                tensor, collectiveArgs.srcOrDst, op, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.reduce(
                tensor, collectiveArgs.srcOrDst, op, async_op=False
            )
            if retFlag:
                return work

    def all_to_all(
        self, collectiveArgs: collectiveArgsHolder, retFlag=False, pair=False, pairIdx=0
    ):
        output_tensors = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_tensors = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.all_to_all(
                output_tensors, input_tensors, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.all_to_all(
                output_tensors, input_tensors, async_op=False
            )
            if retFlag:
                return work

    def all_to_allv(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        output_tensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )
        output_splits = (
            collectiveArgs.opTensor_split
            if not pair
            else collectiveArgs.opTensor_split_pair[pairIdx]
        )
        input_splits = (
            collectiveArgs.ipTensor_split
            if not pair
            else collectiveArgs.ipTensor_split_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.all_to_all_v_single(
                output_tensor, input_tensor, output_splits, input_splits, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.all_to_all_v_single(
                output_tensor, input_tensor, output_splits, input_splits, async_op=False
            )
            if retFlag:
                return work

    def all_to_all_single(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        output_tensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.all_to_all_single(
                output_tensor, input_tensor, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.all_to_all_single(
                output_tensor, input_tensor, async_op=False
            )
            if retFlag:
                return work

    def all_gather(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        tensor_list = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.all_gather(tensor_list, tensor, async_op=True)
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.all_gather(tensor_list, tensor, async_op=False)
            if retFlag:
                return work

    def all_gather_object(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        retObj = objcol.all_gather_object(
            self.torchcomm,
            object_list=(
                collectiveArgs.opTensor
                if not pair
                else collectiveArgs.opTensor_pair[pairIdx]
            ),
            obj=(
                collectiveArgs.ipTensor
                if not pair
                else collectiveArgs.ipTensor_pair[pairIdx]
            ),
        )
        if retFlag:
            return retObj

    def all_gather_base(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        output_tensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.all_gather_single(
                output_tensor, input_tensor, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.all_gather_single(
                output_tensor, input_tensor, async_op=False
            )
            if retFlag:
                return work

    def gather(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        output_list = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.gather(
                output_list, input_tensor, collectiveArgs.srcOrDst, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.gather(
                output_list, input_tensor, collectiveArgs.srcOrDst, async_op=False
            )
            if retFlag:
                return work

    def scatter(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        output_tensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_list = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.scatter(
                output_tensor, input_list, collectiveArgs.srcOrDst, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.scatter(
                output_tensor, input_list, collectiveArgs.srcOrDst, async_op=False
            )
            if retFlag:
                return work

    def reduce_scatter(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        output_tensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_list = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        op = self.get_reduce_op(
            collectiveArgs.op if hasattr(collectiveArgs, "op") else "sum"
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.reduce_scatter(
                output_tensor, input_list, op, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.reduce_scatter(
                output_tensor, input_list, op, async_op=False
            )
            if retFlag:
                return work

    def reduce_scatter_base(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        output_tensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )
        input_tensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )

        op = self.get_reduce_op(
            collectiveArgs.op if hasattr(collectiveArgs, "op") else "sum"
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.reduce_scatter_single(
                output_tensor, input_tensor, op, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.reduce_scatter_single(
                output_tensor, input_tensor, op, async_op=False
            )
            if retFlag:
                return work

    # Many-to-one pattern
    def incast(self, collectiveArgs):
        raise NotImplementedError("Incast not implemented for torchcomms")

    def broadcast(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        tensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )

        if collectiveArgs.asyncOp:
            work = self.torchcomm.broadcast(
                tensor, collectiveArgs.srcOrDst, async_op=True
            )
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.broadcast(
                tensor, collectiveArgs.srcOrDst, async_op=False
            )
            if retFlag:
                return work

    def broadcast_object_list(
        self, collectiveArgs, retFlag=False, pair=False, pairIdx=0
    ):
        retObj = objcol.broadcast_object_list(
            self.torchcomm,
            object_list=(
                collectiveArgs.opTensor
                if not pair
                else collectiveArgs.opTensor_pair[pairIdx]
            ),
            root=collectiveArgs.srcOrDst,
        )
        if retFlag:
            return retObj

    # One-to-many pattern
    def multicast(self, collectiveArgs):
        if collectiveArgs.global_rank == collectiveArgs.srcOrDst:
            # root sends tensor to each of user-specified destination ranks
            for dst_rank in collectiveArgs.dst_ranks:
                self.isend(collectiveArgs, dst_rank)
            # complete outstanding isends if blocking
            if not collectiveArgs.asyncOp:
                self.complete_accel_ops(collectiveArgs, devSync=False)
        elif collectiveArgs.global_rank in collectiveArgs.dst_ranks:
            # recvs tensor from root
            if collectiveArgs.asyncOp:
                self.irecv(collectiveArgs, collectiveArgs.srcOrDst)
            else:
                self.recv(collectiveArgs, collectiveArgs.srcOrDst)

    def send(self, collectiveArgs, retFlag=False, tag=0):
        work = self.torchcomm.send(
            collectiveArgs.ipTensor, collectiveArgs.dst_rank, async_op=False
        )
        if retFlag:
            return work

    def recv(self, collectiveArgs, retFlag=False, tag=0):
        work = self.torchcomm.recv(
            collectiveArgs.opTensor, collectiveArgs.src_rank, async_op=False
        )
        if retFlag:
            return work

    def isend(self, collectiveArgs, retFlag=False, tag=0):
        work = self.torchcomm.send(
            collectiveArgs.ipTensor, collectiveArgs.dst_rank, async_op=True
        )
        collectiveArgs.waitObj.append(work)
        if retFlag:
            return work

    def irecv(self, collectiveArgs, retFlag=False, tag=0):
        work = self.torchcomm.recv(
            collectiveArgs.opTensor, collectiveArgs.src_rank, async_op=True
        )
        collectiveArgs.waitObj.append(work)
        if retFlag:
            return work

    def barrier(self, collectiveArgs, name="dummy", retFlag=False):
        if collectiveArgs.asyncOp:
            work = self.torchcomm.barrier(async_op=True)
            collectiveArgs.waitObj.append(work)
            if retFlag:
                return work
        else:
            work = self.torchcomm.barrier(async_op=False)
            if retFlag:
                return work

    def device_sync(self, collectiveArgs):
        device = self.get_device()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def complete_accel_ops(self, collectiveArgs, devSync=True):
        for waitReq in collectiveArgs.waitObj:
            if waitReq is not None:
                waitReq.wait()

        if devSync:
            self.device_sync(collectiveArgs)

        collectiveArgs.waitObj.clear()
        collectiveArgs.waitObjIds.clear()

    def complete_single_op(self, collectiveArgs, retFlag=False):
        """Only wait on the first op in the queue"""
        if len(collectiveArgs.waitObj) > 0:
            waitReq = collectiveArgs.waitObj.pop(0)
            if waitReq is not None:
                waitReq.wait()

            self.device_sync(collectiveArgs)

    def wait(self, collectiveArgs, retFlag=False):
        # For backwards compatibility, use old wait functionality
        if len(collectiveArgs.waitObjIds) == 0:
            self.complete_single_op(collectiveArgs)
            return

        """Wait on op with the matching reqID"""
        if collectiveArgs.collectiveId in collectiveArgs.waitObjIds:
            waitObj = collectiveArgs.waitObjIds[collectiveArgs.collectiveId]
            if waitObj is not None:
                waitObj.wait()

    def sync_barrier(self, collectiveArgs, desc="dummy"):
        self.complete_accel_ops(collectiveArgs)
        self.barrier(collectiveArgs, name=desc)
        self.complete_accel_ops(collectiveArgs)

    def get_reduce_op(self, opName):
        if isinstance(opName, str):
            return self.reduce_op_map.get(opName.lower(), ReduceOp.SUM)
        else:
            # Assume it's already a ReduceOp or compatible object
            return opName

    # Compute functions
    def compute_mm(self, collectiveArgs):
        self.gemm(collectiveArgs)

    def gemm(self, collectiveArgs):
        # Matrix multiplication as compute kernel
        collectiveArgs.MMout = torch.matmul(collectiveArgs.MMin1, collectiveArgs.MMin2)

    # Memory related
    def get_mem_size(self, collectiveArgs, pair=False, pairIdx=0):
        _sizeBytes = 0
        if isinstance(collectiveArgs.opTensor, list):
            _sizeBytes = sum(
                [t.nelement() * t.element_size() for t in collectiveArgs.opTensor]
            )
        elif isinstance(collectiveArgs.ipTensor, list):
            _sizeBytes = sum(
                [t.nelement() * t.element_size() for t in collectiveArgs.ipTensor]
            )
        elif collectiveArgs.collective in ["reduce_scatter_v", "reduce_scatter_base"]:
            _sizeBytes = (
                collectiveArgs.ipTensor.nelement()
                * collectiveArgs.ipTensor.element_size()
            )
        else:
            _sizeBytes = (
                collectiveArgs.opTensor.nelement()
                * collectiveArgs.opTensor.element_size()
            )
        if pair:
            if isinstance(collectiveArgs.opTensor_pair[pairIdx], list):
                _sizeBytes = sum(
                    [
                        t.nelement() * t.element_size()
                        for t in collectiveArgs.opTensor_pair[pairIdx]
                    ]
                )
            else:
                _sizeBytes = (
                    collectiveArgs.opTensor_pair[pairIdx].nelement()
                    * collectiveArgs.opTensor_pair[pairIdx].element_size()
                )

        return _sizeBytes

    def alloc_random(
        self, sizeArr, curRankDevice="cuda", dtype=torch.float32, scaleFactor=1.0
    ):
        if dtype in (
            torch.int8,
            torch.uint8,
            torch.short,
            torch.int16,
            torch.int32,
            torch.long,
        ):
            ipTensor = torch.randint(
                low=0, high=10, size=tuple(sizeArr), device=curRankDevice, dtype=dtype
            )
        elif dtype == torch.bool:
            ipTensor = (
                torch.rand(sizeArr, device=curRankDevice, dtype=torch.float32) < 0.5
            )
        else:
            ipTensor = torch.rand(sizeArr, device=curRankDevice, dtype=dtype)
            if (scaleFactor) != 0:
                ipTensor = ipTensor / scaleFactor
        return ipTensor

    def alloc_embedding_tables(self, n, m, curRankDevice, dtype):
        # Placeholder implementation - embedding tables not currently supported
        raise NotImplementedError(
            "Embedding tables not supported in torchcomms backend"
        )

    def alloc_empty(self, sizeArr, curRankDevice, dtype):
        return torch.empty(sizeArr, device=curRankDevice, dtype=dtype)

    def clear_memory(self, collectiveArgs):
        del collectiveArgs.ipTensor
        del collectiveArgs.opTensor
        for i in range(len(collectiveArgs.ipTensor_pair) - 1, -1, -1):
            del collectiveArgs.ipTensor_pair[i]
            del collectiveArgs.opTensor_pair[i]

        torch.cuda.empty_cache()

    # Getting world-size and other information
    def get_local_rank(self):
        return self.bootstrap_info.local_rank

    def get_local_size(self):
        return self.bootstrap_info.local_size

    def get_global_rank(self):
        return self.torchcomm.get_rank()

    def get_world_size(self):
        return self.torchcomm.get_size()

    def get_group_rank(self, group):
        # torchcomms doesn't have the same group concept as torch.distributed
        return self.get_global_rank()

    def get_group_size(self, group):
        # torchcomms doesn't have the same group concept as torch.distributed
        return self.get_world_size()

    def get_device(self):
        """Get current device: 'cpu' or 'cuda'"""
        if self.torchcomm:
            return self.torchcomm.get_device()
        else:
            # Fallback based on commsParams
            dev_str = (
                self.commsParams["device"]
                if isinstance(self.commsParams, dict)
                else self.commsParams.device
            )
            my_dev = torch.device(dev_str)
            if dev_str == "cuda":
                ordinal = self.get_local_rank() % torch.cuda.device_count()
                if self.get_local_rank() == -1:
                    logger.warning(
                        "Cannot determine device ordinal since LOCAL_RANK is -1. Try GPU 0 and continue."
                    )
                    ordinal = 0
                my_dev = torch.device(f"cuda:{ordinal}")
            elif dev_str != "cpu":
                raise ValueError(f"{dev_str} is not a valid device option")
            return my_dev

    def get_hw_device(self):
        return self.get_device()

    def get_default_group(self):
        # torchcomms doesn't have the same group concept, return None
        return None

    def get_groups(self):
        return list(self.groups.values())

    def get_num_pgs(self):
        return len(self.groups)

    def get_next_group(self):
        if hasattr(self, "round_robin_group"):
            return next(self.round_robin_group)
        return None

    def get_new_stream(self):
        """Get/allocate a new stream"""
        if self.commsParams.device == "cuda":
            return torch.cuda.Stream(device=self.get_device(), priority=0)
        else:
            return None

    def get_new_event(self, enable_timing=False):
        if self.commsParams.device == "cuda":
            return torch.cuda.Event(enable_timing)
        else:
            return None

    def get_current_stream(self, device: torch.device | None):
        if self.commsParams.device == "cuda":
            return torch.cuda.current_stream(device)
        else:
            return None

    def switch_stream(self, stream, device: torch.device | None):
        """Switch to a new stream and return the old current stream"""
        if device is None:
            device = self.get_device()
        if stream is not None and device.type == "cuda":
            cur_stream = torch.cuda.current_stream(device=device)
            torch.cuda.set_stream(stream)
            return cur_stream
        else:
            return None

    def sync_stream(
        self,
        stream: torch.cuda.Stream | None = None,
        device: torch.device | None = None,
    ):
        """Synchronize a stream with its associated device"""
        if device is None:
            device = self.get_device()
        if device.type == "cuda":
            cur_stream = (
                stream
                if stream is not None
                else torch.cuda.current_stream(device=device)
            )
            cur_stream.synchronize()
        else:
            pass

    def tensor_list_to_numpy(self, tensorList):
        if isinstance(tensorList, list):
            tensorList = [t.cpu().detach().numpy() for t in tensorList]
        return np.array(tensorList)

    # Init functions
    def initialize_backend(
        self, master_ip, master_port, backend="ncclx", eager_mode=False
    ):
        """Initialize the torchcomms backend"""
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend {backend}. Valid backends are {_VALID_BACKENDS}"
            )

        # Forward nccl backend to ncclx backend
        if backend == "nccl":
            backend = "ncclx"

        # Initialize torchcomms
        device = self.get_device()

        # Use the specified backend, defaulting to ncclx for torchcomms
        self.torchcomm = new_comm(backend, device, name="param_bench_comm")

        logger.info(f"Initialized torchcomms backend with {backend} on device {device}")

        # Default 1 group, may be overwritten by user-created groups
        self.groups = {}
        self.groups[0] = self.get_default_group()
        self.num_pgs = len(self.groups)
        self.round_robin_group = cycle(list(self.groups.values()))

    def initialize_groups(
        self,
        groupRanks: dict[int, list[int]] | None = None,
        backend="ncclx",
        force_new_group=False,
    ):
        """Initialize additional process groups if provided"""
        raise NotImplementedError("multi-groups not implemented for torchcomms")

    def benchmark_comms(self, benchTime, commsParams):
        index = 0
        if commsParams.init_only:
            import time

            time.sleep(10)
        else:
            benchTime(index, commsParams, self)
        return

    def __del__(self):
        if self.torchcomm:
            self.torchcomm.finalize()
