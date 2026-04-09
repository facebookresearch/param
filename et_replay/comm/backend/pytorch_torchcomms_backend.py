# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from et_replay.comm.backend.base_backend import BaseBackend, collectiveArgsHolder
from torchcomms import ReduceOp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PyTorchTorchCommsBackend(BaseBackend):
    """Torchcomms-based replay backend for et_replay.

    This is the torchcomms equivalent of PyTorchDistBackend. It implements all
    collectives via the torchcomms API (self.torchcomm.*) instead of
    torch.distributed. Quantization and extend_distributed are not applicable.
    """

    # =========================================================================
    # Collectives
    # =========================================================================
    def all_reduce(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("all_reduce not yet implemented")

    def allreduce_coalesced(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("allreduce_coalesced not yet implemented")

    def reduce(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("reduce not yet implemented")

    def all_to_all(
        self, collectiveArgs: collectiveArgsHolder, retFlag=False, pair=False
    ):
        raise NotImplementedError("all_to_all not yet implemented")

    def all_to_allv(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("all_to_allv not yet implemented")

    def all_gather(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("all_gather not yet implemented")

    def allgather_into_tensor_coalesced(
        self, collectiveArgs, retFlag=False, pair=False
    ):
        raise NotImplementedError("allgather_into_tensor_coalesced not yet implemented")

    def gather(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("gather not yet implemented")

    def scatter(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("scatter not yet implemented")

    def reduce_scatter(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("reduce_scatter not yet implemented")

    def reduce_scatter_base(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("reduce_scatter_base not yet implemented")

    def reduce_scatter_tensor_coalesced(
        self, collectiveArgs, retFlag=False, pair=False
    ):
        raise NotImplementedError("reduce_scatter_tensor_coalesced not yet implemented")

    def all_gather_base(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("all_gather_base not yet implemented")

    def broadcast(self, collectiveArgs, retFlag=False, pair=False):
        raise NotImplementedError("broadcast not yet implemented")

    # =========================================================================
    # P2P
    # =========================================================================
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
        """Async send to destination rank."""
        retObj = self.torchcomm.send(
            collectiveArgs.ipTensor, collectiveArgs.dst_rank, async_op=True
        )
        collectiveArgs.waitObj.append(retObj)
        if retFlag:
            return retObj

    def irecv(self, collectiveArgs, retFlag=False, tag=0):
        """Async receive from source rank."""
        retObj = self.torchcomm.recv(
            collectiveArgs.opTensor, collectiveArgs.src_rank, async_op=True
        )
        collectiveArgs.waitObj.append(retObj)
        if retFlag:
            return retObj

    def P2POp(self, collectiveArgs, retFlag=False, tag=0):
        raise NotImplementedError(
            "P2POp is not supported in torchcomms backend. "
            "Torchcomms does not have a native dist.P2POp equivalent."
        )

    def batch_isend_irecv(self, collectiveArgs, retFlag=False):
        raise NotImplementedError(
            "batch_isend_irecv is not supported in torchcomms backend. "
            "Torchcomms does not have a native dist.batch_isend_irecv equivalent."
        )

    # =========================================================================
    # Sync
    # =========================================================================
    def device_sync(self, collectiveArgs):
        """Synchronize the device."""
        dev_str = self.commsParams.device
        if dev_str == "cuda":
            torch.cuda.synchronize(collectiveArgs.device)

    def complete_accel_ops(self, collectiveArgs, devSync=True):
        """Complete all pending accelerator operations."""
        for waitReq in collectiveArgs.waitObj:
            if waitReq is not None:
                waitReq.wait()
        collectiveArgs.waitObj.clear()
        collectiveArgs.waitObjIds.clear()

        if devSync:
            self.device_sync(collectiveArgs)

    def wait(self, collectiveArgs, retFlag=False):
        """Wait for a specific async operation identified by wait_obj_key."""
        # Wait on op with the matching (pg_id, req_id, is_p2p)
        if collectiveArgs.wait_obj_key in collectiveArgs.waitObjIds:
            work = collectiveArgs.waitObjIds.pop(collectiveArgs.wait_obj_key)
            for i, w in enumerate(collectiveArgs.waitObj):
                if w is work:
                    collectiveArgs.waitObj.pop(i)
                    break
            work.wait()

    def barrier(self, collectiveArgs, name="dummy", retFlag=False):
        """Execute a barrier on the default group."""
        work = self.torchcomm.barrier(async_op=collectiveArgs.asyncOp)
        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(work)

        if retFlag:
            return work

    def barrier_all_ranks(self):
        """Execute a barrier across all ranks."""
        self.torchcomm.barrier(async_op=False)

    def sync_barrier(self, collectiveArgs, desc="dummy"):
        """Synchronize device, execute barrier, then sync again."""
        self.complete_accel_ops(collectiveArgs)
        self.barrier(collectiveArgs, name=desc)
        self.complete_accel_ops(collectiveArgs)

    def get_reduce_op(self, opName):
        """Get the reduce operation for the given operation name."""
        if opName == "sum":
            return ReduceOp.SUM
        elif opName == "max":
            return ReduceOp.MAX
        else:
            return ReduceOp.SUM

    # =========================================================================
    # Compute
    # =========================================================================
    def compute_mm(self, collectiveArgs):
        raise NotImplementedError("compute_mm not yet implemented")

    def gemm(self, collectiveArgs):
        raise NotImplementedError("gemm not yet implemented")

    def add(self, collectiveArgs):
        raise NotImplementedError("add not yet implemented")

    def sub(self, collectiveArgs):
        raise NotImplementedError("sub not yet implemented")

    def add_num(self, collectiveArgs):
        raise NotImplementedError("add_num not yet implemented")

    def sub_num(self, collectiveArgs):
        raise NotImplementedError("sub_num not yet implemented")

    def copy(self, collectiveArgs):
        raise NotImplementedError("copy not yet implemented")

    def emb_lookup(self, collectiveArgs):
        raise NotImplementedError("emb_lookup not yet implemented")

    # =========================================================================
    # Metadata
    # =========================================================================
    def sayHello(self):
        myhost = os.uname()[1]
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        device = self.get_device()

        hello_msg = (
            f"[Rank {global_rank:3}] host {myhost}, device: {device}, "
            f"local_rank: {local_rank} world_size: {world_size}"
        )

        self.store_set(f"hello_msg_{global_rank}", hello_msg)
        if global_rank == 0:
            for rank in range(0, world_size):
                rank_hello_msg = self.store_get(f"hello_msg_{rank}").decode()
                logger.info("Hello from Rank %d: %s", rank, rank_hello_msg)

    def store_get(self, key):
        return self.tcp_store.get(key)

    def store_set(self, key, val):
        self.tcp_store.set(key, val)

    # =========================================================================
    # Memory
    # =========================================================================
    def get_mem_size(self, collectiveArgs, pair=False):
        """Get memory size of input/output tensors in bytes."""
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
            if isinstance(collectiveArgs.opTensor_pair, list):
                _sizeBytes = sum(
                    [
                        t.nelement() * t.element_size()
                        for t in collectiveArgs.opTensor_pair
                    ]
                )
            else:
                _sizeBytes = (
                    collectiveArgs.opTensor_pair.nelement()
                    * collectiveArgs.opTensor_pair.element_size()
                )

        return _sizeBytes

    def alloc_random(
        self,
        sizeArr: list[int],
        curRankDevice="cuda",
        dtype=torch.float32,
        scaleFactor=1.0,
    ):
        """Allocate a tensor with random values on the specified device."""
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
        """Allocate embedding tables with random values."""
        EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

        W = np.random.uniform(
            low=-(np.sqrt(1 / n)), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)

        EE.weight.data = torch.tensor(
            W, dtype=dtype, requires_grad=True, device=curRankDevice
        )
        return EE

    def alloc_empty(self, sizeArr, dtype, curRankDevice):
        """Allocate an empty tensor (uninitialized) on the specified device."""
        return torch.empty(sizeArr, device=curRankDevice, dtype=dtype)

    def clear_memory(self, collectiveArgs):
        """Clear memory by deleting tensors and running garbage collection."""
        if collectiveArgs.ipTensor is not None:
            del collectiveArgs.ipTensor

        if collectiveArgs.opTensor is not None:
            del collectiveArgs.opTensor

        if collectiveArgs.ipTensor_pair is not None:
            del collectiveArgs.ipTensor_pair
            del collectiveArgs.opTensor_pair

        torch.cuda.empty_cache()

    # =========================================================================
    # Rank / World Info
    # =========================================================================
    def get_local_rank(self):
        return self.bootstrap_info.local_rank

    def get_local_size(self):
        return self.bootstrap_info.local_size

    def get_global_rank(self):
        if self.torchcomm is not None:
            return self.torchcomm.get_rank()
        return self.bootstrap_info.global_rank

    def get_world_size(self):
        if self.torchcomm is not None:
            return self.torchcomm.get_size()
        return self.bootstrap_info.world_size

    def get_group_rank(self, group):
        if group is not None and hasattr(group, "get_rank"):
            return group.get_rank()
        return self.get_global_rank()

    def get_group_size(self, group):
        if group is not None and hasattr(group, "get_size"):
            return group.get_size()
        return self.get_world_size()

    def get_device(self):
        """Get current device"""
        dev_str = self.commsParams.device
        my_dev = torch.device(dev_str)
        if dev_str == "cuda":
            # explicitly select the device ordinal based on the local rank
            ordinal = self.get_local_rank()
            if self.get_local_rank() == -1:
                logger.warning(
                    "Cannot determine device ordinal since LOCAL_RANK is -1. Try GPU 0 and continue. "
                )
                ordinal = 0
            my_dev = torch.device(f"cuda:{ordinal}")
        elif dev_str != "cpu":
            # sanity check, such error should be caught when parsing arguments
            raise ValueError(f"{dev_str} is not a valid device option")
        return my_dev

    def get_hw_device(self):
        return self.get_device()

    def get_default_group(self):
        return self.torchcomm

    def get_groups(self):
        return self.groups

    def set_device(self, local_rank, global_rank):
        """Set current device"""
        dev_str = self.commsParams.device
        if dev_str.startswith("cuda"):
            if local_rank >= torch.cuda.device_count():
                raise ValueError(
                    f"Insufficient #GPUs: available {torch.cuda.device_count()} requested {local_rank}"
                )
            torch.cuda.set_device(local_rank)

        logger.info(
            "rank %s set torch device to %s:%s", global_rank, dev_str, local_rank
        )

    # =========================================================================
    # Stream Management
    # =========================================================================
    def get_new_stream(self):
        """Get/allocate a new stream."""
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
        """switch to a new stream and return the current stream"""
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
        if device is not None and device.type == "cuda":
            # if the stream is None, sync on the current default stream
            cur_stream = (
                stream
                if stream is not None
                else torch.cuda.current_stream(device=device)
            )
            cur_stream.synchronize()
        else:
            # no stream available, do nothing
            pass

    def tensor_list_to_numpy(self, tensorList):
        if isinstance(tensorList, list):
            tensorList = [t.cpu().detach().numpy() for t in tensorList]
        return np.array(tensorList)

    # =========================================================================
    # Init
    # =========================================================================
    def __init__(self, bootstrap_info, commsParams):
        raise NotImplementedError("__init__ not yet implemented")

    def initialize_backend(
        self, master_ip, master_port, backend="ncclx", eager_mode=False
    ):
        raise NotImplementedError("initialize_backend not yet implemented")

    def initialize_groups(self, backend="ncclx"):
        raise NotImplementedError("initialize_groups not yet implemented")

    def benchmark_comms(self, benchTime, commsParams):
        raise NotImplementedError("benchmark_comms not yet implemented")

    def __del__(self):
        raise NotImplementedError("__del__ not yet implemented")
