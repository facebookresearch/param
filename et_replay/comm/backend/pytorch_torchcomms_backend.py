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

import torch
from et_replay.comm.backend.base_backend import BaseBackend, collectiveArgsHolder


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
    def isend(self, collectiveArgs, retFlag=False, tag=0):
        raise NotImplementedError("isend not yet implemented")

    def irecv(self, collectiveArgs, retFlag=False, tag=0):
        raise NotImplementedError("irecv not yet implemented")

    def P2POp(self, collectiveArgs, retFlag=False, tag=0):
        raise NotImplementedError("P2POp not yet implemented")

    def batch_isend_irecv(self, collectiveArgs, retFlag=False):
        raise NotImplementedError("batch_isend_irecv not yet implemented")

    # =========================================================================
    # Sync
    # =========================================================================
    def device_sync(self, collectiveArgs):
        raise NotImplementedError("device_sync not yet implemented")

    def complete_accel_ops(self, collectiveArgs, devSync=True):
        raise NotImplementedError("complete_accel_ops not yet implemented")

    def wait(self, collectiveArgs, retFlag=False):
        raise NotImplementedError("wait not yet implemented")

    def barrier(self, collectiveArgs, name="dummy", retFlag=False):
        raise NotImplementedError("barrier not yet implemented")

    def barrier_all_ranks(self):
        raise NotImplementedError("barrier_all_ranks not yet implemented")

    def sync_barrier(self, collectiveArgs, desc="dummy"):
        raise NotImplementedError("sync_barrier not yet implemented")

    def get_reduce_op(self, opName):
        raise NotImplementedError("get_reduce_op not yet implemented")

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
        raise NotImplementedError("store_get not yet implemented")

    def store_set(self, key, val):
        raise NotImplementedError("store_set not yet implemented")

    # =========================================================================
    # Memory
    # =========================================================================
    def get_mem_size(self, collectiveArgs, pair=False):
        raise NotImplementedError("get_mem_size not yet implemented")

    def alloc_random(
        self,
        sizeArr: list[int],
        curRankDevice="cuda",
        dtype=torch.float32,
        scaleFactor=1.0,
    ):
        raise NotImplementedError("alloc_random not yet implemented")

    def alloc_embedding_tables(self, n, m, curRankDevice, dtype):
        raise NotImplementedError("alloc_embedding_tables not yet implemented")

    def alloc_empty(self, sizeArr, dtype, curRankDevice):
        raise NotImplementedError("alloc_empty not yet implemented")

    def clear_memory(self, collectiveArgs):
        raise NotImplementedError("clear_memory not yet implemented")

    # =========================================================================
    # Rank / World Info
    # =========================================================================
    def get_local_rank(self):
        raise NotImplementedError("get_local_rank not yet implemented")

    def get_local_size(self):
        raise NotImplementedError("get_local_size not yet implemented")

    def get_global_rank(self):
        raise NotImplementedError("get_global_rank not yet implemented")

    def get_world_size(self):
        raise NotImplementedError("get_world_size not yet implemented")

    def get_group_rank(self, group):
        raise NotImplementedError("get_group_rank not yet implemented")

    def get_group_size(self, group):
        raise NotImplementedError("get_group_size not yet implemented")

    def get_device(self):
        raise NotImplementedError("get_device not yet implemented")

    def get_hw_device(self):
        raise NotImplementedError("get_hw_device not yet implemented")

    def get_default_group(self):
        raise NotImplementedError("get_default_group not yet implemented")

    def get_groups(self):
        raise NotImplementedError("get_groups not yet implemented")

    def set_device(self, local_rank, global_rank):
        raise NotImplementedError("set_device not yet implemented")

    # =========================================================================
    # Stream Management
    # =========================================================================
    def get_new_stream(self):
        raise NotImplementedError("get_new_stream not yet implemented")

    def get_new_event(self, enable_timing=False):
        raise NotImplementedError("get_new_event not yet implemented")

    def get_current_stream(self, device: torch.device | None):
        raise NotImplementedError("get_current_stream not yet implemented")

    def switch_stream(self, stream, device: torch.device | None):
        raise NotImplementedError("switch_stream not yet implemented")

    def sync_stream(
        self,
        stream: torch.cuda.Stream | None = None,
        device: torch.device | None = None,
    ):
        raise NotImplementedError("sync_stream not yet implemented")

    def tensor_list_to_numpy(self, tensorList):
        raise NotImplementedError("tensor_list_to_numpy not yet implemented")

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
