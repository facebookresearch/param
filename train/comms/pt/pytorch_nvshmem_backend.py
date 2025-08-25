# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)
import torch
import torch.distributed._symmetric_memory as symm_mem
from param_bench.train.comms.pt.pytorch_dist_backend import PyTorchDistBackend


class DummyWork:
    def __init__(self):
        self._work = None

    def wait(self, timeout=None):
        pass


class PyTorchNVShmemBackend(PyTorchDistBackend):
    # Init functions
    def __init__(self, bootstrap_info, commsParams):
        super().__init__(bootstrap_info, commsParams)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    def alloc_empty(self, sizeArr, curRankDevice, dtype):
        return symm_mem.empty(sizeArr, device=curRankDevice, dtype=dtype)

    def alloc_random(
        self, sizeArr, curRankDevice="cuda", dtype=torch.float32, scaleFactor=1
    ):
        torch_tensor = (
            torch.rand(sizeArr, device=curRankDevice, dtype=dtype) / scaleFactor
        )
        tensor = self.alloc_empty(sizeArr, curRankDevice, dtype)
        tensor.copy_(torch_tensor)
        return tensor

    def all_to_allv(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        ipTensor = (
            collectiveArgs.ipTensor
            if not pair
            else collectiveArgs.ipTensor_pair[pairIdx]
        )
        opTensor = (
            collectiveArgs.opTensor
            if not pair
            else collectiveArgs.opTensor_pair[pairIdx]
        )

        torch.ops.symm_mem.all_to_all_vdev(
            ipTensor,
            opTensor,
            collectiveArgs.in_out_splits,
            collectiveArgs.group.group_name,
        )
        # This is really a dummy async work since the above op is a blocking call
        work = DummyWork()

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(work)

        if retFlag:
            return work

    def all_to_all(self, collectiveArgs, retFlag=False, pair=False, pairIdx=0):
        # This is really a dummy async work since the above op is a blocking call

        group_name = collectiveArgs.group.group_name

        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel_per_peer = 10
        numel = self.get_world_size() * numel_per_peer
        inp = symm_mem.empty(numel, dtype=dtype, device=self.get_device()).fill_(
            self.get_global_rank()
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.get_device()).fill_(-1)

        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        torch.ops.symm_mem.nvshmem_all_to_all(inp, out, group_name)
        work = DummyWork()

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(work)

        if retFlag:
            return work
