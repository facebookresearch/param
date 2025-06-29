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

        torch.ops.symm_mem.nvshmem_all_to_all_vdev(
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
