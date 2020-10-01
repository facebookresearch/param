# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from comms_utils import backendFunctions


class PyTorchNCCLBackend(backendFunctions):
    def sayHello(self):
        myhost = os.uname()[1]
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        master_ip = self.comms_world_info.master_ip
        print(
            "\t Running on host: %s g-rank: %d, l-rank: %s world_size: %d master_ip: %s"
            % (myhost, global_rank, local_rank, world_size, master_ip)
        )

    # Collectives
    def all_reduce(self, collectiveArgs, retFlag=False):
        retObj = dist.all_reduce(
            collectiveArgs.ipTensor,
            op=collectiveArgs.op,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if(retFlag):
            return retObj
        else:
            return

    def reduce(self, collectiveArgs, retFlag=False):
        retObj = dist.reduce(
            collectiveArgs.ipTensor,
            dst=collectiveArgs.dst,
            op=collectiveArgs.op,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if retFlag:
            return retObj
        else:
            return

    def all_to_all(self, collectiveArgs, retFlag=False):
        retObj = dist.all_to_all_single(
            collectiveArgs.opTensor,
            collectiveArgs.ipTensor,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if retFlag:
            return retObj
        else:
            return

    def all_to_allv(self, collectiveArgs, retFlag=False):
        retObj = dist.all_to_all_single(
            collectiveArgs.opTensor,
            collectiveArgs.ipTensor,
            collectiveArgs.opTensor_split,
            collectiveArgs.ipTensor_split,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if retFlag:
            return retObj
        else:
            return

    def all_gather(self, collectiveArgs, retFlag=False):
        retObj = dist.all_gather(
            collectiveArgs.tensorList,
            collectiveArgs.ipTensor,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if retFlag:
            return retObj
        else:
            return

    def complete_accel_ops(self, collectiveArgs, initOp=False):
        if initOp is True:
            temp = torch.ones([1], device=collectiveArgs.device)
            dist.all_reduce(temp)
        if collectiveArgs.waitObj is not None:
            collectiveArgs.waitObj.wait()
        torch.cuda.synchronize(collectiveArgs.device)

    def barrier(self, collectiveArgs):
        dist.barrier(collectiveArgs.group)

    def get_reduce_op(self, opName):
        if opName == "sum":
            return dist.ReduceOp.SUM
        elif opName == "max":
            return dist.ReduceOp.MAX
        else:
            return dist.ReduceOp.SUM

    # Compute functions
    def compute_mm(self, collectiveArgs):
        self.gemm(collectiveArgs)

    def gemm(self, collectiveArgs):
        # Matrix multiplication as compute kernel
        collectiveArgs.MMout = torch.mm(collectiveArgs.MMin1, collectiveArgs.MMin2)

    # Memory related
    def get_mem_size(self, collectiveArgs):
        return (
            collectiveArgs.ipTensor.nelement() * collectiveArgs.ipTensor.element_size()
        )

    def alloc_random(self, sizeArr, curRankDevice, dtype, scaleFactor=1.0):
        ipTensor = torch.rand(sizeArr, device=curRankDevice, dtype=dtype)
        if (scaleFactor) != 0:
            ipTensor = ipTensor / scaleFactor
        return ipTensor

    def alloc_embedding_tables(self, n, m, curRankDevice, dtype):
        EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

        W = np.random.uniform(
            low=-(np.sqrt(1 / n)), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)
        # approach 1

        EE.weight.data = torch.tensor(
            W, dtype=dtype, requires_grad=True, device=curRankDevice
        )
        return EE

    def alloc_empty(self, sizeArr, dtype, curRankDevice):
        return torch.empty(sizeArr, device=curRankDevice, dtype=dtype)

    def clear_memory(self):
        torch.cuda.empty_cache()

    # Getting world-size and other information.
    def get_local_rank(self):
        return self.comms_world_info.local_rank

    def get_global_rank(self):
        return self.comms_world_info.global_rank

    def get_world_size(self):
        return self.comms_world_info.world_size

    def get_device(self):
        return torch.device("cuda:%d" % self.get_local_rank())

    def get_group(self, world_size):
        return dist.new_group(i for i in range(world_size))

    # Init functions
    def __init__(self, comms_world_info, commsParams):
        self.comms_world_info = comms_world_info
        self.commsParams = commsParams

    def initialize_backend(self, master_ip, master_port, backend="gloo"):
        global_rank = self.get_global_rank()
        world_size = self.get_world_size()
        # Torch initializaiton
        os.environ["MASTER_ADDR"] = str(master_ip)  # '127.0.0.1'
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(global_rank)

        dist.init_process_group(backend, rank=global_rank, world_size=world_size)

    def benchmark_comms(self):
        self.initialize_backend(
            self.comms_world_info.master_ip,
            self.comms_world_info.master_port,
            self.commsParams.backend,
        )
        index = 0  # used in TPU, where it is not initialized!
        self.commsParams.benchTime(index, self.commsParams, self)
        return

    def __del__(self):
        pass
