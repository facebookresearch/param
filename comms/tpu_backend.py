#!/usr/bin/env python3
import torch
import os
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.nn as nn
import numpy as np
from comms_utils import backendFunctions


class PyTorchTPUBackend(backendFunctions):
    def sayHello(self):
        myhost = os.uname()[1]
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        master_ip = self.comms_world_info.master_ip
        print("\t Running on host: %s g-rank: %d, l-rank: %s world_size: %d master_ip: %s " % (myhost, global_rank, local_rank, world_size, master_ip))

    # Collectives
    def all_reduce(self, collectiveArgs, retFlag=False):
        retObj = xm.all_reduce(collectiveArgs.op, [collectiveArgs.ipTensor])  # group = collectiveArgs.group)  # async_op = True)
        if(retFlag):
            return retObj
        else:
            return

    def reduce(self, collectiveArgs, retFlag=False):
        retObj = xm.reduce(collectiveArgs.op, [collectiveArgs.ipTensor])  # group = collectiveArgs.group)  # async_op = True)
        if(retFlag):
            return retObj
        else:
            return

    def all_to_all(self, collectiveArgs, retFlag=False):
        # the split_size = int(collectiveArgs.numElements/collectiveArgs.world_size) is assumed to ensure equal split and no exception,
        # since all the checks have already been done, as of writing this comment.
        collectiveArgs.opTensor = xm.all_to_all(collectiveArgs.ipTensor, 0, 0, collectiveArgs.world_size)  # , group = collectiveArgs.group)  # async_op = True)

    def all_to_allv(self, collectiveArgs, retFlag=False):
        self.ptTPU_all_to_all(collectiveArgs)

    def complete_accel_ops(self, collectiveArgs, initOp=False):
        if(initOp is True):
            temp = torch.ones([0], device=collectiveArgs.device)
            xm.all_reduce('sum', [temp])
        xm.mark_step()

    def get_reduce_op(self, opName):
        if(opName == 'sum'):
            return 'sum'
        elif(opName == 'max'):
            return 'max'
        else:
            return 'sum'

    # Compute functions
    def compute_mm(self, collectiveArgs):
        pass

    # Memory related
    def get_mem_size(self, collectiveArgs):
        return collectiveArgs.ipTensor.nelement() * collectiveArgs.ipTensor.element_size()

    def alloc_random(self, sizeArr, curRankDevice, dtype, scaleFactor=1.0):
        ipTensor = torch.rand(sizeArr, device=curRankDevice, dtype=dtype)
        if((scaleFactor) != 0):
            ipTensor = ipTensor / scaleFactor
        return ipTensor

    def alloc_embedding_tables(self, n, m, curRankDevice, dtype):
        EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

        W = np.random.uniform(
            low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)
        # approach 1

        EE.weight.data = torch.tensor(W, dtype=dtype, requires_grad=True, device=curRankDevice)
        return EE

    def alloc_empty(self, sizeArr, dtype, curRankDevice):
        return torch.empty(sizeArr, device=curRankDevice, dtype=dtype)

    def clear_memory(self):
        pass  # torch.cuda.empty_cache()

    #Getting world-size and other information.
    def get_local_rank(self,):
        return xm.get_local_ordinal()

    def get_global_rank(self,):
        return xm.get_ordinal()

    def get_world_size(self,):
        return xm.xrt_world_size()

    def get_device(self,):
        return xm.xla_device()

    def get_group(self, world_size):
        pass  # return dist.new_group(i for i in range(world_size))

    # Init functions
    def __init__(self, comms_world_info, commsParams):
        self.commsParams = commsParams

    def initialize_backend(self, master_ip, master_port, backend="gloo"):
        pass

    def benchmark_comms(self):
        self.initialize_backend(self.comms_world_info.master_ip, self.comms_world_info.master_port, self.commsParams.backend)
        #index = 0  # used in TPU, where it is not initialized!
        xmp.spawn(fn=self.commsParams.benchTime, args=(self.commsParams, self), nprocs=self.commsParams.num_tpu_cores)
        return

    def __del__(self):
        pass
