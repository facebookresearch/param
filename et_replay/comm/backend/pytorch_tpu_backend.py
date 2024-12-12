#!/usr/bin/env python3
import os

import numpy as np
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm  # pyre-ignore[21]:
import torch_xla.distributed.xla_multiprocessing as xmp  # pyre-ignore[21]:

from et_replay.comm.backend.base_backend import BaseBackend


class PyTorchTPUBackend(BaseBackend):
    def sayHello(self):
        myhost = os.uname()[1]
        device = self.get_device()
        hw_device = self.get_hw_device()
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        master_ip = self.bootstrap_info.master_ip
        print(
            "\tRunning on host: %s g-rank: %d, l-rank: %s world_size: %d master_ip: %s device: %s (%s)"
            % (
                myhost,
                global_rank,
                local_rank,
                world_size,
                master_ip,
                device,
                hw_device,
            )
        )

    # Collectives
    def all_reduce(self, collectiveArgs, retFlag=False):
        retObj = xm.all_reduce(collectiveArgs.op, [collectiveArgs.ipTensor])
        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)
        if retFlag:
            return retObj

    def reduce(self, collectiveArgs, retFlag=False):
        raise NotImplementedError("Func reduce: not implemented yet on TPU")

    def all_to_all(self, collectiveArgs, retFlag=False):
        retObj = xm.all_to_all(collectiveArgs.ipTensor, 0, 0, collectiveArgs.world_size)
        collectiveArgs.opTensor = retObj
        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)
        if retFlag:
            return retObj

    def all_to_allv(self, collectiveArgs, retFlag=False):
        raise NotImplementedError("Func all_to_allv: not implemented yet on TPU")

    def all_gather(self, collectiveArgs, retFlag=False):
        retObj = xm.all_gather(collectiveArgs.ipTensor, dim=0)
        collectiveArgs.opTensor = retObj
        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)
        if retFlag:
            return retObj

    def complete_accel_ops(self, collectiveArgs):
        xm.mark_step()

    def get_reduce_op(self, opName):
        if opName == "sum":
            return xm.REDUCE_SUM
        elif opName == "max":
            return xm.REDUCE_MAX
        else:
            return xm.REDUCE_SUM

    def barrier(self, collectiveArgs, name="world"):
        xm.rendezvous(name)

    # Compute functions
    def compute_mm(self, collectiveArgs):
        self.gemm(collectiveArgs)

    def gemm(self, collectiveArgs):
        collectiveArgs.MMout = torch.mm(collectiveArgs.MMin1, collectiveArgs.MMin2)

    # Memory related
    def get_mem_size(self, collectiveArgs):
        return (
            collectiveArgs.ipTensor.nelement() * collectiveArgs.ipTensor.element_size()
        )

    def alloc_random(self, sizeArr, curRankDevice, dtype, scaleFactor=1.0):
        if dtype in (torch.int32, torch.long):
            ipTensor = torch.randint(
                0, 1000, sizeArr, device=curRankDevice, dtype=dtype
            )
        else:
            ipTensor = torch.rand(sizeArr, device=curRankDevice, dtype=dtype)
        # ipTensor = torch.full(
        #     sizeArr, self.get_global_rank(), device=curRankDevice, dtype=dtype
        # )
        # print("IP: ", ipTensor, self.get_hw_device())
        if (scaleFactor) != 0:
            ipTensor = ipTensor / scaleFactor
        return ipTensor

    def alloc_embedding_tables(self, n, m, curRankDevice, dtype):
        EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

        W = np.random.uniform(
            low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)
        # approach 1

        EE.weight.data = torch.tensor(
            W, dtype=dtype, requires_grad=True, device=curRankDevice
        )
        return EE

    def alloc_empty(self, sizeArr, dtype, curRankDevice):
        return torch.empty(sizeArr, device=curRankDevice, dtype=dtype)

    def clear_memory(self, collectiveArgs):
        pass  # torch.cuda.empty_cache()

    # Getting world-size and other information.
    def get_local_rank(
        self,
    ):
        return xm.get_local_ordinal()

    def get_local_size(
        self,
    ):
        return self.bootstrap_info.local_size

    def get_global_rank(
        self,
    ):
        return xm.get_ordinal()

    def get_world_size(
        self,
    ):
        return xm.xrt_world_size()

    def get_device(
        self,
    ):
        return xm.xla_device()

    def get_hw_device(
        self,
    ):
        return xm._xla_real_device(xm.xla_device())

    def get_default_group(self):
        pass

    def get_groups(self):
        pass

    def tensor_list_to_numpy(self, tensorList):
        tensorList = torch.transpose(tensorList.view(-1, 1), 0, 1)[0]
        return tensorList.cpu().detach().numpy()

    # Init functions
    def __init__(self, bootstrap_info, commsParams):
        self.bootstrap_info = bootstrap_info
        self.commsParams = commsParams

    def initialize_backend(self, master_ip, master_port, backend="gloo"):
        pass

    def benchmark_comms(self, benchTime, commsParams):
        xmp.spawn(
            fn=benchTime,
            args=(commsParams, self),
            nprocs=self.bootstrap_info.num_tpu_cores,
        )
        return

    def __del__(self):
        pass
