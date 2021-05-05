# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from comms_utils import backendFunctions, collectiveArgsHolder


try:
    from internals import all_to_allv_internal, all_to_all_internal
except ImportError:
    pass


def _downcast(input, bitwidth):
    if bitwidth == 16:
        return input.to(torch.float16)
    elif bitwidth == 8:
        return input.to(torch.int8)
    else:
        raise NotImplementedError("Unsupported bitwidth. Set --bitwidth to 8/16/32")


# a future object or a tensor
# okay to use float32 because a prior check that ensures
# the original dtype is float32.
def _dequantize(obj):
    if obj is None:
        # invoked in a irrelevant rank
        return None
    elif type(obj) == torch.Tensor:
        # only call to() if it is not a float32 tensor
        if obj.dtype != torch.float32:
            return obj.to(torch.float32)
        else:
            return obj
    else:
        resultTensor = obj.value()[0]
        if resultTensor.dtype != torch.float32:
            return resultTensor.to(torch.float32)
        else:
            return resultTensor


class PyTorchDistBackend(backendFunctions):
    def sayHello(self):
        myhost = os.uname()[1]
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        master_ip = self.comms_world_info.master_ip
        device = self.get_device()

        hello_msg = f"[Rank {global_rank:3}] host {myhost}, device: {device}, local_rank: {local_rank} world_size: {world_size}, master_ip: {master_ip}"

        try:
            from mpi4py import MPI
        except ImportError:
            print(hello_msg)
        else:
            # if mpi4py exists, use mpi to collect info and print prettier message :)
            # FIXME: use PG instead?
            comm = MPI.COMM_WORLD

            all_hello_msgs = comm.gather(hello_msg, root=0)
            if global_rank == 0:
                print(all_hello_msgs)

    # Collectives
    def all_reduce(self, collectiveArgs, retFlag=False, pair=False):
        # pair=True mode does not support quantization
        if (collectiveArgs.allreduce_qcomm != 32 and collectiveArgs.allreduce_qcomm > 4
            and collectiveArgs.ipTensor.dtype == torch.float32 and not pair
        ):
            # note: note that quantized is a new tensor
            # that is not collectiveArgs.ipTensor.
            # this means when all_reduce/reduce finished
            # quantized will hold the result instead of collectiveArgs.ipTensor
            # this is intended because we don't want to allocate new buffers
            # every time we call all_reduce (because if we don't, it will be float16 instead of float32).
            # That also means we can't use the output of  quantized all_reduce's for anything other than
            # benchmarking purpose.
            quantized = _downcast(
                collectiveArgs.ipTensor, collectiveArgs.allreduce_qcomm
            )
        else:
            quantized = collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair
        retObj = dist.all_reduce(
            quantized,
            op=collectiveArgs.op,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if collectiveArgs.allreduce_qcomm != 32 and not pair:
            if collectiveArgs.asyncOp:
                retObj = retObj.get_future().then(_dequantize)
            else:
                retObj = _dequantize(quantized)

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def reduce(self, collectiveArgs, retFlag=False, pair=False):
        # pair=True mode does not support quantization
        if collectiveArgs.reduce_qcomm != 32 and not pair:
            assert collectiveArgs.ipTensor.dtype == torch.float32
            quantized = _downcast(
                collectiveArgs.ipTensor, collectiveArgs.allreduce_qcomm
            )
        else:
            quantized = collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair
        retObj = dist.reduce(
            quantized,
            dst=collectiveArgs.srcOrDst,
            op=collectiveArgs.op,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if collectiveArgs.reduce_qcomm != 32 and not pair:
            if collectiveArgs.asyncOp:
                retObj = retObj.get_future().then(_dequantize)
            else:
                retObj = _dequantize(quantized)

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def all_to_all(self, collectiveArgs: collectiveArgsHolder, retFlag=False, pair=False):
        # pair=True mode does not support quantization
        if collectiveArgs.all2all_qcomm and not pair:
            work = all_to_all_internal(collectiveArgs)
        else:
            work = dist.all_to_all_single(
                collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
                collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair,
                None,
                None,
                group=collectiveArgs.group,
                async_op=collectiveArgs.asyncOp,
            )

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(work)

        if retFlag:
            return work

    def all_to_allv(self, collectiveArgs, retFlag=False, pair=False):
        # pair=True mode does not support quantization
        if (
            collectiveArgs.all2all_qcomm
            and collectiveArgs.ipTensor.dtype == torch.float32
            and (
                collectiveArgs.opTensor.nelement() >= collectiveArgs.quan_threshold
                or collectiveArgs.ipTensor.nelement() >= collectiveArgs.quan_threshold
            )
            and not pair
        ):
            work = all_to_allv_internal(collectiveArgs)
        else:
            work = dist.all_to_all_single(
                collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
                collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair,
                collectiveArgs.opTensor_split if not pair else collectiveArgs.opTensor_split_pair,
                collectiveArgs.ipTensor_split if not pair else collectiveArgs.ipTensor_split_pair,
                group=collectiveArgs.group,
                async_op=collectiveArgs.asyncOp,
            )

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(work)

        if retFlag:
            return work

    def all_gather(self, collectiveArgs, retFlag=False, pair=False):
        retObj = dist.all_gather(
            tensor_list=collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
            tensor=collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj


    def reduce_scatter(self, collectiveArgs, retFlag=False, pair=False):
        retObj = dist.reduce_scatter(
            output=collectiveArgs.opTensor,
            input_list=collectiveArgs.ipTensor,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def all_gather_base(self, collectiveArgs, retFlag=False, pair=False):
        retObj = dist._all_gather_base(
            output_tensor=collectiveArgs.opTensor,
            input_tensor=collectiveArgs.ipTensor,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def broadcast(self, collectiveArgs, retFlag=False, pair=False):
        retObj = dist.broadcast(
            tensor=collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
            src=collectiveArgs.srcOrDst,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def send(self, collectiveArgs, dst_rank, retFlag=False, tag=0):
        dist.send(
            tensor=collectiveArgs.ipTensor,
            dst=dst_rank,
            group=collectiveArgs.group,
            tag=tag
        )

    def recv(self, collectiveArgs, src_rank, retFlag=False, tag=0):
        dist.recv(
            tensor=collectiveArgs.opTensor,
            src=src_rank,
            group=collectiveArgs.group,
            tag=tag
        )

    def isend(self, collectiveArgs, dst_rank, retFlag=False, tag=0):
        retObj = dist.isend(
            tensor=collectiveArgs.ipTensor,
            dst=dst_rank,
            group=collectiveArgs.group,
            tag=tag
        )

        collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def irecv(self, collectiveArgs, src_rank, retFlag=False, tag=0):
        retObj = dist.irecv(
            tensor=collectiveArgs.opTensor,
            src=src_rank,
            group=collectiveArgs.group,
            tag=tag
        )

        collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def complete_accel_ops(self, collectiveArgs, initOp=False):
        if initOp is True:
            temp = torch.ones([1], device=collectiveArgs.device)
            dist.all_reduce(temp)
        for waitReq in collectiveArgs.waitObj:
            if waitReq is not None:
                waitReq.wait()
        collectiveArgs.waitObj.clear()

        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        if dev_str == "cuda":
            torch.cuda.synchronize(collectiveArgs.device)

    # retFlag not used
    def complete_single_op(self, collectiveArgs, retFlag=False):
        """ only wait the first op in the queue """
        if len(collectiveArgs.waitObj) > 0:
            waitReq = collectiveArgs.waitObj.pop(0)
            if waitReq is not None:
                waitReq.wait()

            # to ensure GPU collective is completed
            dev_str = (
                self.commsParams["device"]
                if isinstance(self.commsParams, dict)
                else self.commsParams.device
            )
            if dev_str == "cuda":
                torch.cuda.synchronize(collectiveArgs.device)


    def barrier(self, collectiveArgs, name="dummy", retFlag=False):
        retObj = dist.barrier(collectiveArgs.group, async_op=collectiveArgs.asyncOp)

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def sync_barrier(self, collectiveArgs, desc="dummy"):
        self.barrier(collectiveArgs, name=desc)
        self.complete_accel_ops(collectiveArgs)

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
    def get_mem_size(self, collectiveArgs, pair=False):
        _sizeBytes = 0
        # opTensor could be a list of tensor for all_gather/gather, get the aggregated size
        if isinstance(collectiveArgs.opTensor, list):
            _sizeBytes = sum([t.nelement() * t.element_size() for t in collectiveArgs.opTensor])
        #reduce scatter
        elif isinstance(collectiveArgs.ipTensor, list):
            _sizeBytes = sum([t.nelement() * t.element_size() for t in collectiveArgs.ipTensor])
        else:
            _sizeBytes = collectiveArgs.opTensor.nelement() * collectiveArgs.opTensor.element_size()
        if pair:
            if isinstance(collectiveArgs.opTensor_pair, list):
                _sizeBytes = sum([t.nelement() * t.element_size() for t in collectiveArgs.opTensor_pair])
            else:
                _sizeBytes = collectiveArgs.opTensor_pair.nelement() * collectiveArgs.opTensor_pair.element_size()

        return _sizeBytes

    def alloc_random(self, sizeArr, curRankDevice="cuda", dtype=torch.float32, scaleFactor=1.0):
        if dtype in (torch.uint8, torch.int16, torch.int32, torch.long):
            ipTensor = torch.randint(low=0, high=10, size=sizeArr, device=curRankDevice, dtype=dtype)
        else:
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
        """ get current device: 'cpu' or 'cuda' """
        # TODO: this is a temporary workaround; need to unify the type of commsParams in comms and dlrm
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        my_dev = torch.device(dev_str)
        if dev_str == "cuda":
            # explicitly select the device ordinal based on the local rank
            my_dev = torch.device("cuda:%d" % self.get_local_rank())
        elif dev_str != "cpu":
            # sanity check, such error should be catched when parsing arguments
            raise ValueError(f"{dev_str} is not a valid device option")

        return my_dev

    def get_hw_device(self):
        self.get_device()

    def get_default_group(self, world_size):
        # return the world group to always perform collectives on default PG
        return dist.GroupMember.WORLD

    def get_groups(self):
        return self.groups

    def set_device(self):
        """ set current device: 'cpu' or 'cuda' """
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        if dev_str.startswith("cuda"):
            if self.get_local_rank() > torch.cuda.device_count():
                raise ValueError(
                    "Insufficient #GPUs: "
                    f"available {torch.cuda.device_count()} "
                    f"requested {self.get_local_rank()}"
                )
            torch.cuda.set_device(self.get_local_rank())

        logging.info(f"rank {self.get_global_rank()} set torch device to {dev_str}")

    # Init functions
    def __init__(self, comms_world_info, commsParams):
        super().__init__()
        self.comms_world_info = comms_world_info
        self.commsParams = commsParams
        # Add single wait op (Note this is not supported in pytorch_tpu_backend.py now)
        self.collectiveFunc["wait"] = self.complete_single_op

        # Import ucc plugin
        if commsParams.backend == "ucc":
            # try OSS/setup.py
            try:
                import torch_ucc  # noqa
            except ImportError:
                try:
                    from ucc_plugin import initialize_ucc_plugin
                except ImportError:
                    raise RuntimeError("Unable to import initialize_ucc_plugin")
                else:
                    initialize_ucc_plugin(commsParams.backend)

    def initialize_backend(self, master_ip, master_port, backend="gloo"):
        # Set CUDA device before initializing backend
        # Required for backends that don't do lazy initialization, e.g. UCC
        self.set_device()

        global_rank = self.get_global_rank()
        world_size = self.get_world_size()
        # Torch initializaiton
        os.environ["MASTER_ADDR"] = str(master_ip)  # '127.0.0.1'
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(global_rank)

        # default group
        dist.init_process_group(backend, rank=global_rank, world_size=world_size)
        self.groups = []
        self.groups.append(self.get_default_group(self.get_world_size()))

        # non-default groups
        for _ in range(1, self.commsParams.num_pgs):
            pg = dist.new_group(backend=backend)
            self.groups.append(pg)

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
        if dist.is_initialized():
            dist.destroy_process_group()
        pass
