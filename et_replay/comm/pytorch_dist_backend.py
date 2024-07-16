# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from itertools import cycle
from time import sleep
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from et_replay.comm.param_profile import paramProfile
from et_replay.comm.pytorch_backend_utils import backendFunctions, collectiveArgsHolder


try:
    from param_bench.train.comms.pt.fb.internals import (
        all_to_all_internal,
        all_to_allv_internal,
        extend_distributed,
    )

    has_ext_dist = True
except ImportError:
    try:
        # Open-source extend_distributed.py can be found in https://github.com/facebookresearch/dlrm
        import extend_distributed

        has_ext_dist = True
    except ImportError:
        has_ext_dist = False

logger = logging.getLogger(__name__)


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
    elif isinstance(obj, torch.Tensor):
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
    def get_collective_group(self, collectiveArgs):
        if self.use_ext_dist:
            return collectiveArgs.group.my_pg
        else:
            return collectiveArgs.group

    def sayHello(self):
        myhost = os.uname()[1]
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        master_ip = self.bootstrap_info.master_ip
        device = self.get_device()

        hello_msg = f"[Rank {global_rank:3}] host {myhost}, device: {device}, local_rank: {local_rank} world_size: {world_size}, master_ip: {master_ip}"

        self.store_set(f"hello_msg_{global_rank}", hello_msg)
        if global_rank == 0:
            for rank in range(0, world_size):
                rank_hello_msg = self.store_get(f"hello_msg_{rank}").decode()
                print(f"Hello from Rank {rank}: {rank_hello_msg}")

    def store_get(self, key):
        return self.tcp_store.get(key)

    def store_set(self, key, val):
        self.tcp_store.set(key, val)

    # Collectives
    def all_reduce(self, collectiveArgs, retFlag=False, pair=False):
        # pair=True mode does not support quantization
        if (
            collectiveArgs.allreduce_qcomm != 32
            and collectiveArgs.allreduce_qcomm > 4
            and collectiveArgs.ipTensor.dtype == torch.float32
            and not pair
        ):
            # note: note that quantized is a new tensor
            # that is not collectiveArgs.ipTensor.
            # this means when all_reduce/reduce finished
            # quantized will hold the result instead of collectiveArgs.ipTensor
            # this is intended because we don't want to allocate new buffers
            # every time we call all_reduce (because if we don't, it will be float16 instead of float32).
            # That also means we can't use the output of  quantized all_reduce's for anything other than
            # benchmarking purpose.
            with paramProfile(
                timer=collectiveArgs.quant_time,
                description="# PARAM: Allreduce quantization #",
            ):
                quantized = _downcast(
                    collectiveArgs.ipTensor, collectiveArgs.allreduce_qcomm
                )
        else:
            quantized = (
                collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair
            )
        if self.use_ext_dist:
            retObj = collectiveArgs.group.all_reduce(
                tensor=quantized,
                op=collectiveArgs.op,
                async_op=collectiveArgs.asyncOp,
            )  # synchronicity is maintained in runColl
        else:
            retObj = dist.all_reduce(
                quantized,
                op=collectiveArgs.op,
                group=collectiveArgs.group,
                async_op=collectiveArgs.asyncOp,
            )  # synchronicity is maintained in runColl
        if (id(quantized) != id(collectiveArgs.ipTensor)) and not pair:
            if collectiveArgs.asyncOp:
                retObj = retObj.get_future().then(_dequantize)
            else:
                with paramProfile(
                    timer=collectiveArgs.dequant_time,
                    description="# PARAM: Allreduce de-quantization #",
                ):
                    retObj = _dequantize(quantized)

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def reduce(self, collectiveArgs, retFlag=False, pair=False):
        # pair=True mode does not support quantization
        if collectiveArgs.reduce_qcomm != 32 and not pair:
            assert collectiveArgs.ipTensor.dtype == torch.float32
            with paramProfile(
                timer=collectiveArgs.quant_time,
                description="# PARAM: Reduce quantization #",
            ):
                quantized = _downcast(
                    collectiveArgs.ipTensor, collectiveArgs.allreduce_qcomm
                )
        else:
            quantized = (
                collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair
            )

        retObj = dist.reduce(
            quantized,
            dst=collectiveArgs.srcOrDst,
            op=collectiveArgs.op,
            group=self.get_collective_group(collectiveArgs),
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl
        if collectiveArgs.reduce_qcomm != 32 and not pair:
            if collectiveArgs.asyncOp:
                retObj = retObj.get_future().then(_dequantize)
            else:
                with paramProfile(
                    timer=collectiveArgs.dequant_time,
                    description="# PARAM: Reduce de-quantization #",
                ):
                    retObj = _dequantize(quantized)

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def all_to_all(
        self, collectiveArgs: collectiveArgsHolder, retFlag=False, pair=False
    ):
        # pair=True mode does not support quantization
        if collectiveArgs.all2all_qcomm and not pair:
            collectiveArgs.use_ext_dist = self.use_ext_dist
            work = all_to_all_internal(collectiveArgs)
        elif collectiveArgs.num_emb_tables_batched > 0 and self.use_ext_dist:
            work: List[extend_distributed.Request] = []
            dim_sum_per_rank = [
                collectiveArgs.num_emb_tables_batched * collectiveArgs.emb_dim
            ] * collectiveArgs.world_size

            for i in range(collectiveArgs.num_emb_ops):
                pooled_embs = collectiveArgs.emb[i](*collectiveArgs.embRequests[i])
                work += [
                    collectiveArgs.group.alltoall_pooled(
                        pooled_embs.reshape(
                            collectiveArgs.batch_size,
                            -1,
                            collectiveArgs.emb_dim,
                        ),
                        dim_sum_per_rank,
                    )
                ]

            for r in work:
                r.wait()
        else:
            if collectiveArgs.num_emb_tables_batched > 0:
                logger.warn(
                    "Not using batched embedding tables because extend distributed package not in use"
                )

            work = dist.all_to_all(
                collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
                collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair,
                group=self.get_collective_group(collectiveArgs),
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
                collectiveArgs.opTensor.nelement() >= collectiveArgs.quant_threshold
                or collectiveArgs.ipTensor.nelement() >= collectiveArgs.quant_threshold
            )
            and not pair
        ):
            work = all_to_allv_internal(collectiveArgs)
        elif self.use_ext_dist:
            work = collectiveArgs.group.alltoall_single(
                collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
                collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair,
                (
                    collectiveArgs.opTensor_split
                    if not pair
                    else collectiveArgs.opTensor_split_pair
                ),
                (
                    collectiveArgs.ipTensor_split
                    if not pair
                    else collectiveArgs.ipTensor_split_pair
                ),
                async_op=collectiveArgs.asyncOp,
            )
        else:
            work = dist.all_to_all_single(
                collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
                collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair,
                (
                    collectiveArgs.opTensor_split
                    if not pair
                    else collectiveArgs.opTensor_split_pair
                ),
                (
                    collectiveArgs.ipTensor_split
                    if not pair
                    else collectiveArgs.ipTensor_split_pair
                ),
                group=collectiveArgs.group,
                async_op=collectiveArgs.asyncOp,
            )

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(work)

        if retFlag:
            return work

    def all_to_all_single(self, collectiveArgs, retFlag=False, pair=False):
        # does not support quantization
        if collectiveArgs.all2all_qcomm:
            logger.warn("all_to_all_single does not support quantization")
            return

        work = dist.all_to_all_single(
            collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair,
            collectiveArgs.ipTensor if not pair else collectiveArgs.ipTensor_pair,
            group=collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
        )

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(work)

        if retFlag:
            return work

    def all_gather(self, collectiveArgs, retFlag=False, pair=False):
        if self.use_ext_dist:
            retObj = collectiveArgs.group.all_gather(
                tensor_list=(
                    collectiveArgs.opTensor
                    if not pair
                    else collectiveArgs.opTensor_pair
                ),
                tensor=(
                    collectiveArgs.ipTensor
                    if not pair
                    else collectiveArgs.ipTensor_pair
                ),
                async_op=collectiveArgs.asyncOp,
            )
        else:
            retObj = dist.all_gather(
                tensor_list=(
                    collectiveArgs.opTensor
                    if not pair
                    else collectiveArgs.opTensor_pair
                ),
                tensor=(
                    collectiveArgs.ipTensor
                    if not pair
                    else collectiveArgs.ipTensor_pair
                ),
                group=collectiveArgs.group,
                async_op=collectiveArgs.asyncOp,
            )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def gather(self, collectiveArgs, retFlag=False, pair=False):
        if pair:
            ipTensors = collectiveArgs.ipTensor_pair
            opTensors = collectiveArgs.opTensor_pair
        else:
            ipTensors = collectiveArgs.ipTensor
            opTensors = collectiveArgs.opTensor

        retObj = dist.gather(
            gather_list=(
                opTensors
                if (collectiveArgs.global_rank == collectiveArgs.srcOrDst)
                else None
            ),
            tensor=ipTensors,
            dst=collectiveArgs.srcOrDst,
            group=self.get_collective_group(collectiveArgs),
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def scatter(self, collectiveArgs, retFlag=False, pair=False):
        if pair:
            ipTensors = collectiveArgs.ipTensor_pair
            opTensors = collectiveArgs.opTensor_pair
        else:
            ipTensors = collectiveArgs.ipTensor
            opTensors = collectiveArgs.opTensor

        retObj = dist.scatter(
            tensor=opTensors,
            scatter_list=(
                ipTensors
                if (collectiveArgs.global_rank == collectiveArgs.srcOrDst)
                else None
            ),
            src=collectiveArgs.srcOrDst,
            group=self.get_collective_group(collectiveArgs),
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def reduce_scatter(self, collectiveArgs, retFlag=False, pair=False):
        if pair:
            ipTensor = collectiveArgs.ipTensor_pair
            opTensor = collectiveArgs.opTensor_pair
        else:
            ipTensor = collectiveArgs.ipTensor
            opTensor = collectiveArgs.opTensor

        if self.use_ext_dist:
            retObj = collectiveArgs.group.reduce_scatter(
                output=opTensor,
                input_list=ipTensor,
                op=collectiveArgs.op,
                async_op=collectiveArgs.asyncOp,
            )  # synchronicity is maintained in runColl
        else:
            retObj = dist.reduce_scatter(
                output=opTensor,
                input_list=ipTensor,
                op=collectiveArgs.op,
                group=collectiveArgs.group,
                async_op=collectiveArgs.asyncOp,
            )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def reduce_scatter_base(self, collectiveArgs, retFlag=False, pair=False):
        if pair:
            ipTensor = collectiveArgs.ipTensor_pair
            opTensor = collectiveArgs.opTensor_pair
        else:
            ipTensor = collectiveArgs.ipTensor
            opTensor = collectiveArgs.opTensor

        retObj = dist.reduce_scatter_tensor(
            output=opTensor,
            input=ipTensor,
            op=collectiveArgs.op,
            group=self.get_collective_group(collectiveArgs),
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def all_gather_base(self, collectiveArgs, retFlag=False, pair=False):
        if pair:
            ipTensor = collectiveArgs.ipTensor_pair
            opTensor = collectiveArgs.opTensor_pair
        else:
            ipTensor = collectiveArgs.ipTensor
            opTensor = collectiveArgs.opTensor

        retObj = dist.all_gather_into_tensor(
            output_tensor=opTensor,
            input_tensor=ipTensor,
            group=self.get_collective_group(collectiveArgs),
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    # Many-to-one pattern
    def incast(self, collectiveArgs):
        if collectiveArgs.global_rank == collectiveArgs.srcOrDst:
            # root receives tensor from each of user-specified source ranks
            for idx, src_rank in enumerate(collectiveArgs.src_ranks):
                retObj = dist.irecv(
                    tensor=collectiveArgs.opTensor[idx],
                    src=src_rank,
                    group=self.get_collective_group(collectiveArgs),
                    tag=0,
                )
                collectiveArgs.waitObj.append(retObj)
            # complete outstanding irecvs if blocking
            if not collectiveArgs.asyncOp:
                self.complete_accel_ops(collectiveArgs, devSync=False)
        elif collectiveArgs.global_rank in collectiveArgs.src_ranks:
            # send local tensor to root
            if collectiveArgs.asyncOp:
                self.isend(collectiveArgs, collectiveArgs.srcOrDst)
            else:
                self.send(collectiveArgs, collectiveArgs.srcOrDst)

    def broadcast(self, collectiveArgs, retFlag=False, pair=False):
        retObj = dist.broadcast(
            tensor=(
                collectiveArgs.opTensor if not pair else collectiveArgs.opTensor_pair
            ),
            src=collectiveArgs.srcOrDst,
            group=self.get_collective_group(collectiveArgs),
            async_op=collectiveArgs.asyncOp,
        )  # synchronicity is maintained in runColl

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

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
        dist.send(
            tensor=collectiveArgs.ipTensor,
            dst=collectiveArgs.dst_rank,
            group=self.get_collective_group(collectiveArgs),
            tag=tag,
        )

    def recv(self, collectiveArgs, retFlag=False, tag=0):
        dist.recv(
            tensor=collectiveArgs.opTensor,
            src=collectiveArgs.src_rank,
            group=self.get_collective_group(collectiveArgs),
            tag=tag,
        )

    def isend(self, collectiveArgs, retFlag=False, tag=0):
        retObj = dist.isend(
            tensor=collectiveArgs.ipTensor,
            dst=collectiveArgs.dst_rank,
            group=self.get_collective_group(collectiveArgs),
            tag=tag,
        )

        collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def irecv(self, collectiveArgs, retFlag=False, tag=0):
        retObj = dist.irecv(
            tensor=collectiveArgs.opTensor,
            src=collectiveArgs.src_rank,
            group=self.get_collective_group(collectiveArgs),
            tag=tag,
        )

        collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def P2POp(self, collectiveArgs, retFlag=False, tag=0):
        if collectiveArgs.collective in ("send", "isend"):
            op = dist.isend
            tensor = collectiveArgs.ipTensor
            peer = collectiveArgs.dst_rank
        elif collectiveArgs.collective in ("recv", "irecv"):
            op = dist.irecv
            tensor = collectiveArgs.opTensor
            peer = collectiveArgs.src_rank
        else:
            raise RuntimeError(f"Unknown operation type {collectiveArgs.collective}")

        req = dist.P2POp(
            op=op,
            tensor=tensor,
            peer=peer,
            group=self.get_collective_group(collectiveArgs),
            tag=tag,
        )

        collectiveArgs.p2pOps.append(req)

        if retFlag:
            return req

    def batch_isend_irecv(self, collectiveArgs, retFlag=False):
        reqs = dist.batch_isend_irecv(collectiveArgs.p2pOps)

        collectiveArgs.p2pOps.clear()

        for req in reqs:
            collectiveArgs.waitObj.append(req)

    def device_sync(self, collectiveArgs):
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        if dev_str == "cuda":
            torch.cuda.synchronize(collectiveArgs.device)

    def complete_accel_ops(self, collectiveArgs, devSync=True):
        for waitReq in collectiveArgs.waitObj:
            if waitReq is not None:
                waitReq.wait()
        collectiveArgs.waitObj.clear()
        collectiveArgs.waitObjIds.clear()

        if devSync:
            self.device_sync(collectiveArgs)

    # retFlag not used
    def complete_single_op(self, collectiveArgs, retFlag=False):
        """only wait on the first op in the queue"""
        if len(collectiveArgs.waitObj) > 0:
            waitReq = collectiveArgs.waitObj.pop(0)
            if waitReq is not None:
                waitReq.wait()

            # to ensure GPU collective is completed
            self.device_sync(collectiveArgs)

    def wait(self, collectiveArgs, retFlag=False):
        # for backwards compatibility, use old wait functionality.
        if len(collectiveArgs.waitObjIds) == 0:
            self.complete_single_op(collectiveArgs)
            return

        """wait on op with the matching reqID"""
        if collectiveArgs.collectiveId in collectiveArgs.waitObjIds:
            waitObj = collectiveArgs.waitObjIds[collectiveArgs.collectiveId]
            if waitObj is not None:
                waitObj.wait()

    def barrier(self, collectiveArgs, name="dummy", retFlag=False):
        my_dev = self.get_device()
        if self.use_ext_dist:
            retObj = collectiveArgs.group.barrier(
                async_op=collectiveArgs.asyncOp,
                device_ids=(
                    [my_dev.index]
                    if dist.get_backend(collectiveArgs.group.my_pg) == "nccl"
                    else None
                ),
            )
        else:
            retObj = dist.barrier(
                collectiveArgs.group,
                async_op=collectiveArgs.asyncOp,
                device_ids=(
                    [my_dev.index]
                    if dist.get_backend(collectiveArgs.group) == "nccl"
                    else None
                ),
            )

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def sync_barrier(self, collectiveArgs, desc="dummy"):
        # ensure all streams have finished outstanding events before calling barrier
        self.complete_accel_ops(collectiveArgs)

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

    def add(self, collectiveArgs):
        collectiveArgs.compOut = torch.add(
            collectiveArgs.compIn1, collectiveArgs.compIn2, alpha=2
        )

    def sub(self, collectiveArgs):
        collectiveArgs.compOut = torch.sub(
            collectiveArgs.compIn1, collectiveArgs.compIn2, alpha=2
        )

    def add_num(self, collectiveArgs):
        collectiveArgs.compOut = torch.add(collectiveArgs.compIn1, 20)

    def sub_num(self, collectiveArgs):
        collectiveArgs.compOut = torch.add(collectiveArgs.compIn1, 20)

    def copy(self, collectiveArgs):
        collectiveArgs.compIn1.copy_(collectiveArgs.compOut)

    def emb_lookup(self, collectiveArgs):
        # If we are using the batched embedding lookup with alltoall, don't do the embedding
        # lookup here, but pool it with the alltoalls in the collective
        if not (
            collectiveArgs.collective == "all_to_all"
            and collectiveArgs.num_emb_tables_batched != -1
            and self.use_ext_dist
        ):
            # Embedding table lookup as compute kernel
            # If forward pass
            if collectiveArgs.direction == "forward":
                for i in range(len(collectiveArgs.embRequests)):
                    (indices, offsets, weights) = collectiveArgs.embRequests[i]
                    collectiveArgs.LookupOut = collectiveArgs.emb[i].forward(
                        indices,
                        offsets,
                        weights,
                    )
            # Otherwise backward pass
            else:
                for i in range(len(collectiveArgs.embRequests)):
                    (indices, offsets, weights) = collectiveArgs.embRequests[i]
                    collectiveArgs.LookupOut.backward(
                        collectiveArgs.grad_output,
                        retain_graph=collectiveArgs.reuseTensors,
                    )

    # Memory related
    def get_mem_size(self, collectiveArgs, pair=False):
        _sizeBytes = 0
        # opTensor could be a list of tensor for all_gather/gather/incast, get the aggregated size
        if isinstance(collectiveArgs.opTensor, list):
            _sizeBytes = sum(
                [t.nelement() * t.element_size() for t in collectiveArgs.opTensor]
            )
        # reduce scatter
        elif isinstance(collectiveArgs.ipTensor, list):
            _sizeBytes = sum(
                [t.nelement() * t.element_size() for t in collectiveArgs.ipTensor]
            )
        # reduce_scatter_base and reduce_scatter_v should use input tensor for total memory size
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

    def clear_memory(self, collectiveArgs):
        del collectiveArgs.ipTensor
        del collectiveArgs.opTensor
        if collectiveArgs.ipTensor_pair is not None:
            del collectiveArgs.ipTensor_pair
            del collectiveArgs.opTensor_pair

        torch.cuda.empty_cache()

    # Getting world-size and other information.
    def get_local_rank(self):
        return self.bootstrap_info.local_rank

    def get_local_size(self):
        return self.bootstrap_info.local_size

    def get_global_rank(self):
        return dist.get_rank()

    def get_world_size(self):
        return dist.get_world_size()

    def get_group_rank(self, group):
        if self.use_ext_dist:
            return dist.get_rank(group.my_pg)
        else:
            return dist.get_rank(group)

    def get_group_size(self, group):
        if self.use_ext_dist:
            return dist.get_world_size(group.my_pg)
        else:
            return dist.get_world_size(group)

    def get_device(self):
        """get current device: 'cpu' or 'cuda'"""
        # TODO: this is a temporary workaround; need to unify the type of commsParams in comms and dlrm
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
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
            # sanity check, such error should be catched when parsing arguments
            raise ValueError(f"{dev_str} is not a valid device option")

        return my_dev

    def get_hw_device(self):
        self.get_device()

    def get_default_group(self):
        if self.use_ext_dist:
            return extend_distributed.from_process_group(dist.GroupMember.WORLD)
        else:
            # return the world group to always perform collectives on default PG
            return dist.GroupMember.WORLD

    def get_groups(self):
        return self.groups

    def get_num_pgs(self):
        return self.num_pgs

    def get_next_group(self):
        return next(self.round_robin_group)

    def set_device(self, local_rank, global_rank):
        """set current device: 'cpu' or 'cuda'"""
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        if dev_str.startswith("cuda"):
            if local_rank > torch.cuda.device_count():
                raise ValueError(
                    "Insufficient #GPUs: "
                    f"available {torch.cuda.device_count()} "
                    f"requested {local_rank}"
                )
            torch.cuda.set_device(local_rank)

        logger.info(f"rank {global_rank} set torch device to {dev_str}:{local_rank}")

    def get_new_stream(self):
        """get/allocate a new stream"""
        if self.commsParams.device == "cuda":
            # TODO: optional to use high-priority stream
            return torch.cuda.Stream(device=self.get_device(), priority=0)
        else:
            return None

    def get_new_event(self, enable_timing=False):
        if self.commsParams.device == "cuda":
            return torch.cuda.Event(enable_timing)
        else:
            return None

    def get_current_stream(self, device: Optional[torch.device]):
        if self.commsParams.device == "cuda":
            return torch.cuda.current_stream(device)
        else:
            return None

    def switch_stream(self, stream, device: Optional[torch.device]):
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
        stream: Optional[torch.cuda.Stream] = None,
        device: Optional[torch.device] = None,
    ):
        """Synchronize a stream with its associated device"""
        if device.type == "cuda":
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

    # Init functions
    def __init__(self, bootstrap_info, commsParams):
        super().__init__()
        self.use_ext_dist = commsParams.use_ext_dist
        self.bootstrap_info = bootstrap_info
        self.commsParams = commsParams
        # extra ops supported (Note these are not supported in pytorch_tpu_backend.py)
        self.collectiveFunc["wait"] = (
            self.wait
        )  # a noop until all collective operations can post a wait operation or specify async vs not async
        self.collectiveFunc["send"] = self.send
        self.collectiveFunc["recv"] = self.recv
        self.collectiveFunc["isend"] = self.isend
        self.collectiveFunc["irecv"] = self.irecv
        self.collectiveFunc["batch_isend_irecv"] = self.batch_isend_irecv
        self.collectiveFunc["pt2pt"] = (
            self.noop
        )  # dummy entry to support pt2pt benchmark

        self.computeFunc["emb_lookup"] = self.emb_lookup
        self.computeFunc["add"] = self.add
        self.computeFunc["sub"] = self.sub
        self.computeFunc["add_num"] = self.add_num
        self.computeFunc["sub_num"] = self.sub_num
        self.computeFunc["copy"] = self.copy

        backend = (
            self.commsParams["backend"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.backend
        )
        # Import ucc plugin
        if backend == "ucc":
            # try OSS/setup.py
            try:
                import torch_ucc  # noqa
            except ImportError:
                try:
                    from ucc_plugin import initialize_ucc_plugin
                except ImportError:
                    raise RuntimeError("Unable to import initialize_ucc_plugin")
                else:
                    initialize_ucc_plugin(backend)
        # Import Fairring
        if backend == "fairring":
            try:
                import fairring  # noqa
            except ImportError:
                raise RuntimeError("Unable to import Fairring")

    def get_new_pg(self, group_ranks, backend):
        if self.use_ext_dist:
            return extend_distributed.new_extend_process_group(
                ranks=group_ranks, backend=backend
            )
        else:
            return dist.new_group(ranks=group_ranks, backend=backend)

    def tensor_list_to_numpy(self, tensorList):
        if isinstance(tensorList, list):
            tensorList = [t.cpu().detach().numpy() for t in tensorList]
        return np.array(tensorList)

    def initialize_tcpstore(self, master_ip, master_port):
        global_rank = self.bootstrap_info.global_rank
        world_size = self.bootstrap_info.world_size
        self.tcp_store = dist.TCPStore(
            master_ip,
            int(master_port),
            world_size,
            is_master=(global_rank == 0),
            use_libuv=True,
        )

    def initialize_backend(
        self, master_ip, master_port, backend="gloo", eager_mode=False
    ):
        # Set CUDA device before initializing backend
        # Required for backends that don't do lazy initialization, e.g. UCC
        self.set_device(self.bootstrap_info.local_rank, self.bootstrap_info.global_rank)

        global_rank = self.bootstrap_info.global_rank
        world_size = self.bootstrap_info.world_size

        if has_ext_dist and self.use_ext_dist:
            extend_distributed.init_distributed(
                rank=global_rank,
                size=world_size,
                backend=backend,
                init_method=self.commsParams.init_method,
            )
            self.tcp_store = extend_distributed.my_store
        else:
            self.use_ext_dist = False

        if self.tcp_store is None:
            # TCP store initializaiton for generic CPU data
            self.initialize_tcpstore(master_ip, master_port)

        if not dist.is_initialized():
            # init default process group if not yet initialized or extend_distributed failed or is disabled
            dist.init_process_group(
                backend,
                rank=global_rank,
                world_size=world_size,
                store=self.tcp_store if self.commsParams.init_method is None else None,
                init_method=self.commsParams.init_method,
                device_id=(
                    torch.device(f"cuda:{self.bootstrap_info.local_rank}")
                    if eager_mode
                    else None
                ),
            )

        # default 1 group, maybe overwritten by user created groups via initialize_groups
        self.groups = {}
        self.groups[0] = self.get_default_group()
        self.num_pgs = len(self.groups)
        self.round_robin_group = cycle(list(self.groups.values()))

    def initialize_groups(self, backend="gloo"):
        groups = {}
        world_size = self.get_world_size()

        # create additional groups
        for pg_id, group_ranks in self.commsParams.groupRanks.items():
            if (
                len(group_ranks) > world_size
            ):  # this means that --auto-shrink is enabled, only use default pg
                groups.clear()
                break
            if (
                len(group_ranks) == world_size
            ):  # this is the default group, it has already been created
                pg = self.get_default_group()
            else:
                pg = self.get_new_pg(group_ranks=group_ranks, backend=backend)
                global_rank = self.get_global_rank()
                if global_rank in group_ranks:
                    logger.info(
                        f"initialize_groups: Rank {global_rank} creates new group pg_id {pg_id} {pg} with {group_ranks}"
                    )
            groups[pg_id] = pg

        # if additional groups are created, overwrite the default groups list
        if len(groups):
            self.groups = groups

        self.num_pgs = len(self.groups)

        self.round_robin_group = cycle(list(self.groups.values()))

    def benchmark_comms(self, benchTime, commsParams):
        index = 0  # used in TPU, where it is not initialized!
        if commsParams.init_only:
            sleep(10)
        else:
            benchTime(index, commsParams, self)
        return

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        pass
