#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os

# pytorch
import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.nn as nn
import numpy as np
import sys
import argparse
import os
import json
import subprocess

# custom libraries
#data generation
import dlrm_data_pytorch as dp

#utils
from pytorch_nccl_backend import PyTorchNCCLBackend
import comms_utils

# Global parameters
rowDim = 0
colDim = 1
sparseFrac = 1.0  # PENDING: Should set it to a value which ensures we don't transfer the whole of the embedding table.
my_size = 1


def initializeData(curDevice, backendFuncs, global_rank, topMLP, botMLP, embedLayersDim):
    # Generate data for each device.
    curProcData = {}
    topLayers = []
    botLayers = []
    embedLayers = nn.ModuleList()  # embedLayers = []
    #curDevice = torch.device("cpu")

    for layerIdx, curLayer in enumerate(topMLP):
        #curLayerData = torch.rand(curLayer[rowDim], curLayer[colDim], device=curDevice)
        curLayerData = backendFuncs.alloc_random([curLayer[rowDim], curLayer[colDim]], curDevice, torch.float)
        if(global_rank == 0):
            print("\t Top-Layer-%d data: %s " % (layerIdx, curLayerData[0][0]))
        topLayers.append(curLayerData)

    for layerIdx, curLayer in enumerate(botMLP):
        #curLayerData = torch.rand(curLayer[rowDim], curLayer[colDim], device=curDevice)
        curLayerData = backendFuncs.alloc_random([curLayer[rowDim] , curLayer[colDim]], curDevice, torch.float)
        if(global_rank == 0):
            print("\t Bot-Layer-%d data: %s n: %d m: %d " % (layerIdx, curLayerData[0][0], curLayer[rowDim], curLayer[colDim]))
        botLayers.append(curLayerData)

    host = os.uname()[1]
    for layerIdx, curLayer in enumerate(embedLayersDim):
        #curLayerData = torch.rand(curLayer[rowDim], curLayer[colDim], device=curDevice)
        #print("\t Embed-Layer-%d host: %s data: %s " % (layerIdx, host, curLayerData[0][0]))
        #embedLayers.append(curLayerData)
        curLayerData = backendFuncs.alloc_embedding_tables(curLayer[rowDim], curLayer[colDim], curDevice, torch.float)
        if(layerIdx == 0):
            print("\t Embed-Layer-%d host: %s data: %s n: %d m: %d " % (layerIdx, host, curLayerData.weight.data[0][0], curLayer[rowDim], curLayer[colDim]))
        embedLayers.append(curLayerData)

    curProcData['topLayers'] = topLayers
    curProcData['botLayers'] = botLayers
    curProcData['embedLayers'] = embedLayers
    return curProcData


def apply_emb(lS_o, lS_i, emb_l, mixed_dim=False):
    # WARNING: notice that we are processing the batch at once. We implicitly
    # assume that the data is laid out such that:
    # 1. each embedding is indexed with a group of sparse indices,
    #   corresponding to a single lookup
    # 2. for each embedding the lookups are further organized into a batch
    # 3. for a list of embedding tables there is a list of batched lookups

    ly = []
    for k, sparse_index_group_batch in enumerate(lS_i):
        sparse_offset_group_batch = lS_o[k]

        # embedding lookup
        # We are using EmbeddingBag, which implicitly uses sum operator.
        # The embeddings are represented as tall matrices, with sum
        # happening vertically across 0 axis, resulting in a row vector
        E = emb_l[k]
        V = E(sparse_index_group_batch, sparse_offset_group_batch)

        ly.append(V)

    # print(ly)
    if(mixed_dim):
        ly = torch.cat(ly, dim=1)
    else:
        ly = torch.stack(ly)  # avs
    return ly


def get_split_lengths_by_len(n, global_rank, world_size):
    k, m = divmod(n, world_size)
    if m == 0:
        splits = [k] * world_size
        my_len = k
    else:
        splits = [(k + 1) if i < m else k for i in range(world_size)]
        my_len = splits[global_rank]
    return (my_len, splits)


def get_my_slice(n, global_rank, world_size):
    k, m = divmod(n, world_size)
    return slice(
        global_rank * k + min(global_rank, m), (global_rank + 1) * k + min(global_rank + 1, m), 1
    )


def create_mlp(global_rank, ln):
    # build MLP layer by layer
    if(global_rank == 0):
        print("\n\t Creating new set of MLP layers..")
    mlp_layers = []
    for i in range(0, ln.size - 1):
        n = ln[i]
        m = ln[i + 1]
        if(global_rank == 0):
            print("\t MLP layer: %d size: %d x %d " % (i, n, m))
        mlp_layers.append([m, n])  # as per DLRM benchmark dlrm_s_pytroch.py
    return mlp_layers


def create_emb(global_rank, local_emb_dims, ln):
    embed_layers = []
    for i in range(0, ln.size):
        n = ln[i]
        m = local_emb_dims[i]
        if(global_rank == 0):
            if(i % 50 == 0):
                print("\t Embedding layer: %d size: %d x %d " % (i, n, m))
        embed_layers.append([n, m])  # as per DLRM benchmark dlrm_s_pytroch.py
    return embed_layers


def dp_sparse_data(lS_o, lS_i, local_emb_slice):
    """ Just get the slice based on the rank """

    lS_o = lS_o[local_emb_slice]
    lS_i = lS_i[local_emb_slice]

    return lS_o, lS_i

# All-to-all function, borrowed from extended_distribtued HPC trainer code.


class All2AllInfo(object):
    pass


class All2All_Scatter_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        # logger.info("All2All_Scatter_Wait:forward")
        ctx.a2ai = myreq.a2ai
        #for r in myreq.req:
        #    r.wait()
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        import torch.distributed as dist
        global myreq
        # logger.info("All2All_Scatter_Wait:backward")
        assert len(grad_output) == my_size
        scatter_list = [t.contiguous() for t in grad_output]
        a2ai = ctx.a2ai
        mb_split_lengths = a2ai.gNS if a2ai.gNS else a2ai.lN
        # emb_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        grad_input = grad_output[0].new_empty([a2ai.N, a2ai.E * a2ai.lS])
        gather_list = list(grad_input.split(mb_split_lengths, dim=0))
        req_list = []
        for i in range(my_size):
            req = dist.gather(
                scatter_list[i],
                gather_list if i == my_rank else [],
                dst=i,
                async_op=True,
            )
            req_list.append(req)
        myreq.req = req_list
        myreq.tensor = grad_input
        return grad_output


class Request(object):
    def __init__(self):
        self.req = None
        self.tensor = None
        self.WaitFunction = All2All_Scatter_Wait
        #self.WaitFunction = None

    def wait(self):
        ret = self.WaitFunction.apply(*self.tensor)
        self.req = None
        self.tensor = None
        return ret


class All2Allv_Req(Function):
    @staticmethod
    def forward(ctx, a2ai, *inputs):
        global myreq
        global backendFuncs
        global collectiveArgs
        global measured_regions
        global commDetails

        # logger.info("All2Allv_Req:forward")
        mb_split_lengths = [m * sum(a2ai.E) for m in a2ai.gNS]
        emb_split_lengths = [a2ai.lN * e for e in a2ai.gSS]
        input = torch.cat(inputs, dim=1).view([-1])
        output = input.new_empty(sum(emb_split_lengths))

        cur_iter_memory = input.element_size() * input.nelement()
        measured_regions['fwd_a2a']['memory'].append(cur_iter_memory)
        commDetails.append(
            {
                "comms" : "all_to_all",
                "msg_size" : cur_iter_memory,
                "in_split" : mb_split_lengths,
                "out_split" : emb_split_lengths,
                "dtype" : str(input.dtype),
            }
        )

        if(1):  # with record_function("## alltoall_bwd_single ##"):
            #print("\t emb_split_lengths: %s mb_split_lengths: %s " % (emb_split_lengths, mb_split_lengths))
            collectiveArgs.opTensor = output
            collectiveArgs.ipTensor = input
            collectiveArgs.opTensor_split = emb_split_lengths
            collectiveArgs.ipTensor_split = mb_split_lengths
            collectiveArgs.asyncOp = True

            backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)  # Adding it to ensure we isolate time for fwd-pass all2all right.
            collectiveArgs.timers['fwd_a2a_start'] = time.monotonic()
            req = backendFuncs.all_to_allv(collectiveArgs, retFlag=True)

        myreq.req = req
        myreq.tensor = []
        myreq.tensor.append(output)
        myreq.tensor = tuple(myreq.tensor)
        a2ai.mb_split_lengths = mb_split_lengths
        a2ai.emb_split_lengths = emb_split_lengths
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        global collectiveArgs

        backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)  # Making sure operation is definitively finished.
        collectiveArgs.timers['bwd_a2a_end'] = time.monotonic()

        a2ai = ctx.a2ai
        myreq.req.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.view([a2ai.N, -1]).split(a2ai.E, dim=1)
        grad_inputs = [gin.contiguous() for gin in grad_inputs]
        myreq.tensor = None
        return (None, *grad_inputs)


class All2Allv_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        global collectiveArgs

        myreq.req.wait()
        backendFuncs.complete_accel_ops(collectiveArgs)  # Making sure operation is definitively finished.
        collectiveArgs.timers['fwd_a2a_end'] = time.monotonic()

        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req = None
        myreq.tensor = None
        outputs = tuple(
            out.view([a2ai.lN, -1]) for out in output[0].split(a2ai.emb_split_lengths)
        )
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        global myreq
        global backendFuncs
        global collectiveArgs
        global measured_regions
        global commDetails

        a2ai = ctx.a2ai
        grad_outputs = [gout.contiguous().view([-1]) for gout in grad_outputs]
        grad_output = torch.cat(grad_outputs)
        grad_input = grad_output.new_empty([a2ai.N * sum(a2ai.E)])

        cur_iter_memory = grad_input.element_size() * grad_input.nelement()
        measured_regions['bwd_a2a']['memory'].append(cur_iter_memory)
        commDetails.append(
            {
                "comms" : "all_to_all",
                "msg_size" : cur_iter_memory,
                "in_split" : a2ai.emb_split_lengths,
                "out_split" : a2ai.mb_split_lengths,
                "dtype" : str(grad_output.dtype),
            }
        )

        if(1):  # with record_function("## alltoall_bwd_single ##"):
            collectiveArgs.opTensor = grad_input
            collectiveArgs.ipTensor = grad_output
            collectiveArgs.opTensor_split = a2ai.mb_split_lengths
            collectiveArgs.ipTensor_split = a2ai.emb_split_lengths
            collectiveArgs.asyncOp = True

            backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)  # Adding it to ensure we isolate time for bwd-pass all2all right.
            collectiveArgs.timers['bwd_a2a_start'] = time.monotonic()
            req = backendFuncs.all_to_allv(collectiveArgs, retFlag=True)

        myreq.req = req
        myreq.tensor = grad_input
        return (grad_output,)


def alltoallv(inputs, global_rank, out_split, per_rank_split_lengths):
    global myreq
    N, _ = inputs[0].size()
    E = [e.size()[1] for e in inputs]
    a2ai = All2AllInfo()
    a2ai.lS = len(inputs)
    if out_split:
        a2ai.gSS = out_split
    else:
        # all the embs have the same dimension
        a2ai.gSS = [s * E[0] for s in per_rank_split_lengths]
    a2ai.lN, a2ai.gNS = get_split_lengths_by_len(N, global_rank, len(out_split))
    a2ai.E = E
    a2ai.N = N
    a2ai.S = sum(a2ai.gSS) if a2ai.gSS else a2ai.lS * my_size

    All2Allv_Req.apply(a2ai, *inputs)
    myreq.WaitFunction = All2Allv_Wait

    return myreq


####### AlltoAllPooled #############


def recat_pooled_embedding_mixed_dim_grad_out(grad_output, dim_sum_per_rank):
    pass

class All2All_Pooled_Req(Function):
    @staticmethod
    def forward(ctx, global_rank, world_size, myreq, a2ai, input_embeddings):
        global backendFuncs
        global collectiveArgs
        global measured_regions
        global commDetails

        if a2ai.mixed_dim:
            (B_global, D_local_sum) = input_embeddings.shape
            a2ai.D = -1  # won't use
        else:
            (B_global, T_local, D) = input_embeddings.shape
            D_local_sum = T_local * D
            a2ai.D = D
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        B_local = B_global // dist.get_world_size()
        a2ai.B_local = B_local
        assert B_global % dist.get_world_size() == 0

        sharded_input_embeddings = input_embeddings.view(
            world_size, B_local, D_local_sum
        )
        D_global_sum = sum(dim_sum_per_rank)
        sharded_output_embeddings = torch.empty(
            B_local * D_global_sum,
            dtype=input_embeddings.dtype,
            device=input_embeddings.device,
        )
        out_split_sizes = [B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank]

        cur_iter_memory = sharded_input_embeddings.element_size() * sharded_input_embeddings.nelement()
        measured_regions['fwd_a2a']['memory'].append(cur_iter_memory)
        commDetails.append(
            {
                "comms" : "all_to_all",
                "msg_size" : cur_iter_memory,
                "in_split" : [],
                "out_split" : out_split_sizes,
                "dtype" : str(sharded_input_embeddings.dtype),
            }
        )

        if(1):  # with record_function("## alltoall_fwd_single ##"):
            # req = dist.all_to_all_single(
            #     output=sharded_output_embeddings,
            #     input=sharded_input_embeddings,
            #     output_split_sizes=out_split_sizes,
            #     input_split_sizes=None,
            #     async_op=True,
            # )

            collectiveArgs.opTensor = sharded_output_embeddings
            collectiveArgs.ipTensor = sharded_input_embeddings
            collectiveArgs.opTensor_split = out_split_sizes
            collectiveArgs.ipTensor_split = None
            collectiveArgs.asyncOp = True

            backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)  # Adding it to ensure we isolate time for fwd-pass all2all right.
            collectiveArgs.timers['fwd_a2a_start'] = time.monotonic()
            req = backendFuncs.all_to_allv(collectiveArgs, retFlag=True)

        assert (
            sum(B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank)
            == B_local * D_global_sum
        )

        myreq.req = req
        myreq.tensor = (sharded_output_embeddings,)
        myreq.a2ai = a2ai
        myreq.wait_function = All2All_Pooled_Wait
        ctx.myreq = myreq
        ctx.mixed_dim = a2ai.mixed_dim
        return myreq.tensor

    @staticmethod
    def backward(ctx, *unused):
        global collectiveArgs

        myreq = ctx.myreq
        myreq.req.wait()
        backendFuncs.complete_accel_ops(collectiveArgs)  # Making sure operation is definitively finished.
        collectiveArgs.timers['bwd_a2a_end'] = time.monotonic()

        myreq.req = None
        grad_output = myreq.tensor
        if ctx.mixed_dim:
            (W, B_local, D_local_sum) = grad_output.shape
            grad_input = grad_output.view(W * B_local, D_local_sum)
        else:
            (W, B_local, T_local, D) = grad_output.shape
            grad_input = grad_output.view(W * B_local, T_local, D)
        myreq.tensor = None
        return (None, None, None, None, grad_input)


class All2All_Pooled_Wait(Function):
    @staticmethod
    #def forward(ctx, global_rank, world_size, myreq, sharded_output_embeddings):
    def forward(ctx, sharded_output_embeddings):
        global myreq
        global collectiveArgs

        myreq.req.wait()
        backendFuncs.complete_accel_ops(collectiveArgs)  # Making sure operation is definitively finished.
        collectiveArgs.timers['fwd_a2a_end'] = time.monotonic()

        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req = None
        myreq.tensor = None
        #ctx.ext_pg = ext_pg
        ctx.myreq = myreq
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        B_local = a2ai.B_local
        mixed_dim = a2ai.mixed_dim

        outputs_by_rank = sharded_output_embeddings.split(
            [B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank]
        )
        if mixed_dim:
            result = torch.cat(
                [output.view(B_local, -1) for output in outputs_by_rank], dim=1
            )
        else:
            D = a2ai.D
            result = torch.cat(
                [output.view(B_local, -1, D) for output in outputs_by_rank], dim=1
            )
        return result

    @staticmethod
    #def backward(ctx, global_rank, world_size, grad_output):
    def backward(ctx, grad_output):
        global backendFuncs
        global collectiveArgs
        global measured_regions
        global commDetails

        myreq = ctx.myreq
        a2ai = ctx.a2ai
        #ext_pg = ctx.ext_pg
        dim_sum_per_rank = a2ai.dim_sum_per_rank

        D_local_sum = dim_sum_per_rank[dist.get_rank()]
        if a2ai.mixed_dim:
            (B_local, D_global_sum) = grad_output.shape
            sharded_grad_input_sizes = (dist.get_world_size(), B_local, D_local_sum)
        else:
            (B_local, T_global, D) = grad_output.shape
            D_global_sum = T_global * D
            grad_output = grad_output.view(B_local, -1)
            T_local = D_local_sum // D
            sharded_grad_input_sizes = (dist.get_world_size(), B_local, T_local, D)
        assert sum(dim_sum_per_rank) == D_global_sum

        sharded_grad_output = recat_pooled_embedding_mixed_dim_grad_out(
            grad_output.contiguous(), dim_sum_per_rank
        )
        sharded_grad_input = torch.empty(
            sharded_grad_input_sizes, device=grad_output.device, dtype=grad_output.dtype
        )
        in_split_sizes = [B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank]

        cur_iter_memory = sharded_grad_output.element_size() * sharded_grad_output.nelement()
        measured_regions['bwd_a2a']['memory'].append(cur_iter_memory)
        commDetails.append(
            {
                "comms" : "all_to_all",
                "msg_size" : cur_iter_memory,
                "in_split" : in_split_sizes,
                "out_split" : [],
                "dtype" : str(sharded_grad_output.dtype),
            }
        )

        if(1):  # with record_function("## alltoall_bwd_single ##"):
            # req = dist.all_to_all_single(
            #     output=sharded_grad_input,
            #     input=sharded_grad_output,
            #     output_split_sizes=None,
            #     input_split_sizes=in_split_sizes,
            #     async_op=True,
            # )
            collectiveArgs.opTensor = sharded_grad_input
            collectiveArgs.ipTensor = sharded_grad_output
            collectiveArgs.opTensor_split = None
            collectiveArgs.ipTensor_split = in_split_sizes
            collectiveArgs.asyncOp = True

            backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)  # Adding it to ensure we isolate time for bwd-pass all2all right.
            collectiveArgs.timers['bwd_a2a_start'] = time.monotonic()
            req = backendFuncs.all_to_allv(collectiveArgs, retFlag=True)

        myreq.req = req
        myreq.tensor = sharded_grad_input
        # Note - this mismatch is by design! We return sharded_grad_output to allow PyTorch shape matching to proceed correctly.
        #return (None, None, sharded_grad_output)
        return (sharded_grad_output)


def alltoall_pooled(global_rank, world_size, a2a_pooled_embs_tensor, dim_sum_per_rank, mixed_dim=False):
    global myreq
    a2ai = All2AllInfo()
    a2ai.dim_sum_per_rank = dim_sum_per_rank
    a2ai.mixed_dim = mixed_dim
    All2All_Pooled_Req.apply(global_rank, world_size, myreq, a2ai, a2a_pooled_embs_tensor)
    myreq.WaitFunction = All2All_Pooled_Wait
    return myreq
####### AlltoAllPooled #############


def _decum(tensor):
    first = torch.unsqueeze(tensor[0], dim=0)
    return torch.cat([first, tensor[1:] - tensor[:-1]])


def calculateLengths(feature_count, offsets, indices):
    lengths_list = []
    indices_list = []
    for curFeature in range(feature_count):
        curFeat_offsets = offsets[curFeature]
        curFeat_indices = indices[curFeature]

        if(len(curFeat_offsets) > 0):
            last_size = len(curFeat_indices) - curFeat_offsets[-1].item()
            # pyre-fixme[16]: `Tensor` has no attribute `roll`.
            curFeat_offsets = _decum(curFeat_offsets.roll(-1, dims=0))
            curFeat_offsets[-1] = last_size

        lengths_list.append(curFeat_offsets)
        indices_list.append(curFeat_indices)

    return (torch.cat(lengths_list), torch.cat(indices_list))


def lengthsToOffsets(lengths, curDevice):
    zero = torch.tensor([0], device=curDevice)
    mlengths = lengths[:-1]
    catTensor = torch.cat([zero, mlengths], dim=0)
    tbl_offsets = torch.cumsum(catTensor, dim=0)
    return tbl_offsets  # torch.tensor([tbl_offsets])


def splitPerTable(lengths, indices, batch_size, num_my_features, world_size, global_rank, curDevice):

    all_feature_lengths = []
    indicesSplitLengths = []
    for _ in range(num_my_features):
        all_feature_lengths.append(torch.empty([0], device=curDevice))

    for r in range(world_size):
        r_offset = num_my_features * batch_size * r
        for f in range(num_my_features):
            cur_feature_lengths = all_feature_lengths[f]
            f_offset = f * batch_size
            start = r_offset + f_offset
            end = start + batch_size
            curSlice = lengths[start: end]
            accum = torch.sum(curSlice)
            indicesSplitLengths.append(accum.view(1, -1))
            #print("\t rank: %d f: %d r: %d f_offset: %d r_offset: %d start: %d end: %d curSlice: %s accum: %s " % (global_rank, f, r, f_offset, r_offset, start, end, curSlice, accum))
            cur_feature_lengths = torch.cat([cur_feature_lengths, lengths[start:end]]).to(dtype=torch.int64)
            all_feature_lengths[f] = cur_feature_lengths

    all_feature_indices = []
    for f in range(num_my_features):
        cur_feature_lengths = all_feature_lengths[f]
        #print("\t rank: %d f: %d cur_feature_lengths: %s " % (global_rank, f, cur_feature_lengths))
        all_feature_indices.append(torch.empty([0], device=curDevice))

    indicesSplitLengths = torch.cat(indicesSplitLengths)
    indicesSplitLengths = indicesSplitLengths.to('cpu')
    #print("\t indicesSplitLengths: %s " % (indicesSplitLengths))
    accum = 0
    for idx, curSplitLen in enumerate(indicesSplitLengths):
        cur_feature = idx % num_my_features
        cur_feature_indices = all_feature_indices[cur_feature]
        curSlice = indices[accum : accum + curSplitLen]
        all_feature_indices[cur_feature] = torch.cat([cur_feature_indices, curSlice]).to(dtype=torch.int64)
        #print("\t cur_feature: %d accum: %s accum+curSplitLne: %s " % (cur_feature, accum, accum + curSplitLen))
        accum = accum + curSplitLen

    offsets = []
    #host = os.uname()[1]
    for f in range(num_my_features):
        cur_feature_indices = all_feature_indices[f]
        cur_feature_lengths = all_feature_lengths[f]
        cur_feature_offsets = lengthsToOffsets(cur_feature_lengths, curDevice)
        offsets.append(cur_feature_offsets)
        #print("\t rank: %d f: %d host: %s lengths: offsets: %s lengths: %s indices.shape: %s " % (global_rank, f, host, cur_feature_offsets.shape, cur_feature_lengths.shape, cur_feature_indices.shape))
        #print("\t rank: %d f: %d \n cur_feature_offsets: %s \n cur_feature_lengths: %s \nhost: %s cur_feature_indices: %s " % (global_rank, f, cur_feature_offsets, cur_feature_lengths, host, cur_feature_indices))

    return (offsets, all_feature_indices)


class SparseFeatures:
    # Should be called AFTER lengths and indices are pushed to device.
    def __init__(self, count, batch_size, offsets, ip_indices, curDevice, global_rank):
        global backendFuncs
        global collectiveArgs

        self.count = count
        self.batch_size = batch_size
        lengths, indices = calculateLengths(count, offsets, ip_indices)
        collectiveArgs.timers['length_calc_end'] = time.monotonic()
        #print("\t rank: %d offsets: %s lengths: %s ip_indices: %s indices: %s " % (global_rank, offsets, lengths, ip_indices, indices))
        self.lengths = lengths.to(curDevice)
        self.indices = indices.to(curDevice)

        backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
        collectiveArgs.timers['mem_push_idx_end'] = time.monotonic()
        #print("\t self.lengths.shape: %s self.indices.shape: %s" % (len(self.lengths), len(self.indices)))

    def get_indices_memory_size(self):
        return self.indices.nelement() * self.element_size()


def SparseDataDist(num_features_per_rank, input_features, global_rank, world_size, timers):
    global backendFuncs
    global collectiveArgs

    if len(num_features_per_rank) == 1:
        return

    batch_size = input_features.batch_size
    device = input_features.lengths.device
    cpu_device = torch.device("cpu")

    #print("\t <SDD> input_features.lengths.shape: %s input_features.indices.shape: %s" % (len(input_features.lengths), len(input_features.indices)))
    #print("\t <SDD> input_features.count: %s input_features.batch_size: %s" % (input_features.count, input_features.batch_size))
    # Start input_lengths_per_feature copy to host.
    input_lengths_per_feature = (
        input_features.lengths.view(input_features.count, input_features.batch_size)
        .sum(dim=1)
        .to(cpu_device, non_blocking=True)
    )

    # Distribute lengths first
    # as we need to know output lengths to indices and weights.
    # Then distribute indices and weights in parallel.
    num_my_features = num_features_per_rank[global_rank]
    output_lengths = torch.empty(
        num_my_features * batch_size * world_size,
        device=device,
        dtype=input_features.lengths.dtype,
    )
    #with record_function("## all2all_data:lengths ##"):
    out_splits = [num_my_features * batch_size] * world_size
    in_splits = [
        num_features * batch_size
        for num_features in num_features_per_rank
    ]
    #print("\t rank: %s len(out): %s len(in): %s in_split: %s out_split: %s " % (global_rank, output_lengths.shape, input_features.lengths.shape, in_splits, out_splits))
    collectiveArgs.opTensor = output_lengths
    collectiveArgs.ipTensor = input_features.lengths
    collectiveArgs.opTensor_split = out_splits
    collectiveArgs.ipTensor_split = in_splits
    collectiveArgs.asyncOp = False
    backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)

    timers['offset_xchg_start'] = time.monotonic()
    backendFuncs.all_to_allv(collectiveArgs)
    backendFuncs.complete_accel_ops(collectiveArgs)
    timers['offset_xchg_end'] = time.monotonic()

    cur_iter_memory = output_lengths.element_size() * output_lengths.nelement()
    measured_regions['offset_xchg']['memory'].append(cur_iter_memory)
    global commDetails
    prev_a2a_details = {
        "comms" : "all_to_all",
        "msg_size" : cur_iter_memory,
        "in_split" : in_splits,
        "out_split" : out_splits,
        "dtype" : str(input_features.lengths.dtype),
    }
    commDetails.append(prev_a2a_details)

    #temp_out_lengths = output_lengths.to(cpu_device)
    #print("\t global_rank: %d temp_out_lengths: %s " % (global_rank, temp_out_lengths))

    # Start alltoall request for 'indices'.
    output_indices = torch.empty(
        output_lengths.sum().item(),
        device=device,
        dtype=torch.int64  # input_features.indices.dtype,
    )
    output_indices_splits = (
        output_lengths.view(world_size, -1)
        .sum(dim=1)
        .to(cpu_device)
        .numpy()
    )

    input_features_splits = []
    feature_offset = 0
    input_lengths_per_feature = input_lengths_per_feature.numpy()
    for feature_count in num_features_per_rank:
        feature_length = sum(
            input_lengths_per_feature[
                feature_offset : feature_offset + feature_count
            ]
        )
        input_features_splits.append(feature_length)
        feature_offset += feature_count

    collectiveArgs.opTensor = output_indices
    collectiveArgs.ipTensor = input_features.indices
    collectiveArgs.opTensor_split = output_indices_splits
    collectiveArgs.ipTensor_split = input_features_splits
    collectiveArgs.asyncOp = False

    backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
    timers['idx_xchg_start'] = time.monotonic()
    backendFuncs.all_to_allv(collectiveArgs)
    backendFuncs.complete_accel_ops(collectiveArgs)
    timers['idx_xchg_end'] = time.monotonic()

    cur_iter_memory = output_indices.element_size() * output_indices.nelement()
    measured_regions['idx_xchg']['memory'].append(cur_iter_memory)
    cur_a2a_details = {
        "comms" : "all_to_all",
        "msg_size" : cur_iter_memory,
        "in_split" : np.array(input_features_splits, dtype=np.int32).tolist(),
        "out_split" : np.array(output_indices_splits, dtype=np.int32).tolist(),
        "dtype" : str(input_features.indices.dtype)
    }
    commDetails.append(cur_a2a_details)

    #if(global_rank == 0):
    offsets, indices = splitPerTable(output_lengths, output_indices, batch_size, num_my_features, world_size, global_rank, device)
    return (offsets, indices)


def getMemSizes(curDeviceData, backendFuncs, collectiveArgs):
    memSizes = {}
    m_topLayer = []
    m_botLayer = []
    for curLayer in curDeviceData['topLayers']:
        collectiveArgs.ipTensor = curLayer
        m_topLayer.append(backendFuncs.get_mem_size(collectiveArgs))

    for curLayer in curDeviceData['botLayers']:
        collectiveArgs.ipTensor = curLayer
        m_botLayer.append(backendFuncs.get_mem_size(collectiveArgs))

    memSizes['top'] = m_topLayer
    memSizes['bot'] = m_botLayer
    return memSizes


def new_printTiming(global_rank, wamrupIters, measuredIters, measured_regions, world_size, curDevice):
    if(measuredIters != 0):
        global collectiveArgs
        global backendFuncs

        #sorted_regions = ['offset_xchg', 'idx_xchg', 'fwd_a2a', 'bwd_a2a', 'bwd_top_ar', 'bwd_bot_ar', 'iter_time']
        all_timers = ['intermed_calc_length', 'mem_push_idx', 'intermed_bef_offset_xchg', 'offset_xchg', 'intermed_btw_offset_idx_xchg',
        'idx_xchg', 'intermed_post_idx_xchg_sparse_dist', 'intermed_emb_lookup_to_a2a_start', 'fwd_a2a', 'intermed_fwd_a2a_grad_push',
        'mem_push_gradients', 'bwd_top_ar', 'intermed_top_ar_end_to_bwd_a2a_start', 'bwd_a2a', 'intermed_bwd_a2a_bot_ar', 'bwd_bot_ar',
        'iter_time', 'iter_data_prep', 'iter_fwd_a2a', 'iter_bwd_top_ar', 'iter_bwd_a2a']

        combined_latency_list = []
        combined_memory_list = []
        for cur_region in all_timers:
            combined_latency_list.append(measured_regions[cur_region]['samples'])
            combined_memory_list.append(measured_regions[cur_region]['memory'])

        #print("\t 0. rank: %d combined_latency_list: %s" % (global_rank, combined_latency_list))
        timeElapsedTensor = torch.tensor(combined_latency_list, device=curDevice)
        tensor_list = [torch.ones_like(timeElapsedTensor) for _ in range(world_size)]
        #print("\t 1. rank: %d before-tensor_list: %s" % (global_rank, tensor_list))
        collectiveArgs.ipTensor = timeElapsedTensor
        collectiveArgs.tensorList = tensor_list
        collectiveArgs.asyncOp = False
        collectiveArgs.dataSize = (
            timeElapsedTensor.nelement() * timeElapsedTensor.element_size()
        )
        collectiveArgs.numElements = timeElapsedTensor.nelement()
        collectiveArgs.waitObj = backendFuncs.all_gather(collectiveArgs, retFlag=True)
        backendFuncs.complete_accel_ops(collectiveArgs)

        memory_tensor = torch.tensor(combined_memory_list, device=curDevice)
        memory_tensor_list = [torch.ones_like(memory_tensor) for _ in range(world_size)]
        collectiveArgs.ipTensor = memory_tensor
        collectiveArgs.tensorList = memory_tensor_list
        collectiveArgs.waitObj = backendFuncs.all_gather(collectiveArgs, retFlag=True)
        backendFuncs.complete_accel_ops(collectiveArgs)

        #print("\t 2. rank: %d after-tensor_list: %s" % (global_rank, tensor_list))
        sum_latency = 0.0
        sum_mean_latency = 0.0
        if(global_rank == 0):
            cpu_tensor_latency = []
            cpu_tensor_memory = []
            for cur_region in range(world_size):
                cpu_tensor_latency.append(tensor_list[cur_region].to('cpu'))
                cpu_tensor_memory.append(memory_tensor_list[cur_region].to('cpu'))

            res_mean_percentiles = []
            res_percentiles = []
            print("\t iters \t region \t memory (B) \t\t Latency(us):min\tp50\tp75\t\tp95")
            for region_idx, cur_region in enumerate(all_timers):
                all_rank_latency = []
                all_rank_memory = []
                all_rank_mean_latency = []
                for cur_rank in range(world_size):
                    cur_rank_latency = cpu_tensor_latency[cur_rank][region_idx]
                    cur_rank_memory = cpu_tensor_memory[cur_rank][region_idx][wamrupIters:]
                    #print("\t rank: %d cur_rank_latency: %s cur_rank_memory: %s " % (cur_rank, cur_rank_latency, cur_rank_memory))
                    all_rank_latency.append(cur_rank_latency)
                    all_rank_memory.append(cur_rank_memory)
                    all_rank_mean_latency.append(torch.mean(cur_rank_latency))
                all_rank_latency = torch.cat(all_rank_latency)
                all_rank_memory = torch.cat(all_rank_memory)
                #print("\t region_idx: %s all_rank_latency: %s all_rank_memory: %s " % (region_idx, all_rank_latency, all_rank_memory))

                latencyAcrossRanks = np.array(all_rank_latency)
                min_lat = torch.min(all_rank_latency)
                p50 = np.percentile(latencyAcrossRanks, 50)
                p75 = np.percentile(latencyAcrossRanks, 75)
                p95 = np.percentile(latencyAcrossRanks, 95)

                mean_latencyAcrossRanks = np.array(all_rank_mean_latency)
                mean_min_lat = min(all_rank_mean_latency)
                mean_p50 = np.percentile(mean_latencyAcrossRanks, 50)
                mean_p75 = np.percentile(mean_latencyAcrossRanks, 75)
                mean_p95 = np.percentile(mean_latencyAcrossRanks, 95)

                memoryAcrossRanks = np.array(all_rank_memory)
                mem_p50 = np.percentile(memoryAcrossRanks, 50)
                if('iter' not in cur_region):
                    sum_latency = sum_latency + p50
                    sum_mean_latency = sum_mean_latency + mean_p50
                res_percentiles_line = "\t%d\t%36s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s" % (measuredIters, cur_region, '%d' % (mem_p50), '%.3f' % (min_lat),
                '%.3f' % (p50), '%.3f' % (p75), '%.3f' % (p95), '%.3f' % (sum_latency))
                res_mean_percentiles_line = "\t%d\t%36s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s" % (measuredIters, cur_region, '%d' % (mem_p50), '%.3f' % (mean_min_lat),
                '%.3f' % (mean_p50), '%.3f' % (mean_p75), '%.3f' % (mean_p95), '%.3f' % (sum_mean_latency))

                res_percentiles.append(res_percentiles_line)
                res_mean_percentiles.append(res_mean_percentiles_line)

            for cur_line in res_percentiles:
                if('iter_time' in cur_line):
                    print("\n")
                print(cur_line)

            print("\t%d\t%36s\t%12s\t%12s\t%12s" % (measuredIters, "total_time", "0.0", "0.0", '%.3f' % (sum_latency)))
            print("\n\n -----------------------------------------------------------------------------------------------------------------------------\n\n")
            for cur_line in res_mean_percentiles:
                if('iter_time' in cur_line):
                    print("\n")
                print(cur_line)
            print("\t%d\t%36s\t%12s\t%12s\t%12s" % (measuredIters, "total_time", "0.0", "0.0", '%.3f' % (sum_mean_latency)))
            print("\n\n -----------------------------------------------------------------------------------------------------------------------------\n\n")


def printTiming(global_rank, wamrupIters, measuredIters, measured_regions, world_size, curDevice):
    if(measuredIters != 0):
        new_printTiming(global_rank, wamrupIters, measuredIters, measured_regions, world_size, curDevice)
        return
        print("\n\n")
        all_timers = ['intermed_calc_length', 'mem_push_idx', 'intermed_bef_offset_xchg', 'offset_xchg', 'intermed_btw_offset_idx_xchg',
        'idx_xchg', 'intermed_post_idx_xchg_sparse_dist', 'intermed_emb_lookup_to_a2a_start', 'fwd_a2a', 'intermed_fwd_a2a_grad_push',
        'mem_push_gradients', 'bwd_top_ar', 'intermed_top_ar_end_to_bwd_a2a_start', 'bwd_a2a', 'intermed_bwd_a2a_bot_ar', 'bwd_bot_ar',
        'iter_time', 'iter_data_prep', 'iter_fwd_a2a', 'iter_bwd_top_ar', 'iter_bwd_a2a']
        regions = all_timers
        if(global_rank == 0):
            print("\t iters \t region \t memory (B) \t avg_time (us)")
            for cur_region in regions:
                accuml_time = sum(measured_regions[cur_region]['samples'])
                avg_time = accuml_time // measuredIters
                accuml_memory = sum(measured_regions[cur_region]['memory'][wamrupIters:])
                avg_memory = accuml_memory // measuredIters
                print("\t%d\t%36s\t%12s\t%12s" % (measuredIters, cur_region, '%d' % (avg_memory), '%.3f' % (avg_time)))


def resetTimers(timers):
    """ Serves as both a way to add a timer and reset when it is already present """
    timers['iter_start'] = 0.0

    timers['length_calc_end'] = 0.0
    timers['mem_push_idx_end'] = 0.0

    timers['offset_xchg_start'] = 0.0
    timers['offset_xchg_end'] = 0.0
    timers['idx_xchg_start'] = 0.0
    timers['idx_xchg_end'] = 0.0

    timers['bef_emb_lookup'] = 0.0
    timers['fwd_a2a_start'] = 0.0
    timers['fwd_a2a_end'] = 0.0
    timers['grad_push_start'] = 0.0

    timers['bwd_top_ar_start'] = 0.0
    timers['bwd_top_ar_end'] = 0.0
    timers['bwd_a2a_start'] = 0.0
    timers['bwd_a2a_end'] = 0.0
    timers['bwd_bot_ar_start'] = 0.0
    timers['bwd_bot_ar_end'] = 0.0


def configure_regions(measured_regions, region_name, start_timer, end_timer):
    """ Add a new measure-region, by specifying the timers """
    measured_regions[region_name] = {}
    measured_regions[region_name]['start'] = start_timer
    measured_regions[region_name]['end'] = end_timer
    measured_regions[region_name]['samples'] = []
    measured_regions[region_name]['memory'] = []


def intermed_region_memory(measured_regions, timers):
    intermed_regions = ['intermed_calc_length', 'mem_push_idx', 'intermed_bef_offset_xchg', 'intermed_btw_offset_idx_xchg', 'intermed_post_idx_xchg_sparse_dist',
        'intermed_emb_lookup_to_a2a_start', 'intermed_fwd_a2a_grad_push', 'mem_push_gradients', 'intermed_top_ar_end_to_bwd_a2a_start', 'intermed_bwd_a2a_bot_ar',
        'iter_time', 'iter_data_prep', 'iter_fwd_a2a', 'iter_bwd_top_ar', 'iter_bwd_a2a']

    for cur_region in intermed_regions:
        measured_regions[cur_region]['memory'].append(0)


def compute_times(measured_regions, timers):
    #print("\t timers: %s " % (timers))
    for cur_region in measured_regions:
        start_time = timers[measured_regions[cur_region]['start']]
        end_time = timers[measured_regions[cur_region]['end']]
        #print ("\t cur_region: %s end_time: %.3f start_time: %.3f " % (cur_region, end_time, start_time))
        time_spent = (end_time - start_time) * 1e6  # nanoseconds
        measured_regions[cur_region]['samples'].append(time_spent)
    resetTimers(timers)


def initializeTimers():
    timers = {}
    resetTimers(timers)
    global measured_regions
    measured_regions = {}

    configure_regions(measured_regions, 'intermed_calc_length', 'iter_start', 'length_calc_end')
    configure_regions(measured_regions, 'mem_push_idx', 'length_calc_end', 'mem_push_idx_end')
    configure_regions(measured_regions, 'intermed_bef_offset_xchg', 'mem_push_idx_end', 'offset_xchg_start')
    configure_regions(measured_regions, 'offset_xchg', 'offset_xchg_start', 'offset_xchg_end')
    configure_regions(measured_regions, 'intermed_btw_offset_idx_xchg', 'offset_xchg_end', 'idx_xchg_start')
    configure_regions(measured_regions, 'idx_xchg', 'idx_xchg_start', 'idx_xchg_end')
    configure_regions(measured_regions, 'intermed_post_idx_xchg_sparse_dist', 'idx_xchg_end', 'bef_emb_lookup')

    configure_regions(measured_regions, 'intermed_emb_lookup_to_a2a_start', 'bef_emb_lookup', 'fwd_a2a_start')
    configure_regions(measured_regions, 'fwd_a2a', 'fwd_a2a_start', 'fwd_a2a_end')
    configure_regions(measured_regions, 'intermed_fwd_a2a_grad_push', 'fwd_a2a_end', 'grad_push_start')

    configure_regions(measured_regions, 'mem_push_gradients', 'grad_push_start', 'bwd_top_ar_start')
    configure_regions(measured_regions, 'bwd_top_ar', 'bwd_top_ar_start', 'bwd_top_ar_end')

    configure_regions(measured_regions, 'intermed_top_ar_end_to_bwd_a2a_start', 'bwd_top_ar_end', 'bwd_a2a_start')
    configure_regions(measured_regions, 'bwd_a2a', 'bwd_a2a_start', 'bwd_a2a_end')
    configure_regions(measured_regions, 'intermed_bwd_a2a_bot_ar', 'bwd_a2a_end', 'bwd_bot_ar_start')

    configure_regions(measured_regions, 'bwd_bot_ar', 'bwd_bot_ar_start', 'bwd_bot_ar_end')
    configure_regions(measured_regions, 'iter_time', 'iter_start', 'bwd_bot_ar_end')

    configure_regions(measured_regions, 'iter_data_prep', 'iter_start', 'bef_emb_lookup')
    configure_regions(measured_regions, 'iter_fwd_a2a', 'iter_start', 'grad_push_start')
    configure_regions(measured_regions, 'iter_bwd_top_ar', 'iter_start', 'bwd_top_ar_end')
    configure_regions(measured_regions, 'iter_bwd_a2a', 'iter_start', 'bwd_bot_ar_start')

    return timers, measured_regions


def per_device_DLRM(mpi_env_params, comms_world_info, expt_config, args):
    global myreq
    global my_size
    global my_rank
    global backendFuncs
    global collectiveArgs
    global commDetails

    collectiveArgs = comms_utils.collectiveArgsHolder()
    backendFuncs = ""
    if(expt_config['backend'] == "pytorch"):  # TODO: change it topytorch-nccl
        backendFuncs = PyTorchNCCLBackend(comms_world_info, expt_config)  # WARNING: expt_config is different from comms_params but using it as a placeholder!
        backendFuncs.initialize_backend(comms_world_info.master_ip, comms_world_info.master_port, backend="nccl")
    else:
        print("\t Input backend: %s not supported! " % (expt_config['backend']))
        sys.exit()

    local_rank, global_rank, world_size, group, curDevice = comms_utils.get_rank_details(backendFuncs)
    backendFuncs.sayHello()
    mConfig = getLayerDimensions(global_rank, world_size, args)

    collectiveArgs.device = curDevice
    collectiveArgs.waitObj = None
    myreq = Request()
    my_size = world_size
    my_rank = global_rank
    collectiveArgs.group = group

    curDeviceData = initializeData(curDevice, backendFuncs, global_rank, mConfig.topMLP, mConfig.botMLP, mConfig.embedLayers)

    host = os.uname()[1]
    memSizes = getMemSizes(curDeviceData, backendFuncs, collectiveArgs)
    timers, measured_regions = initializeTimers()
    collectiveArgs.timers = timers
    commDetails = []

    backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)

    for batchNum, (_, lS_o, lS_i, _) in enumerate(mConfig.train_ld):
        timers['iter_start'] = time.monotonic()
        if(global_rank >= 0):
            if(batchNum == 0):
                print("\n\t ****** Rank: G: %d L: %d host: %s starting new epoch: %d model: %s ***** \n" % (global_rank, local_rank, host, batchNum, args.model))
        #PENDING:
        # 1. What about embeddings backward pass? input-shapes? should update mConfig.a2a_memory based on that.
        # Currently we have indices, offsets for current rank's mini-batch for all the tables
        curIterSparseFeatures = SparseFeatures(mConfig.num_sparse_fea,
                                                expt_config['mini_batch_size'],
                                                lS_o,
                                                lS_i,
                                                curDevice, global_rank)
        # Exchange it among all the ranks, so that we have indices and offsets of only the tables we have but for the global mini-batch, not just local mini-batch.s
        g_offsets, g_indices = SparseDataDist(mConfig.n_emb_per_rank, curIterSparseFeatures, global_rank, world_size, timers)

        # Begin with reading the embedding table.
        backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
        timers['bef_emb_lookup'] = time.monotonic()
        ly = apply_emb(g_offsets, g_indices, curDeviceData['embedLayers'], mixed_dim=mixedDimFlag)
        #print("\t Rank: %d mixedDimFlag: %s ly.size(): %s " % (global_rank, mixedDimFlag, ly.size()))

        # Start with fwd pass all-to-all, this is blocking communication.
        if(mixedDimFlag):
            a2a_req = alltoall_pooled(global_rank, world_size, ly, mConfig.dims_sum_per_rank, mixed_dim=mixedDimFlag)
        else:
            a2a_req = alltoallv(ly, global_rank, mConfig.dims_sum_per_rank, mConfig.n_emb_per_rank)
        B = a2a_req.wait()

        # Preparing backward pass data, the time for this is NOT accounted.
        tempB = ""
        if(mixedDimFlag):
            tempB = B
        else:
            tempB = torch.cat(B, dim=1)
        collectiveArgs.timers['grad_push_start'] = time.monotonic()
        C = tempB
        backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)  # else data won't actually be moved, evidently!

        if(args.perf_debug):
            backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
            backendFuncs.barrier(collectiveArgs)

        #back-prop: top layer, non-blocking between the top-layers.
        timers['bwd_top_ar_start'] = time.monotonic()
        for curLayerIdx in range(len(curDeviceData['topLayers'])):
            collectiveArgs.ipTensor = curDeviceData['topLayers'][curLayerIdx]
            collectiveArgs.asyncOp = True
            collectiveArgs.op = backendFuncs.get_reduce_op('sum')
            backendFuncs.all_reduce(collectiveArgs)
            commDetails.append(
                {
                    "comms" : "all_reduce",
                    "msg_size" : curDeviceData['topLayers'][curLayerIdx].nelement() * curDeviceData['topLayers'][curLayerIdx].element_size(),
                    "dtype" : str(curDeviceData['topLayers'][curLayerIdx].dtype),
                }
            )
        backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
        timers['bwd_top_ar_end'] = time.monotonic()

        if(args.perf_debug):
            backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
            backendFuncs.barrier(collectiveArgs)

        # back-prop: embedding update, blocking, since we are waiting for it to complete.
        tempB.backward(C)

        if(args.perf_debug):
            backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
            backendFuncs.barrier(collectiveArgs)

        #back-prop: bottom layer, non-blocking between the layers.
        timers['bwd_bot_ar_start'] = time.monotonic()
        for curLayerIdx in range(len(curDeviceData['botLayers'])):
            collectiveArgs.ipTensor = curDeviceData['botLayers'][curLayerIdx]
            collectiveArgs.asyncOp = True
            collectiveArgs.op = backendFuncs.get_reduce_op('sum')
            backendFuncs.all_reduce(collectiveArgs)
            commDetails.append(
                {
                    "comms" : "all_reduce",
                    "msg_size" : curDeviceData['botLayers'][curLayerIdx].nelement() * curDeviceData['botLayers'][curLayerIdx].element_size(),
                    "dtype" : str(curDeviceData['botLayers'][curLayerIdx].dtype),
                }
            )
        backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
        timers['bwd_bot_ar_end'] = time.monotonic()
        measured_regions['bwd_top_ar']['memory'].append(sum(memSizes['top']))
        measured_regions['bwd_bot_ar']['memory'].append(sum(memSizes['bot']))

        if(batchNum >= expt_config['warmup_batches']):
            compute_times(measured_regions, timers)
        intermed_region_memory(measured_regions, timers)

    if(args.print_comms):
        global model
        folder = str(args.model) + "_np" + str(world_size)
        try:
            subprocess.check_output(["mkdir", "-p", str(folder)], universal_newlines=True)
        except Exception as err:
            print("\t Error: %s while creating directory: %s " % (err, folder))
            pass
        comms_file = str(folder) + "/rank" + str(global_rank) + ".json"
        with open(comms_file, "w") as write_file:
            json.dump(commDetails, write_file)

    #if(global_rank == 0):
    measuredIters = expt_config['numBatches'] - expt_config['warmup_batches']
    printTiming(global_rank, expt_config['warmup_batches'], measuredIters, measured_regions, world_size, curDevice)  # printTiming(global_rank, mConfig.a2a_memory, measuredIters, perIterTime_us, topLTime_us, botLTime_us, fwda2aTime_us, bwda2aTime_us, memSizes)


def read_args():
    parser = argparse.ArgumentParser(
        description="Deep Learning Recommendation Model (DLRM)- communications benchmark"
    )

    parser.add_argument("--warmup-batches", type=int, default=5)  # Number of batches within an epoch.
    parser.add_argument("--backend", type=str, default='pytorch')  # Number of batches within an epoch.
    parser.add_argument("--model", type=str, default='opensource')
    parser.add_argument("--model-path", type=str, default='./short.json')
    # Copied from DLRM benchmark.
    parser.add_argument("--arch-sparse-feature-size", type=int, default=4)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--master-ip", type=str, default='127.0.0.1')  # The master-IP to coordinate.
    parser.add_argument("--master-port", type=int, default=29500)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--round-targets", type=bool, default=False)
    parser.add_argument("--data-generation", type=str, default="random")  # synthetic or random
    parser.add_argument("--rand-data-dist", type=str, default="uniform")  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--num-tpu-cores", type=int, default=1)  # required for TPU
    parser.add_argument("--print-comms", action='store_true')
    parser.add_argument("--perf-debug", action='store_true')
    dtypeMap = {
        'float32' : torch.float32,
        'int32' : torch.int32,
        'float16' : torch.half,
        'float64' : torch.double,
    }
    supportedDtype = list(dtypeMap.keys())
    parser.add_argument("--embed-dtype", type=torch.dtype, default=torch.float32)  # will be overwritten based on args.data_type and dtypeMap.
    parser.add_argument("--embed-data-type", type=str, default='float32')  # The network stack to profile.

    args = parser.parse_args()
    if(args.embed_data_type not in supportedDtype):
        print("\t ERROR: Specified dtype: %d is not one of the supported commstyle: %s" % (args.data_type, str(supportedDtype)))
        sys.exit()

    args.embed_dtype = dtypeMap[args.embed_data_type]
    return args


def get_slice_sparse(global_rank, num_emb_per_rank, world_size):
    if global_rank == 0:
        return slice(0, num_emb_per_rank[0], 1)
    cum_sum = sum(num_emb_per_rank[0:global_rank])
    return slice(cum_sum, cum_sum + num_emb_per_rank[global_rank], 1)


class modelConfig():
    def __init__(self, ipDict):
        self.num_sparse_fea = ipDict['num_sparse_fea']
        self.n_emb_per_rank = ipDict['n_emb_per_rank']
        self.local_emb_slice = ipDict['local_emb_slice']
        self.dims_sum_per_rank = ipDict['dims_sum_per_rank']
        #self.dims = ipDict['dims']
        #self.dims_per_rank = ipDict['dims_per_rank']

        self.ln_top = ipDict['ln_top']
        self.ln_bot = ipDict['ln_bot']

        self.topMLP = ipDict['topMLP']
        self.botMLP = ipDict['botMLP']
        self.embedLayers = ipDict['embedLayers']

        self.train_ld = ipDict['train_ld']
        self.a2a_memory = ipDict['a2a_memory']
        self.a2a_dims = ipDict['a2a_dims']


def getEmbTableDimensions(global_rank, world_size, args):
    local_emb_dims = []
    ln_emb = ""
    ipConfig = {}
    global model
    # Following logic is borrowed from dlrm_s_pytorch.py or opensource dlrm model.
    ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
    n_emb = len(ln_emb)
    _, n_emb_per_rank = get_split_lengths_by_len(n_emb, global_rank, world_size)
    dims_per_rank = []
    local_emb_slice = []
    for i in range(world_size):
        tempArr = get_slice_sparse(i, n_emb_per_rank, world_size)
        p = ln_emb[tempArr]

        if(i == global_rank):
            local_emb_slice = ln_emb[tempArr]
            local_emb_dims = [args.arch_sparse_feature_size for i in p]

        temp = []
        for _ in p:
            temp.append(args.arch_sparse_feature_size)  # PENDING/TODO: Update it based on trainer/feeds model.
        dims_per_rank.append(temp)

        if(global_rank == 0):
            print("\t Rank: %d tempArr: %s p: %s " % (i, tempArr, p))
    #dims = [s for e in dims_per_rank for s in e]
    dims_sum_per_rank = [sum(s) for s in dims_per_rank]

    #print("\t dims_per_rank: %s dims: %s dims_sum_per_rank: %s " % (dims_per_rank, dims, dims_sum_per_rank))
    if(global_rank == 0):
        print("\tdims_sum_per_rank: %s " % (dims_sum_per_rank))

    #Package the parameters to be used by DLRM function # PENDING/TODO: Replace it with a class?
    ipConfig['num_sparse_fea'] = ln_emb.size
    ipConfig['n_emb_per_rank'] = n_emb_per_rank
    ipConfig['local_emb_slice'] = local_emb_slice
    ipConfig['dims_sum_per_rank'] = dims_sum_per_rank
    ipConfig['local_emb_dims'] = local_emb_dims

    return (local_emb_dims,
        ln_emb,
        ipConfig)


def getLayerDimensions(global_rank, world_size, args):
    ### parse command line arguments ###
    (local_emb_dims,
        ln_emb,
        ipConfig) = getEmbTableDimensions(global_rank, world_size, args)
    if(global_rank == 0):
        print("\t ipConfig['num_sparse_fea']: %s " % (ipConfig['num_sparse_fea']))
        print("\t ipConfig['n_emb_per_rank']: %s " % (ipConfig['n_emb_per_rank']))
        print("\t ipConfig['dims_sum_per_rank']: %s " % (ipConfig['dims_sum_per_rank']))

        print("\t ipConfig['local_emb_dims']: %s " % (ipConfig['local_emb_dims']))
    #     print("\t ipConfig['local_emb_slice']: %s " % (ipConfig['local_emb_slice']))

    # Genereously borrowed from public DLRM benchmark.
    # input and target at random
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    n_emb = ln_emb.size
    #print("\t type(ln_emb): %s ln_emb: %s " % (type(ln_emb), ln_emb))
    # Parameters required for determining num_int and creating top-layer dimensions.
    #m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    m_den = ln_bot[0]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )

    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    if(global_rank == 0):
        print("\t ln_top: %s \n\t ln_bot: %s \n\t n_emb: %s " % (ln_top, ln_bot, n_emb))

    embedDataSize = torch.ones([0], dtype=args.embed_dtype).element_size()
    #ipConfig['a2a_memory'] = sum(ipConfig['dims_sum_per_rank']) * args.mini_batch_size * embedDataSize
    ipConfig['a2a_memory'] = ipConfig['dims_sum_per_rank'][global_rank] * args.mini_batch_size * embedDataSize
    ipConfig['a2a_dims'] = [len(ipConfig['local_emb_slice']), args.mini_batch_size, args.arch_sparse_feature_size]  # TODO/PENDING: won't work when sparse_feature_size is different!

    topMLP = create_mlp(global_rank, ln_top)  # Creates actual dimensions of each MLP layer
    botMLP = create_mlp(global_rank, ln_bot)
    ipConfig['ln_top'] = ln_top
    ipConfig['ln_bot'] = ln_bot
    ipConfig['topMLP'] = topMLP
    ipConfig['botMLP'] = botMLP

    #ln_emb = ipConfig['local_emb_slice']  # Local slice is sufficient, similar to what is done for jan_v0 model
    #print("\t rank: %d local_emb_slice: %s local_emb_dims: %s " % (global_rank, ipConfig['local_emb_slice'], local_emb_dims))
    print("\t rank: %d len(local_emb_slice): %s local_emb_dims: %s " % (global_rank, len(ipConfig['local_emb_slice']), len(local_emb_dims)))
    embedLayers = create_emb(global_rank, ipConfig['local_emb_dims'], ipConfig['local_emb_slice'])  # WARNING: Assuming that for allocating memory, we should only use local-emb-slice.
    ipConfig['embedLayers'] = embedLayers

    args.numpy_rand_seed = global_rank
    train_ld = ""
    if args.data_generation in ["random", "synthetic"]:
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)

    ipConfig['train_ld'] = train_ld

    myModelConfig = modelConfig(ipConfig)
    print("\t rank: %s n_emb: %s modelConfig: %s " % (global_rank, n_emb, myModelConfig.dims_sum_per_rank))
    #args.num_batches if args.num_batches > 0 else len(train_ld)
    return myModelConfig  # (topMLP, botMLP, embedLayers, train_ld, n_emb_per_rank, local_emb_slice)


def main():

    mpi_env_params = comms_utils.read_mpi_env_vars()
    args = read_args()  # Since main function was too big according to lint, moved argument parsing into a function of its own.
    args.data_size = mpi_env_params['world_size'] * args.num_batches * args.mini_batch_size
    args.num_workers = mpi_env_params['world_size']  # parameter required by dlrm_data_pytorch
    if(mpi_env_params['global_rank'] == 0):
        print("\t mpi-params: %s" % (mpi_env_params))

    global model
    global mixedDimFlag
    model = args.model
    print("\t rank: %s args.model: %s model: %s " % (mpi_env_params['global_rank'], args.model, model))
    mixedDimFlag = False
    mixedDimFlag = False

    expt_config = {}
    supportedBackends = ["pytorch"]
    if(not (args.backend in supportedBackends)):
        print("\t Input backend: %s is not supported! " % (args.backend))

    # Once layer-dimensions are inferred, we can use the rest of the code (I think!)
    expt_config['numDevices'] = mpi_env_params['world_size']
    expt_config['numBatches'] = args.num_batches  # WARNING: Should ensure that dataSize = int(N) * numDevices * batchSize
    expt_config['numBatchesPerEpoch'] = args.mini_batch_size
    expt_config['dataSize'] = mpi_env_params['world_size'] * expt_config['numBatches'] * expt_config['numBatchesPerEpoch']
    expt_config['embedLayers'] = []  # scaledEmbedLayers
    expt_config['mini_batch_size'] = args.mini_batch_size
    expt_config['arch_sparse_feature_size'] = args.arch_sparse_feature_size
    expt_config['mpi_env_params'] = mpi_env_params
    expt_config['backend'] = args.backend
    expt_config['collective'] = 'all_reduce'  # dummy params for now
    expt_config['warmup_batches'] = args.warmup_batches

    if(mpi_env_params['global_rank'] == 0):
        print("\t expt_config: %s " % (expt_config))
    time.sleep(1)
    comms_world_info = comms_utils.comms_world_info_holder(args.master_ip, args.master_port, args.num_tpu_cores, mpi_env_params)
    per_device_DLRM(mpi_env_params, comms_world_info, expt_config, args)


if __name__ == "__main__":
    main()
