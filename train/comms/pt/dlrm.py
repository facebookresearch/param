#!/usr/bin/env python3

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import logging
import os
import subprocess
import sys

import time

import dlrm_data as dd
import numpy as np

import torch
import torch.nn as nn
from comms_utils import paramCommsBench

from param_bench.train.comms.pt import comms_utils
from pytorch_dist_backend import PyTorchDistBackend
from torch.autograd import Function

logger = logging.getLogger(__name__)


# All-to-all classes/functions
class All2AllInfo:
    pass


class All2All_Scatter_Wait(Function):
    @staticmethod
    def forward(ctx, myreq, *output):
        ctx.a2ai = myreq.a2ai
        ctx.myreq = myreq
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        import torch.distributed as dist

        myreq = ctx.myreq
        my_size = myreq.bench.comm_size
        my_rank = myreq.bench.my_rank
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
        return (None, grad_output)


class Request:
    def __init__(self, bench):
        self.req = None
        self.tensor = None
        self.WaitFunction = All2All_Scatter_Wait
        self.bench = bench

    def wait(self):
        ret = self.WaitFunction.apply(self, *self.tensor)
        self.req = None
        self.tensor = None
        return ret


class All2Allv_Req(Function):
    @staticmethod
    def forward(ctx, myreq, a2ai, *inputs):
        measured_regions = myreq.bench.measured_regions
        commDetails = myreq.bench.commDetails
        collectiveArgs = myreq.bench.collectiveArgs
        backendFuncs = myreq.bench.backendFuncs

        # logger.info("All2Allv_Req:forward")
        mb_split_lengths = [m * sum(a2ai.E) for m in a2ai.gNS]
        emb_split_lengths = [a2ai.lN * e for e in a2ai.gSS]
        input = torch.cat(inputs, dim=1).view([-1])
        output = input.new_empty(sum(emb_split_lengths))

        cur_iter_memory = input.element_size() * input.nelement()
        measured_regions["fwd_a2a"]["memory"].append(cur_iter_memory)
        commDetails.append(
            {
                "comms": "all_to_all",
                "msg_size": cur_iter_memory,
                "in_split": mb_split_lengths,
                "out_split": emb_split_lengths,
                "dtype": str(input.dtype),
            }
        )

        # print("\t emb_split_lengths: %s mb_split_lengths: %s " % (emb_split_lengths, mb_split_lengths))
        collectiveArgs.opTensor = output
        collectiveArgs.ipTensor = input
        collectiveArgs.opTensor_split = emb_split_lengths
        collectiveArgs.ipTensor_split = mb_split_lengths
        collectiveArgs.asyncOp = True

        backendFuncs.sync_barrier(
            collectiveArgs
        )  # Adding it to ensure we isolate time for fwd-pass all2all right.
        collectiveArgs.timers["fwd_a2a_start"] = time.monotonic()
        req = backendFuncs.all_to_allv(collectiveArgs, retFlag=True)

        myreq.req = req
        myreq.tensor = []
        myreq.tensor.append(output)
        myreq.tensor = tuple(myreq.tensor)
        a2ai.mb_split_lengths = mb_split_lengths
        a2ai.emb_split_lengths = emb_split_lengths
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        ctx.myreq = myreq
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        myreq = ctx.myreq
        collectiveArgs = myreq.bench.collectiveArgs
        backendFuncs = myreq.bench.backendFuncs

        backendFuncs.sync_barrier(
            collectiveArgs
        )  # Making sure operation is definitively finished.
        collectiveArgs.timers["bwd_a2a_end"] = time.monotonic()

        a2ai = ctx.a2ai
        myreq.req.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.view([a2ai.N, -1]).split(a2ai.E, dim=1)
        grad_inputs = [gin.contiguous() for gin in grad_inputs]
        myreq.tensor = None
        return (None, None, *grad_inputs)


class All2Allv_Wait(Function):
    @staticmethod
    def forward(ctx, myreq, *output):
        collectiveArgs = myreq.bench.collectiveArgs
        backendFuncs = myreq.bench.backendFuncs

        myreq.req.wait()
        backendFuncs.complete_accel_ops(
            collectiveArgs
        )  # Making sure operation is definitively finished.
        collectiveArgs.timers["fwd_a2a_end"] = time.monotonic()

        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req = None
        myreq.tensor = None
        outputs = tuple(
            out.view([a2ai.lN, -1]) for out in output[0].split(a2ai.emb_split_lengths)
        )
        ctx.myreq = myreq
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        myreq = ctx.myreq
        backendFuncs = myreq.bench.backendFuncs
        collectiveArgs = myreq.bench.collectiveArgs
        measured_regions = myreq.bench.measured_regions
        commDetails = myreq.bench.commDetails

        a2ai = ctx.a2ai
        grad_outputs = [gout.contiguous().view([-1]) for gout in grad_outputs]
        grad_output = torch.cat(grad_outputs)
        grad_input = grad_output.new_empty([a2ai.N * sum(a2ai.E)])

        cur_iter_memory = grad_input.element_size() * grad_input.nelement()
        measured_regions["bwd_a2a"]["memory"].append(cur_iter_memory)
        commDetails.append(
            {
                "comms": "all_to_all",
                "msg_size": cur_iter_memory,
                "in_split": a2ai.emb_split_lengths,
                "out_split": a2ai.mb_split_lengths,
                "dtype": str(grad_output.dtype),
            }
        )

        collectiveArgs.opTensor = grad_input
        collectiveArgs.ipTensor = grad_output
        collectiveArgs.opTensor_split = a2ai.mb_split_lengths
        collectiveArgs.ipTensor_split = a2ai.emb_split_lengths
        collectiveArgs.asyncOp = True

        backendFuncs.sync_barrier(
            collectiveArgs
        )  # Adding it to ensure we isolate time for bwd-pass all2all right.
        collectiveArgs.timers["bwd_a2a_start"] = time.monotonic()
        req = backendFuncs.all_to_allv(collectiveArgs, retFlag=True)

        myreq.req = req
        myreq.tensor = grad_input
        return (None, grad_output)


def _decum(tensor):
    first = torch.unsqueeze(tensor[0], dim=0)
    return torch.cat([first, tensor[1:] - tensor[:-1]])


def calculateLengths(feature_count, offsets, indices):
    lengths_list = []
    indices_list = []
    for curFeature in range(feature_count):
        curFeat_offsets = offsets[curFeature]
        curFeat_indices = indices[curFeature]

        if len(curFeat_offsets) > 0:
            last_size = len(curFeat_indices) - curFeat_offsets[-1].item()
            # pyre-fixme[16]: `Tensor` has no attribute `roll`.
            curFeat_offsets = _decum(curFeat_offsets.roll(-1, dims=0))
            curFeat_offsets[-1] = last_size

        lengths_list.append(curFeat_offsets)
        indices_list.append(curFeat_indices)

    return (torch.cat(lengths_list), torch.cat(indices_list))


def lengthsToOffsets(lengths, curDevice):
    # Opposit of _decum, which is borrowed from trainer code.
    zero = torch.tensor([0], device=curDevice)
    mlengths = lengths[:-1]
    catTensor = torch.cat([zero, mlengths], dim=0)
    tbl_offsets = torch.cumsum(catTensor, dim=0)
    return tbl_offsets  # torch.tensor([tbl_offsets])


class SparseFeatures:
    # This is the interface SparseDataDist expects.
    # Calculate lengths (from offsets), which is required to figure out how to split data between various ranks.
    # Push data (lengths, indices) to device
    def __init__(
        self,
        count,
        batch_size,
        offsets,
        ip_indices,
        curDevice,
        global_rank,
        backendFuncs,
        collectiveArgs,
    ):
        self.count = count
        self.batch_size = batch_size
        lengths, indices = calculateLengths(count, offsets, ip_indices)
        collectiveArgs.timers["length_calc_end"] = time.monotonic()
        self.lengths = lengths.to(curDevice)
        self.indices = indices.to(curDevice)

        backendFuncs.sync_barrier(collectiveArgs)
        collectiveArgs.timers["mem_push_idx_end"] = time.monotonic()

    def get_indices_memory_size(self):
        return self.indices.nelement() * self.element_size()


class modelConfig:
    def __init__(self, ipDict):
        self.num_sparse_fea = ipDict["num_sparse_fea"]
        self.n_emb_per_rank = ipDict["n_emb_per_rank"]
        self.local_emb_slice = ipDict["local_emb_slice"]
        self.dims_sum_per_rank = ipDict["dims_sum_per_rank"]

        self.ln_top = ipDict["ln_top"]
        self.ln_bot = ipDict["ln_bot"]

        self.topMLP = ipDict["topMLP"]
        self.botMLP = ipDict["botMLP"]
        self.embedLayers = ipDict["embedLayers"]

        self.train_ld = ipDict["train_ld"]


class paramDLRM_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rowDim = 0
        self.colDim = 1
        self.sparseFrac = 1.0  # PENDING: Should set it to a value which ensures we don't transfer the whole of the embedding table.

    def initializeData(
        self, curDevice, backendFuncs, global_rank, topMLP, botMLP, embedLayersDim
    ):
        # Generate data for each device.
        curProcData = {}
        topLayers = []
        botLayers = []
        embedLayers = nn.ModuleList()  # embedLayers = []

        for layerIdx, curLayer in enumerate(topMLP):
            curLayerData = backendFuncs.alloc_random(
                [curLayer[self.rowDim], curLayer[self.colDim]], curDevice, torch.float
            )
            if global_rank == 0:
                print("\t Top-Layer-%d data: %s " % (layerIdx, curLayerData[0][0]))
            topLayers.append(curLayerData)

        for layerIdx, curLayer in enumerate(botMLP):
            curLayerData = backendFuncs.alloc_random(
                [curLayer[self.rowDim], curLayer[self.colDim]], curDevice, torch.float
            )
            if global_rank == 0:
                print(
                    "\t Bot-Layer-%d data: %s n: %d m: %d "
                    % (
                        layerIdx,
                        curLayerData[0][0],
                        curLayer[self.rowDim],
                        curLayer[self.colDim],
                    )
                )
            botLayers.append(curLayerData)

        host = os.uname()[1]
        for layerIdx, curLayer in enumerate(embedLayersDim):
            curLayerData = backendFuncs.alloc_embedding_tables(
                curLayer[self.rowDim], curLayer[self.colDim], curDevice, torch.float
            )
            if layerIdx == 0:
                print(
                    "\t Embed-Layer-%d host: %s data: %s n: %d m: %d "
                    % (
                        layerIdx,
                        host,
                        curLayerData.weight.data[0][0],
                        curLayer[self.rowDim],
                        curLayer[self.colDim],
                    )
                )
            embedLayers.append(curLayerData)

        curProcData["topLayers"] = topLayers
        curProcData["botLayers"] = botLayers
        curProcData["embedLayers"] = embedLayers
        return curProcData

    def apply_emb(self, lS_o, lS_i, emb_l, mixed_dim=False):
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

        if mixed_dim:
            ly = torch.cat(ly, dim=1)
        else:
            ly = torch.stack(ly)  # avs
        return ly

    def get_split_lengths_by_len(self, n, global_rank, world_size):
        k, m = divmod(n, world_size)
        if m == 0:
            splits = [k] * world_size
            my_len = k
        else:
            splits = [(k + 1) if i < m else k for i in range(world_size)]
            my_len = splits[global_rank]
        return (my_len, splits)

    def create_mlp(self, global_rank, ln):
        # build MLP layer by layer
        if global_rank == 0:
            print("\n\t Creating new set of MLP layers..")
        mlp_layers = []
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            if global_rank == 0:
                print("\t MLP layer: %d size: %d x %d " % (i, n, m))
            mlp_layers.append([m, n])  # as per DLRM benchmark dlrm_s_pytroch.py
        return mlp_layers

    def create_emb(self, global_rank, local_emb_dims, ln):
        embed_layers = []
        for i in range(0, ln.size):
            n = ln[i]
            m = local_emb_dims[i]
            if global_rank == 0:
                if i % 50 == 0:
                    print("\t Embedding layer: %d size: %d x %d " % (i, n, m))
            embed_layers.append([n, m])  # as per DLRM benchmark dlrm_s_pytroch.py
        return embed_layers

    def get_slice_sparse(self, global_rank, num_emb_per_rank, world_size):
        if global_rank == 0:
            return slice(0, num_emb_per_rank[0], 1)
        cum_sum = sum(num_emb_per_rank[0:global_rank])
        return slice(cum_sum, cum_sum + num_emb_per_rank[global_rank], 1)

    def splitPerTable(
        self,
        lengths,
        indices,
        batch_size,
        num_my_features,
        world_size,
        global_rank,
        curDevice,
    ):
        # By now we have received the lengths, and indices of local-table for the global-batch
        # We need to split the lengths and indices per table

        all_feature_lengths = []
        indicesSplitLengths = []
        for _ in range(num_my_features):
            all_feature_lengths.append(torch.empty([0], device=curDevice))

        # Lengths can be split based on the mini-batch-size (i.e. local-batch-size).
        # The lengths array size should be num_tables * num_ranks * mini_batch_size
        # Data is arranged in the following format --> where each rI-tJ-length represents mini-batch-size number of elements
        # [rank0-table-1-length, r0-t2-length, .., r0-tm-length, rank1-table-1-length, r1-t2-length, .., r1-tm-length, .. rN-t1-length, ..rN-tm-length]
        for r in range(world_size):
            r_offset = num_my_features * batch_size * r
            for f in range(num_my_features):
                cur_feature_lengths = all_feature_lengths[f]
                f_offset = f * batch_size
                start = r_offset + f_offset
                end = start + batch_size
                curSlice = lengths[start:end]  # Get all the lengths per rank
                accum = torch.sum(
                    curSlice
                )  # Computes the sum of the current slice of mini-batch-size elements, which would be used to split the elements per table
                indicesSplitLengths.append(accum.view(1, -1))
                # print("\t rank: %d f: %d r: %d f_offset: %d r_offset: %d start: %d end: %d curSlice: %s accum: %s " % (global_rank, f, r, f_offset, r_offset, start, end, curSlice, accum))
                cur_feature_lengths = torch.cat(
                    [cur_feature_lengths, lengths[start:end]]
                ).to(dtype=torch.int64)
                all_feature_lengths[f] = cur_feature_lengths

        all_feature_indices = []
        for f in range(num_my_features):
            cur_feature_lengths = all_feature_lengths[f]
            # print("\t rank: %d f: %d cur_feature_lengths: %s " % (global_rank, f, cur_feature_lengths))
            all_feature_indices.append(torch.empty([0], device=curDevice))

        indicesSplitLengths = torch.cat(indicesSplitLengths)
        indicesSplitLengths = indicesSplitLengths.to("cpu")
        accum = 0
        # indicesSplitLengths has the num_ranks * num_tables elements.
        # Data arrangement [rank0-table-1-size, r0-t2-size,.. r0-tm-size, r1-t1-size, r1-t2-size, .., r1-tm-size, ..., rN-t1-size, .. rN-tm-size]
        for idx, curSplitLen in enumerate(indicesSplitLengths):
            cur_feature = idx % num_my_features
            cur_feature_indices = all_feature_indices[cur_feature]
            curSlice = indices[
                accum : accum + curSplitLen
            ]  # from lengths, we know that the next curSplitLen elements belong to this table
            all_feature_indices[cur_feature] = torch.cat(
                [cur_feature_indices, curSlice]
            ).to(dtype=torch.int64)
            # print("\t cur_feature: %d accum: %s accum+curSplitLne: %s " % (cur_feature, accum, accum + curSplitLen))
            accum = accum + curSplitLen

        offsets = []
        for f in range(num_my_features):
            cur_feature_indices = all_feature_indices[f]
            cur_feature_lengths = all_feature_lengths[f]
            cur_feature_offsets = lengthsToOffsets(
                cur_feature_lengths, curDevice
            )  # convert length back to offsets.
            offsets.append(cur_feature_offsets)
            # print("\t rank: %d f: %d host: %s lengths: offsets: %s lengths: %s indices.shape: %s " % (global_rank, f, host, cur_feature_offsets.shape, cur_feature_lengths.shape, cur_feature_indices.shape))
            # print("\t rank: %d f: %d \n cur_feature_offsets: %s \n cur_feature_lengths: %s \nhost: %s cur_feature_indices: %s " % (global_rank, f, cur_feature_offsets, cur_feature_lengths, host, cur_feature_indices))

        return (offsets, all_feature_indices)

    def getEmbTableDimensions(self, global_rank, world_size, args):
        local_emb_dims = []
        ln_emb = ""
        ipConfig = {}
        if args.model == "dlrm":  # open-source DLRM.
            ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
            n_emb = len(ln_emb)
            if n_emb < world_size:
                raise ValueError(
                    "Embedding size should match process count, please fix '--arch-embedding-size' and try again"
                )
            _, n_emb_per_rank = self.get_split_lengths_by_len(
                n_emb, global_rank, world_size
            )
            dims_per_rank = []
            local_emb_slice = []

            # Following logic is borrowed from ctr_mbl_feed_jan_v0.py:_gen_embs_dims (which inturn depends on _generate_sparse_param_args)
            for i in range(world_size):
                tempArr = self.get_slice_sparse(i, n_emb_per_rank, world_size)
                p = ln_emb[tempArr]

                if i == global_rank:
                    local_emb_slice = ln_emb[tempArr]
                    local_emb_dims = [args.arch_sparse_feature_size for i in p]

                temp = []
                for _ in p:
                    temp.append(
                        args.arch_sparse_feature_size
                    )  # PENDING/TODO: Update it based on trainer/feeds model.
                dims_per_rank.append(temp)

                if global_rank == 0:
                    print("\t Rank: %d tempArr: %s p: %s " % (i, tempArr, p))

            dims_sum_per_rank = [sum(s) for s in dims_per_rank]

            if global_rank == 0:
                print("\tdims_sum_per_rank: %s " % (dims_sum_per_rank))

            # Package the parameters to be used by DLRM function
            ipConfig["num_sparse_fea"] = ln_emb.size
            ipConfig["n_emb_per_rank"] = n_emb_per_rank
            ipConfig["local_emb_slice"] = local_emb_slice
            ipConfig["dims_sum_per_rank"] = dims_sum_per_rank
            ipConfig["local_emb_dims"] = local_emb_dims
        else:
            print("Model " + args.model + " not supported...Abort!")
            sys.exit()

        return (local_emb_dims, ln_emb, ipConfig)

    def getLayerDimensions(self, global_rank, world_size, args):
        ### parse command line arguments ###
        (local_emb_dims, ln_emb, ipConfig) = self.getEmbTableDimensions(
            global_rank, world_size, args
        )
        if global_rank == 0:
            print("\t ipConfig['num_sparse_fea']: %s " % (ipConfig["num_sparse_fea"]))
            print("\t ipConfig['n_emb_per_rank']: %s " % (ipConfig["n_emb_per_rank"]))
            print(
                "\t ipConfig['dims_sum_per_rank']: %s "
                % (ipConfig["dims_sum_per_rank"])
            )

            print("\t ipConfig['local_emb_dims']: %s " % (ipConfig["local_emb_dims"]))

        # Genereously borrowed from public DLRM benchmark.
        ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
        n_emb = ln_emb.size

        # Parameters required for determining num_int and creating top-layer dimensions.
        # m_spa = args.arch_sparse_feature_size
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

        if args.arch_project_size > 0:
            num_int = num_fea * args.arch_project_size + m_den_out

        arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
        if global_rank == 0:
            print(
                "\t ln_top: %s \n\t ln_bot: %s \n\t n_emb: %s "
                % (ln_top, ln_bot, n_emb)
            )

        topMLP = self.create_mlp(
            global_rank, ln_top
        )  # Creates actual dimensions of each MLP layer
        botMLP = self.create_mlp(global_rank, ln_bot)
        ipConfig["ln_top"] = ln_top
        ipConfig["ln_bot"] = ln_bot
        ipConfig["topMLP"] = topMLP
        ipConfig["botMLP"] = botMLP

        print(
            "\t rank: %d len(local_emb_slice): %s local_emb_dims: %s "
            % (global_rank, len(ipConfig["local_emb_slice"]), len(local_emb_dims))
        )
        embedLayers = self.create_emb(
            global_rank, ipConfig["local_emb_dims"], ipConfig["local_emb_slice"]
        )  # WARNING: Assuming that for allocating memory, we should only use local-emb-slice.
        ipConfig["embedLayers"] = embedLayers

        args.numpy_rand_seed = global_rank
        train_data, train_ld = dd.data_loader(args, ln_emb, m_den)

        ipConfig["train_ld"] = train_ld

        myModelConfig = modelConfig(
            ipConfig
        )  # A holding object to carry all the parameters (for both type of models) that have been calculated.
        print(
            "\t rank: %s n_emb: %s modelConfig: %s "
            % (global_rank, n_emb, myModelConfig.dims_sum_per_rank)
        )
        return myModelConfig


class commsDLRMBench(paramCommsBench):
    def __init__(self):
        # TODO: only support pytorch-dist as the nw-stack now
        super().__init__(supportedNwstacks=["pytorch-dist"])
        self.model = ""
        self.mixedDimFlag = False
        self.measured_regions = {}
        self.commDetails = []
        self.paramNN = paramDLRM_Net()
        self.myreq = Request(self)

    def readArgs(self, parser, defaultModel="dlrm"):
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--warmup-batches",
            type=int,
            default=5,
            help="number of warm-up batches per epoch",
        )  # Number of warmup batches within an epoch.
        # Added specifically to dlrm-comms
        parser.add_argument(
            "--model", type=str, default=defaultModel, help="Model to be benchmarked"
        )
        parser.add_argument("--print-comms", action="store_true")
        parser.add_argument("--perf-debug", action="store_true")
        # Copied from DLRM benchmark.
        parser.add_argument("--arch-sparse-feature-size", type=int, default=4)
        parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
        # MLP layers
        parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
        parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
        parser.add_argument(
            "--arch-project-size",
            type=int,
            default=0,
            help="project size for the interaction features",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="number of workers for DataLoader",
        )
        parser.add_argument("--arch-interaction-op", type=str, default="dot")
        parser.add_argument(
            "--arch-interaction-itself", action="store_true", default=False
        )
        # data
        parser.add_argument("--data-size", type=int, default=1)
        parser.add_argument(
            "--num-batches", type=int, default=10, help="Number of batches to be run"
        )  # ensure num-bathes is always larger than warmup-batches
        parser.add_argument(
            "--synthetic-data-folder",
            type=str,
            default="./synthetic_data/syn_data_bs65536/",
        )
        # training
        parser.add_argument("--mini-batch-size", type=int, default=1)
        parser.add_argument("--num-indices-per-lookup", type=int, default=10)
        parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
        parser.add_argument("--round-targets", type=bool, default=False)
        parser.add_argument(
            "--data-generation", type=str, default="random", help="Input data generator"
        )  # synthetic or random
        parser.add_argument(
            "--rand-data-dist", type=str, default="uniform"
        )  # uniform or gaussian
        parser.add_argument("--rand-data-min", type=float, default=0)
        parser.add_argument("--rand-data-max", type=float, default=1)
        parser.add_argument("--rand-data-mu", type=float, default=-1)
        parser.add_argument("--rand-data-sigma", type=float, default=1)
        parser.add_argument(
            "--data-trace-file", type=str, default="./input/dist_emb_j.log"
        )
        parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
        parser.add_argument("--numpy-rand-seed", type=int, default=123)

        parser.add_argument(
            "--embed-dtype", type=torch.dtype, default=torch.float32
        )  # will be overwritten based on args.embed_data_type and dtypeMap.
        parser.add_argument(
            "--embed-data-type",
            type=str,
            default="float32",
            help="Data type to be used, supports " + str(self.supportedDtype),
        )

        return parser.parse_args()

    def checkArgs(self, args):
        super().checkArgs(args)
        if args.embed_data_type not in self.supportedDtype:
            print(
                "\t ERROR: Specified dtype: %d is not one of the supported commstyle: %s"
                % (args.embed_data_type, str(self.supportedDtype))
            )
            comms_utils.gracefulExit()

    def SparseDataDist(
        self, num_features_per_rank, input_features, global_rank, world_size, timers
    ):
        if len(num_features_per_rank) == 1:
            return

        batch_size = input_features.batch_size
        device = input_features.lengths.device
        cpu_device = torch.device("cpu")

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
        # with record_function("## all2all_data:lengths ##"):
        out_splits = [num_my_features * batch_size] * world_size
        in_splits = [
            num_features * batch_size for num_features in num_features_per_rank
        ]
        self.collectiveArgs.opTensor = output_lengths
        self.collectiveArgs.ipTensor = input_features.lengths
        self.collectiveArgs.opTensor_split = out_splits
        self.collectiveArgs.ipTensor_split = in_splits
        self.collectiveArgs.asyncOp = False
        self.backendFuncs.sync_barrier(self.collectiveArgs)

        timers["offset_xchg_start"] = time.monotonic()
        self.backendFuncs.all_to_allv(self.collectiveArgs)
        self.backendFuncs.complete_accel_ops(self.collectiveArgs)
        timers["offset_xchg_end"] = time.monotonic()

        cur_iter_memory = output_lengths.element_size() * output_lengths.nelement()
        self.measured_regions["offset_xchg"]["memory"].append(cur_iter_memory)

        prev_a2a_details = {
            "comms": "all_to_all",
            "msg_size": cur_iter_memory,
            "in_split": in_splits,
            "out_split": out_splits,
            "dtype": str(input_features.lengths.dtype),
        }
        self.commDetails.append(prev_a2a_details)

        # Start alltoall request for 'indices'.
        output_indices = torch.empty(
            output_lengths.sum().item(),
            device=device,
            dtype=torch.int64,  # input_features.indices.dtype,
        )
        output_indices_splits = (
            output_lengths.view(world_size, -1).sum(dim=1).to(cpu_device).numpy()
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

        self.collectiveArgs.opTensor = output_indices
        self.collectiveArgs.ipTensor = input_features.indices
        self.collectiveArgs.opTensor_split = output_indices_splits
        self.collectiveArgs.ipTensor_split = input_features_splits
        self.collectiveArgs.asyncOp = False

        self.backendFuncs.sync_barrier(self.collectiveArgs)
        timers["idx_xchg_start"] = time.monotonic()
        self.backendFuncs.all_to_allv(self.collectiveArgs)
        self.backendFuncs.complete_accel_ops(self.collectiveArgs)
        timers["idx_xchg_end"] = time.monotonic()

        cur_iter_memory = output_indices.element_size() * output_indices.nelement()
        self.measured_regions["idx_xchg"]["memory"].append(cur_iter_memory)
        cur_a2a_details = {
            "comms": "all_to_all",
            "msg_size": cur_iter_memory,
            "in_split": np.array(input_features_splits, dtype=np.int32).tolist(),
            "out_split": np.array(output_indices_splits, dtype=np.int32).tolist(),
            "dtype": str(input_features.indices.dtype),
        }
        self.commDetails.append(cur_a2a_details)

        # By now we have received the lengths, and indices of local-table for the global-batch
        # We need to split the lengths and indices per table -- logic explained in splitPerTable
        offsets, indices = self.paramNN.splitPerTable(
            output_lengths,
            output_indices,
            batch_size,
            num_my_features,
            world_size,
            global_rank,
            device,
        )
        return (offsets, indices)

    # FIXME: can we refere to extend_distributed.ExtendProcessGroup.alltoallv?
    def alltoallv(self, inputs, global_rank, out_split, per_rank_split_lengths):
        N, _ = inputs[0].size()
        E = [e.size()[1] for e in inputs]
        a2ai = All2AllInfo()
        a2ai.lS = len(inputs)
        if out_split:
            a2ai.gSS = out_split
        else:
            # all the embs have the same dimension
            a2ai.gSS = [s * E[0] for s in per_rank_split_lengths]
        a2ai.lN, a2ai.gNS = self.paramNN.get_split_lengths_by_len(
            N, global_rank, len(out_split)
        )
        a2ai.E = E
        a2ai.N = N
        a2ai.S = sum(a2ai.gSS) if a2ai.gSS else a2ai.lS * self.my_size

        All2Allv_Req.apply(self.myreq, a2ai, *inputs)
        self.myreq.WaitFunction = All2Allv_Wait

        return self.myreq

    def resetTimers(self, timers):
        """Serves as both a way to add a timer and reset when it is already present"""
        timers["iter_start"] = 0.0

        timers["length_calc_end"] = 0.0
        timers["mem_push_idx_end"] = 0.0

        timers["offset_xchg_start"] = 0.0
        timers["offset_xchg_end"] = 0.0
        timers["idx_xchg_start"] = 0.0
        timers["idx_xchg_end"] = 0.0

        timers["bef_emb_lookup"] = 0.0
        timers["fwd_a2a_start"] = 0.0
        timers["fwd_a2a_end"] = 0.0
        timers["grad_push_start"] = 0.0

        timers["bwd_top_ar_start"] = 0.0
        timers["bwd_top_ar_end"] = 0.0
        timers["bwd_a2a_start"] = 0.0
        timers["bwd_a2a_end"] = 0.0
        timers["bwd_bot_ar_start"] = 0.0
        timers["bwd_bot_ar_end"] = 0.0

    def setTimerRegions(self, region_name, start_timer, end_timer):
        """Add a new measure-region, by specifying the timers"""
        self.measured_regions[region_name] = {}
        self.measured_regions[region_name]["start"] = start_timer
        self.measured_regions[region_name]["end"] = end_timer
        self.measured_regions[region_name]["samples"] = []
        self.measured_regions[region_name]["memory"] = []

    def intermed_region_memory(self, timers):
        # regions which are in between communication sizes won't have memory, which we are fixing by adding an entry for each timer every iteration
        # This is needed because we want to average the memory for all the all-to-all operations (whose size varies every iteration).
        intermed_regions = [
            "intermed_calc_length",
            "mem_push_idx",
            "intermed_bef_offset_xchg",
            "intermed_btw_offset_idx_xchg",
            "intermed_post_idx_xchg_sparse_dist",
            "intermed_emb_lookup_to_a2a_start",
            "intermed_fwd_a2a_grad_push",
            "mem_push_gradients",
            "intermed_top_ar_end_to_bwd_a2a_start",
            "intermed_bwd_a2a_bot_ar",
            "iter_time",
            "iter_data_prep",
            "iter_fwd_a2a",
            "iter_bwd_top_ar",
            "iter_bwd_a2a",
        ]

        for cur_region in intermed_regions:
            self.measured_regions[cur_region]["memory"].append(0)

    def getMemSizes(self, curDeviceData):
        memSizes = {}
        m_topLayer = []
        m_botLayer = []
        for curLayer in curDeviceData["topLayers"]:
            self.collectiveArgs.ipTensor = curLayer
            m_topLayer.append(self.backendFuncs.get_mem_size(self.collectiveArgs))

        for curLayer in curDeviceData["botLayers"]:
            self.collectiveArgs.ipTensor = curLayer
            m_botLayer.append(self.backendFuncs.get_mem_size(self.collectiveArgs))

        memSizes["top"] = m_topLayer
        memSizes["bot"] = m_botLayer
        return memSizes

    def computeTimes(self, timers):
        # At the end of the iteration, measure the time taken for a given region.
        for cur_region in self.measured_regions:
            start_time = timers[self.measured_regions[cur_region]["start"]]
            end_time = timers[self.measured_regions[cur_region]["end"]]
            time_spent = (end_time - start_time) * 1e6  # nanoseconds
            self.measured_regions[cur_region]["samples"].append(time_spent)
        self.resetTimers(timers)

    def initTimers(self):
        timers = {}
        self.resetTimers(timers)
        # add different measured "region" to the dictinary called "measured_regions".
        # <dictionary> <name> <start-timer> <end-timer>
        self.setTimerRegions("intermed_calc_length", "iter_start", "length_calc_end")
        self.setTimerRegions("mem_push_idx", "length_calc_end", "mem_push_idx_end")
        self.setTimerRegions(
            "intermed_bef_offset_xchg", "mem_push_idx_end", "offset_xchg_start"
        )
        self.setTimerRegions("offset_xchg", "offset_xchg_start", "offset_xchg_end")
        self.setTimerRegions(
            "intermed_btw_offset_idx_xchg", "offset_xchg_end", "idx_xchg_start"
        )
        self.setTimerRegions("idx_xchg", "idx_xchg_start", "idx_xchg_end")
        self.setTimerRegions(
            "intermed_post_idx_xchg_sparse_dist", "idx_xchg_end", "bef_emb_lookup"
        )

        self.setTimerRegions(
            "intermed_emb_lookup_to_a2a_start", "bef_emb_lookup", "fwd_a2a_start"
        )
        self.setTimerRegions("fwd_a2a", "fwd_a2a_start", "fwd_a2a_end")
        self.setTimerRegions(
            "intermed_fwd_a2a_grad_push", "fwd_a2a_end", "grad_push_start"
        )

        self.setTimerRegions(
            "mem_push_gradients", "grad_push_start", "bwd_top_ar_start"
        )
        self.setTimerRegions("bwd_top_ar", "bwd_top_ar_start", "bwd_top_ar_end")

        self.setTimerRegions(
            "intermed_top_ar_end_to_bwd_a2a_start", "bwd_top_ar_end", "bwd_a2a_start"
        )
        self.setTimerRegions("bwd_a2a", "bwd_a2a_start", "bwd_a2a_end")
        self.setTimerRegions(
            "intermed_bwd_a2a_bot_ar", "bwd_a2a_end", "bwd_bot_ar_start"
        )

        self.setTimerRegions("bwd_bot_ar", "bwd_bot_ar_start", "bwd_bot_ar_end")
        self.setTimerRegions("iter_time", "iter_start", "bwd_bot_ar_end")

        self.setTimerRegions("iter_data_prep", "iter_start", "bef_emb_lookup")
        self.setTimerRegions("iter_fwd_a2a", "iter_start", "grad_push_start")
        self.setTimerRegions("iter_bwd_top_ar", "iter_start", "bwd_top_ar_end")
        self.setTimerRegions("iter_bwd_a2a", "iter_start", "bwd_bot_ar_start")

        return timers

    def reportBenchTime(
        self, global_rank, wamrupIters, measuredIters, world_size, curDevice
    ):
        if measuredIters != 0:
            all_timers = [
                "intermed_calc_length",
                "mem_push_idx",
                "intermed_bef_offset_xchg",
                "offset_xchg",
                "intermed_btw_offset_idx_xchg",
                "idx_xchg",
                "intermed_post_idx_xchg_sparse_dist",
                "intermed_emb_lookup_to_a2a_start",
                "fwd_a2a",
                "intermed_fwd_a2a_grad_push",
                "mem_push_gradients",
                "bwd_top_ar",
                "intermed_top_ar_end_to_bwd_a2a_start",
                "bwd_a2a",
                "intermed_bwd_a2a_bot_ar",
                "bwd_bot_ar",
                "iter_time",
                "iter_data_prep",
                "iter_fwd_a2a",
                "iter_bwd_top_ar",
                "iter_bwd_a2a",
            ]

            # Each rank makes a list (2D tensor) of all the samples for each measured-region. Do the same for memory as well.
            combined_latency_list = []
            combined_memory_list = []
            for cur_region in all_timers:
                combined_latency_list.append(
                    self.measured_regions[cur_region]["samples"]
                )
                combined_memory_list.append(self.measured_regions[cur_region]["memory"])

            # All-gather to exchange the samples (memory and latency)
            timeElapsedTensor = torch.tensor(combined_latency_list, device=curDevice)
            tensor_list = [
                torch.ones_like(timeElapsedTensor) for _ in range(world_size)
            ]
            self.collectiveArgs.ipTensor = timeElapsedTensor
            self.collectiveArgs.opTensor = tensor_list
            self.collectiveArgs.asyncOp = False
            self.collectiveArgs.dataSize = (
                timeElapsedTensor.nelement() * timeElapsedTensor.element_size()
            )
            self.collectiveArgs.numElements = timeElapsedTensor.nelement()
            self.backendFuncs.all_gather(self.collectiveArgs)
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)

            memory_tensor = torch.tensor(combined_memory_list, device=curDevice)
            memory_tensor_list = [
                torch.ones_like(memory_tensor) for _ in range(world_size)
            ]
            self.collectiveArgs.ipTensor = memory_tensor
            self.collectiveArgs.opTensor = memory_tensor_list
            self.backendFuncs.all_gather(self.collectiveArgs)
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)

            sum_latency = 0.0
            sum_mean_latency = 0.0
            if global_rank == 0:
                cpu_tensor_latency = []
                cpu_tensor_memory = []
                for cur_region in range(world_size):
                    cpu_tensor_latency.append(tensor_list[cur_region].to("cpu"))
                    cpu_tensor_memory.append(memory_tensor_list[cur_region].to("cpu"))

                res_mean_percentiles = []
                res_percentiles = []
                print(
                    "\t{}\t{:>36}\t{:>12}\t{:>12}\t{:>12}\t{:>12}\t{:>12}\t{:>12}".format(
                        "iters",
                        "region",
                        "memory (B)",
                        "Latency(us):min",
                        "p50",
                        "p75",
                        "p95",
                        "sum(p50)",
                    )
                )
                for region_idx, cur_region in enumerate(all_timers):
                    # For each region, get data from different ranks. Compute percentiles for a given region.
                    all_rank_latency = []
                    all_rank_memory = []
                    all_rank_mean_latency = []
                    for cur_rank in range(world_size):
                        cur_rank_latency = cpu_tensor_latency[cur_rank][region_idx]
                        cur_rank_memory = cpu_tensor_memory[cur_rank][region_idx][
                            wamrupIters:
                        ]

                        all_rank_latency.append(cur_rank_latency)
                        all_rank_memory.append(cur_rank_memory)
                        all_rank_mean_latency.append(torch.mean(cur_rank_latency))

                    all_rank_latency = torch.cat(all_rank_latency)
                    all_rank_memory = torch.cat(all_rank_memory)

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
                    # Printing two sets of results --
                    # 1. Percentiles based on samples across all the ranks (so #samples = num_iterations * num_ranks)
                    # 2. Percentiles based on average latency at each rank (so #samples = num_ranks)

                    if "iter" not in cur_region:
                        sum_latency = sum_latency + p50
                        sum_mean_latency = sum_mean_latency + mean_p50
                    res_percentiles_line = (
                        "\t%d\t%36s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s"
                        % (
                            measuredIters,
                            cur_region,
                            "%d" % (mem_p50),
                            "%.3f" % (min_lat),
                            "%.3f" % (p50),
                            "%.3f" % (p75),
                            "%.3f" % (p95),
                            "%.3f" % (sum_latency),
                        )
                    )
                    res_mean_percentiles_line = (
                        "\t%d\t%36s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s"
                        % (
                            measuredIters,
                            cur_region,
                            "%d" % (mem_p50),
                            "%.3f" % (mean_min_lat),
                            "%.3f" % (mean_p50),
                            "%.3f" % (mean_p75),
                            "%.3f" % (mean_p95),
                            "%.3f" % (sum_mean_latency),
                        )
                    )

                    res_percentiles.append(res_percentiles_line)
                    res_mean_percentiles.append(res_mean_percentiles_line)

                for cur_line in res_percentiles:
                    if "iter_time" in cur_line:
                        print("\n")
                    print(cur_line)

                print(
                    "\t%d\t%36s\t%12s\t%12s\t%12s"
                    % (
                        measuredIters,
                        "total_time",
                        "N/A",
                        "N/A",
                        "%.3f" % (sum_latency),
                    )
                )
                print(
                    "\n\n -----------------------------------------------------------------------------------------------------------------------------\n\n"
                )
                for cur_line in res_mean_percentiles:
                    if "iter_time" in cur_line:
                        print("\n")
                    print(cur_line)
                print(
                    "\t%d\t%36s\t%12s\t%12s\t%12s"
                    % (
                        measuredIters,
                        "total_time",
                        "N/A",
                        "N/A",
                        "%.3f" % (sum_mean_latency),
                    )
                )
                print(
                    "\n\n -----------------------------------------------------------------------------------------------------------------------------\n\n"
                )

    def benchTime(
        self,
        global_rank,
        world_size,
        timers,
        mConfig,
        curDevice,
        curDeviceData,
        commsDlrmParams,
    ):
        memSizes = self.getMemSizes(curDeviceData)
        for batchNum, (_, lS_o, lS_i, _) in enumerate(mConfig.train_ld):
            timers["iter_start"] = time.monotonic()

            curIterSparseFeatures = SparseFeatures(
                mConfig.num_sparse_fea,
                commsDlrmParams.mini_batch_size,
                lS_o,
                lS_i,
                curDevice,
                global_rank,
                self.backendFuncs,
                self.collectiveArgs,
            )
            # Exchange it among all the ranks, so that we have indices and offsets of only the tables we have but for the global mini-batch, not just local mini-batch.s
            g_offsets, g_indices = self.SparseDataDist(
                mConfig.n_emb_per_rank,
                curIterSparseFeatures,
                global_rank,
                world_size,
                timers,
            )

            # Begin with reading the embedding table.
            self.backendFuncs.sync_barrier(self.collectiveArgs)
            timers["bef_emb_lookup"] = time.monotonic()
            ly = self.paramNN.apply_emb(
                g_offsets,
                g_indices,
                curDeviceData["embedLayers"],
                mixed_dim=self.mixedDimFlag,
            )

            # Start with fwd pass all-to-all, this is blocking communication.
            a2a_req = ""
            B = ""
            tempB = ""
            emb_memory = ly.nelement() * ly.element_size()
            if not self.mixedDimFlag:
                a2a_req = self.alltoallv(
                    ly, global_rank, mConfig.dims_sum_per_rank, mConfig.n_emb_per_rank
                )
                B = a2a_req.wait()
                tempB = torch.cat(B, dim=1)

            self.collectiveArgs.timers["grad_push_start"] = time.monotonic()
            C = tempB
            self.backendFuncs.sync_barrier(
                self.collectiveArgs
            )  # else data won't actually be moved, evidently!

            if commsDlrmParams.perf_debug:
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            # back-prop: top layer, non-blocking between the top-layers.
            timers["bwd_top_ar_start"] = time.monotonic()
            for curLayerIdx in range(len(curDeviceData["topLayers"])):
                # Prepare collective arguments
                self.collectiveArgs.ipTensor = curDeviceData["topLayers"][curLayerIdx]
                self.collectiveArgs.asyncOp = True
                self.collectiveArgs.op = self.backendFuncs.get_reduce_op("sum")
                self.backendFuncs.all_reduce(self.collectiveArgs)
                # Prepare communication details, logging to understand performance.
                self.commDetails.append(
                    {
                        "comms": "all_reduce",
                        "msg_size": curDeviceData["topLayers"][curLayerIdx].nelement()
                        * curDeviceData["topLayers"][curLayerIdx].element_size(),
                        "dtype": str(curDeviceData["topLayers"][curLayerIdx].dtype),
                    }
                )
            self.backendFuncs.sync_barrier(self.collectiveArgs)
            timers["bwd_top_ar_end"] = time.monotonic()

            if commsDlrmParams.perf_debug:
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            # back-prop: embedding update, blocking, since we are waiting for it to complete.
            if self.mixedDimFlag:
                self.collectiveArgs.timers["bwd_a2a_start"] = time.monotonic()
                tempB.backward(C)
                self.collectiveArgs.timers["bwd_a2a_end"] = time.monotonic()
                self.measured_regions["bwd_a2a"]["memory"].append(
                    emb_memory
                )  # this is not quite right in case of , just ensuring that we have a non-zero entry for ads-feeds model
            else:
                tempB.backward(C)

            if commsDlrmParams.perf_debug:
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            # back-prop: bottom layer, non-blocking between the layers.
            timers["bwd_bot_ar_start"] = time.monotonic()
            for curLayerIdx in range(len(curDeviceData["botLayers"])):
                self.collectiveArgs.ipTensor = curDeviceData["botLayers"][curLayerIdx]
                self.collectiveArgs.asyncOp = True
                self.collectiveArgs.op = self.backendFuncs.get_reduce_op("sum")
                self.backendFuncs.all_reduce(self.collectiveArgs)
                self.commDetails.append(
                    {
                        "comms": "all_reduce",
                        "msg_size": curDeviceData["botLayers"][curLayerIdx].nelement()
                        * curDeviceData["botLayers"][curLayerIdx].element_size(),
                        "dtype": str(curDeviceData["botLayers"][curLayerIdx].dtype),
                    }
                )
            self.backendFuncs.sync_barrier(self.collectiveArgs)
            timers["bwd_bot_ar_end"] = time.monotonic()
            self.measured_regions["bwd_top_ar"]["memory"].append(sum(memSizes["top"]))
            self.measured_regions["bwd_bot_ar"]["memory"].append(sum(memSizes["bot"]))

            if batchNum >= commsDlrmParams.warmup_batches:
                self.computeTimes(timers)
            self.intermed_region_memory(timers)

    def runBench(self, bootstrap_info, commsDlrmParams, args):
        """Run num-batches iterations of the model (only comms-operations)"""
        if commsDlrmParams.nw_stack == "pytorch-dist":
            self.backendFuncs = PyTorchDistBackend(bootstrap_info, commsDlrmParams)
            self.backendFuncs.initialize_backend(
                bootstrap_info.master_ip,
                bootstrap_info.master_port,
                backend=commsDlrmParams.backend,
            )
        else:
            logger.error(
                "\t Input nw_stack {} or backend {} not supported! ".format(
                    commsDlrmParams.nw_stack, commsDlrmParams.backend
                )
            )
            sys.exit()

        (
            local_rank,
            global_rank,
            world_size,
            group,
            curDevice,
            curHwDevice,
        ) = comms_utils.get_rank_details(self.backendFuncs)
        self.backendFuncs.sayHello()
        mConfig = self.paramNN.getLayerDimensions(
            global_rank, world_size, args
        )  # supports reading model parameters from json file, or from opensource DLRM CLI format.
        self.collectiveArgs.device = curDevice
        self.collectiveArgs.waitObj = []
        self.collectiveArgs.group = group
        self.comm_size = world_size
        self.my_rank = global_rank

        # Initializes the data for MLP, local embedding table
        curDeviceData = self.paramNN.initializeData(
            curDevice,
            self.backendFuncs,
            global_rank,
            mConfig.topMLP,
            mConfig.botMLP,
            mConfig.embedLayers,
        )
        host = os.uname()[1]
        timers = self.initTimers()
        self.collectiveArgs.timers = timers

        self.backendFuncs.sync_barrier(
            self.collectiveArgs
        )  # To ensure everyone starts at the same point

        if global_rank >= 0:
            print(
                "\n\t ****** Rank: G: %d L: %d host: %s starting new epoch, model: %s ***** \n"
                % (global_rank, local_rank, host, self.model)
            )
        # start runing benchmark and measuring latency
        self.benchTime(
            global_rank,
            world_size,
            timers,
            mConfig,
            curDevice,
            curDeviceData,
            commsDlrmParams,
        )

        if commsDlrmParams.print_comms:
            folder = str(self.model) + "_np" + str(world_size)
            try:
                subprocess.check_output(["mkdir", "-p", str(folder)], text=True)
            except Exception as err:
                print("\t Error: {} while creating directory: {} ".format(err, folder))
                pass
            comms_file = str(folder) + "/rank" + str(global_rank) + ".json"
            with open(comms_file, "w") as write_file:
                json.dump(self.commDetails, write_file)

        measuredIters = commsDlrmParams.numBatches - commsDlrmParams.warmup_batches
        self.reportBenchTime(
            global_rank,
            commsDlrmParams.warmup_batches,
            measuredIters,
            world_size,
            curDevice,
        )

    def initBench(self, args, mpi_env_params) -> None:
        """
        placeholder function to initializes DLRM benchmarking parameters.
        """
        args.embed_dtype = self.dtypeMap[args.embed_data_type]
        args.data_size = (
            mpi_env_params["world_size"] * args.num_batches * args.mini_batch_size
        )
        if mpi_env_params["global_rank"] == 0:
            print("\t mpi-params: %s" % (mpi_env_params))

        self.model = args.model
        print(
            "\t rank: %s args.model: %s model: %s "
            % (mpi_env_params["global_rank"], args.model, self.model)
        )


def main():
    mpi_env_params = comms_utils.read_comms_env_vars()

    dlrmBench = commsDLRMBench()
    parser = argparse.ArgumentParser(
        description="PARAM-Comms DLRM Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    args = dlrmBench.readArgs(parser)
    dlrmBench.checkArgs(args)
    dlrmBench.initBench(args, mpi_env_params)

    time.sleep(1)
    bootstrap_info = comms_utils.bootstrap_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, mpi_env_params
    )
    commsDlrmParams = comms_utils.commsDlrmParamsHolder(args, mpi_env_params)
    dlrmBench.runBench(bootstrap_info, commsDlrmParams, args)


if __name__ == "__main__":
    main()
