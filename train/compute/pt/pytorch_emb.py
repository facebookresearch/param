# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# measuring embeddingbag performance using pytorch
# 1. Table lookup index is generated together for all iterations
#    Currently we see very similar performance across iterations,
#    even with same indices for each iteration
#    So, use same indices for different iteration now.
#
# 2. Test case:
#    python3 pytorch_emb.py --features=30000000 --embdim=128 --nnz=100 --batch=8192 --testgpu=1 --verify=1 --steps=10
#
# 3. Using XlaEmbedding Bag to replace torch.nn.embeddingbag with XLA on TPU now
#
import time
import torch
import torch.nn as nn

class XlaEmbeddingBag(nn.Module):


    """
    nn.EmbeddingBag is not lowered just yet to xla.
    This performs the same functionality, in an xla compatible, sub-optimal way.
    Warning!: only works with constant offsets atm.
    """

    def __init__(self, n, m, mode, offset, *args, **kwargs):
        super(XlaEmbeddingBag, self).__init__()
        self.n = n
        self.m = m
        self.mode = mode
        self.offset = offset
        self.embtable = nn.Embedding(n, m, *args, **kwargs)

    def forward(self, sparse_index_group_batch, sparse_offset_group_batch):
        emb = self.embtable(sparse_index_group_batch)
        # XXX: only works w/ constant offset atm
        bsz = emb.size(0) // self.offset
        emb = emb.reshape(bsz, self.offset, *emb.size()[1:])
        reduce_fn = getattr(torch, self.mode)
        return reduce_fn(emb, axis=1)

def measure_cpu(warmups, steps, h_emb, h_indices, h_offsets):

    emb_times = 0
    start1 = time.perf_counter()
    for i in range(warmups + steps):
        start = time.perf_counter()
        results = h_emb(h_indices, h_offsets)
        # results = emb1(h_indices)
        end  = time.perf_counter()
        # print("Time {0:.6f} ".format(end - start))
        if (i >= warmups):
            emb_times += end - start
    end1 = time.perf_counter()

    return end1 - start1, emb_times, results


def measure_gpu(warmups, steps, h_emb, h_indices, h_offsets):

    ncuda = torch.cuda.device_count()
    print("There are {} cuda devices".format(ncuda))
    print("The current cuda device name is {} ".format(torch.cuda.get_device_name()))
    cuda0 = torch.device('cuda:0')
    emb_times = 0
    with torch.cuda.device(cuda0):
        g_emb  = h_emb.to(cuda0)
        g_indices = h_indices.to(cuda0)
        g_offsets = h_offsets.to(cuda0)
        torch.cuda.synchronize()

        start1 = time.perf_counter()
        for i in range(warmups + steps):
            start = time.perf_counter()
            results = g_emb(g_indices, g_offsets)
            torch.cuda.synchronize()
            end = time.perf_counter()
            # print("Time: {0:.6f} ".format(end - start))

            if (i >= warmups):
                emb_times += end - start

        end1 = time.perf_counter()

    return end1 - start1, emb_times, results


def measure_tpu(warmups, steps, h_emb, h_indices, h_offsets, args):

    import torch_xla
    import torch_xla.core.xla_model as xm
    import os
    # import math
    # import torch_xla.debug.metrics as met

    tsize = int(os.environ.get("MODEL_PARTITION_SIZE", 3000000))

    def syncTPU(tensor):
        torch_xla._XLAC._xla_sync_multi([tensor], devices=[], wait=True, sync_xla_data=True)

    alldev = xm.get_xla_supported_devices()
    allrealdev = xm.xla_real_devices(alldev)
    print("Found {0} devices: {1}".format(len(allrealdev), allrealdev))

    dev = xm.xla_device()
    # dev = xm.xla_device(n=2, devkind='TPU')
    if (args.features > tsize):
        if args.usexlabag:
            tsplit = torch.split(h_emb.embtable.weight, tsize, dim=0)
        else:
            tsplit = torch.split(h_emb.weight, tsize, dim=0)
        tsplit = list(tsplit)
        for i, chunk in enumerate(tsplit):
            tsplit[i] = chunk.to(dev)

        t = nn.Parameter(torch.ones(10, 10))
        if args.usexlabag:
            h_emb.embtable.weight = t
            t_emb = h_emb.to(dev)
            tsplit = torch.cat(tsplit)
            t_emb.embtable.weight = nn.Parameter(tsplit)
            print("Xla EMB weight shape: ", t_emb.embtable.weight.shape, " on device: ", str(dev))
        else:
            h_emb.weight = t
            t_emb = h_emb.to(dev)
            tsplit = torch.cat(tsplit)
            t_emb.weight = nn.Parameter(tsplit)
            print("EMB weight shape: ", t_emb.weight.shape, " on device: ", str(dev))
    else:
        t_emb = h_emb.to(dev)

    t_indices = h_indices.to(dev)
    t_offsets = h_offsets.to(dev)

    emb_times = 0.0
    start1 = time.perf_counter()
    for i in range(warmups + steps):
        start = time.perf_counter()
        results = t_emb(t_indices, t_offsets)
        syncTPU(results)
        end = time.perf_counter()
        print("Time: {0:.6f} ".format(end - start))
        if (i >= warmups):
            emb_times += end - start

    end1 = time.perf_counter()
    # print(met.metrics_report())

    return end1 - start1, emb_times, results


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Measure the performance of pytorch embeddingbag"
    )
    # model related parameters
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--embdim", type=int, default=64)
    parser.add_argument("--nnz", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--randomseed", type=int, default=0)
    parser.add_argument("--testcpu", action='store_true', default=False)
    parser.add_argument("--testgpu", action='store_true', default=False)
    parser.add_argument("--testtpu", action='store_true', default=False)
    parser.add_argument("--verify", action='store_true', default=False)
    parser.add_argument("--usexlabag", action='store_true', default=False)

    args = parser.parse_args()

    num_features = args.features
    embed_dim = args.embdim
    nnz = args.nnz
    batch_size = args.batch
    steps = args.steps
    warmups = args.warmups

    random_seed = args.randomseed

    print("Test problem size:")
    print("Number of features : ", num_features)
    print("Embedding size     : ", embed_dim)
    print("Nnz_per_input      : ", nnz)
    print("Number of batches  : ", batch_size)

    torch.manual_seed(random_seed)

    # 1. measure on CPU first
    # h_indices  = torch.randint(0, num_features, (warmups+steps, batch_size, nnz))
    h_indices = torch.randint(0, num_features, (batch_size*nnz,))
    # h_indices = torch.randint(0, num_features, (batch_size, nnz))
    h_offsets = torch.zeros(batch_size, dtype=torch.int64)
    for i in range(batch_size):
        h_offsets[i] = i * nnz
    print("Finished generating indices")
    if (not args.usexlabag):
        h_emb = nn.EmbeddingBag(num_features, embed_dim, mode='sum')
        total_bytes = batch_size * nnz * embed_dim * h_emb.weight.element_size()
    else:
        print("using XlaBag instead of torch.nn.EmbeddingBag now")
        h_emb = XlaEmbeddingBag(num_features, embed_dim, "sum", nnz)
        total_bytes = batch_size * nnz * embed_dim * h_emb.embtable.weight.element_size()
    print("Finished generating tables. Using mem: ", total_bytes)
    h_results = torch.zeros(batch_size, embed_dim)
    g_results = torch.zeros(batch_size, embed_dim)
    t_results = torch.zeros(batch_size, embed_dim)

    total_times = 0

    if (args.testcpu or args.verify):

        total_times, emb_times, h_results = measure_cpu(warmups, steps, h_emb, h_indices, h_offsets)
        print("CPU: total test time: {0:.6f} seconds, emb {1:.6f} seconds  for {2:6d} steps ".format(total_times, emb_times, steps))
        print("CPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes * 1.0 * steps / emb_times / 1.0e9))
        print("CPU results: ", h_results)

    if (args.testgpu and torch.cuda.is_available()):

        total_times, emb_times, g_results = measure_gpu(warmups, steps, h_emb, h_indices, h_offsets)
        print("---------")
        print("GPU: total test time: {0:.6f} seconds, emb {1:.6f} seconds  for {2:6d} steps ".format(total_times, emb_times, steps))
        print("GPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes * 1.0 * steps / emb_times / 1.0e9))
        print("---------")
        print("GPU results: ", g_results)
        g_results = g_results.to('cpu')

        if (args.verify and args.testcpu):
            if (torch.equal(h_results, g_results)):
                print("Success! CPU results and GPU results match!\n")
            else:
                print("Failed!  CPU and GPU results does not match!\n")
                print("CPU results:")
                print(h_results)
                print("GPU results:")
                print(g_results)

    if (args.testtpu):

        total_times, emb_times, t_results = measure_tpu(warmups, steps, h_emb, h_indices, h_offsets, args)

        print("---------")
        print("TPU: total test time: {0:.6f} seconds, emb {1:.6f} seconds  for {2:6d} steps ".format(total_times, emb_times, steps))
        print("TPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes * 1.0 * steps / emb_times / 1.0e9))
        print("---------")
        print("TPU results: ", t_results)
        t_results = t_results.to('cpu')

        if (args.verify and args.testcpu):
            if (torch.equal(h_results, t_results)):
                print("Success! CPU results and TPU results match!\n")
            else:
                print("Failed!  CPU and TPU results does not match!\n")

if __name__ == "__main__":
    main()
