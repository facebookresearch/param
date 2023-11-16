# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

import numpy as np
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
        bsz = emb.size(0) // self.offset
        emb = emb.reshape(bsz, self.offset, *emb.size()[1:])
        reduce_fn = getattr(torch, self.mode)
        return reduce_fn(emb, axis=1)


def measure_cpu(warmups, steps, h_emb, h_indices, h_offsets):

    start = time.perf_counter()
    for i in range(warmups + steps):
        results = h_emb(h_indices, h_offsets)
        if i < warmups:
            start = time.perf_counter()
    end = time.perf_counter()

    return end - start, results


def measure_gpu(warmups, steps, h_emb, h_indices, h_offsets):

    # ncuda = torch.cuda.device_count()
    # print("There are {} cuda devices".format(ncuda))
    # print("The current cuda device name is {} ".format(torch.cuda.get_device_name()))
    cuda0 = torch.device("cuda:0")
    with torch.cuda.device(cuda0):
        g_emb = h_emb.to(cuda0)
        g_indices = h_indices.to(cuda0)
        g_offsets = h_offsets.to(cuda0)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for i in range(warmups + steps):
            results = g_emb(g_indices, g_offsets)
            # torch.cuda.synchronize()
            if i < warmups:
                torch.cuda.synchronize()
                start = time.perf_counter()
        torch.cuda.synchronize()
        end = time.perf_counter()

    return end - start, results


def measure_tpu(warmups, steps, h_emb, h_indices, h_offsets, usexlabag, batch, nnz):

    import os

    import torch_xla
    import torch_xla.core.xla_model as xm

    # If emb table is too large will cause protobuf error,
    # we have to split them
    tsize = int(os.environ.get("MODEL_PARTITION_SIZE", 60000000))

    def syncTPU(tensor):
        torch_xla._XLAC._xla_sync_multi(
            [tensor], devices=[], wait=True, sync_xla_data=True
        )

    # alldev = xm.get_xla_supported_devices()
    # allrealdev = xm.xla_real_devices(alldev)
    # print("Found {0} devices: {1}".format(len(allrealdev), allrealdev))

    dev = xm.xla_device()
    if usexlabag:
        features = h_emb.n
    else:
        features = h_emb.weight.shape[0]

    if features > tsize:
        if usexlabag:
            tsplit = torch.split(h_emb.embtable.weight, tsize, dim=0)
        else:
            tsplit = torch.split(h_emb.weight, tsize, dim=0)
        tsplit = list(tsplit)
        for i, chunk in enumerate(tsplit):
            tsplit[i] = chunk.to(dev)

        t = nn.Parameter(torch.ones(10, 10))
        if usexlabag:
            h_emb.embtable.weight = t
            t_emb = h_emb.to(dev)
            tsplit = torch.cat(tsplit)
            t_emb.embtable.weight = nn.Parameter(tsplit)
            # print("Xla EMB weight shape: ", t_emb.embtable.weight.shape, " on device: ", str(dev))
        else:
            h_emb.weight = t
            t_emb = h_emb.to(dev)
            tsplit = torch.cat(tsplit)
            t_emb.weight = nn.Parameter(tsplit)
            # print("EMB weight shape: ", t_emb.weight.shape, " on device: ", str(dev))
    else:
        t_emb = h_emb.to(dev)

    t_indices = h_indices.to(dev)
    t_offsets = h_offsets.to(dev)

    start = time.perf_counter()
    results = t_emb(t_indices, t_offsets)
    start = time.perf_counter()
    for _ in range(steps):
        results = t_emb(t_indices, t_offsets)
        syncTPU(results)

    end = time.perf_counter()
    results = results.cpu()

    return end - start, results


def init_indices(alpha, features, batch, nnz):

    if alpha == 0.0:
        indices = torch.randint(0, features, (batch * nnz,))
    else:
        # Zipf (i.e. zeta) distribution
        pmf = np.power(np.arange(1, features + 1, dtype=np.float64), -alpha)
        pmf = pmf / pmf.sum()
        # oversample and then remove duplicates to obtain sampling without replacement
        indices = np.random.choice(features, size=(batch, 2 * nnz), replace=True, p=pmf)
        for b in range(batch):
            r = set()
            for x in indices[b, :]:
                if x in r:
                    continue
                else:
                    r.add(x)
                    if len(r) == nnz:
                        break

            indices[b, :nnz] = list(r)
        indices = torch.flatten(torch.tensor(indices[:, :nnz])).to(torch.int64)

    return indices


def run_single(args, features, embdim, nnz, batch):

    device = args.device
    random_seed = args.randomseed
    warmups = args.warmups
    steps = args.steps

    torch.manual_seed(random_seed)

    h_indices = init_indices(args.alpha, features, batch, nnz)
    h_offsets = torch.zeros(batch, dtype=torch.int64)
    for i in range(batch):
        h_offsets[i] = i * nnz

    # Use xlabag now on TPU
    # EmbeddingBag is not optimized by PyTorch/XLA currently
    if not device == "tpu" or not args.usexlabag:
        h_emb = nn.EmbeddingBag(features, embdim, mode="sum")
        total_bytes = batch * nnz * embdim * h_emb.weight.element_size()
    else:
        # print("using XlaBag instead of torch.nn.EmbeddingBag now")
        h_emb = XlaEmbeddingBag(features, embdim, "sum", nnz)
        total_bytes = batch * nnz * embdim * h_emb.embtable.weight.element_size()

    h_results = torch.zeros(batch, embdim)

    if device == "cpu":
        emb_times, h_results = measure_cpu(warmups, steps, h_emb, h_indices, h_offsets)

    elif device == "gpu":
        if torch.cuda.is_available():
            emb_times, h_results = measure_gpu(
                warmups, steps, h_emb, h_indices, h_offsets
            )
        else:
            print("CUDA is not available, could not run on GPU")
            sys.exit(1)

    else:
        emb_times, t_results = measure_tpu(
            warmups, steps, h_emb, h_indices, h_offsets, args.usexlabag, batch, nnz
        )

    return emb_times, total_bytes


def run(args, dataset):

    print(
        "---------------------------------------------------------------------------------"
    )
    print(
        "    Features    embdim    nnz     batch      Time(s)/step   Data(MB)   BW(GB/s)"
    )
    print(
        "---------------------------------------------------------------------------------"
    )

    for i in range(len(dataset)):
        features, embdim, nnz, batch = dataset[i]
        elap, total_bytes = run_single(args, features, embdim, nnz, batch)
        elap /= args.steps
        total_bytes /= 1.0e6
        print(
            "{0:10},  {1:6},  {2:6},  {3:8},    {4:10.6f}, {5:10.1f},  {6:8.3f}".format(
                features,
                embdim,
                nnz,
                batch,
                elap,
                total_bytes,
                total_bytes / elap / 1.0e3,
            )
        )


def main() -> None:

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
    parser.add_argument("-t", "--dtype", type=str, default="float32")
    parser.add_argument(
        "-d", "--device", choices=["cpu", "gpu", "tpu"], type=str, default="cpu"
    )
    parser.add_argument("--usexlabag", action="store_true")
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="Zipf param. Use uniform if == 0.0"
    )

    args = parser.parse_args()

    num_features = args.features
    embdim = args.embdim
    nnz = args.nnz
    batch = args.batch
    steps = args.steps
    warmups = args.warmups

    d = [(num_features, embdim, nnz, batch)]
    run(args, d)


if __name__ == "__main__":
    main()  # pragma: no cover
