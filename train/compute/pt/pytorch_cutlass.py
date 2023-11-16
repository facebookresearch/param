# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

import torch


def measure_blas(a, b, steps):

    global c
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    c.to("cpu")
    return end - start


def measure_tlass(a, b, steps):
    torch.ops.load_library("//caffe2/torch/fb/cutlass:cutlass_gemm")

    global c
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        c = torch.ops.fb.mm(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    c.to("cpu")
    return end - start


def run_single(args, m, n, k, func):

    dtype = args.dtype
    warmups = args.warmups
    steps = args.steps

    dt = torch.float32
    if dtype == "float16" or dtype == "half":
        dt = torch.float16
    elif dtype == "bfloat16":
        dt = torch.bfloat16

    torch.manual_seed(0)

    elap = 0.0

    a = torch.randn(
        m,
        k,
    ).to(dt)
    b = torch.randn(k, n).to(dt)
    c = torch.zeros(m, n).to(dt)

    if torch.cuda.is_available():
        # ncuda = torch.cuda.device_count()
        # print("There are {} cuda devices".format(ncuda))
        # print("The first cuda device name is {} ".format(torch.cuda.get_device_name()))
        cuda0 = torch.device("cuda:0")
        with torch.cuda.device(cuda0):
            acuda = a.to(cuda0)
            bcuda = b.to(cuda0)
            if func == "blas":
                measure_blas(acuda, bcuda, warmups)
                elap = measure_blas(acuda, bcuda, steps)
            else:
                measure_tlass(acuda, bcuda, warmups)
                elap = measure_tlass(acuda, bcuda, steps)
    else:
        print("CUDA is not available")
        sys.exit(1)

    return elap


def run(args, dataset):

    print("----------------------------------------------------------------")
    print("         M         N          K          Time(s)      Rate(TF/s)")
    print("----------------------------------------------------------------")
    for i in range(len(dataset)):
        m, n, k = dataset[i]
        elap = run_single(args, m, n, k, "blas")
        elap /= args.steps
        print(
            "{0:10}, {1:10}, {2:10},     {3:10.6f}     {4:.3f} ".format(
                m, n, k, elap, m * n * k * 2 * 1.0 / elap / 1.0e12
            )
        )
        elap = run_single(args, m, n, k, "tlass")
        elap /= args.steps
        print(
            "{0:10}, {1:10}, {2:10},     {3:10.6f}     {4:.3f} ".format(
                m, n, k, elap, m * n * k * 2 * 1.0 / elap / 1.0e12
            )
        )


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description="Measure and compare the performance of GEMM cuBlas and cuTlass"
    )
    # model related parameters
    parser.add_argument("-m", "--msize", type=int, default=1024)
    parser.add_argument("-n", "--nsize", type=int, default=1024)
    parser.add_argument("-k", "--ksize", type=int, default=1024)
    parser.add_argument("-t", "--dtype", type=str, default="float32")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()

    d = [(args.msize, args.nsize, args.ksize)]
    run(args, d)


if __name__ == "__main__":
    main()  # pragma: no cover
