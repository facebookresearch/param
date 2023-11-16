# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

import torch


def measure_cpu(a, b, steps):

    global c
    start = time.perf_counter()
    for i in range(steps):
        c = torch.mm(a, b)
    end = time.perf_counter()
    c.to("cpu")
    return end - start


def measure_gpu(a, b, steps):

    global c
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(steps):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    c.to("cpu")
    return end - start


def measure_xla(a, b, steps):

    import torch_xla

    def sync(tensor, dev):
        torch_xla._XLAC._xla_sync_multi(
            [tensor], devices=[str(dev)], wait=True, sync_xla_data=True
        )

    c = torch.mm(a, b)

    start = time.perf_counter()
    for _ in range(steps):
        # Add data dependency to prevent loop elimination
        # The PyTorch/XLA lazy evaluation will eliminate the loop
        # Simplier data dependency will not work
        b[0] = torch.min(c[0], b[0])
        c = torch.min(c, torch.mm(a, b))

    sync(c, c.device)
    end = time.perf_counter()
    # c.to('cpu')
    return end - start


def run_single(args, m, n, k):

    dtype = args.dtype
    device = args.device
    warmups = args.warmups
    steps = args.steps

    dt = torch.float32
    if dtype == "float16" or dtype == "half":
        dt = torch.float16
    elif dtype == "bfloat16":
        dt = torch.bfloat16
    elif dtype == "tf32":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    torch.manual_seed(0)

    elap = 0.0

    a = torch.randn(m, k).to(dt)
    b = torch.randn(k, n).to(dt)
    c = torch.zeros(m, n).to(dt)

    if device == "cpu":

        measure_cpu(a, b, warmups)
        elap = measure_cpu(a, b, steps)

    elif device == "gpu":

        if torch.cuda.is_available():
            # ncuda = torch.cuda.device_count()
            # print("There are {} cuda devices".format(ncuda))
            # print("The first cuda device name is {} ".format(torch.cuda.get_device_name()))
            cuda0 = torch.device("cuda:0")
            with torch.cuda.device(cuda0):
                acuda = a.to(cuda0)
                bcuda = b.to(cuda0)
                measure_gpu(acuda, bcuda, warmups)
                elap = measure_gpu(acuda, bcuda, steps)
        else:
            print("CUDA is not available")
            sys.exit(1)

    else:
        # import torch_xla
        import torch_xla.core.xla_model as xm

        # alldev = xm.get_xla_supported_devices()
        # allrealdev = xm.xla_real_devices(alldev)
        # print("Found {0} XLA devices: {1}".format(len(allrealdev), allrealdev))

        dev = xm.xla_device()
        a = a.to(dev)
        b = b.to(dev)
        c = c.to(dev)
        measure_xla(a, b, warmups)
        xm.mark_step()
        elap = measure_xla(a, b, steps)
        xm.mark_step()

    return elap


def run(args, dataset):

    print("----------------------------------------------------------------")
    print("         M         N          K          Time(s)      Rate(TF/s)")
    print("----------------------------------------------------------------")
    for i in range(len(dataset)):
        m, n, k = dataset[i]
        elap = run_single(args, m, n, k)
        elap /= args.steps
        print(
            "{0:10}, {1:10}, {2:10},     {3:10.6f}     {4:.3f} ".format(
                m, n, k, elap, m * n * k * 2 * 1.0 / elap / 1.0e12
            )
        )


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description="Measure the performance of GEMM using mm, or matmul"
    )
    # model related parameters
    parser.add_argument("-m", "--msize", type=int, default=1024)
    parser.add_argument("-n", "--nsize", type=int, default=1024)
    parser.add_argument("-k", "--ksize", type=int, default=1024)
    parser.add_argument("-t", "--dtype", type=str, default="float32")
    parser.add_argument(
        "-d", "--device", choices=["cpu", "gpu", "tpu"], type=str, default="cpu"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()

    d = [(args.msize, args.nsize, args.ksize)]
    run(args, d)


if __name__ == "__main__":
    main()  # pragma: no cover
