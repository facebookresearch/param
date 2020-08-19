
# measuring gemm (matmul, mm) performance using pytorch
# using one matrix
# exaple: python pytorch_gemm.py -m 4096 -n 4096 -k 4096  --verify  --testgpu --dtype=float16

import time
#import numpy as np
import sys
import torch

def measure_cpu(a, b, steps, m):


    global c
    start = time.perf_counter()
    for i in range(steps):
        c = torch.mm(a, b)
        i1 = i % m
        a[i1][0] = a[i1][0] + c[i1][0]   # prevent mm done only once, seems not necessary ?
    end = time.perf_counter()
    c.to('cpu')
    return end - start

def measure_gpu(a, b, steps, m):


    global c
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(steps):
        c = torch.mm(a, b)
        i1 = i % m
        a[i1][0] = a[i1][0] + c[i1][0]   # prevent mm done only once
    torch.cuda.synchronize()
    end = time.perf_counter()
    c.to('cpu')
    return end - start

def measure_xla(a, b, steps, m):


    import torch_xla

    def sync(tensor, dev):
        torch_xla._XLAC._xla_sync_multi([tensor], devices=[str(dev)], wait=True, sync_xla_data=True)

    global c
    start = time.perf_counter()
    for _ in range(steps):
        c = torch.mm(a, b)
        # i1 = i % m
        # a[i1][0] = a[i1][0] + c[i1][0]   #This will slow down TPU performance significantly
        sync(c, c.device)
    end = time.perf_counter()
    c.to('cpu')
    return end - start

if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser(
        description="Measure the performance of GEMM using mm, or matmul")
    # model related parameters
    parser.add_argument("-m", "--msize", type=int, default=1024)
    parser.add_argument("-n", "--nsize", type=int, default=1024)
    parser.add_argument("-k", "--ksize", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--testcpu", action='store_true')
    parser.add_argument("--testgpu", action='store_true')
    parser.add_argument("--testtpu", action='store_true')
    parser.add_argument("--verify", action='store_true')
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()

    m = args.msize
    n = args.nsize
    k = args.ksize
    dt = torch.float32
    if (args.dtype == "float16" or args.dtype == "half"):
        dt = torch.float16
    elif (args.dtype == "bfloat16"):
        dt = torch.bfloat16

    print("Test problem size for m, n, k are : ", m, n, k)
    print("Test problem data type : ", dt)

    torch.manual_seed(0)

    warmups = args.warmups
    steps = args.steps
    elap1 = 0.0
    elap2 = 0.0

    a = torch.randn(m, k).to(dt)
    b = torch.randn(k, n).to(dt)
    c = torch.zeros(m, n).to(dt)

    # cpu and gpu returns the same results
    a_save0 = torch.zeros(m)
    a_save = a_save0.to(dt)
    for i in range(m):
        a_save[i] = a[i][0]

    if (args.testcpu):
        measure_cpu(a, b, warmups, m)
        elap1 = measure_cpu(a, b, steps, m)

        print("c device: ", c.device, type(c), c.dtype)
        print("c[2x2] : ", c.narrow(0, 0, 2).narrow(1, 0, 2))
        print("------")
        print("CPU Time is {0:.6f} seconds, rate {1:.3f} GFlops for iter {2}".format(elap1,
               m * n * k * 2 * 1.0/(elap1 * 1000000000 / steps), steps))
        print("------\n")

        c.fill_(0)
        for i in range(m):
            a[i][0] = a_save[i]

    # 2. measure on GPU
    is_cuda = torch.cuda.is_available()
    if (args.testgpu and is_cuda):
        ncuda = torch.cuda.device_count()
        print("There are {} cuda devices".format(ncuda))
        print("The first cuda device name is {} ".format(torch.cuda.get_device_name()))
        cuda0 = torch.device('cuda:0')
        with torch.cuda.device(cuda0):
            acuda = a.to(cuda0)
            bcuda = b.to(cuda0)
            measure_gpu(acuda, bcuda, warmups, m)
            elap1 = measure_gpu(acuda, bcuda, steps, m)

            print("c device: ", c.device, type(c), c.dtype)
            print("c[2x2] : ", c.narrow(0, 0, 2).narrow(1, 0, 2))
            print("------")
            print("GPU Time is {0:.6f} seconds, rate {1:.3f} GFlops for iter {2} ".format(elap1,
                   m * n * k * 2 * 1.0 / (elap1 * 1000000000 / steps), steps))
            print("------\n")

    if (args.testtpu):
        # import torch_xla
        import torch_xla.core.xla_model as xm

        alldev = xm.get_xla_supported_devices()
        allrealdev = xm.xla_real_devices(alldev)
        print("Found {0} XLA devices: {1}".format(len(allrealdev), allrealdev))
        print(torch.__version__)

        dev = xm.xla_device()
        a = a.to(dev)
        b = b.to(dev)
        c = c.to(dev)
        measure_xla(a, b, warmups, m)
        elap1 = measure_xla(a, b, steps, m)

        print("c device: ", c.device, type(c), c.dtype)
        print("c[2x2] : ", c.narrow(0, 0, 2).narrow(1, 0, 2))
        print("------")
        print("TPU(xla) Time is {0:.6f} seconds, rate {1:.3f} GFlops for iter {2} ".format(elap1,
              m * n * k * 2 * 1.0 / (elap1 * 1000000000 / steps), steps))
        print("------\n")
