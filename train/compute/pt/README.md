# PARAM benchmark -- compute benchmarks

Unified compute kernel benchmarks for DLRM and other important AI workloads
under PyTorch interface.

Currently there are three kernels are identified, 
* GEMM (or MatMul) : Measure GEMM performance for matrix Z(m,n) = X(m,k) x Y(k, n)
* MLP (multilayer perceptron) : measure a series of FC layer performance
* EmbeddingBag : Measure the EmbeddingBag performance for table lookup

The benchmark is developed to measure the performance of individual 
operation or kernel and used to measure the performance across 
different platforms, such as CPU, GPU, or TPU. 

The TPU implementation is through PyTorch/XLA.

## Usage

A driver (`driver.py`) is developed, which can be used to run different kernels.
For each kernel, one or more datasets have been defined as 'A', 'B', 'C', etc.

```bash
python3 driver.py -h
usage: driver.py [-h] [--warmups WARMUPS] [--steps STEPS] --device {cpu,gpu,tpu} {gemm,emb,linear} ...

Measuring the Compute Kernel Performance Using PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --warmups WARMUPS     warmup times
  --steps STEPS         repeat times
  --device {cpu,gpu,tpu}
                        valid devices

kernels:
  {gemm,emb,linear}
    gemm                measure mm performance (m,k)*(k,n)=(m,n)
    emb                 measure EmbeddingBag performance
    linear              measure mlp performance
```

### Testing GEMM :

```bash
python3 driver.py --steps=100 --device='cpu' gemm --dataset='A'

Measuring the performance of  gemm  on device =  cpu
Steps = 100  warmups = 10
with matrix dataset  A , Data type:  float32

----------------------------------------------------------------
         M         N          K          Time(s)      Rate(GF/s)
----------------------------------------------------------------
       128,       4096,       4096,       0.519193     827.240
       256,       4096,       4096,       1.005778     854.058
       512,       4096,       4096,       2.214854     775.666
      1024,       4096,       4096,       3.388758     1013.933
       128,       1024,       1024,       0.555641     48.311
       256,       1024,       1024,       0.145774     368.291
       512,       1024,       1024,       0.177422     605.189
      1024,       1024,       1024,       0.215082     998.447

```

### Testing EmbeddingBag
```bash
python3 driver.py --steps=100 --device='cpu' emb --dataset='A'

Measuring the performance of  emb  on device =  cpu
Steps = 10  warmup = 1
with emb data A.
---------------------------------------------------------------------------------
    Features    embdim    nnz     batch      Time(s)/step   Data(MB)   BW(GB/s)
---------------------------------------------------------------------------------
  14000000,     128,      30,      2048,      0.002067,       31.5,    15.222
  14000000,     128,      30,      4096,      0.004611,       62.9,    13.644
  14000000,     128,      30,      8192,      0.006464,      125.8,    19.466
  14000000,     128,      30,     16384,      0.009102,      251.7,    27.649

```
Note that on TPU, due to the current performance concern of EmbeddingBag, we
also support an alternative implementation, XlaEmbeddingBag, which can be
invoked through --usexlabag

Example: Measure the performance of a MLP with 18 hidden layer, layer size 1024
```bash
python pytorch_linear.py --device gpu --layer-num 18  --batch-size 128 --input-size 1024 \
       --hidden-size 1024  --output-size 1024 --steps 100 \
       --dtype=float16 --optimizer-type=sgd
```

### Testing MLP Linear
```bash
python3 driver.py --steps=100 --device='cpu' linear --dataset='A'

Measuring the performance of  linear  on device =  cpu
Steps = 10  warmups = 1
with linear dataset  A , Data type:  float
--------------------------------------------------------------------------------
 #Layer    Input    Hidden    Output   Batch   Time(s)/step  QPS      Rate(GF/s)
--------------------------------------------------------------------------------

    18,    1024,    1024,    1024,     128,    0.344426,     371.6,       46.8
    18,    1024,    1024,    1024,     256,    0.206910,    1237.3,      155.7
    18,    1024,    1024,    1024,     512,    0.279407,    1832.5,      230.6

```

In addition, you can run individual kernels using
```bash
python3 pytorch_gemm.py ...
```
```bash
python3 pytorch_emb.py ...
```
```bash
python3 pytorch_linear.py ...
```
