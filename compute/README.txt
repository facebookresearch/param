
Unified compute kernel benchmarks for DLRM under PyTorch interface.

Currently two important compute kernels are identified.  
One is GEMM (or MatMul) and the other is EmbeddingBag operation. 

A unified PyTorch implementation is developed to measure the performance
on different platforms, including CPUs, GPUs, and TPUs.
The TPU implementation is through PyTorch/XLA.

Usage:

A) GEMM
Measuring the performance of Z(m,n) = X(m,k) x Y(k,n)
Parameters:
-m, -n, -k : matrix dimension
--dtype : data type
--testcpu : measure CPU performance
--testgpu : measure GPU performance
--testtpu : measure TPU performance

Running example:

1. Measuring the performance on GPU for data type float32 
python pytorch_gemm.py -m 1024 -n 2048 -k 1024 --testgpu --dtype=float32

Following are frequently used matrix shapes for DLRM, 
#BATCH is the batch size, could be 128, 256, 512, 1024.

M	N	K
#Batch	4096	4096
#Batch	1024	1024
4096    4096    #Batch
1024    1024    #Batch

2) EmbeddingBag
Measure the EmbeddingBag performance for a table(features, embdim).
Parameters:
--features : size of the dictionary of embedding
--embdim : size of embedding vector
--batch : batch size
--nnz : pooling fatcor size

Running Example:

Measuring the performance on TPU:
python pytorch_emb.py --features=14000000 --embdim=128 --nnz=28 --batch=512 --testtpu

Note that on TPU, due to the current performance concern of EmbeddingBag usng PyTorch/XLA, 
we also support an alternative implementation, XlaEmbeddingBag, which can be
invoked through --usexlabag


