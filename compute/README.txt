
Unified compute kernel benchmarks for DLRM under PyTorch interface.

Currently there are two kernels are identified, one is GEMM (or MatMul) and the
other is EmbeddingBag operations. The benchmark is developed to measure the
performance of individual operations.

They could be used to measure the performance across different platforms, such
as CPU, GPU, or TPU. The TPU implementation is through PyTorch/XLA.

Here are some run examples:

1. Measure GEMM performance on GPU for matrix Z(m,n) = X(m,k) x Y(k, n):
python pytorch_gemm.py -m 1024 -n 2048 -k 1024 --testgpu --dtype=float32

2. Measure the EmbeddingBag performance for a table(features, embdim) with
batch size 512 and pooling factor 28 on TPU:

python pytorch_emb.py --features=26000000 --embdim=128 --nnz=28 --batch=512
--testtpu

Note that on TPU, due to the current performance concern of EmbeddingBag, we
also support an alternative implementation, XlaEmbeddingBag, which can be
invoked through --usexlabag

