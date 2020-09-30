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

### GEMM :

```bash
python pytorch_gemm.py \
    --msize/-m \
    --nsize/-n \
    --ksize/-k \
    --dtype \
    --testgpu \
    --testcpu \
    --testtpu \
    --steps \
    --warmups

```
Example: measure float performance on GPU:
```bash
python pytorch_gemm.py -m 1024 -n 2048 -k 1024 --testgpu --dtype=float32
```

### MLP
```bash
python pytorch_linear.py \
    --device \
    --optimizer-type \
    --data-type \
    --layer-num \
    --batch-size \
    --input-size \
    --output-size \
    --hidden-size \
    --num-batches
```

Example: Measure the performance of a MLP with 18 hidden layer, layer size 1024
```bash
python pytorch_linear.py --device gpu --layer-num 18  --batch-size 128 --input-size 1024 \
       --hidden-size 1024  --output-size 1024 --num-batches 100 \
       --data-type=float16 --optimizer-type=sgd
```

### EmbeddingBag
```bash
python pytorch_emb.py \
    --features \
    --embdim \
    --nnz \
    --batch \
    --steps \
    --warmups \
    --testcpu \
    --testgpu \
    --testtpu \
    --randomseed \
    --usexlabag
```

Note that on TPU, due to the current performance concern of EmbeddingBag, we
also support an alternative implementation, XlaEmbeddingBag, which can be
invoked through --usexlabag

Example: Measure the EmbeddingBag performance for a table(26000000, 128) with 
batch size 512 and pooling factor 28 on TPU:
```bash
python pytorch_emb.py --features=26000000 --embdim=128 --nnz=28 --batch=512
--testtpu
```

