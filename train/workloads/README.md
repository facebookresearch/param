# Distributed DLRM

The implementation is developed based on [DLRM dist_exp branch](https://github.com/facebookresearch/dlrm/tree/dist_exp)
and add Facebook features and optimizations.

Currently you need to download [the following PR](https://github.com/facebookresearch/dlrm/pull/127)
to get the latest update. (Will be fixed soon.)

## Usage

Currently, it is launched with mpirun on multi-nodes. The hostfile need to be created or
a host list should be given. The DLRM parameters should be given in the same way as single
node master branch.
```bash
mpirun -np 128 -hostfile hostfile python dlrm_s_pytorch.py ...
```

## Example

large_arch_emb=$(printf '14000%.0s' {1..64})
large_arch_emb=${large_arch_emb_ads//"01"/"0-1"}

```bash
python dlrm_s_pytorch.py
   --arch-sparse-feature-size=128
   --arch-mlp-bot="2000-1024-1024-128"
   --arch-mlp-top="4096-4096-4096-1"
   --arch-embedding-size=$large_arch_emb
   --data-generation=random
   --loss-function=bce
   --round-targets=True
   --learning-rate=0.1
   --mini-batch-size=2048
   --print-freq=10240
   --print-time
   --test-mini-batch-size=16384
   --test-num-workers=16
   --num-indices-per-lookup-fixed=1
   --num-indices-per-lookup=100
   --arch-projection-size 30
   --use-gpu
```

Please check the README.md in the PR for more details.
