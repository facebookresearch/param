DLRM comms is an effort to develop unified benchmarking framework to
characterize training platform backends.

The DLRM-Comms benchmark offers a single point solution to perform both top-down
(DLRM application) and bottoms-up (collectives) operations for any given
communication backend.

Currently the benchmark supports Pytorch-NCCL backend, and PyTorch-XLA backend.

The bottoms-up benchmark (comms.py) is designed similar to nccl-tests, and the
top-down benchmark (dlrm.py) is similar to opensource dlrm benchmark except it
only implements communication primitives.


Usage:

Comms.py
mpirun -np <num-processes> -N <processes per node> ./comms.py \
    --master-ip <master-node-ip-address>
    --b <begin-size> \
    --e <end-size> \
    --n <num-iters> \
    --f <step-factor> \
    --z <blocking/non-blocking> \
    --collective <collective-to-test>

example:
mpirun -np 16 -N 8 ./comms.py --master-ip <node-1-ip> --b 8 --e 256M --n 100 \
    --f 2 --z 1 --collective all-to-all

dlrm.py
mpirun -np <num-processes> -N <processes per node> ./dlrm.py \
    --master-ip <master-node-ip-address>
    --arch-sparse-feature-size <spare-feature-size> \
    --arch-embedding-size <embedding-table-sizes> \
    --arch-mlp-bot <layer-dimensions of bottom layers> \
    --arch-mlp-top <layer-dimensions of top layers> \
    --mini-batch-size <mini-batch-sizes> \
    --num-batches <number-of-batches>

example:
mpirun -np 16 -N 8 ./dlrm.py --master-ip <node-1-ip> --mini-batch-size 32 \
    --num-batches 100 \
    --arch-mlp-bot 1024-256 \
    --arch-sparse-feature-size 64 \
    --arch-embedding-size "10000-10000"
