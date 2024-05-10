# PARAM benchmark - Communication benchmarks

PARAM-Comms is an effort to develop a unified benchmarking framework to
characterize training platform backends. Currently, the benchmark supports
Pytorch Distributed and PyTorch-XLA backends.

The PARAM-Comms benchmark offers a single point solution to perform both top-down
(DLRM application) and bottoms-up (collectives) operations for any given
communication backend.

The Collective-Comms benchmark (`comms.py`) is designed similar to nccl-tests
for evaluating collective operations, such as All-reduce and All-to-all, through PyTorch backends.
The DLRM-Comms benchmark (`dlrm.py`) is similar to the open-source DLRM benchmark except it
only implements communication primitives.
The Trace Replay benchmark (`commsTraceReplay.py`) is designed to replay the communication patterns captured
from any distributed PyTorch workloads.

## Usage:

### Collective-Comms benchmark (`comms.py`)
```bash
mpirun -np <num-processes> -N <processes per node> --hostfile <file contains host list> ./comms.py \
    --master-ip 127.0.0.1
    --b <begin-size> \
    --e <end-size> \
    --n <num-iters> \
    --f <step-factor> \
    --z <blocking/non-blocking> \
    --collective <collective-to-test>
```
Example:
```bash
mpirun -np 16 -N 8 --hostfile ./hfile ./comms.py --master-ip $(head -n 1 ./hfile.txt) --b 8 --e 256M --n 100 \
    --f 2 --z 1 --collective all_to_all --backend nccl --device cuda --log INFO
```

### DLRM-Comms benchmark (`dlrm.py`)
```bash
mpirun -np <num-processes> -N <processes per node> --hostfile <file contains host list> ./dlrm.py \
    --master-ip <master-node-ip-address>
    --arch-sparse-feature-size <spare-feature-size> \
    --arch-embedding-size <embedding-table-sizes> \
    --arch-mlp-bot <layer-dimensions of bottom layers> \
    --arch-mlp-top <layer-dimensions of top layers> \
    --mini-batch-size <mini-batch-sizes> \
    --num-batches <number-of-batches>
```
Example:
```bash
mpirun -np 16 -N 8 --hostfile ./hfile ./dlrm.py --master-ip $(head -n 1 ./hfile.txt) --mini-batch-size 32 \
    --num-batches 100 \
    --arch-mlp-bot 1024-256 \
    --arch-sparse-feature-size 64 \
    --arch-embedding-size "10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000-10000"
```

### Trace Replay benchmark (`commsTraceReplay.py`)
```bash
mpirun -np <num-processes> -N <processes per node> --hostfile <file contains host list> ./commsTraceReplay.py \
    --master-ip 127.0.0.1 --trace-path /path/to/traces --dry-run
```
Example:
```bash
mpirun -np 16 -N 8 --hostfile ./hfile ./commsTraceReplay.py --master-ip $(head -n 1 ./hfile.txt) \
    --backend nccl --device cuda \
    --trace-path /path/to/commTraces
```
Note that there should be one trace file (in JSON format) per rank.
