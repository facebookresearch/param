# Execution Trace Replay (et_replay)
`et_replay` is a tool designed for replaying Chakra Execution Traces (ET) from machine learning models.

## Installation
To install `param`, use the following commands:

```bash
$ git clone --recurse-submodules git@github.com:facebookresearch/param.git
$ conda create -n param python=3.8.0
$ conda activate param
$ cd param
$ pip3 install -r requirements.txt
$ cd train/comms/pt/
$ pip3 install .
$ cd -
$ cd train/compute/python/
$ pip3 install -r requirements.txt
$ pip3 install .
$ cd -
```

## Running et_replay
To use et_replay, execution traces are required.
Start by collecting an execution trace using the command below. This command runs a benchmark with specific configurations and enables execution tracing.
```bash
$ python -m param_bench.train.compute.python.pytorch.run_benchmark -c train/compute/python/examples/pytorch/configs/simple_add.json --et
```

After collecting the trace, replay it with the following command. Set the warm-up iteration count to at least 1 to exclude tensor transfer time to GPUs.
```bash
$ python -m param_bench.train.compute.python.tools.et_replay --input <trace_path> --warmup-iter 10 --iter 50 --compute --profile-replay
```

> Note: When analyzing performance values from et_replay, refer to the collected Kineto traces rather than the execution time reported by et_replay. Kineto traces are only collected when --profile-replay is provided.
