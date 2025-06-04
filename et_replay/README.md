# License
Chakra replay is released under the Apache 2.0 license. Please see the LICENSE file for more information.

# Execution Trace Replay (chakra_replay)
`chakra_replay` is a tool designed for replaying Chakra Execution Traces (ET) from machine learning models.

## Installation
To install `chakra_replay`, use the following commands:

```bash
$ git clone --recurse-submodules git@github.com:pytorch-labs/chakra_replay.git
$ conda create -n chakra_replay python=3.10
$ conda activate chakra_replay
$ cd chakra_replay
$ pip3 install -r requirements.txt
$ pip3 install .
```

## Running et_replay
Replay the trace with the following command. Set the warm-up iteration count to at least 1 to exclude tensor transfer time to GPUs.
```bash
$ python -m tools.et_replay --input <trace_path> --warmup-iter 10 --iter 50 --compute --profile-replay
```

> Note: When analyzing performance values from et_replay, refer to the collected Kineto traces rather than the execution time reported by et_replay. Kineto traces are only collected when --profile-replay is provided.
