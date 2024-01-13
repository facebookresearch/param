# PARAM Compute Tools

This README provides information on using the `et_replay` and `trace_link.py` tools for working with execution traces.

## Installation
To install the tools:
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

## Execution Trace Replay (et_replay)
The `et_replay` tool replays collected Chakra execution traces.

To collect a trace:
```bash
$ python -m param_bench.train.compute.python.pytorch.run_benchmark \
  -c train/compute/python/examples/pytorch/configs/simple_add.json --et
```

To replay the trace:
```bash
$ python -m param_bench.train.compute.python.tools.et_replay \
  --input <trace_path> --warmup-iter 10 --iter 50 --compute --profile-replay
```
> Note: When analyzing performance values from et_replay, refer to the collected Kineto traces rather than the execution time reported by et_replay. Kineto traces are only collected when --profile-replay is provided.

## Trace Linker (trace_link)
The `trace_link.py` tool links PyTorch execution traces and Kineto traces into a PyTorch ET+ format.
This merged format retains event timing information.

Example usage:
```bash
$ python tools/trace_link.py\
  --pytorch-et-file <PyTorch execution trace file> \
  --kineto-file <Kineto trace file> \
  --output-file <PyTorch execution trace plus file>
```

The merged PyTorch ET+ trace can then be converted into the Chakra format for simulation and analysis.
