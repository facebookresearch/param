# PARAM Compute Benchmark

## Overview
The compute benchmarks have gone under development to support new use cases. The previous [standalone](standalone) benchmarks remain unchanged.

The general motivation and design philosophy for the new microbenchmarks are:
* Support a wide variety of PyTorch operators and workloads.
* Decoupled workload configuration from Python code. It allows building tools in a pipeline to collect operator configuration from production runs, generate microbenchmark inputs, gather metrics.
* Generates human readable and easy to parse output data format for downstream tools.
* A library interface allowing exeternal tools to run and collect data results from the microbenchmarks.
* Support replay of workload through PyTorch execution graph.

For design and implementation details, please look at the [development documentation](development.md).

## Installation
We use `setuptools` to install/uninstall the `parambench-train-compute` package:

```shell
# Inside dir "param/train/compute/pytnon"
# Install package
> python setup.py install

# Uninstall package
> python -m pip uninstall parambench-train-compute
```

## Usage
After install `parambench-train-compute` a package using the `setuptools`, it can be run as:
```shell
# Run benchmark tool script module
> python -m param_bench.train.compute.python.pytorch_benchmark --config test/pytorch/test_op.json
```

### Options
```shell
python -m param_bench.train.compute.python.pytorch_benchmark [-h] --config CONFIG [--warmup WARMUP] [--iter ITER] [--metric] [--device DEVICE] [--out-file-name OUT_FILE_NAME] [-v]
```