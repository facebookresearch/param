# PARAM Training Compute Benchmark

## Overview
The training compute benchmarks have gone under development to support new use cases. The previous [standalone](../pt) benchmarks remain unchanged.

The general motivation and design philosophy for the new microbenchmarks are:
* Support a wide variety of PyTorch operators and workloads.
* Decoupled workload configuration from Python code. It allows building tools in a pipeline to collect operator configuration from production runs, generate microbenchmark inputs, gather metrics.
* Generates human readable and easy to parse output data format for downstream tools.
* A library interface allowing external tools to run and collect data results from the microbenchmarks.
* Support replay of workload through PyTorch execution trace.

For design and implementation details or make a contribution to the project, please look at the [development documentation](development.md).

## Installation
We use `setuptools` to install/uninstall the `parambench-train-compute` package:

```shell
# Inside dir "param/train/compute/pytnon"

# Install required dependencies
> pip install -r requirements.txt

# Install PARAM Compute package
> pip install .

# Uninstall package
> python -m pip uninstall parambench-train-compute
```

The installed packages are under **`param_bench.train.compute.python`**.

To use the [`FBGEMM_GPU`](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu) library and its operator benchmark workload ([`split_table_batched_embeddings_ops.py`](workloads/pytorch/split_table_batched_embeddings_ops.py)), please follow its set up instruction to download and install. It's not required for the compute benchmarks. During initialization, if an operator fail to import, it'll be ignored and will not affect other benchmarks.

Please make sure to install the `parambench-train-comms` package (`train/comms/pt`). This is important because some functions in this package reference those in the comms package.

## Usage
The bundled tool scripts such as [`run_benchmark.py`](pytorch/run_benchmark.py) are written using relative import paths as part of the `parambench-train-compute` package, so they must be ran as a module using the `python -m` option.

A reliable way to run the benchmarks is install `parambench-train-compute` as a package following the above instructions. Afterward, it can be ran as:
```shell
# Run benchmark tool script module
> python -m param_bench.train.compute.python.pytorch.run_benchmark -c examples/pytorch/configs/simple_add.json
```

Without installing the package, you can run a tool script as a module in the source directory:
```shell
# Inside dir "param/train/compute"
> python -m python.pytorch.run_benchmark -c python/examples/pytorch/configs/simple_add.json
```
However, this method may conflict with other packages (such as `fbgemm_gpu.split_table_batched_embeddings_ops`) that have its own modules under a `python` package.

Additional example configs can be found in [`examples/pytorch/configs/`](examples/pytorch/configs/).

### Benchmark Library
As a library, it can be used as any regular Python package:
```python
from param_bench.train.compute.python.lib.config import BenchmarkConfig
```
A complete example to generate benchmark config, run the benchmark, then get the results can be found in [`run_op.py`](examples/pytorch/run_op.py)

## PyTorch Benchmark Options
```
=> python -m param_bench.train.compute.python.pytorch.run_benchmark -h
usage: run_benchmark.py [-h] [-c CONFIG] [-w WARMUP] [-i ITERATION] [-b] [-d DEVICE] [-o OUTPUT_PREFIX] [-r RESUME_ID] [-s STOP_ID] [-a] [--cuda-l2-cache [{on,off}]] [--ncu] [--ncu-bin NCU_BIN] [--ncu-args-file NCU_ARGS_FILE] [--ncu-warmup NCU_WARMUP]
                        [--ncu-iteration NCU_ITERATION] [--nsys] [--nsys-bin NSYS_BIN] [--nsys-args-file NSYS_ARGS_FILE] [--nsys-warmup NSYS_WARMUP] [--nsys-iteration NSYS_ITERATION] [--run-batch-size RUN_BATCH_SIZE] [--batch-cuda-device BATCH_CUDA_DEVICE]
                        [--batch-cmd BATCH_CMD] [--exec-mode [{discrete,continuous,continuous_events}]] [-p] [-l LOG_LEVEL] [--version]

PyTorch Microbenchmarks

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        The benchmark config file.
  -w WARMUP, --warmup WARMUP
                        Number of warm up iterations.
  -i ITERATION, --iteration ITERATION
                        Number of benchmark iterations.
  -b, --backward        Include backward pass.
  -d DEVICE, --device DEVICE
                        Target device for benchmark.
  -o OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
                        File name prefix to write benchmark results.
  -r RESUME_ID, --resume-id RESUME_ID
                        Define a resume op_run_id to continue benchmark, skip all previous configs.
  -s STOP_ID, --stop_id STOP_ID
                        Define a stop op_run_id (exclusive) to stop benchmark, skip remaining configs.
  -a, --append          Append to output file, rather than overwrite.
  --cuda-l2-cache [{on,off}]
                        Set option for CUDA GPU L2 cache between iterations in discrete mode.
  --ncu                 Run NSight Compute to collect metrics.
  --ncu-bin NCU_BIN     Path to the NSight Compute (ncu) binary.
  --ncu-args-file NCU_ARGS_FILE
                        NSight Compute extra command line options (metrics etc.).
  --ncu-warmup NCU_WARMUP
                        NSight Systems number of warmup runs.
  --ncu-iteration NCU_ITERATION
                        NSight Systems number of measured iteration runs.
  --nsys                Run NSight Systems to collect metrics.
  --nsys-bin NSYS_BIN   Path to the NSight Systems (nsys) binary.
  --nsys-args-file NSYS_ARGS_FILE
                        NSight Systems extra command line options (metrics etc.).
  --nsys-warmup NSYS_WARMUP
                        NSight Systems number of warmup runs.
  --nsys-iteration NSYS_ITERATION
                        NSight Systems number of measured iteration runs.
  --run-batch-size RUN_BATCH_SIZE
                        Batch run input size (number of input configs to run in one launch), used by both NCU and NSYS.
  --batch-cuda-device BATCH_CUDA_DEVICE
                        CUDA GPU device ID to run batch job.
  --batch-cmd BATCH_CMD
                        Run batch job command.
  --exec-mode [{discrete,continuous,continuous_events}]
                        Set execution mode of the operators (discrete, continuous, continuous_events). Default=discrete
  -p, --profile         Enable profiler and tracing.
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        Log output verbosity.
  --version             Print version.
```

## Benchmark Configuration File
Benchmark configurations are defined in a JSON format. It can be stored in a file on disk, or being passed between external callers and the benchmark’s library interface. There are two types of configurations:
* Build configuration (optional)
  * Defines arguments used to construct and initialize the operator.
  * It’s optional for operators that do not require initialization.
* Input configuration
  * Defines arguments used to execute the operator.

An operator may or may not need to have a build configuration. For example, `torch.matmul` can be called directly with input arguments, so there's no need to specify a build configuration. Other operators require creating the operator first before running it:

```python
embedding = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse).to(device=device)
embedding(input, offset)
```
The library expects the benchmark configuration in the following JSON format (the notes in `<>` are comments):
```json
{
  "operator_name": {
    "build_iterator": "(optional) build iterator name",
    "input_iterator": "(optional) input iterator name",
    "build_data_generator": "(optional) data generator name",
    "input_data_generator": "(required) data generator name",
    "config": [
      <a list of config spec>
      {
        "build": [
          <(optional) a list of build spec>
        ]
        "input": [
          <(required) a list of input spec>
        ]
      }
    ]
  }
  <additional key:value of operator and its spec>
}
```
The **`"operator_name"`** is a key mapped to a concrete workload implementation defined inside [`workloads`](workloads) directory, for example [`workloads/pytorch/native_basic_ops.py`](workloads/pytorch/native_basic_ops.py).

## Default PyTorch Config Specification
For each **`"build"`** and **`"input"`** configuration, the __`"args"`__ and __`"kwargs"`__ JSON keys should be specified with a list of argument data specifications (see [PyTorch Data Types](#pyTorch-data-types)). This is synonymous to Python's __`*args`__ and __`**kwargs`__. As expected, **`"args"`** is positional and defined as a list. __`"kwargs"`__ is defined as a dictionary of `"kwarg_name": <data_type_dict>`.

**Example**
```json
{
  "torch.baddbmm": {
    "input_data_generator": "PyTorch:DefaultDataGenerator",
    "config": [
      {
        "input": [
          {
            "args": [
              {
                "dtype": "float",
                "shape": [2, 1, 512],
                "type": "tensor"
              },
              {
                "dtype": "float",
                "shape": [2, 512, 512],
                "type": "tensor"
              },
              {
                "dtype": "float",
                "shape": [2, 512, 512],
                "type": "tensor"
              }
            ],
            "kwargs": {
              "beta": {
                "type": "int",
                "value": 1
              },
              "alpha": {
                "type": "int",
                "value": 1
              }
            }
          }
        ]
      }
    ]
  }
}
```

### PyTorch Data Types Specification
Current supported data types and examples are listed here:
```json
{
  "type": "int",
  "value": 237
},
{
  "type": "int",
  "value_range": [100, 1000]
},
{
  "type": "long",
  "value": 8328
},
{
  "type": "long",
  "value_range": [1000, 10000]
},
{
  "type": "float",
  "value": 1.2
},
{
  "type": "float",
  "value_range": [0.0, 2.0]
},
{
  "type": "double",
  "value": 3.4
},
{
  "type": "double",
  "value_range": [0.5, 5.5]
},
{
  "type": "bool",
  "value": false
},
{
  "type": "device",
  "value": "cpu"
},
{
  "type": "str",
  "value": "a string value"
},
{
  "type": "genericlist",
  "value": [
    {
      "type": "int",
      "value": 237,
    },
    {
      "type": "tensor",
      "dtype": "float",
      "shape": [16, 32],
    }]
},
{
  "type": "tuple",
  "value": [
    {
      "type": "tensor",
      "dtype": "float",
      "shape": [16, 32],
    },
    {
      "type": "tensor",
      "dtype": "float",
      "shape": [16, 32],
    }]
},
{
  "type": "tensor",
  "dtype": "float",
  "shape": [128, 256]
}
```
**Notes**
* `"value_range"` specifies a random value in the `[min, max]` range to be generated for the argument.

To help construct benchmark configs, some utility functions are available in [examples/pytorch/run_op.py](examples/pytorch/run_op.py).

### Configuration Customization
Users are able to implement custom specs for **`"build"`** and **`"input"`** to support a wide variety of operators. Should there's a need, the benchmark specification allows implementing new [`ConfigIterator`](lib/config.py) and [`DataGenerator`](lib/data.py) for your specific use case.

## Development Contributions
For more details, please take a look at the [development document](development.md).
