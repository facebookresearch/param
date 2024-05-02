>>> ./README.md
# PARAM

PARAM Benchmarks is a repository of communication and compute micro-benchmarks as well as full workloads for evaluating training and inference platforms.

PARAM complements two broad categories of commonly used benchmarks:
1. C++ based stand-alone compute and communication benchmarks using cuDNN, MKL, NCCL, MPI libraries - e.g., NCCL tests (https://github.com/NVIDIA/nccl-tests), OSU MPI benchmarks (https://mvapich.cse.ohio-state.edu/benchmarks/), and DeepBench (https://github.com/baidu-research/DeepBench).
2. Application benchmarks such as Deep Learning Recommendation Model (DLRM) and the broader MLPerf benchmarks. Its worth noting that while MLPerf is the de-facto industry standard for benchmarking ML applications we hope to compliment this effort with broader workloads that are of more interest to Facebook with more in-depth analysis of each within this branch of Application benchmarks.

Our initial release of PARAM benchmarks focuses on AI training and comprises of:
1. Communication: PyTorch based collective benchmarks across arbitrary message sizes, effectiveness of compute-communication overlap, and DLRM communication patterns in fwd/bwd pass
2. Compute: PyTorch based GEMM, embedding lookup, and linear layer
3. DLRM: tracks the `ext_dist` branch of DRLM benchmark use Facebook's DLRM benchmark (https://github.com/facebookresearch/dlrm). In short, PARAM fully relies on DLRM benchmark for end-to-end workload evaluation; with additional extensions as required for scale-out AI training platforms.
4. PyTorch Execution Trace (ET) replay based tests: The PyTorch ET capturing capabilities, which have recently been introduced, allow for the recording of runtime information of a model at the operator level. This capability enables the creation of replay-based benchmarks (https://dl.acm.org/doi/abs/10.1145/3579371.3589072) to accurately reproduce the original performance.


In essence, PARAM bridges the gap between stand-alone C++ benchmarks and PyTorch/Tensorflow based application benchmarks. This enables us to gain deep insights into the inner workings of the system architecture as well as identify framework-level overheads by stressing all subcomponents of a system.

## Version

0.1 : Initial release

## Requirements

- pytorch
- future
- numpy
- apex

## License

PARAM benchmarks is released under the MIT license. Please see the [`LICENSE`](LICENSE) file for more information.

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for more info.

>>> ./comm/README.md
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

>>> ./README.md
# PARAM

PARAM Benchmarks is a repository of communication and compute micro-benchmarks as well as full workloads for evaluating training and inference platforms.

PARAM complements two broad categories of commonly used benchmarks:
1. C++ based stand-alone compute and communication benchmarks using cuDNN, MKL, NCCL, MPI libraries - e.g., NCCL tests (https://github.com/NVIDIA/nccl-tests), OSU MPI benchmarks (https://mvapich.cse.ohio-state.edu/benchmarks/), and DeepBench (https://github.com/baidu-research/DeepBench).
2. Application benchmarks such as Deep Learning Recommendation Model (DLRM) and the broader MLPerf benchmarks. Its worth noting that while MLPerf is the de-facto industry standard for benchmarking ML applications we hope to compliment this effort with broader workloads that are of more interest to Facebook with more in-depth analysis of each within this branch of Application benchmarks.

Our initial release of PARAM benchmarks focuses on AI training and comprises of:
1. Communication: PyTorch based collective benchmarks across arbitrary message sizes, effectiveness of compute-communication overlap, and DLRM communication patterns in fwd/bwd pass
2. Compute: PyTorch based GEMM, embedding lookup, and linear layer
3. DLRM: tracks the `ext_dist` branch of DRLM benchmark use Facebook's DLRM benchmark (https://github.com/facebookresearch/dlrm). In short, PARAM fully relies on DLRM benchmark for end-to-end workload evaluation; with additional extensions as required for scale-out AI training platforms.
4. PyTorch Execution Trace (ET) replay based tests: The PyTorch ET capturing capabilities, which have recently been introduced, allow for the recording of runtime information of a model at the operator level. This capability enables the creation of replay-based benchmarks (https://dl.acm.org/doi/abs/10.1145/3579371.3589072) to accurately reproduce the original performance.


In essence, PARAM bridges the gap between stand-alone C++ benchmarks and PyTorch/Tensorflow based application benchmarks. This enables us to gain deep insights into the inner workings of the system architecture as well as identify framework-level overheads by stressing all subcomponents of a system.

## Version

0.1 : Initial release

## Requirements

- pytorch
- future
- numpy
- apex

## License

PARAM benchmarks is released under the MIT license. Please see the [`LICENSE`](LICENSE) file for more information.

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for more info.

>>> ./comm/README.md
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

>>> ./tools/README.md
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
$ python -m param.train.compute.python.pytorch.run_benchmark -c train/compute/python/examples/pytorch/configs/simple_add.json --et
```

After collecting the trace, replay it with the following command. Set the warm-up iteration count to at least 1 to exclude tensor transfer time to GPUs.
```bash
$ python -m param.train.compute.python.tools.et_replay --input <trace_path> --warmup-iter 10 --iter 50 --compute --profile-replay
```

> Note: When analyzing performance values from et_replay, refer to the collected Kineto traces rather than the execution time reported by et_replay. Kineto traces are only collected when --profile-replay is provided.

>>> comp/README.md
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

The installed packages are under **`param.train.compute.python`**.

To use the [`FBGEMM_GPU`](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu) library and its operator benchmark workload ([`split_table_batched_embeddings_ops.py`](workloads/pytorch/split_table_batched_embeddings_ops.py)), please follow its set up instruction to download and install. It's not required for the compute benchmarks. During initialization, if an operator fail to import, it'll be ignored and will not affect other benchmarks.

Please make sure to install the `parambench-train-comms` package (`train/comms/pt`). This is important because some functions in this package reference those in the comms package.

## Usage
The bundled tool scripts such as [`run_benchmark.py`](pytorch/run_benchmark.py) are written using relative import paths as part of the `parambench-train-compute` package, so they must be ran as a module using the `python -m` option.

A reliable way to run the benchmarks is install `parambench-train-compute` as a package following the above instructions. Afterward, it can be ran as:
```shell
# Run benchmark tool script module
> python -m param.train.compute.python.pytorch.run_benchmark -c examples/pytorch/configs/simple_add.json
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
from param.train.compute.python.lib.config import BenchmarkConfig
```
A complete example to generate benchmark config, run the benchmark, then get the results can be found in [`run_op.py`](examples/pytorch/run_op.py)

## PyTorch Benchmark Options
```
=> python -m param.train.compute.python.pytorch.run_benchmark -h
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

>>> docs/using_ET.md
# Using Execution Trace in PARAM Benchmark

This section includes how to collect Chakra Execution Trace from a PyTorch training workload, as well as how to run PARAM replay on top of the collected ET.


## Execution Trace Collection
Execution Trace collection logic has to be added in the main training loop. This includes three steps:

### Step 1: Set up Execution Trace Observer
The first step is to create a Execution Trace Observer object and register a. temporary file for ET store.

```
from torch.profiler import ExecutionTraceObserver

et_ob = ExecutionTraceObserver()
fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
fp.close()
et_ob.register_callback(fp.name)
```

### Step 2: Define your function to dump Execution Trace
You have to define a function to store/dump/upload your collected ET trace for further use. Here is an example:

```
def dump_execution_trace(tmp_et_path):
    et_dir.mkdir(exist_ok=True, parents=True)
    et_path = DUMP_DIR / f"rank-{global_rank}.et.json.gz"
    with open(tmp_et_path) as fin:
        with gzip.open(et_path, "wt") as fout:
            fout.writelines(fin)
    os.remove(tmp_et_path)
    print(f"Finished Rank {global_rank} ET collection at {et_path}")
```

### Step 3: Collect Execution Trace in the training loop
This is the key step to collect ET. You have to insert the collection logic into the main training loop of your workload.
TWO parameters have to be set:
- ET_START_ITER: the iteration to start ET collection
- ET_END_ITER: the iteration to stop ET collection

```
<START of training loop>
while step < TRAINING_STEPS:
    ...
    ...
    # Collect Execution Trace Logic

    # Start ET collection
    if et_ob and step == ET_START_ITER:
        et_ob.start()

        # First record process group(PG) mapping
        pg_config_info = (
            torch.distributed.distributed_c10d._world.pg_config_info
        )
        rf_handle = torch.autograd._record_function_with_args_enter(
            "## process_group:init ##", json.dumps(pg_config_info)
        )
        torch.autograd._record_function_with_args_exit(rf_handle)

    # Stop ET collection
    elif et_ob and state.step == ET_END_ITER:
        et_ob.stop()
        tmp_et_path = et_ob.get_output_file_path()
        et_ob.unregister_callback()
        dump_execution_trace(tmp_et_path)

    ...
    ...
    step += 1
<END of training loop>
```

Note that process group information collection is not automatically covered by ET observer, because process_group initialization happens before the main training loop. Therefore, you have to manually add pg information collection, as the code shown above.




## PARAM Comms Replay on Execution Trace
Execution Trace now is fully supported in PARAM benchmark. In order to replay an ET trace, just need to specify `--trace-type=et` and the benchmark will parse your ET and replay the collective communication operators.

An example command:

```
/bin/mpirun -np 8 commsTraceReplay.par --trace-path <ET-PATH> --trace-type et
```
