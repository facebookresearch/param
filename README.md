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
