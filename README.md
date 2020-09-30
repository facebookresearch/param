# PARAM

PARAM Benchmarks is a repository of communication and compute micro-benchmarks as well as full workloads for evaluating training and inference platforms.

PARAM complements two broad categories of commonly used benchmarks:
1. C++ based stand-alone compute and communication benchmarks using cuDNN, MKL, NCCL, MPI libraries - e.g., NCCL tests (https://github.com/NVIDIA/nccl-tests), OSU MPI benchmarks (https://mvapich.cse.ohio-state.edu/benchmarks/), and DeepBench (https://github.com/baidu-research/DeepBench).
2. Application benchmarks such as Deep Learning Recommendation Model (DLRM) and the broader MLPerf benchmarks. MLPerf is a de-facto industry standard benchmark covering a wide range of AI workloads including computer vision and natural language processing. Recent addition of DLRM to MLPerf 0.7 is a great step towards making it more representative of FB’s AI workloads.

Our initial release of PARAM benchmarks focuses on AI training and comprises of:
1. Communication: PyTorch based collective benchmarks across arbitrary message sizes, effectiveness of compute-communication overlap, and DLRM communication patterns in fwd/bwd pass
2. Compute: PyTorch based GEMM, embedding lookup, and linear layer
3. DLRM: tracks the `ext_dist` branch of DRLM benchmark use Facebook's DLRM benchmark (https://github.com/facebookresearch/dlrm). In short, PARAM fully relies on DLRM benchmark for end-to-end workload evaluation; with additional extensions as required for scale-out AI training platforms.

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
