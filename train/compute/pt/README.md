# PARAM Compute Benchmarks

## Overview
The compute benchmarks are under active development to support new use cases. The previous [standalone](standalone) benchmarks remain unchanged.

The general motivation and design philosophy for the new microbenchmarks are:
* Support a wide variety of PyTorch operators and workloads.
* Decoupled workload configuration from Python code. It allows building tools in a pipeline to collect operator configuration from production runs, generate microbenchmark inputs, gather metrics.
* Generates human readable and easy to parse output data format for downstream tools.
* A library interface allowing exeternal tools to run and collect data results from the microbenchmarks.
* Support replay of workload through PyTorch execution graph.
