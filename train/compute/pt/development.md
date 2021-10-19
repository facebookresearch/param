# PARAM Compute Benchmark Development

## Main Components

### Configuration
Benchmark configurations are defined in a JSON format. It can be stored in a file on disk, or being passed between external callers and the benchmark’s library interface. There are two types of configurations:
* Build configuration (optional)
  * Defines arguments used to construct and initialize the operator.
  * It’s optional for operators that do not require initialization.
* Input configuration
  * Defines arguments used to execute the operator.

An operator may or may not need to have a build configuration, such as torch.matmul. Others will need to create the operator before running it:

```python
embedding = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse).to(device=device)
embedding(input, offset)
```

It’s important to note that configuration is a specification for data, not the actual data itself. From a configuration, we need to actually generate the data as the arguments to an operator.

### Data Generator
The role of the data generator is given a configuration specification, it generates actual data (scalar, boolean, string, tensor, etc.) for the building or executing an operator. Current implementations:
* `DefaultDataGenerator`

### Configuration Iterator
Given a list of configurations (build or input), we need some mechanism to iterate over them. The overall logic is simple (for illustration, not actual code):

```python
for build in build_configs:
  op = Op.build(build.args, build.kwargs)
  for input in input_configs:
    op.forward(input.args, input.kwargs)
```

There are some finer details:
Often we want to quickly generate many variations of build and input configurations without explicitly specifying each of them.
The configuration is only a specification, further it may need to be materialized (based on the macros) before generating the data.
Current implementations:
* `DefaultConfigIterator`
* `RangeConfigIterator`
* `DummyConfigIterator`

### Timer
Timer is essential in measuring operator latency. Some devices (GPU) are async and require special steps to run in blocking or synchronized mode. Depending on where the operator will run, the proper timer should be used:
* CPU
* GPU
* TPU

### Auto Discovery of Workloads
Python pkgutil.iter_modules provides a mechanism for discovering and importing modules dynamically. This allows adding workloads through the following simple steps:
* Create or add to an operator workload python file in `param/workloads` directory
* Implement the OperatorInterface
* Register the new operator through one of the following
  * `register_operator(name: str, operator: OperatorInterface)`
  * `register_operators(op_dict: Dict[str, OperatorInterface])`

The benchmark driver will be able to load configuration files and instantiate corresponding operators for benchmarking. Two categories of of operators:
* PyTorch native
  * Operators have no dependencies other than official PyTorch release.
* External
  * Operators require additional installation.

For users who do not have certain external operators in their environment, automatically importing these can cause errors. Auto import will try/catch these errors and skip these operators.
