# PARAM Compute Benchmark Development

## File Structures

Directories

* [`python`](.)
  * Base dir for Python benchmarks, including driver scripts.
* [`python/lib`](./lib)
  * Benchmark library modules and utilities.
* [`python/workloads`](./workloads)
  * Implementation of workloads (operators).
* [`python/test`](./test)
  * Unit tests and test config files.

ML framework specific modules and files are in separate directories (e.g. `pytorch`) under these top level directories.

Because the benchmark library and workloads are intended to be used both inside and outside of Facebook, we need to make sure that they work consistently in both scenarios. **Within the benchmark library package itself, we prefer to use relative imports**, for example:

```python
from ..config import OperatorConfig
from ..iterator import ConfigIterator
from ..operator import OperatorInterface
from ..timer import Timer
```
This allows the top level package name to change without affecting the library code itself.

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
It can be used by the users as any regular Python package:
```python
from param_bench.train.compute.python.lib.config import BenchmarkConfig
```

The bundled driver scripts such as `pytorch_benchmark.py` are written using relative import paths as part of the `parambench-train-compute` package, so they must be run as a module using the `python -m` option.

Without installing the package, run driver script as module in the source directory:
```shell
# Inside dir "param/train/compute"
> python -m python.pytorch_benchmark --config python/test/pytorch/test_op.json
```

After install `parambench-train-compute` a package using the `setuptools`, it can be run as:
```shell
# Run benchmark driver script module
> python -m param_bench.train.compute.python.pytorch_benchmark --config test/pytorch/test_op.json
```

## Configuration
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
The **`"operator_name"`** is a key mapped to a concrete workload implementation defined inside [`workloads`](workloads) directory.

### Configuration Customization
Users are free to implement custom specs for **`"build"`** and **`"input"`** to support a wide variety of operators. Should there's a need, the specification allows custom implementation of [`ConfigIterator`](lib/config.py) and [`DataGenerator`](lib/data.py) for your specific use case.

### Default PyTorch Config Specification
For each **`"build"`** and **`"input"`** configuration, the __`"args"`__ and __`"kwargs"`__ JSON keys should be specified with a list of argument data specifications (see [PyTorch Data Types](#pyTorch-data-types)). This is synonymous to Python's __`*args`__ and __`**kwargs`__. As expected, **`"args"`** is positional and defined as a list. __`"kwargs"`__ is defined as a dictionary of `"kwarg_name": data_type_dict`.

**Example**
```json
{
  "torch.baddbmm": {
    "input_data_generator": "PyTorch::DefaultDataGenerator",
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

### PyTorch Data Types
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
  "type": "tensor",
  "dtype": "float",
  "shape": [128, 256]
}
```
**Notes**
* `"value_range"` specifies a random value in the `[min, max]` range to be generated for the argument.

It’s important to note that **the configuration is a specification for data**, not the actual data itself. From a configuration, we need to actually materialize and generate the data as the arguments for an operator.

## Data Generator
The role of the data generator is given a configuration specification, it generates actual data (scalar, boolean, string, tensor, etc.) for building or executing an operator.

In current implementations we provide a default data generator that supports PyTorch data types (see [PyTorch Data Types](#pyTorch-data-types)):
* [`PyTorch::DefaultDataGenerator`](lib/pytorch/data_impl.py)

New implementation of that supports the [`DataGenerator`](lib/data.py) interface can be registered using
[`register_data_generator(name: str, data_gen_class: Type[DataGenerator])`](lib/data.py).

## Configuration Iterator
Given a list of configurations (**build** or **input**), we need some mechanism to iterate over them. The overall logic is simple (for illustration, not actual code):

```python
for build in build_configs:
  build_args, build_kwargs = materialize_config(build)
  op = Op.build(build_args, build_kwargs)
  for input in input_configs:
    input_args, input_kwargs = materialize_config(input)
    op.forward(input_args, input_kwargs)
```

There are some finer details:
* Often we want to quickly generate many variations of build and input configurations without explicitly specifying each of them. This demands some mechanism for [**macros**](#macros).
* The configuration is only a specification, further it may need to be materialized (based on the macros) before generating the data.
* Current implementations:
  * `DefaultConfigIterator`
  * `RangeConfigIterator`

New implementation of that supports the [`ConfigIterator`](lib/iterator.py) interface can be registered using
[`register_config_iterator(name: str, iterator_class: Type[ConfigIterator])`](lib/iterator.py)

## Macros
Macros are for convenience to reduce the number of configurations to be specified manually.

### `__range__`
**`__range__`** defines a list of attributes with range specification.
<pre>
<b>"__range__"</b>: ["attr_name_1",...]
</pre>

**Example**
```json
"args": [
  {
    "type": "tensor",
    "dtype": "float",
    "shape": [512, [512, 514, 1], 30],
    "__range__": ["shape"]
  }
]
```
In above example, the argument is a `tensor` type. It has `"__range__"` macro specifies the `"shape"` attribute has range values: `[512, [512, 514, 1], 30]`. The second value the shape is a list `[512, 514, 1]`, it's represents `[min, max, step]`. During configuration iteration, multiple configurations will be generated, each with a different `"shape"` attribute after expansion:
* `[512, 512, 30]`
* `[512, 513, 30]`
* `[512, 514, 30]`

`"__range__"` macro also works for non-numeric values like `bool`, `str`, etc. These values can be specified in a list, i.e.,
```json
  {
    "type": "bool",
    "value": [true, false],
    "__range__": ["value"]
  }
```

### `__copy__`
**Only `tensor` data type in positional `"args"` is supported.**

In some instances, we need to ensure certain values are consistent between two attributes. For example, the input of a `matmul` operator has two tensors of shapes `A = [m, n]` and `B = [j, k]` where `n == j` for the inputs to be valid. As each of these values can vary between each input configuration, to ensure `j = n`, `__copy__` macro is applied to the data type attributes after tensor shape `A` is specified and copies the value of `n` to the value of `j` in tensor shape `B`.
<pre>
<b>"__copy__"</b>: [{"src_attr_name":[i, [j, k]]},...]
</pre>
Defines a list of attributes and where to copy their values from.
* `"src_attr_name"`: source attribute name
* `i`: target element index
* `j`: source **argument** index
* `k`: source **element** index
Copy value from source argument at `j`, element index `k`, to the current argument attribute element at index `i`.

**Example**
```json
"input": [
  {
    "type": "tensor",
    "dtype": "float",
    "shape": [-1, 64, 128],
    "__copy__": [
      {
        "shape": [0, [1, 2]]
      }
    ]
  },
  {
    "type": "tensor",
    "dtype": "float",
    "shape": [8, 16, 32]
  }
]
```
In above example of a tensor argument, its shape's value at element index `0` (with a `-1` value), will get the value of argument a position `1`, and its `"shape"` attribute's value at element index `2` (with value '32'). After the copy macro is applied, the tensor argument at index `0`, will have shape `[32, 64, 128]`.

## Operator Interface
The [`OperatorInterface`](lib/operator.py) specifies the interface each workload should support. At a minimum it should implement the `forward(*args, **kwargs)` method.

* `build(*args, **kwargs)`: [optional]
  * initialize and constructs all necessary data and objects to run the operator workload. It takes positional and keyword arguments from the configuration file.
* `cleanup()`: [optional]
  * release and delete any data and objects retained by this operator, its state should reset to before `build()` is called. This is called after a benchmark is run, so subsequent benchmarks do not run out of resource.
* `forward(*args, **kwargs)`: [required]
  * runs the forward pass of the operator and stores the output for running `backward()`.
* `create_grad()`: [optional]
  * create the gradient needed to run the `backward()` pass. This step is explicit to avoid counting this part in the benchmark latency for the backward pass.
* `backward()`: [optional]
  * Use the result from `forward()` and gradient generated in `create_grad()` to run the backward pass.

### Auto Discovery of Workloads
Python `pkgutil.iter_modules` provides a mechanism for discovering and importing modules dynamically. This allows adding workloads through the following simple steps:
* Create or add to an operator workload python file in [`workloads`](workloads) directory
* Implement the [`OperatorInterface`](lib/operator.py)
* Register the new operator through one of the following
  * [`register_operator(name: str, operator: OperatorInterface)`](lib/operator.py)
  * [`register_operators(op_dict: Dict[str, OperatorInterface])`](lib/operator.py)

The benchmark driver will be able to load configuration files and instantiate corresponding operators for benchmarking. Two categories of of operators:
* PyTorch native
  * Operators have no dependencies other than official PyTorch release.
* External
  * Operators require additional installation.

For users who do not have certain external operators in their environment, automatically importing these can cause errors. Auto import will try/catch these errors and skip these operators.

## Timer
Timer is essential in measuring operator latency. Some devices (GPU) are async and require special steps to run in blocking or synchronized mode. Depending on where the operator will run, the proper timer should be used:
* CPU
* GPU (PyTorch)

In the future, we may support timers for other device types.
