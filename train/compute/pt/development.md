# PARAM Compute Benchmark Development

## Main Components

### Configuration
Benchmark configurations are defined in a JSON format. It can be stored in a file on disk, or being passed between external callers and the benchmark’s library interface. There are two types of configurations:
Build configuration
Defines arguments used to construct and initialize the operator.
It’s optional for operators that do not require initialization.
Input configuration
Defines arguments used to execute the operator.
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
DefaultDataGenerator

### Configuration Iterator
Given a list of configurations (build or input), we need some mechanism to iterate over them. The overall logic is simple (for illustration, not actual code):
