import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import PoolingMode

from ...lib import pytorch as lib_pytorch
from ...lib.config import make_op_config
from ...lib.init_helper import load_modules
from ...lib.pytorch.config_util import (
    create_op_args,
    create_op_info,
    ExecutionPass,
    get_benchmark_options,
)
from ...lib.pytorch.op_executor import OpExecutor
from ...workloads import pytorch as workloads_pytorch


def main():

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    # Important to set num of threads to 1, some ops are not thread safe and
    # also improves measurement stability.
    torch.set_num_threads(1)

    op_name = "SplitTableBatchedEmbeddingBagsCodegen"
    op_info = create_op_info()
    op_info[
        "input_data_generator"
    ] = "SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator"
    print(op_info)

    # Get the default benchmark options.
    run_options = get_benchmark_options()

    # Create OperatorConfig that initializes the actual operator workload and
    # various generators to create inputs for the operator.
    op_config = make_op_config(op_name, op_info, run_options["device"])

    # By default, benchmark will run the forward pass.
    # By setting backward (which requires running forward pass), the benchmark
    # will run both forward and backward pass.
    run_options["pass_type"] = ExecutionPass.BACKWARD

    # Define config parameters required for input data generation and operator building.
    num_tables = 1
    rows = 228582
    dim = 128
    batch_size = 512
    pooling_factor = 50
    weighted = True
    weights_precision = "fp16"
    optimizer = "exact_row_wise_adagrad"

    # Construct configuration for input data generator.
    data_generator_config = create_op_args(
        [
            {"type": "int", "name": "num_tables", "value": num_tables},
            {"type": "int", "name": "rows", "value": rows},
            {"type": "int", "name": "dim", "value": dim},
            {"type": "int", "name": "batch_size", "value": batch_size},
            {"type": "int", "name": "pooling_factor", "value": pooling_factor},
            {"type": "bool", "name": "weighted", "value": weighted},
            {"type": "str", "name": "weights_precision", "value": weights_precision},
        ],
        {"optimizer": {"type": "str", "value": optimizer}},
    )

    # Generate the actual data for inputs.
    input_data_gen = op_config.input_data_generator()
    (input_args, input_kwargs) = input_data_gen.get_data(
        data_generator_config, run_options["device"]
    )

    # Construct and initialize the SplitTableBatchedEmbeddingBagsCodegen operator.
    op_config.op.build(
        num_tables,
        rows,
        dim,
        PoolingMode.SUM,
        weighted,
        weights_precision,
        optimizer,
    )

    # Create an OpExecutor to run the actual workload.
    op_exe = OpExecutor(op_name, op_config.op, run_options)

    # Run and collect the result metrics.
    result = op_exe.run(input_args, input_kwargs, "0:0:0")

    # Loop through and print the metrics.
    print("### Benchmark Results ###")
    for pass_name, pass_data in result.items():
        print(f"pass: {pass_name}")
        for metric_name, metrics in pass_data.items():
            print(metric_name)
            print(metrics)


if __name__ == "__main__":
    main()
