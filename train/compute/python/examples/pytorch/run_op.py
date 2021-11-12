from ...lib import pytorch as lib_pytorch
from ...lib.config import make_op_config
from ...lib.init_helper import load_modules
from ...lib.pytorch.benchmark import OpExecutor, ExecutionPass, get_benchmark_options
from ...lib.pytorch.config_util import create_bench_config, create_data
from ...workloads import pytorch as workloads_pytorch


def main():

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    op_name = "torch.mm"
    bench_config = create_bench_config(op_name)
    tensor_1 = create_data("tensor")
    tensor_1["shape"] = [128, 128]
    tensor_2 = create_data("tensor")
    tensor_2["shape"] = [128, 128]

    op_info = bench_config[op_name]

    # Add the two tensors as first and second positional args for the operator.
    op_info["config"][0]["input"][0]["args"] = [tensor_1, tensor_2]
    print(op_info)

    # Get the default benchmark options
    run_options = get_benchmark_options()

    # Create OperatorConfig that initialize the actual operator workload and
    # various generators to create inputs for the operator.
    op_config = make_op_config(op_name, op_info, run_options["device"])

    # By default, benchmark will run the forward pass.
    # By setting backward (which requires running forward pass), the benchmark
    # will run both forward and backward pass.
    run_options["pass_type"] = ExecutionPass.BACKWARD

    # Generate the actual data for inputs. For operators that require a build
    # step, a similar data generation is needed for build config.
    input_config = op_info["config"][0]["input"][0]
    input_data_gen = op_config.input_data_generator()
    (input_args, input_kwargs) = input_data_gen.get_data(
        input_config, run_options["device"]
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
