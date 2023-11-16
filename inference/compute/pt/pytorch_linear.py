# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num):
        super(Net, self).__init__()
        self.layer_num = layer_num
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_hid_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(self.layer_num)]
        )
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_in(x)
        x = F.relu(x)
        for linear_hid in self.linear_hid_list:
            x = linear_hid(x)
            x = F.relu(x)
        x = self.linear_out(x)
        x = F.softmax(x, dim=1)
        return x


def infer_nnpi(model, device, data_type, input_size, output_size, batch_size, args):
    import torch_glow

    # Detailed structure for spec can be found at https://fburl.com/diffusion/79q4efud
    # Create compilation spec
    spec = torch_glow.CompilationSpec()
    spec.get_settings().set_glow_backend("NNPI")
    # Create compilation group and update settings.
    # Compilation group contains compilation specific information like
    # fp16 settings, enableRemoveMutation, anything that changes
    # the Glow graph compiled for example.
    compilation_group = torch_glow.CompilationGroup()
    compilation_group_settings = compilation_group.get_settings()
    compilation_group_settings.set_convert_to_fp16(True)
    compilation_group_settings.set_replication_count(1)
    compilation_group_settings.backend_specific_opts_insert("NNPI_IceCores", "1")

    data = torch.randn(batch_size, input_size)
    # Create input spec and add it into compilation group.
    # This is used for shape inference when lowering the model to Glow.
    data_spec = torch_glow.InputSpec()
    data_spec.set_same_as(data)
    compilation_group.input_sets_append([data_spec])

    spec.compilation_groups_append(compilation_group)

    traced_model = torch.jit.trace(model, (data))
    lowered_model = torch_glow.to_glow(traced_model, spec)

    start_time = time.time()
    for i in range(args.steps + args.warmups):
        lowered_model(data)
        if i < args.warmups:
            start_time = time.time()
    return time.time() - start_time


def infer_cpu(model, device, data_type, input_size, output_size, batch_size, args):
    start_time = time.time()

    for i in range(args.steps + args.warmups):
        data = torch.randn(batch_size, input_size, device=device)
        if data_type == "float16":
            data = data.half()

        model(data).float()
        if i < args.warmups:
            start_time = time.time()

    return time.time() - start_time

    return 0


def infer_gpu(model, device, data_type, input_size, output_size, batch_size, args):
    data = torch.randn(batch_size, input_size, device="cuda")

    if data_type == "float16":
        data = data.half()
        model_final = model.half()
        if args.use_trt:
            print("Creating TRT model")
            from torch_tensorrt.fx.lower import compile
            from torch_tensorrt.fx.utils import LowerPrecision

            model_final = compile(
                model_final,
                [data],
                max_batch_size=batch_size,
                explicit_batch_dimension=False,
                max_workspace_size=4 << 30,
                lower_precision=LowerPrecision.FP16,
            )
    else:
        model_final = model

    if args.use_migraphx:
        torch.onnx.export(
            model_final,
            torch.randn(
                batch_size,
                input_size,
                device="cuda",
                dtype=torch.float16 if data_type == "float16" else torch.float32,
            ),
            "benchmark.onnx",
            input_names=["input"],
            output_names=["output"],
        )
        import migraphx

        migraphx_program = migraphx.parse_onnx("benchmark.onnx")
        migraphx_program.compile(migraphx.get_target("gpu"), offload_copy=False)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    for i in range(args.steps + args.warmups):
        data = torch.randn(batch_size, input_size, device="cuda")

        if data_type == "float16":
            data = data.half()

        if args.use_migraphx:
            params = {}
            for key, value in migraphx_program.get_parameter_shapes().items():
                params[key] = migraphx.to_gpu(migraphx.generate_argument(value))

        if i >= args.warmups:
            start_event.record()

        if args.use_migraphx:
            migraphx_program.run(params)
        else:
            model_final(data)

        if i >= args.warmups:
            if args.use_migraphx:
                torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) * 1.0e-3

    return total_time


def infer(model, device, data_type, input_size, output_size, batch_size, args):

    if device == "cpu":
        elap = infer_cpu(
            model,
            device,
            data_type,
            input_size,
            output_size,
            batch_size,
            args,
        )

    elif device == "gpu":
        elap = infer_gpu(
            model,
            device,
            data_type,
            input_size,
            output_size,
            batch_size,
            args,
        )

    elif device == "nnpi":
        elap = infer_nnpi(
            model,
            device,
            data_type,
            input_size,
            output_size,
            batch_size,
            args,
        )

    return elap


def run_single(args, layer_num, input_size, hidden_size, output_size, batch_size):

    device = args.device
    data_type = args.dtype

    if device == "cpu":
        dev = torch.device("cpu")
        with torch.no_grad():
            model = Net(input_size, hidden_size, output_size, layer_num).to(dev)
            model.eval()

            elap = infer(
                model, device, data_type, input_size, output_size, batch_size, args
            )

    elif device == "gpu":

        assert torch.cuda.is_available(), "cuda not available"

        dev = torch.device("cuda:0")
        with torch.no_grad():
            model = Net(input_size, hidden_size, output_size, layer_num).to(dev)
            model.eval()

            elap = infer(
                model, device, data_type, input_size, output_size, batch_size, args
            )

    elif device == "nnpi":

        import os

        # Set to 1 to use the card, set to 0 to use the emulator
        os.environ["USE_INF_API"] = "1"

        with torch.no_grad():
            model = Net(input_size, hidden_size, output_size, layer_num)
            model.eval()

            elap = infer(
                model, device, data_type, input_size, output_size, batch_size, args
            )
    return elap


def run(args, dataset):

    print(
        "--------------------------------------------------------------------------------"
    )
    print(
        " #Layer    Input    Hidden    Output   Batch   Time(s)/step  QPS      Rate(TF/s)"
    )
    print(
        "--------------------------------------------------------------------------------"
    )

    for i in range(len(dataset)):
        layer_num, input_size, hidden_size, output_size, batch_size = dataset[i]
        elap = run_single(
            args, layer_num, input_size, hidden_size, output_size, batch_size
        )
        elap /= args.steps

        flops = batch_size * (
            hidden_size * hidden_size * layer_num
            + hidden_size * input_size
            + hidden_size * output_size
        )
        flops *= 2

        QPS = batch_size / elap

        print(
            "{0:6},  {1:6},  {2:6},  {3:6},  {4:6},  {5:10.6f},  {6:8.1f}, {7:10.1f}".format(
                layer_num,
                input_size,
                hidden_size,
                output_size,
                batch_size,
                elap,
                QPS,
                flops / elap / 1.0e12,
            )
        )


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(description="Measure the performance of MLP")
    parser.add_argument("--device", required=True, choices=["cpu", "gpu", "nnpi"])
    parser.add_argument(
        "--dtype",
        default="float",
        help="data type",
        choices=["float", "float16"],
    )
    parser.add_argument(
        "--layer-num", type=int, default=20, help="Number of Linear layers"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--input-size", type=int, default=1024, help="Input layer size")
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="Number of hidden_sizes per layer"
    )
    parser.add_argument(
        "--output-size", type=int, default=1024, help="Output layer size"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--use-trt", default=False, action="store_true")
    parser.add_argument("--use-migraphx", default=False, action="store_true")

    args = parser.parse_args()
    layer_num = args.layer_num
    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    output_size = args.output_size
    # num_batches = args.num_batches

    d = [(layer_num, input_size, hidden_size, output_size, batch_size)]
    run(args, d)


if __name__ == "__main__":
    main()  # pragma: no cover
