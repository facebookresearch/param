# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, layers_size):
        super(Net, self).__init__()
        self.linear_layer_list = nn.ModuleList(
            [
                nn.Linear(layers_size[i], layers_size[i + 1])
                for i in range(len(layers_size) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.linear_layer_list:
            x = layer(x)
            x = F.relu(x)
        x = F.softmax(x, dim=1)
        return x


def train_cpu(
    model, device, optimizer, data_type, input_size, output_size, batch_size, args
):
    loss_f = nn.CrossEntropyLoss()

    # model.train()
    start_time = time.time()

    for i in range(args.steps + args.warmups):
        data = torch.randn(batch_size, input_size, device=device)
        target = torch.randint(
            output_size, [batch_size], device=device, dtype=torch.long
        )
        # data, target = data.to(device), target.to(device)
        if data_type == "float16":
            data = data.half()

        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if i < args.warmups:
            start_time = time.time()

    return time.time() - start_time, loss


def train_gpu(
    model, device, optimizer, data_type, input_size, output_size, batch_size, args
):
    if data_type == "tf32":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    import apex

    loss_f = nn.CrossEntropyLoss().to(device)

    if data_type == "float16":
        model = apex.fp16_utils.network_to_half(model)

    # model.train()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    for i in range(args.steps + args.warmups):
        data = torch.randn(batch_size, input_size, device=device)
        target = torch.randint(
            output_size, [batch_size], device=device, dtype=torch.long
        )
        # data, target = data.to(device), target.to(device)
        if data_type == "float16":
            data = data.half()

        if i >= args.warmups:
            start_event.record()

        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if i >= args.warmups:
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) * 1.0e-3

    return total_time, loss


def train_tpu(
    model, device, optimizer, data_type, input_size, output_size, batch_size, args
):
    import torch_xla.core.xla_model as xm

    loss_f = nn.CrossEntropyLoss().to(device)

    # model.train()
    start_time = time.time()

    for i in range(args.steps + args.warmups):
        data = torch.randn(batch_size, input_size, device=device)
        target = torch.randint(
            output_size, [batch_size], device=device, dtype=torch.long
        )
        # data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        xm.mark_step()
        if i < args.warmups:
            start_time = time.time()

    return time.time() - start_time, loss


def train(
    model, device, optimizer, data_type, input_size, output_size, batch_size, args
):

    if device.type == "cpu":
        elap, loss = train_cpu(
            model,
            device,
            optimizer,
            data_type,
            input_size,
            output_size,
            batch_size,
            args,
        )

    elif device.type == "cuda":
        elap, loss = train_gpu(
            model,
            device,
            optimizer,
            data_type,
            input_size,
            output_size,
            batch_size,
            args,
        )

    elif device.type == "xla":
        elap, loss = train_tpu(
            model,
            device,
            optimizer,
            data_type,
            input_size,
            output_size,
            batch_size,
            args,
        )

    return elap, loss


def run_single(args, layers_size, batch_size):

    device = args.device
    optimizer_type = args.optimizer_type
    data_type = args.dtype

    torch.manual_seed(1)

    lr = 0.01

    if device == "cpu":
        dev = torch.device("cpu")
        model = Net(layers_size).to(dev)
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            assert 0, "Unsupported optimizer type"

    elif device == "gpu":

        assert torch.cuda.is_available(), "cuda not available"

        import apex

        dev = torch.device("cuda:0")
        model = Net(layers_size).to(dev)
        if optimizer_type == "sgd":
            optimizer = apex.optimizers.FusedSGD(
                model.parameters(), lr=lr, set_grad_none=True
            )
        elif optimizer_type == "lamb":
            optimizer = apex.optimizers.FusedLAMB(
                model.parameters(), lr=lr, set_grad_none=True
            )
        else:
            assert 0, "Unsupported optimizer type"

        if data_type == "float16":
            apex.amp.initialize(model, optimizer, opt_level="O2", verbosity=0)

    elif device == "tpu":

        import torch_xla.core.xla_model as xm

        # alldev = xm.get_xla_supported_devices()
        # allrealdev = xm.xla_real_devices(alldev)
        # print("Found {0} XLA devices: {1}".format(len(allrealdev), allrealdev))

        dev = xm.xla_device()
        # print("Using device:", dev)
        model = Net(layers_size).to(dev)
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            assert 0, "Unsupported optimizer type"

    elap, loss = train(
        model, dev, optimizer, data_type, layers_size[0], layers_size[-1], batch_size, args
    )
    return elap, loss


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
        layers_size, batch_size = dataset[i]
        elap, loss = run_single(
            args, layers_size, batch_size
        )
        elap /= args.steps

        flops = 0
        for i in range(len(layers_size)-1):
            flops += layers_size[i] * layers_size[i+1]
        flops *= batch_size

        # Forward 2x and Backward 4x
        flops *= 6

        QPS = batch_size / elap

        # The hidden layer size could vary, but for now keeping for backward
        # compatibility
        print(
            "{0:6},  {1:6},  {2:6},  {3:6},  {4:6},  {5:10.6f},  {6:8.1f}, {7:10.1f}".format(
                len(layers_size),
                layers_size[0],
                layers_size[1],
                layers_size[-1],
                batch_size,
                elap,
                QPS,
                flops / elap / 1.0e12,
            )
        )

        print("layers size: {}".format(layers_size))


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            if val:
                int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


if __name__ == "__main__":

    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="Measure the performance of MLP")
    parser.add_argument("--device", required=True, choices=["cpu", "gpu", "tpu"])
    parser.add_argument(
        "--optimizer-type",
        default="sgd",
        help="Optimizer: SGD",
        choices=["sgd", "lamb"],
    )
    parser.add_argument(
        "--dtype",
        default="float",
        help="data type",
        choices=["float", "float16", "bfloat16", "tf32"],
    )
    parser.add_argument(
        "--layer-num", type=int, default=20, help="Number of Linear layers"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--input-size", type=int, default=1024, help="Input layer size")
    parser.add_argument("--hidden-size", type=int, default=1024,help="Number of hidden_sizes per layer")
    parser.add_argument(
        "--layers-size", #"1024-1024-1024-1024" for example
        type=dash_separated_ints,
        default="",
        help="dash separated shape for all layers",
    )
    parser.add_argument(
        "--output-size", type=int, default=1024, help="Output layer size"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)

    args = parser.parse_args()
    batch_size = args.batch_size

    layers_size = []
    if args.layers_size:
        print("Using 'layers-size'")
        layers_size = np.fromstring(args.layers_size, dtype=int, sep="-")
        layers_size = [int(size) for size in layers_size]
    else:
        print("Using 'input_size', 'hidden_size' and 'output_size'.")
        layers_size.append(args.input_size)
        for _ in range(args.layer_num):
            layers_size.append(args.hidden_size)
        layers_size.append(args.output_size)

    d = [(layers_size, batch_size)]
    run(args, d)
