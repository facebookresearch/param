# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def run_single(args, layer_num, input_size, hidden_size, output_size, batch_size):

    device = args.device
    optimizer_type = args.optimizer_type
    data_type = args.dtype

    torch.manual_seed(1)

    lr = 0.01

    if device == "cpu":
        dev = torch.device("cpu")
        model = Net(input_size, hidden_size, output_size, layer_num).to(dev)
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            assert 0, "Unsupported optimizer type"

    elif device == "gpu":

        assert torch.cuda.is_available(), "cuda not available"

        import apex

        dev = torch.device("cuda:0")
        model = Net(input_size, hidden_size, output_size, layer_num).to(dev)
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
        model = Net(input_size, hidden_size, output_size, layer_num).to(dev)
        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            assert 0, "Unsupported optimizer type"

    elap, loss = train(
        model, dev, optimizer, data_type, input_size, output_size, batch_size, args
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
        layer_num, input_size, hidden_size, output_size, batch_size = dataset[i]
        elap, loss = run_single(
            args, layer_num, input_size, hidden_size, output_size, batch_size
        )
        elap /= args.steps

        flops = batch_size * (
            hidden_size * hidden_size * layer_num
            + hidden_size * input_size
            + hidden_size * output_size
        )
        # Forward 2x and Backward 4x
        flops *= 6

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


if __name__ == "__main__":

    import argparse

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
        choices=["float", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--layer-num", type=int, default=20, help="Number of Linear layers"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--input-size", type=int, default=1024, help="Input layer size")
    parser.add_argument(
        "--hidden-size", type=int, default=1024, help="Number of hidden_sizes per layer"
    )
    parser.add_argument(
        "--output-size", type=int, default=1024, help="Output layer size"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)

    args = parser.parse_args()
    layer_num = args.layer_num
    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    output_size = args.output_size

    d = [(layer_num, input_size, hidden_size, output_size, batch_size)]
    run(args, d)
