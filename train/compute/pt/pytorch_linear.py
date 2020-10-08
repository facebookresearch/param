# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import apex
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(
        self, batch_size, num_batches, input_size, output_size, transform=None
    ):
        self.transform = transform
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return self.batch_size * self.num_batches

    def __getitem__(self, idx):
        input_sample = torch.FloatTensor(self.input_size).uniform_(-1, 1)
        output_label = torch.randint(0, self.output_size, (1,), dtype=torch.long)[0]
        return input_sample, output_label


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num):
        super(Net, self).__init__()
        self.layer_num = layer_num
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_hid = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_in(x)
        x = F.relu(x)
        for _ in range(self.layer_num):
            x = self.linear_hid(x)
            x = F.relu(x)
        x = self.linear_out(x)
        x = F.softmax(x, dim=1)
        return x


def train(
    model,
    device,
    train_loader,
    optimizer,
    data_type,
    batch_size,
    num_batches,
    input_size,
    hidden_size,
    output_size,
    layer_num,
):

    loss_f = nn.CrossEntropyLoss().to(device)

    if data_type == "float16":
        model = apex.fp16_utils.network_to_half(model)

    model.train()
    start_time = time.time()

    # for batch_idx, (data, target) in enumerate(train_loader):
    for _ in range(num_batches):
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
    total_time = time.time() - start_time
    total_examples = batch_size * num_batches

    print("---------------------------------------------")
    print("QPS: {:.6f}".format(total_examples / total_time))
    print("global_step/sec: {:.6f}".format(num_batches / total_time))
    print("Total time: {:.6f}".format(total_time))
    flops = batch_size * (
        hidden_size * hidden_size * layer_num
        + hidden_size * input_size
        + hidden_size * output_size
    )
    # Forward 2x and Backward 4x
    flops *= 6 * num_batches
    print("TFLOPS: {}".format(flops / total_time / 1e12))


@click.command()
@click.option("--use-gpu/--no-use-gpu", default=True, help="Use GPU/CPU")
@click.option("--layer-num", type=int, default=20, help="Number of Linear layers")
@click.option("--batch-size", type=int, default=512, help="Batch size")
@click.option("--input-size", type=int, default=1024, help="Input layer size")
@click.option(
    "--hidden-size", type=int, default=128, help="Number of hidden_sizes per layer"
)
@click.option("--output-size", type=int, default=1024, help="Output layer size")
@click.option("--num-batches", type=int, default=100, help="Number of batches to train")
@click.option(
    "--data-type",
    default="float16",
    type=click.Choice(["float", "float16"]),
    help="data type",
)
@click.option(
    "--optimizer-type",
    default="sgd",
    type=click.Choice(["sgd"]),
    help="Optimizers: SGD",
)
def main(
    use_gpu,
    layer_num,
    batch_size,
    input_size,
    hidden_size,
    output_size,
    num_batches,
    data_type,
    optimizer_type,
):
    # print("PyTorch VERSION:", torch.__version__)
    # print("CuDNN version: {}".format(torch.backends.cudnn.version()))
    # print("PyTorch config: {}".format(torch.__config__.show()))

    torch.manual_seed(1)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_gpu else {}
    train_loader = torch.utils.data.DataLoader(
        myDataset(batch_size, num_batches, input_size, output_size),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    use_gpu = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device:", device)

    model = Net(input_size, hidden_size, output_size, layer_num).to(device)
    if optimizer_type == "sgd":
        optimizer = apex.optimizers.FusedSGD(
            model.parameters(),
            lr=0.01,
            set_grad_none=True,
        )
    else:
        assert 0, "Unsupported optimizer type"

    if data_type == "float16":
        apex.amp.initialize(
            model, optimizer, opt_level="O2", verbosity=1
        )

    train(
        model,
        device,
        train_loader,
        optimizer,
        data_type,
        batch_size,
        num_batches,
        input_size,
        hidden_size,
        output_size,
        layer_num,
    )


if __name__ == "__main__":
    main()
