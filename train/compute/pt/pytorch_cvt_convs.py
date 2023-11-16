# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Benchmarks for:
# (1) CvT's convolutional token embedding layers;
# (2) CvT's convolutional projection layers.

import collections
import statistics
import time
from itertools import repeat
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def benchmark_requests(
    requests: List[Tuple[torch.FloatTensor]],
    func: Callable[[torch.Tensor], torch.Tensor],
    check_median: bool = False,
) -> float:
    times = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    for x in requests:
        start_time = time.time()
        if torch.cuda.is_available():
            start_event.record()
        func(x)
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            it_time = start_event.elapsed_time(end_event) * 1.0e-3
            times.append(it_time)
        else:
            it_time = time.time() - start_time
            times.append(it_time)
    avg_time = sum(times) / len(requests)
    median_time = statistics.median(times)
    return median_time if check_median else avg_time


# The conv. proj. and conv. token embedding classes are based on MSFT's CvT repo, cls_cvt.py.
class cvt_convolutional_projection(torch.nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, method):
        super().__init__()
        if method == "dw_bn":
            self.proj = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                dim_in,
                                dim_in,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=False,
                                groups=dim_in,
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(dim_in)),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )
        elif method == "avg":
            self.proj = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "avg",
                            nn.AvgPool2d(
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                ceil_mode=True,
                            ),
                        ),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )

    def forward(self, x):
        return self.proj(x)


def _ntuple(n):
    def parse(x):
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class cvt_convolutional_token_embedding(nn.Module):
    """Image to Conv Embedding"""

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x


def run(layer_type, input_shape, kwargs, device, warmups, steps, forward_only):

    if layer_type == "conv_proj":
        conv_block = cvt_convolutional_projection(**kwargs)
    elif layer_type == "patch_embed":
        conv_block = cvt_convolutional_token_embedding(**kwargs)
    else:
        raise ValueError(f"{layer_type} is invalid.")

    conv_block = conv_block.to(device)

    # get output shape
    if not forward_only:
        x = torch.rand(input_shape).to(device)
        output_shape = conv_block(x).shape

    # create requests
    requests = [torch.rand(input_shape).to(device) for _ in range(warmups + steps)]
    warmup_requests, requests = requests[:warmups], requests[warmups:]

    if forward_only:
        # forward
        _ = benchmark_requests(warmup_requests, lambda x: conv_block(x))
        time_per_iter = benchmark_requests(requests, lambda x: conv_block(x))
    else:
        # foward & backward
        grad_output = torch.randn(output_shape).to(device)
        _ = benchmark_requests(
            warmup_requests, lambda x: conv_block(x).backward(grad_output)
        )
        time_per_iter = benchmark_requests(
            requests, lambda x: conv_block(x).backward(grad_output)
        )

    # compute flops
    stride, padding = kwargs["stride"], kwargs["padding"]
    if "conv_proj" == layer_type:
        conv_filter = [
            input_shape[0],
            kwargs["dim_in"],
            kwargs["kernel_size"],
            kwargs["kernel_size"],
        ]
    else:
        conv_filter = [
            input_shape[0],
            kwargs["in_chans"],
            kwargs["patch_size"],
            kwargs["patch_size"],
        ]

    n = conv_filter[1] * conv_filter[2] * conv_filter[3]
    flops_per_instance = n + 1

    num_instances_per_filter = (
        (input_shape[1] - conv_filter[2] + 2 * padding) / stride
    ) + 1  # for rows
    num_instances_per_filter *= (
        (input_shape[2] - conv_filter[3] + 2 * padding) / stride
    ) + 1  # multiplying with cols

    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_in_layer = (
        flops_per_filter * conv_filter[0]
    )  # multiply with number of filters

    # TO DO:
    # Factor nn.BatchNorm2d into FLOPS calc for conv projection.
    # Factor nn.layernorm into FLOPS calc for conv token embedding.

    global_flops = total_flops_in_layer
    global_elap = time_per_iter
    global_bytes = num_instances_per_filter * conv_filter[0]

    return global_elap, global_bytes, global_flops


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description="Measuring CvT's Convolution Projection layer and Convolutional Token Embedding layer performance using PyTorch"
    )
    parser.add_argument("--warmups", type=int, default=10, help="warmup times")
    parser.add_argument("--steps", type=int, default=100, help="repeat times")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        required=True,
        help="valid devices",
    )
    parser.add_argument("--randomseed", type=int, default=0)

    parser.add_argument("--forward_only", dest="forward_only", action="store_true")
    parser.set_defaults(forward_only=False)
    args = parser.parse_args()

    torch.manual_seed(args.randomseed)

    # Tensor sizes used in this benchmark are taken from the CvT paper.
    # The authors trained CvT using input tensors of size 32 batches, 3 channels,
    # 224 rows, 224 columns.  Using that input size creates intermediate tensors
    # of sizes that are benchmarked here.

    # Note: conv_proj_k and conv_proj_v are identical in hyperparameters.
    # So conv_proj_k is benchmarked, while conv_proj_v is excluded as redundant.
    benchmark_cfgs_l = [
        {
            "layer_name": "cvt.stage0.block0.conv_proj_q",
            "input_shape": torch.Size([32, 64, 56, 56]),
            "kwargs": {
                "dim_in": 64,
                "dim_out": 64,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage0.block0.conv_proj_k\nSkip redundant benchmark of cvt.stage0.block0.conv_proj_v",
            "input_shape": torch.Size([32, 64, 56, 56]),
            "kwargs": {
                "dim_in": 64,
                "dim_out": 64,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block0.conv_proj_q",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block0.conv_proj_k\nSkip redundant benchmark of cvt.stage1.block0.conv_proj_v",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block1.conv_proj_q",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage1.block1.conv_proj_k\nSkip redundant benchmark of cvt.stage1.block1.conv_proj_v)",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "dim_in": 192,
                "dim_out": 192,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage2.blocks0.conv_proj_q\nSkip redundant benchmarks of cvt.stage2.blocks{1-9}.conv_proj_q",
            "input_shape": torch.Size([32, 384, 14, 14]),
            "kwargs": {
                "dim_in": 384,
                "dim_out": 384,
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
                "method": "dw_bn",
            },
        },
        # Blocks 0 through 9 in cvt.stage2 have identical hyperparameters
        {
            "layer_name": "cvt.stage2.blocks0.conv_proj_k\nSkip redundant benchmarks of cvt.stage2.blocks{1-9}.conv_proj_k and cvt.stage2.blocks{0-9}.conv_proj_v",
            "input_shape": torch.Size([32, 384, 14, 14]),
            "kwargs": {
                "dim_in": 384,
                "dim_out": 384,
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,
                "method": "dw_bn",
            },
        },
        {
            "layer_name": "cvt.stage0.patch_embed",
            "input_shape": torch.Size([32, 3, 224, 224]),
            "kwargs": {
                "patch_size": 7,
                "in_chans": 3,
                "embed_dim": 64,
                "stride": 4,
                "padding": 2,
                "norm_layer": nn.LayerNorm,
            },
        },
        {
            "layer_name": "cvt.stage1.patch_embed",
            "input_shape": torch.Size([32, 64, 56, 56]),
            "kwargs": {
                "patch_size": 3,
                "in_chans": 64,
                "embed_dim": 192,
                "stride": 2,
                "padding": 1,
                "norm_layer": nn.LayerNorm,
            },
        },
        {
            "layer_name": "cvt.stage2.patch_embed",
            "input_shape": torch.Size([32, 192, 28, 28]),
            "kwargs": {
                "patch_size": 3,
                "in_chans": 192,
                "embed_dim": 384,
                "stride": 2,
                "padding": 1,
                "norm_layer": nn.LayerNorm,
            },
        },
    ]

    for cfg in benchmark_cfgs_l:
        layer_name = cfg["layer_name"]
        input_shape = cfg["input_shape"]
        kwargs = cfg["kwargs"]
        device = "cuda" if args.device == "gpu" else args.device
        if "conv_proj" in layer_name:
            layer_type = "conv_proj"
        elif "patch_embed" in layer_name:
            layer_type = "patch_embed"
        else:
            raise ValueError(f"{layer_name} is invalid.")

        print("Benchmarking", layer_name)
        global_elap, global_bytes, global_flops = run(
            layer_type,
            input_shape,
            kwargs,
            device,
            args.warmups,
            args.steps,
            args.forward_only,
        )
        benchmark_metrics = {
            "GB/s": global_bytes / global_elap / 1.0e3,
            "TF/s": global_flops / global_elap / 1.0e12,
            "ELAP": global_elap,
            "FLOPS": global_flops,
        }
        print(benchmark_metrics)
        print("")


if __name__ == "__main__":
    main()  # pragma: no cover
