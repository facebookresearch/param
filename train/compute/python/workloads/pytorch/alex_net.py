import gc
from typing import Callable

import torch
import torch.nn as nn

from ...lib.operator import OperatorInterface, register_operator

class AlexNet(nn.Module):
    """
    Ref: https://pytorch.org/vision/master/_modules/torchvision/models/alexnet.html
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNetOp(OperatorInterface):
    """
    A wrapper for an implementation of AlexNet.
    """

    def __init__(self):
        super(AlexNetOp, self).__init__()
        self.alex_net: Callable = None
        self.fwd_out: torch.tensor = None
        self.grad_in = None

    def build(self):
        self.alex_net = AlexNet().to(torch.device(self.device))

    def cleanup(self):
        self.op = None
        self.grad_in = None
        self.fwd_out = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        self.fwd_out = self.alex_net.forward(*args, **kwargs)
        return self.fwd_out

    def create_grad(self):
        self.grad_in = torch.ones_like(self.fwd_out)

    def backward(self):
        self.fwd_out.backward(self.grad_in)



register_operator(
    "pytorch:alex_net", AlexNetOp()
)
