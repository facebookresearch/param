import time

import torch


# Timer
class Timer:
    def __init__(self, device: str):
        self.device: str = device
        if self.device is None:
            self.torch_device = None
        else:
            self.torch_device = torch.device(self.device)
        self.start_time: float = 0
        self.end_time: float = 0

    def start(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.torch_device)
        self.start_time = time.perf_counter_ns()

    def stop(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.torch_device)
        self.end_time = time.perf_counter_ns()

    # Return result in milliseconds.
    def elapsed_time_ms(self) -> float:
        return (self.end_time - self.start_time) / 1e6

    def elapsed_time_sec(self) -> float:
        return (self.end_time - self.start_time) / 1e9
