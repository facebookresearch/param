import gc

import torch

from ..init_helper import get_logger

logger = get_logger()


def log_cuda_memory_usage():
    cuda_allocated = torch.cuda.memory_allocated() / 1048576
    cuda_reserved = torch.cuda.memory_reserved() / 1048576
    logger.info(
        f"CUDA memory allocated = {cuda_allocated:.3f} MB, reserved = {cuda_reserved:.3f} MB"
    )


def free_torch_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()
