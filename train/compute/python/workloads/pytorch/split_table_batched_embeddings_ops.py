import copy
import gc
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    CacheAlgorithm,
    ComputeDevice,
    EmbeddingLocation,
    OptimType,
    PoolingMode,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
    WeightDecayMode,
)

from ...lib.data import register_data_generator
from ...lib.generator import full_range, IterableList, ListProduct, TableProduct
from ...lib.init_helper import get_logger
from ...lib.iterator import (
    ConfigIterator,
    genericList_to_list,
    register_config_iterator,
    remove_meta_attr,
)
from ...lib.operator import OperatorInterface, register_operator

logger = get_logger()


class SplitTableBatchedEmbeddingBagsCodegenInputIterator(ConfigIterator):
    def __init__(
        self,
        configs: Dict[str, Any],
        key: str,
        device: str,
    ):
        super(SplitTableBatchedEmbeddingBagsCodegenInputIterator, self).__init__(
            configs, key, device
        )
        logger.debug(f"build_input_config: {configs}")
        build_config = configs["build"]
        logger.debug(f"build_config: {build_config}")
        self.num_tables = build_config["args"][0]
        self.rows = build_config["args"][1]
        self.dim = build_config["args"][2]
        self.weighted = build_config["args"][4]
        self.weights_precision = build_config["args"][5]
        self.generator = self._generator()

    def _generator(self):
        inputs = self.configs[self.key]
        var_id = 0
        for input in inputs:
            input_config = copy.deepcopy(input)
            args = []
            for arg in input_config["args"]:
                if "__range__" in arg:
                    arg["value"] = full_range(*arg["value"])
                if "__list__" in arg:
                    arg["value"] = IterableList(arg["value"])
                args.append(TableProduct(arg))

            config_id = 0
            for arg_config in ListProduct(args):
                batch_size = arg_config[0]
                pooling_factor = arg_config[1]
                result = {
                    "args": [
                        self.num_tables,
                        self.rows,
                        self.dim,
                        batch_size,
                        pooling_factor,
                        self.weighted,
                        self.weights_precision,
                    ],
                    "kwargs": {},
                }
                yield (f"{var_id}_{config_id}", remove_meta_attr(result))
                config_id += 1

    def __next__(self):
        return next(self.generator)


register_config_iterator(
    "SplitTableBatchedEmbeddingBagsCodegenInputIterator",
    SplitTableBatchedEmbeddingBagsCodegenInputIterator,
)


def generate_requests(
    B: int,  # batch size
    L: int,  # pooling factor
    E: int,  # emb size
    offset_start: int,  # indices offset from previous generator
    # alpha <= 1.0: use uniform distribution
    # alpha > 1.0: use zjpf distribution
    alpha: float = 1.0,
    weighted: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    indices_size = B * L
    # indices
    if alpha == 0:
        # linear sequence by pooling factor
        indices = torch.arange(0, indices_size).long() % L
    elif alpha <= 0.5:
        # linear sequence by embedding size
        indices = torch.arange(0, indices_size).long() % E
    elif alpha <= 1.0:
        indices = torch.randint(
            low=0,
            high=E,
            size=(indices_size,),
            dtype=torch.int64,
        )
    else:
        indices = torch.as_tensor(np.random.zipf(a=alpha, size=indices_size)).long() % E

    # offsets
    lengths = np.ones(B, dtype=np.int64) * L
    # here we want to add the start of previous offset to all the offsets
    # if offset_start = 0, we insert it in the beginning
    if offset_start == 0:
        offsets = torch.tensor(np.cumsum([0] + lengths.tolist()))
    else:
        offsets = torch.tensor(offset_start + np.cumsum(lengths))

    # weights
    weights_tensor = (
        torch.randn(indices_size, dtype=torch.float32) if weighted else None
    )

    return (indices, offsets, weights_tensor)


class SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator:
    def get_data(self, config, device, alpha=1):
        logger.debug(f"data generator config: {config}")
        # batch size * pooling_factor
        num_tables = config["args"][0]["value"]

        if num_tables > 1:
            rows = config["args"][1]["value"]
            pooling_factors = config["args"][4]["value"]
        else:
            rows = [config["args"][1]["value"]]
            pooling_factors = [config["args"][4]["value"]]
        batch_size = config["args"][3]["value"]
        weighted = config["args"][5]["value"]

        indices_list = []
        offsets_list = []
        per_sample_weights_list = []
        offset_start = 0
        distribution = os.getenv("split_embedding_distribution")
        if distribution is None:
            distribution = 1
        distribution = alpha
        logger.debug(f"distribution = {distribution}")

        target_device = torch.device(device)

        indices_file = None
        offsets_file = None
        weights_file = None
        if ("indices_tensor" in config["args"][4]) and (
            "offsets_tensor" in config["args"][4]
        ):
            indices_file = config["args"][4]["indices_tensor"]
            offsets_file = config["args"][4]["offsets_tensor"]
            if weighted and "weights_tensor" in config["args"][4]:
                weights_file = config["args"][4]["weights_tensor"]
        else:
            indices_file = os.getenv("split_embedding_indices")
            offsets_file = os.getenv("split_embedding_offsets")
            if weighted:
                weights_file = os.getenv("split_embedding_weights")

        logger.debug(f"indices_file: {indices_file}, offsets_file: {offsets_file}")
        if indices_file is not None and offsets_file is not None:
            indices_tensor = torch.load(indices_file, map_location=target_device)
            offsets_tensor = torch.load(offsets_file, map_location=target_device)
            per_sample_weights_tensor = None
            if weights_file:
                per_sample_weights_tensor = torch.load(
                    weights_file, map_location=target_device
                )
        else:
            for i in range(num_tables):
                indices, offsets, per_sample_weights = generate_requests(
                    batch_size,
                    pooling_factors[i],
                    rows[i],
                    offset_start,
                    float(distribution),
                    weighted,
                )
                indices_list.append(indices)
                offsets_list.append(offsets)
                # update to the offset_start to the last element of current offset
                offset_start = offsets[-1].item()
                if weighted:
                    per_sample_weights_list.append(per_sample_weights)

            indices_tensor = torch.cat(indices_list)
            offsets_tensor = torch.cat(offsets_list)

            # check for per sample weights
            per_sample_weights_tensor = (
                torch.cat(per_sample_weights_list) if weighted else None
            )

        logger.debug(f"indices: {indices_tensor.shape}")
        logger.debug(f"offsets: {offsets_tensor.shape}")
        if per_sample_weights_tensor is not None:
            logger.debug(
                f"per_sample_weights: {per_sample_weights_tensor.shape}, {per_sample_weights_tensor}"
            )

        return (
            [
                indices_tensor.to(target_device),
                offsets_tensor.to(target_device),
                per_sample_weights_tensor.to(target_device) if weighted else None,
            ],
            {},
        )


register_data_generator(
    "SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator",
    SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator,
)

# Callable ops are ops can be called in the form of op(*args, **kwargs)
class SplitTableBatchedEmbeddingBagsCodegenOp(OperatorInterface):
    def __init__(
        self,
    ):
        super(SplitTableBatchedEmbeddingBagsCodegenOp, self).__init__()
        self.op = None
        self.fwd_out: torch.tensor = None
        self.grad_in: torch.tensor = None

    def build(
        self,
        num_tables: int,
        rows: Union[int, list],
        dims: Union[int, list],
        pooling: int,
        weighted: bool,
        weights_precision: str,
        optimizer: str,
        lr: float = 0.01,
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
    ):
        logger.debug(
            f"build: [{num_tables}, {rows}, {dims}, {pooling}, {weighted}, {weights_precision}, \
            {optimizer}, {lr}, {eps}, {weight_decay}, {WeightDecayMode}]"
        )
        rows_list = rows if isinstance(rows, list) else [rows]
        dims_list = dims if isinstance(dims, list) else [dims]
        if self.device.startswith("cpu"):
            compute_device = ComputeDevice.CPU
            location = EmbeddingLocation.HOST
        elif self.device.startswith("cuda"):
            compute_device = ComputeDevice.CUDA
            location = EmbeddingLocation.DEVICE
        else:
            raise ValueError(f"Unknown compute device {self.device}")

        # split_table op options from actual runs of
        # caffe2/torch/fb/module_factory/proxy_module/grouped_sharded_embedding_bag.py
        self.op = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    rows_list[i],
                    dims_list[i],
                    location,
                    compute_device,
                )
                for i in range(num_tables)
            ],
            optimizer=OptimType(optimizer),
            pooling_mode=PoolingMode(pooling),
            weights_precision=SparseType(weights_precision),
            stochastic_rounding=True,
            cache_algorithm=CacheAlgorithm.LFU,
            cache_load_factor=0.0,
            cache_reserved_memory=12.0,
            device=torch.device(self.device),
            learning_rate=lr,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_mode=weight_decay_mode,
        )

        logger.debug(f"op embedding_specs: {self.op.embedding_specs}")

    def cleanup(self):
        logger.debug("op cleanup")
        self.op = None
        self.grad_in = None
        self.fwd_out = None

    def forward(self, *args, **kwargs):
        self.fwd_out = self.op.forward(args[0], args[1], args[2])
        return self.fwd_out

    def create_grad(self):
        self.grad_in = torch.ones_like(self.fwd_out)

    def backward(self, grad=None):
        if grad is not None:
            self.fwd_out.backward(grad)
        else:
            if self.grad_in is None:
                self.create_grad()
            self.fwd_out.backward(self.grad_in)


register_operator(
    "SplitTableBatchedEmbeddingBagsCodegen", SplitTableBatchedEmbeddingBagsCodegenOp()
)
