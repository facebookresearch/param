import copy
import gc
import logging
import os
from typing import (
    Dict,
    Set,
    List,
    Tuple,
    Any,
    Callable,
    Iterable,
    Type,
    TextIO,
    Optional,
)

import numpy as np
import torch
from fbgemm_gpu import split_table_batched_embeddings_ops
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    CacheAlgorithm,
    OptimType,
    SparseType,
    PoolingMode,
    EmbeddingLocation,
    ComputeDevice,
)
from torch import Tensor

from ...lib.data import data_generator_map, register_data_generator
from ...lib.generator import full_range, TableProduct, IterableList, ListProduct
from ...lib.iterator import (
    ConfigIterator,
    RangeConfigIterator,
    DefaultConfigIterator,
    config_iterator_map,
    remove_meta_attr,
    register_config_iterator,
    genericList_to_list,
)
from ...lib.operator import OperatorInterface, register_operator

FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)
# torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")


class SplitTableBatchedEmbeddingInputIterator(ConfigIterator):
    def __init__(
        self,
        configs: Dict[str, Any],
        key: str,
        device: str,
    ):
        super(SplitTableBatchedEmbeddingInputIterator, self).__init__(
            configs, key, device
        )
        logging.debug(configs)
        build_configs = configs["build"]
        logging.debug(build_configs)
        self.num_tables = build_configs["args"][0]
        self.rows = build_configs["args"][1]
        self.dim = build_configs["args"][2]
        self.propagation = build_configs["args"][6]
        self.weighted = build_configs["args"][7]
        self.weights_precision = build_configs["args"][8]
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
                logging.debug(arg_config)
                batch_size = arg_config[0]
                pooling_factor = arg_config[1]
                result = {
                    "args": [
                        self.num_tables,
                        self.rows,
                        self.dim,
                        batch_size,
                        pooling_factor,
                        self.propagation,
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
    "SplitTableBatchedEmbeddingInputIterator", SplitTableBatchedEmbeddingInputIterator
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

    logging.debug(indices)
    logging.debug(offsets)
    logging.debug(weights_tensor)
    return (indices, offsets, weights_tensor)


class SplitTableBatchedEmbeddingInputDataGenerator:
    def get_data(self, config, device):
        logging.debug(config)
        # batch size * pooling_factor
        num_tables = config["args"][0]["value"]
        if num_tables > 1:
            rows = genericList_to_list(config["args"][1])
            dims = genericList_to_list(config["args"][2])
            pooling_factors = genericList_to_list(config["args"][4])
        else:
            rows = [config["args"][1]["value"]]
            dims = [config["args"][2]["value"]]
            pooling_factors = [config["args"][4]["value"]]
        batch_size = config["args"][3]["value"]
        propagation = config["args"][5]["value"]
        weighted = config["args"][6]["value"]

        indices_list = []
        offsets_list = []
        per_sample_weights_list = []
        offset_start = 0
        distribution = os.getenv("split_embedding_distribution")
        if distribution is None:
            distribution = 1
        logging.debug(f"distribution = {distribution}")

        target_device = torch.device(device)

        if ("indices_tensor" in config["args"][4]) and (
            "offsets_tensor" in config["args"][4]
        ):
            indices_file = config["args"][4]["indices_tensor"]
            offsets_file = config["args"][4]["offsets_tensor"]
            if "weights_tensor" in config["args"][4]:
                weights_file = config["args"][4]["weights_tensor"]
        else:
            indices_file = os.getenv("split_embedding_indices")
            offsets_file = os.getenv("split_embedding_offsets")
            weights_file = os.getenv("split_embedding_weights")

        logging.debug(f"indices_file: {indices_file}, offsets_file: {offsets_file}")
        if indices_file is not None and offsets_file is not None:
            indices_tensor = torch.load(indices_file, map_location=target_device)
            offsets_tensor = torch.load(offsets_file, map_location=target_device)
            per_sample_weights_tensor = (
                torch.load(weights_file, map_location=target_device)
                if weighted
                else None
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

            indices_tensor = torch.cat([t for t in indices_list])
            offsets_tensor = torch.cat([t for t in offsets_list])

            # check for per sample weights
            per_sample_weights_tensor = (
                torch.cat([t for t in per_sample_weights_list]) if weighted else None
            )

        # check for backward
        grads_tensor = (
            torch.randn(batch_size, sum(dims)) if (propagation == 1) else None
        )
        logging.debug(f"indices: {indices_tensor.shape}, {indices_tensor}")
        logging.debug(f"offsets: {offsets_tensor.shape}, {offsets_tensor}")
        if per_sample_weights_tensor is not None:
            logging.debug(
                f"per_sample_weights: {per_sample_weights_tensor.shape}, {per_sample_weights_tensor}"
            )
        if grads_tensor is not None:
            logging.debug(f"grads: {grads_tensor.shape}, {grads_tensor}")

        return (
            [
                indices_tensor.to(target_device),
                offsets_tensor.to(target_device),
                per_sample_weights_tensor.to(target_device) if weighted else None,
                grads_tensor.to(target_device) if (propagation == 1) else None,
            ],
            {},
        )


register_data_generator(
    "SplitTableBatchedEmbeddingInputDataGenerator",
    SplitTableBatchedEmbeddingInputDataGenerator,
)

# Callable ops are ops can be called in the form of op(*args, **kwargs)
class SplitTableBatchedEmbeddingOp(OperatorInterface):
    def __init__(
        self,
    ):
        super(SplitTableBatchedEmbeddingOp, self).__init__()
        self.cleanup()

    def build(
        self,
        num_tables,
        rows,
        dims,
        location,
        device,
        pooling,
        propagation,
        weighted,
        weights_precision,
        optimizer,
    ):
        logging.debug(
            f"Op build {num_tables}, {rows}, {dims}, {location}, {device}, {pooling}, {propagation}, {weighted}, {weights_precision}, {optimizer}"
        )
        if num_tables == 1:
            rows_list = [rows]
            dims_list = [dims]
        else:
            rows_list = rows
            dims_list = dims

        self.propagation = propagation
        # split_table op options from actual runs of
        # caffe2/torch/fb/module_factory/proxy_module/grouped_sharded_embedding_bag.py
        self.op = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                [
                    (
                        rows_list[i],
                        dims_list[i],
                        EmbeddingLocation(location),
                        ComputeDevice(device),
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
            )
        )
        if device == 1:
            logging.debug("Op using CUDA")
            self.op = self.op.cuda()
        logging.debug(
            f"Op built: {self.op.weights_precision} {self.op.embedding_specs}"
        )

    def cleanup(self):
        logging.debug("Op cleanup")
        self.op = None
        self.propagation = None
        gc.collect()

    def forward(self, *args, **kwargs):
        if self.propagation == 0:
            logging.debug("Op Forward")
            self.op.forward(args[0], args[1], args[2])
        elif self.propagation == 1:
            logging.debug("Op Forward + Backward")
            self.op(args[0], args[1], args[2]).backward(args[3])
        else:
            raise ValueError(f"Invalid propagation value: {self.propagation}")


register_operator("split_table_batched_embedding_bags", SplitTableBatchedEmbeddingOp())
