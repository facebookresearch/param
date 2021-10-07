import abc
import json
import logging
from typing import Dict, Set, Tuple, List, Any, Callable, Iterable, Type, TextIO

import torch
from .op.split_table_batched_embeddings import SplitTableBatchedEmbeddingOp
from .pytorch_op_config import config_iterator_map
from .pytorch_op_interface import OperatorInterface, InPlaceOpByName, CallableOp
from .pytorch_op_util import data_generator_map

op_map: Dict[str, OperatorInterface] = {
    "Optimizer.step#FusedLAMB.step": None,
    "aten::_embedding_bag_backward": None,
    "aten::add": CallableOp(torch.add),
    "aten::add_": InPlaceOpByName("add_"),
    "aten::baddbmm": CallableOp(torch.baddbmm),
    "aten::binary_cross_entropy_with_logits": CallableOp(
        torch.nn.functional.binary_cross_entropy_with_logits
    ),
    "aten::binary_cross_entropy_with_logits_backward": None,
    "aten::bmm": CallableOp(torch.bmm),
    "aten::cat": CallableOp(torch.cat),
    "aten::clamp_": InPlaceOpByName("clamp_"),
    "aten::conj": None,
    "aten::contiguous": None,
    "aten::copy_": None,
    "aten::cumsum": None,
    "aten::detach": None,
    "aten::detach_": None,
    "aten::div": None,
    "aten::embedding_bag": None,
    "aten::empty": None,
    "aten::empty_like": None,
    "aten::expand": None,
    "aten::flatten": None,
    "aten::gather": None,
    "aten::gather_backward": None,
    "aten::layer_norm": None,
    "aten::le": None,
    "aten::log": None,
    "aten::matmul": CallableOp(torch.matmul),
    "aten::mean": CallableOp(torch.mean),
    "aten::mm": CallableOp(torch.mm),
    "aten::mul": CallableOp(torch.mul),
    "aten::narrow": None,
    "aten::native_layer_norm_backward": None,
    "aten::neg": None,
    "aten::new_empty_strided": None,
    "aten::ones_like": None,
    "aten::permute": None,
    "aten::pin_memory": None,
    "aten::record_stream": None,
    "aten::relu": CallableOp(torch.nn.functional.relu),
    "aten::reshape": CallableOp(torch.reshape),
    "aten::rsub": None,
    "aten::select": None,
    "aten::select_backward": None,
    "aten::set_": None,
    "aten::sigmoid": None,
    "aten::sigmoid_backward": None,
    "aten::slice": None,
    "aten::split_with_sizes": None,
    "aten::squeeze": None,
    "aten::sub": None,
    "aten::sum": None,
    "aten::t": None,
    "aten::tanh": None,
    "aten::tanh_backward": None,
    "aten::threshold_backward": None,
    "aten::to": None,
    "aten::transpose": None,
    "aten::unsqueeze": None,
    "aten::view": None,
    "aten::where": None,
    "aten::zero_": None,
    "aten::zeros": None,
    "aten::zeros_like": None,
    "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__": None,
    "fb::asynchronous_complete_cumsum": None,
    "fb::asynchronous_exclusive_cumsum": None,
    "fb::offsets_range": None,
    "fb::permute_pooled_embs_auto_grad": None,
    "fb::permute_sparse_features": None,
    "fb::split_embedding_codegen_lookup_rowwise_adagrad_function": None,
    "split_table_batched_embedding_bags_codegen": SplitTableBatchedEmbeddingOp(),
}


class OperatorConfig:
    def __init__(self, config_file_name: str, device: str, filters: Set[str] = None):
        self.type = type
        self.filters = filters
        self.device = device
        with open(config_file_name) as config_file:
            ops_data: TextIO
            self.op_configs = json.load(config_file)

    # Returns all or filtered ops in the config
    def get_selected_ops(self):
        ops = []

        def make_op(op_name: str):
            if (op_name not in op_map) or (not op_map[op_name]):
                logging.warning("{op_name} has no valid callable defined, skipped.")
                return
            op = op_map[op_name]
            op_info = self.op_configs[op_name]
            op_run_configs = op_info["configs"]
            build_iterator = (
                config_iterator_map[op_info["build_iterator"]]
                if "build_iterator" in op_info
                else None
            )
            input_iterator = (
                config_iterator_map[op_info["input_iterator"]]
                if "input_iterator" in op_info
                else None
            )
            if not input_iterator:
                logging.error(f"Invalid input_iterator: {input_iterator}")

            op.set_iterator(build_iterator, input_iterator)

            build_data_generator = (
                data_generator_map[op_info["build_data_generator"]]
                if "build_data_generator" in op_info
                else None
            )
            input_data_generator = (
                data_generator_map[op_info["input_data_generator"]]
                if "input_data_generator" in op_info
                else None
            )

            op.set_data_generator(build_data_generator, input_data_generator)
            return (op_name, op, op_run_configs)

        if self.filters:
            for op_name in self.filters:
                if op_name not in self.op_configs:
                    logging.warning("{op_name} not in configuration file, skipped.")
                    continue
                ops.append(make_op(op_name))
        else:
            for op_name in self.op_configs:
                ops.append(make_op(op_name))

        return ops

    def has_op(self, op: str):
        return (op in self.op_configs) and (op in op_map)
