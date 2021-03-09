import copy
import json
import logging
import random
from enum import Enum
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

import torch
from generator import full_range, IterableList, ListProduct, TableProduct

from pytorch_op_def import get_pytorch_ops, pytorch_dtype_map


OpConfigType = Enum("OpConfigType", "SAMPLE RANGE")


"""
OperatorConfig
- load all ops and configs in a file
- iterate all (or filtered) ops with configs or range configs
- for each config generate data to run in benchmark
- if generating range configs, cache previous data to avoid regenerate the same data.
- stream through the ops rather than generate all the data for all the ops at once.

- get a list of all ops in the config, so external caller can iterate and request data for each op.
- provide interface for iterate over one op and all its configs
- user shouldn't care if it's sample or ranged
- it just request arg data through an iterator for (OperatorConfig, operator)

- given an op and its configurations, provide iterator to generate data for a all the arguments for each configuration.
  inputs: op, configs, device
  outputs: iterator for Dict["op_name": List[List[arg_data]]]

- given an op and one configuration, generate data for a all the arguments or
  a subset of arg indices through a mask.
  inputs: op, config, device, arg_data, mask
  outputs: Dict["op_name": List[arg_data]]

- given an op and all of its range configuration, provides an interface to iterator
  to get all possible combinations of configurations.
  inputs: op, range_config, device
  outputs: iterator for Dict["op_name": List[arg_data]]

"""
# Special meta attributes in range based configs
__ATTR_COPY__ = "__copy__"
__ATTR_RANGE__ = "__range__"
META_ATTRS = {__ATTR_COPY__, __ATTR_RANGE__}


class OperatorConfig:
    def __init__(
        self, json_file: str, type: OpConfigType, device: str, filters: Set[str] = None
    ):
        self.type = type
        self.filters = filters
        self.device = device
        with open(json_file) as ops_config:
            ops_data: TextIO
            self.ops = json.load(ops_config)

    # Returns all or filtered ops in the config
    def get_selected_ops(self):
        if self.filters:
            return [x for x in self.filters if x in self.ops]
        return self.ops.keys()

    def has_op(self, op: str):
        return op in self.ops

    def get_next_args(self, op: str):
        if self.type == OpConfigType.RANGE:
            yield from self.__generate_op_args_in_range(op)
        elif self.type == OpConfigType.SAMPLE:
            yield from self.__generate_op_args(op)
        else:
            raise NotImplementedError("OpConfigType {self.type} not implemented")

    def __valid_op(self, op: str):
        ops_map = get_pytorch_ops()
        # check if this is in the op filter, skip if not
        if not ops_map[op]:
            logging.info(f"{op} is not yet supported, skipping.")
            return False
        return True

    def __generate_op_args_in_range(self, op: str) -> Dict[str, Any]:
        def apply_copy(args: List[Any]):
            for arg in args:
                if __ATTR_COPY__ in arg:
                    copy_list = arg[__ATTR_COPY__]
                    for attr_map in copy_list:
                        for attr, mapping in attr_map.items():
                            # current arg's attribute at index mapping[0]
                            elem_idx, (src_arg_idx, src_elem_idx) = mapping
                            arg[attr][elem_idx] = args[src_arg_idx][attr][src_elem_idx]

        def remove_meta_attr(args: List[Any]):
            result_args = copy.deepcopy(args)
            for arg in result_args:
                for attr in META_ATTRS:
                    arg.pop(attr, None)
            return result_args

        def find_updates(last_configs: List[Any], configs: List[Any]):
            updates = set()
            for i, vals in enumerate(zip(last_configs, configs)):
                if vals[0] != vals[1]:
                    updates.add(i)

            logging.debug(f"  last: {last_configs}")
            logging.debug(f"  curr: {configs}")
            logging.debug(f"  updt: {updates}")
            return updates

        def generate_data():
            if arg_indices or kwarg_map:
                for i, arg_pos in enumerate(arg_indices):
                    if update_args:
                        if arg_pos in update_args:
                            logging.debug(f"  {arg_pos}: {arg_configs[arg_pos]}")
                            op_args[i] = self.create_arg(
                                arg_configs[arg_pos], self.device
                            )
                    else:
                        logging.debug(f"  {arg_pos}: {arg_configs[arg_pos]}")
                        op_args[i] = self.create_arg(arg_configs[arg_pos], self.device)
                for kw, arg_pos in kwarg_map.items():
                    if update_args:
                        if arg_pos in update_args:
                            logging.debug(f"  {kw}:{arg_configs[arg_pos]}")
                            op_kwargs[kw] = self.create_arg(
                                arg_configs[arg_pos], self.device
                            )
                    else:
                        logging.debug(f"  {kw}:{arg_configs[arg_pos]}")
                        op_kwargs[kw] = self.create_arg(
                            arg_configs[arg_pos], self.device
                        )
            else:
                for i, arg in enumerate(arg_configs):
                    if update_args:
                        if i in update_args:
                            logging.debug(f"  {i}: {arg}")
                            op_args[i] = self.create_arg(arg, self.device)
                    else:
                        logging.debug(f"  {i}: {arg}")
                        op_args[i] = self.create_arg(arg, self.device)

        ops_map = get_pytorch_ops()
        if not self.__valid_op(op):
            return
        op_to_run = {"name": op}
        logging.info(f"Loading {op} args range")
        logging.info(f"{op}, arg range variants: {len(self.ops[op]['inputs'])}")
        # loop operator args in each input variants
        # check for special arg index and kwargs handling
        arg_indices = ops_map[op].get_arg_indices()
        kwarg_map = ops_map[op].get_kwarg_map()
        var_id = 0
        # each input has: a list of range arg names, a list of args (some with range info).
        for arg_ranges in self.ops[op]["inputs"]:
            # create iterables for each arg with ranges.
            generated_configs = []
            for arg in arg_ranges:
                generated_configs.append(self.create_range_iter(arg))
            logging.debug(generated_configs)
            # stores the actual positional and keyward args with real data
            op_args = []
            op_kwargs = {}
            if arg_indices or kwarg_map:
                op_args = [None] * len(arg_indices)
                op_kwargs = dict.fromkeys(kwarg_map)
            else:
                op_args = [None] * len(generated_configs)

            # keep track/cache last arg_config so we only generate data for
            # args that's different from previous iteration.
            last_arg_configs = [None] * len(generated_configs)
            # start generating combinations
            config_id = 0
            for arg_configs in ListProduct(generated_configs):
                # apply __copy__
                apply_copy(arg_configs)

                # remove meta attributes
                actual_configs = remove_meta_attr(arg_configs)

                # find the arg config that changed from previous iteration
                update_args = find_updates(last_arg_configs, actual_configs)

                # do the real work of generating data
                generate_data()
                op_to_run["args"] = (
                    f"{var_id}_{config_id}",
                    op_args,
                    op_kwargs,
                    actual_configs,
                )
                # cache arg configs for next iteration to compare.
                last_arg_configs = copy.deepcopy(actual_configs)
                yield op_to_run
                config_id += 1
            var_id += 1
        logging.info(f"Finished all {op} args")

    def create_range_iter(self, arg: Dict[str, Any]):
        def create_tensor(attr: Dict[str, Any]):
            logging.debug(f"{attr}")
            result = copy.copy(attr)
            # if ranges exists, create iterator
            if __ATTR_RANGE__ in attr:
                ranges = set(attr[__ATTR_RANGE__])
                for key, val in attr.items():
                    if key in ranges:
                        result[key] = arg_factory_iter[key](val)
                    else:
                        result[key] = val
                return TableProduct(result)
            # otherwise return unchanged
            return result

        def create_float(attr: Dict[str, Any]):
            # Not supporting range float values, any use cases?
            return copy.copy(attr)

        def create_int(attr: Dict[str, Any]):
            result = copy.copy(attr)
            if __ATTR_RANGE__ in attr:
                ranges = set(attr[__ATTR_RANGE__])
                if "value" in ranges:
                    result["value"] = full_range(*attr["value"])
                    return TableProduct(result)
            return result

        def create_bool(attr: Dict[str, Any]):
            result = copy.copy(attr)
            if __ATTR_RANGE__ in attr:
                ranges = set(attr[__ATTR_RANGE__])
                if "value" in ranges:
                    result["value"] = IterableList(attr["value"])
                    return TableProduct(result)
            return result

        def create_none(attr: Dict[str, Any]):
            return copy.copy(attr)

        def create_dtype(values: List[str]):
            return IterableList(values)

        def create_shape(values: List[Any]):
            shape = []
            for val in values:
                if type(val) is list:
                    shape.append(full_range(*val))
                else:
                    shape.append(val)
            return ListProduct(shape)

        def create_device(attr: Dict[str, Any]):
            result = copy.copy(attr)
            if __ATTR_RANGE__ in attr:
                ranges = set(attr[__ATTR_RANGE__])
                if "value" in ranges:
                    result["value"] = IterableList(attr["value"])
                    return TableProduct(result)
            return result

        def create_genericlist(attr: List[Any]):
            result = copy.copy(attr)
            if __ATTR_RANGE__ in attr:
                ranges = set(attr[__ATTR_RANGE__])
                if "value" in ranges:
                    values = []
                    for item in attr["value"]:
                        values.append(arg_factory_iter[item["type"]](item))
                    result["value"] = ListProduct(values)
                    return TableProduct(result)
            return result

        arg_factory_iter: Dict[str, Callable] = {
            "tensor": create_tensor,
            "float": create_float,
            "double": create_float,
            "int": create_int,
            "long": create_int,
            "none": create_none,
            "bool": create_bool,
            "dtype": create_dtype,
            "shape": create_shape,
            "device": create_device,
            "genericlist": create_genericlist,
        }
        return arg_factory_iter[arg["type"]](arg)

    # Loads input args configuration and generates the arg data for an op.
    # Returns a dictionary of op -> List[args data] that can be passed to
    # benchmarks.
    def __generate_op_args(self, op: str) -> Dict[str, Any]:
        ops_map = get_pytorch_ops()
        # check if this is in the op filter, skip if not
        if not self.__valid_op(op):
            return
        op_to_run = {"name": op}
        logging.info(f"Loading {op} args")
        logging.info(f"{op}, arg variants: {len(self.ops[op]['inputs'])}")
        # loop through the different input variants
        idx = 0
        for arg_configs in self.ops[op]["inputs"]:
            op_args = []
            op_kwargs = {}
            # loop operator args in each input variants
            # check for special arg index and kwargs handling
            arg_map = ops_map[op].get_arg_indices()
            kwarg_map = ops_map[op].get_kwarg_map()
            if arg_map or kwarg_map:
                for i in arg_map:
                    logging.debug(f"  {arg_configs[i]}")
                    op_args.append(self.create_arg(arg_configs[i], self.device))
                for kw, i in kwarg_map.items():
                    logging.debug(f"  {kw}:{arg_configs[i]}")
                    op_kwargs[kw] = self.create_arg(arg_configs[i], self.device)
            else:
                for arg in arg_configs:
                    logging.debug(f"  {arg}")
                    op_args.append(self.create_arg(arg, self.device))
            op_to_run["args"] = (idx, op_args, op_kwargs, arg_configs)
            yield op_to_run
            idx += 1
        logging.info(f"Finished all {op} args")

    # Given an arg configuration, generate the test data for that arg.
    def create_arg(self, arg: Dict[str, Any], device: str):
        def create_tensor(attr: Dict[str, Any]):
            shape = attr["shape"]
            if len(shape) > 0:
                if attr["dtype"] == "float" or attr["dtype"] == "double":
                    return torch.rand(
                        *shape, requires_grad=False, device=torch.device(device)
                    )
                elif attr["dtype"] == "int" or attr["dtype"] == "long":
                    return torch.randint(
                        -10,
                        10,
                        tuple(shape),
                        requires_grad=False,
                        device=torch.device(device),
                    )
            # Single value
            else:
                return torch.tensor(
                    random.uniform(-10.0, 10.0),
                    dtype=pytorch_dtype_map[attr["dtype"]],
                    requires_grad=False,
                    device=torch.device(device),
                )

        def create_float(attr: Dict[str, Any]):
            if "value" in attr:
                return attr["value"]
            return random.uniform(attr["value_range"][0], attr["value_range"][1])

        def create_int(attr: Dict[str, Any]):
            # check "value" key exists, attr["value"] = 0 could be eval to False
            if "value" in attr:
                return attr["value"]
            return random.randint(attr["value_range"][0], attr["value_range"][1])

        def create_bool(attr: Dict[str, Any]):
            return attr["value"]

        def create_none(attr: Dict[str, Any]):
            return None

        def create_device(attr: Dict[str, Any]):
            return torch.device(attr["value"])

        def create_genericlist(attr: List[Any]):
            result = []
            for item in attr["value"]:
                result.append(arg_factory[item["type"]](item))
            return result

        arg_factory: Dict[str, Callable] = {
            "tensor": create_tensor,
            "float": create_float,
            "double": create_float,
            "int": create_int,
            "long": create_int,
            "none": create_none,
            "bool": create_bool,
            "device": create_device,
            "genericlist": create_genericlist,
        }
        return arg_factory[arg["type"]](arg)


class OpDataIter:
    def __init__(self, op_config: OperatorConfig, op: str):
        self.op_config = op_config
        self.op = op
        self.generator = None
        if self.op_config.has_op(self.op):
            self.generator = self.op_config.get_next_args(self.op)

    def __iter__(self):
        return self

    def __next__(self):
        if self.generator:
            return next(self.generator)
        else:
            raise StopIteration
