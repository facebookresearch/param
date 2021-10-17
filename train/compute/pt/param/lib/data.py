import abc
import copy
import json
import logging
import random
from enum import Enum
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

import torch
from .generator import full_range, IterableList, ListProduct, TableProduct

pytorch_dtype_map: Dict[str, torch.dtype] = {
    "float": torch.float,
    "double": torch.double,
    "int": torch.int,
    "long": torch.long,
}

# Given an arg configuration, generate the test data for that arg.
def create_arg(arg: Dict[str, Any], device: str):
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

    def create_str(attr: Dict[str, Any]):
        # check "value" key exists, attr["value"] = 0 could be eval to False
        if "value" in attr:
            return attr["value"]
        return ""

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
        "str": create_str,
        "genericlist": create_genericlist,
    }
    return arg_factory[arg["type"]](arg)


class DataGenerator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "get_data")
            and callable(subclass.get_data)
            or NotImplemented
        )

    def __init__(self):
        pass

    # Loads arg configurations and generates the arg data for an op.
    @abc.abstractmethod
    def get_data(self):
        raise NotImplementedError


class DefaultDataGenerator(DataGenerator):
    def __init__(self, cache: bool = False):
        super(DefaultDataGenerator, self).__init__()
        # keep track/cache last arg_config so we only generate data for
        # args that's different from previous iteration.
        self.cache = cache
        self.prev_config = None
        self.op_args = []
        self.op_kwargs = {}

    def _find_updates(self, config: Dict[str, Any]):
        if not self.prev_config:
            return (None, None)
        arg_updates = set()
        kwarg_updates = set()
        if "args" in config:
            for i, vals in enumerate(zip(self.prev_config["args"], config["args"])):
                if vals[0] != vals[1]:
                    arg_updates.add(i)
        if "kwargs" in config:
            for key in self.prev_config["kwargs"]:
                if self.prev_config["kwargs"][key] != config["kwargs"][key]:
                    kwarg_updates.add(key)

        logging.debug(f"  prev: {self.prev_config}")
        logging.debug(f"  curr: {config}")
        logging.debug(f"  updt: {arg_updates} {kwarg_updates}")
        return (arg_updates, kwarg_updates)

    def _generate_data(
        self,
        config: Dict[str, Any],
        device: str,
        arg_updates: Set[Any],
        kwarg_updates: Set[Any],
    ):
        if len(self.op_args) == 0:
            self.op_args = [None] * len(config["args"])
        if "args" in config:
            for i, arg in enumerate(config["args"]):
                if arg_updates:
                    if i in arg_updates:
                        self.op_args[i] = create_arg(arg, device)
                else:
                    self.op_args[i] = create_arg(arg, device)

        if "kwargs" in config:
            for key, arg in config["kwargs"].items():
                if kwarg_updates:
                    if key in kwarg_updates:
                        self.op_kwargs[key] = create_arg(arg, device)
                else:
                    self.op_kwargs[key] = create_arg(arg, device)

    def get_data(self, config: Dict[str, Any], device: str):
        if self.cache:
            # find the arg config that changed from previous iteration
            arg_updates, kwarg_updates = self._find_updates(config)
            # cache arg configs for next iteration to compare.
            self.prev_config = copy.deepcopy(config)
            self._generate_data(config, device, arg_updates, kwarg_updates)
        else:
            self._generate_data(config, device, None, None)

        return (self.op_args, self.op_kwargs)


def register_data_generator(name: str, data_gen_class: Type[DataGenerator]):
    global data_generator_map
    if name not in data_generator_map:
        data_generator_map[name] = data_gen_class
    else:
        raise ValueError(f'Duplicate data generator registration name: "{name}"')


data_generator_map: Dict[str, Type[DataGenerator]] = {}

register_data_generator("DefaultDataGenerator", DefaultDataGenerator)
