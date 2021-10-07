import abc
import copy
import json
import logging
import random
from enum import Enum
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

import torch
from .generator import full_range, IterableList, ListProduct, TableProduct
from .pytorch_op_util import (
    create_arg,
    create_range_iter,
    ATTR_COPY,
    ATTR_RANGE,
    META_ATTRS,
)


class ConfigIterator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__init__")
            and callable(subclass.__init__)
            and hasattr(subclass, "__iter__")
            and callable(subclass.__iter__)
            and hasattr(subclass, "__next__")
            and callable(subclass.__next__)
            or NotImplemented
        )

    def __init__(self, configs: Dict[str, Any], key: str, device: str):
        self.configs = configs
        self.key = key
        self.device = device

    def __iter__(self):
        return self

    # Loads arg configurations and generates the arg data for an op.
    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError


def remove_meta_attr(config: Dict[str, Any]):
    result_config = copy.deepcopy(config)
    # TODO lofe: support kwargs too.
    for arg in result_config["args"]:
        for attr in META_ATTRS:
            arg.pop(attr, None)
    return result_config


class RangeConfigIterator(ConfigIterator):
    def __init__(
        self,
        configs: Dict[str, Any],
        key: str,
        device: str,
    ):
        super(RangeConfigIterator, self).__init__(configs, key, device)
        self.generator = self._generate()

    def _apply_copy(self, config: Dict[str, Any]):
        args = config["args"]
        # TODO lofe: support kwargs too.
        for arg in args:
            if ATTR_COPY in arg:
                copy_list = arg[ATTR_COPY]
                for attr_map in copy_list:
                    for attr, mapping in attr_map.items():
                        # current arg's attribute at index mapping[0]
                        elem_idx, (src_arg_idx, src_elem_idx) = mapping
                        arg[attr][elem_idx] = args[src_arg_idx][attr][src_elem_idx]

    def _generate(self):
        # loop operator args in each input variants
        # check for special arg index and kwargs handling
        var_id = 0
        # each input has: a list of range arg names, a list of args (some with range info).
        for config in self.configs[self.key]:
            # create iterables for each arg with ranges.
            config_generator = {}
            args = config["args"] if "args" in config else None
            kwargs = config["kwargs"] if "kwargs" in config else None

            if args:
                arg_iters = []
                for arg in args:
                    arg_iters.append(create_range_iter(arg))
                config_generator["args"] = ListProduct(arg_iters)
            if kwargs:
                kwarg_iters = {}
                for kw, arg in kwargs.items():
                    kwarg_iters[kw] = create_range_iter(arg)
                config_generator["kwargs"] = TableProduct(kwarg_iters)

            # start generating combinations
            config_id = 0
            for arg_config in TableProduct(config_generator):
                logging.debug(arg_config)
                # apply __copy__
                self._apply_copy(arg_config)

                # remove meta attributes
                actual_config = remove_meta_attr(arg_config)

                yield (
                    f"{var_id}_{config_id}",
                    actual_config,
                )
                config_id += 1
            var_id += 1

    def __next__(self):
        return next(self.generator)


class SampleConfigIterator(ConfigIterator):
    def __init__(
        self,
        configs: Dict[str, Any],
        key: str,
        device: str,
    ):
        super(SampleConfigIterator, self).__init__(configs, key, device)
        self.idx = 0
        self.configs = configs[key]
        self.num_configs = len(self.configs)

    def __next__(self):
        # check if this is in the op filter, skip if not
        # loop through the different input variants
        if self.idx < self.num_configs:
            result = (self.idx, self.configs[self.idx])
            self.idx += 1
            return result
        else:
            raise StopIteration


class DummyConfigIterator(ConfigIterator):
    def __init__(
        self,
        configs: Dict[str, Any],
        key: str,
        device: str,
    ):
        super(DummyConfigIterator, self).__init__(configs, key, device)
        self.called_once = False

    def __next__(self):
        if not self.called_once:
            self.called_once = True
            return (0, {"args": [], "kwargs": {}})
        else:
            raise StopIteration


config_iterator_map: Dict[str, ConfigIterator] = {
    "SampleConfigIterator": SampleConfigIterator,
    "RangeConfigIterator": RangeConfigIterator,
}
