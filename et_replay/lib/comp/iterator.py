import abc
import copy
from typing import Any, Callable, Dict, List, Type

from .generator import full_range, IterableList, ListProduct, TableProduct
from .init_helper import get_logger

logger = get_logger()

# Special meta attributes in range based configs
ATTR_COPY = "__copy__"
ATTR_RANGE = "__range__"
ATTR_LIST = "__list__"
META_ATTRS = {ATTR_COPY, ATTR_RANGE, ATTR_LIST}


def genericList_to_list(genericList: Dict[str, Any]):
    result = []
    for item in genericList["value"]:
        result.append(item["value"])
    return result


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


def create_range_iter(arg: Dict[str, Any]):
    def create_tensor(attr: Dict[str, Any]):
        logger.debug(f"{attr}")
        result = copy.copy(attr)
        # if ranges exists, create iterator
        if ATTR_RANGE in attr:
            ranges = set(attr[ATTR_RANGE])
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
        if ATTR_RANGE in attr:
            ranges = set(attr[ATTR_RANGE])
            if "value" in ranges:
                result["value"] = full_range(*attr["value"])
                return TableProduct(result)
        return result

    def create_str(attr: Dict[str, Any]):
        result = copy.copy(attr)
        if ATTR_RANGE in attr:
            ranges = set(attr[ATTR_RANGE])
            if "value" in ranges:
                result["value"] = IterableList(attr["value"])
                return TableProduct(result)
        return result

    def create_bool(attr: Dict[str, Any]):
        result = copy.copy(attr)
        if ATTR_RANGE in attr:
            ranges = set(attr[ATTR_RANGE])
            if "value" in ranges:
                result["value"] = IterableList(attr["value"])
                return TableProduct(result)
        return result

    def create_none(attr: Dict[str, Any]):
        return copy.copy(attr)

    # Called for a list of data types to be iterated
    def create_dtype(values: List[str]):
        return IterableList(values)

    def create_shape(values: List[Any]):
        shape = []
        for val in values:
            # TODO lofe: should also check for ATTR_RANGE
            if type(val) is list:
                shape.append(full_range(*val))
            else:
                shape.append(val)
        return ListProduct(shape)

    def create_device(attr: Dict[str, Any]):
        result = copy.copy(attr)
        if ATTR_RANGE in attr:
            ranges = set(attr[ATTR_RANGE])
            if "value" in ranges:
                result["value"] = IterableList(attr["value"])
                return TableProduct(result)
        return result

    def create_genericlist(attr: List[Any]):
        result = copy.copy(attr)
        if ATTR_RANGE in attr:
            ranges = set(attr[ATTR_RANGE])
            if "value" in ranges:
                values = []
                for item in attr["value"]:
                    values.append(arg_factory_iter[item["type"]](item))
                result["value"] = ListProduct(values)
                return TableProduct(result)
        return result

    def create_tuple(attr: List[Any]):
        result = copy.copy(attr)
        if ATTR_RANGE in attr:
            ranges = set(attr[ATTR_RANGE])
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
        "str": create_str,
        "none": create_none,
        "bool": create_bool,
        "dtype": create_dtype,
        "shape": create_shape,
        "device": create_device,
        "genericlist": create_genericlist,
        "tuple": create_tuple,
    }
    return arg_factory_iter[arg["type"]](arg)


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
            if ATTR_COPY in arg and arg["type"] == "tensor":
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
                logger.debug(arg_config)
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


class DefaultConfigIterator(ConfigIterator):
    def __init__(
        self,
        configs: Dict[str, Any],
        key: str,
        device: str,
    ):
        super(DefaultConfigIterator, self).__init__(configs, key, device)
        self.idx = 0
        self.configs = configs[key]
        self.num_configs = len(self.configs)

    def __next__(self):
        # loop through the different config variants
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


def register_config_iterator(name: str, iterator_class: Type[ConfigIterator]):
    global config_iterator_map
    logger.debug(f"register iterator: {name}")
    if name not in config_iterator_map:
        config_iterator_map[name] = iterator_class
    else:
        raise ValueError(f"Duplicate iterator registration name: {name}")


config_iterator_map: Dict[str, Type[ConfigIterator]] = {}

register_config_iterator("DefaultConfigIterator", DefaultConfigIterator)
register_config_iterator("RangeConfigIterator", RangeConfigIterator)
