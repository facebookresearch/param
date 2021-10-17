import json
import logging
from typing import Dict, Set, Tuple, List, Any, Callable, Iterable, Type, TextIO

from .data import DataGenerator, DefaultDataGenerator, data_generator_map
from .iterator import ConfigIterator, config_iterator_map
from .operator import OperatorInterface, op_map

class OperatorConfig:
    def __init__(self, name, config, op):
        self._name = name
        self._config = config
        self._op = op

    @property
    def name(self) -> str:
        return self._name

    @property
    def op(self) -> OperatorInterface:
        return self._op

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def build_iterator(self) -> Type[ConfigIterator]:
        return self._build_iterator

    @build_iterator.setter
    def build_iterator(self, value: Type[ConfigIterator]):
        self._build_iterator = value

    @property
    def input_iterator(self) -> Type[ConfigIterator]:
        return self._input_iterator

    @input_iterator.setter
    def input_iterator(self, value: Type[ConfigIterator]):
        self._input_iterator = value

    @property
    def build_data_generator(self) -> Type[DataGenerator]:
        return self._build_data_generator

    @build_data_generator.setter
    def build_data_generator(self, value: Type[DataGenerator]):
        self._build_data_generator = value

    @property
    def input_data_generator(self) -> Type[DataGenerator]:
        return self._input_data_generator

    @input_data_generator.setter
    def input_data_generator(self, value: Type[DataGenerator]):
        self._input_data_generator = value


class BenchmarkConfig:
    def __init__(self, config_file_name: str, device: str):
        self.device = device
        with open(config_file_name) as config_file:
            ops_data: TextIO
            self.bench_config = json.load(config_file)

            self._op_configs = []
            for op_name in self.bench_config:
                op_config = self._make_op_config(op_name)
                if op_config is not None:
                    self._op_configs.append(op_config)

    @property
    def op_configs(self) -> List[OperatorConfig]:
        return self._op_configs

    def _make_op_config(self, op_name: str):
        if (op_name not in op_map) or (not op_map[op_name]):
            logging.warning(f"{op_name} has no valid callable defined, skipped.")
            return None
        op = op_map[op_name]
        op_info = self.bench_config[op_name]
        configs = op_info["configs"]
        op_config = OperatorConfig(op_name, configs, op)

        op_config.build_iterator = (
            config_iterator_map[op_info["build_iterator"]]
            if "build_iterator" in op_info
            else None
        )

        op_config.input_iterator = (
            config_iterator_map[op_info["input_iterator"]]
            if "input_iterator" in op_info
            else None
        )
        # input_iterator is required
        if not op_config.input_iterator:
            raise ValueError(f"Invalid input_iterator: {op_config.input_iterator}")

        # If no data generator defined, the default is assumed.
        op_config.build_data_generator = (
            data_generator_map[op_info["build_data_generator"]]
            if "build_data_generator" in op_info
            else DefaultDataGenerator
        )
        op_config.input_data_generator = (
            data_generator_map[op_info["input_data_generator"]]
            if "input_data_generator" in op_info
            else DefaultDataGenerator
        )
        return op_config

    def has_op(self, op: str):
        return (op in self.op_configs) and (op in op_map)
