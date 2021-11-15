import copy
import json
from typing import Dict, List, Any
from typing import Type

from .data import DataGenerator, data_generator_map
from .init_helper import get_logger
from .iterator import ConfigIterator, config_iterator_map, DefaultConfigIterator
from .operator import OperatorInterface, op_map

logger = get_logger()


class OperatorConfig:
    def __init__(self, name: str, info: Dict[str, Any], op: OperatorInterface):
        self._name = name
        self._info = info
        self._op = op

    @property
    def name(self) -> str:
        return self._name

    @property
    def op(self) -> OperatorInterface:
        return self._op

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

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


def make_op_config(op_name: str, op_info: Dict[str, Any], device: str):
    global op_map
    if (op_name not in op_map) or (not op_map[op_name]):
        logger.warning(f"{op_name} has no valid callable defined, skipped.")
        return None
    op = op_map[op_name]
    op.device = device
    configs = op_info["config"]
    op_config = OperatorConfig(op_name, op_info, op)

    def get(key, table, default):
        nonlocal op_info
        if key in op_info:
            result = op_info[key]
            if result:
                return table[result]
        return default

    op_config.build_iterator = get(
        "build_iterator", config_iterator_map, DefaultConfigIterator
    )
    op_config.input_iterator = get(
        "input_iterator", config_iterator_map, DefaultConfigIterator
    )
    op_config.build_data_generator = get(
        "build_data_generator", data_generator_map, None
    )
    op_config.input_data_generator = get(
        "input_data_generator", data_generator_map, None
    )

    # input_data_generator is required
    if not op_config.input_data_generator:
        raise ValueError(
            f"Invalid input_data_generator: {op_config.input_data_generator}"
        )

    return op_config


class BenchmarkConfig:
    """
    BenchmarkConfig stores loaded configuration data.
    """

    def __init__(self, run_options: Dict[str, Any]):
        self.run_options = run_options
        self._op_configs = []
        self.bench_config = None

    def _process_bench_config(self):
        for op_name, op_info in self.bench_config.items():
            op_config = make_op_config(op_name, op_info, self.run_options["device"])
            if op_config is not None:
                self._op_configs.append(op_config)

    def load_json_file(self, config_file_name: str):
        with open(config_file_name) as config_file:
            self.bench_config = json.load(config_file)
            self._process_bench_config()

    def load_json(self, config_json: str):
        self.bench_config = json.loads(config_json)
        self._process_bench_config()

    def load(self, config: Dict[str, Any]):
        self.bench_config = copy.deepcopy(config)
        self._process_bench_config()

    @property
    def op_configs(self) -> List[OperatorConfig]:
        return self._op_configs

    def has_op(self, op: str):
        return (op in self.op_configs) and (op in op_map)
