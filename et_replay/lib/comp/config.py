import copy
import json
from typing import Any, Dict, List, Optional, Type

from .data import data_generator_map, DataGenerator
from .init_helper import get_logger
from .iterator import config_iterator_map, ConfigIterator, DefaultConfigIterator
from .operator import op_map, OperatorInterface


logger = get_logger()


class OperatorConfig:
    def __init__(
        self, name: str, info: Dict[str, Any], op: Optional[OperatorInterface] = None
    ):
        self._name: str = name
        self._info: Dict[str, Any] = info
        self._op: Optional[OperatorInterface] = op

    @property
    def name(self) -> str:
        return self._name

    @property
    def op(self) -> Optional[OperatorInterface]:
        return self._op

    @op.setter
    def op(self, value: OperatorInterface):
        self._op = value

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
    if op_name in op_map:
        op = op_map[op_name]
        op.device = device
    else:
        op = None
    op_config = OperatorConfig(op_name, op_info, op)

    def get(key, table, default):
        nonlocal op_info
        if key in op_info:
            result = op_info[key]
            if result and result in table:
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
        logger.warning(
            f"{op_name} has invalid input_data_generator: {op_config.input_data_generator}"
        )
        return None

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
