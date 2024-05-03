import abc
from typing import Dict, Type

from .init_helper import get_logger

logger = get_logger()


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


def register_data_generator(name: str, data_gen_class: Type[DataGenerator]):
    global data_generator_map
    logger.debug(f"register data generator: {name}")
    if name not in data_generator_map:
        data_generator_map[name] = data_gen_class
    else:
        raise ValueError(f"Duplicate data generator registration name: {name}")


data_generator_map: Dict[str, Type[DataGenerator]] = {}
