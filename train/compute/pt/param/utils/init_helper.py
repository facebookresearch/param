

import importlib
import logging
import pkgutil

def init_logging():
    FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)


def load_benchmarks(package):
    # See https://packaging.python.org/guides/creating-and-discovering-plugins/
    benchmark_modules = pkgutil.iter_modules(package.__path__, package.__name__ + ".")
    for _, name, _ in benchmark_modules:
        logging.debug(f"Loading benchmark module: {name}")
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            logging.error(f"Failed to import module: {name}")
