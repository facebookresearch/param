import importlib
import logging
import pkgutil


logger = None


def get_logger():
    global logger
    if logger:
        return logger
    else:
        return init_logging(logging.INFO)


def init_logging(log_level):
    global logger
    FORMAT = "[%(asctime)s] %(filename)s:%(lineno)-3d [%(levelname)s]: %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("param_bench")
    logger.setLevel(log_level)
    return logger


def load_modules(package):
    """
    Given a package, load/import all the modules in that package.
    See https://packaging.python.org/guides/creating-and-discovering-plugins/
    """
    modules = pkgutil.iter_modules(package.__path__, package.__name__ + ".")
    for _, name, _ in modules:
        logger.debug(f"Loading module: {name}")
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            logger.warning(f"Failed to import module: {name}. ModuleNotFoundError: {error}")


def load_package(package) -> bool:
    """
    Try to load third-party modules, return false if failed.
    """
    logger.debug(f"Loading package: {package}")
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as error:
        logger.warning(f"Failed to import package: {package}. ModuleNotFoundError: {error}")
        return False
    return True
