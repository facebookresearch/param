import importlib
import logging
import pkgutil


_logger = None
_logger_stream_handler = None


def get_logger():
    global _logger
    if _logger:
        return _logger
    else:
        return init_logging(logging.INFO)


def init_logging(log_level):
    global _logger
    global _logger_stream_handler
    if log_level is logging.DEBUG:
        FORMAT = "[%(asctime)s] %(process)d %(filename)s:%(lineno)-3d [%(levelname)s]: %(message)s"
    else:
        FORMAT = "[%(asctime)s] %(process)d [%(levelname)s]: %(message)s"
    _logger = logging.getLogger("param_bench")
    _logger.setLevel(log_level)
    # Reset the stream handlers to avoid multiple outputs.
    _logger.handlers.clear()
    # Do not use parent logger to avoid duplicate messages.
    _logger.propagate = False
    _logger_stream_handler = logging.StreamHandler()
    _logger_stream_handler.setLevel(log_level)
    formatter = logging.Formatter(FORMAT)
    _logger_stream_handler.setFormatter(formatter)
    _logger.addHandler(_logger_stream_handler)
    return _logger


logger = get_logger()


def load_modules(package):
    """
    Given a package, load/import all the modules in that package.
    See https://packaging.python.org/guides/creating-and-discovering-plugins/
    """
    modules = pkgutil.iter_modules(package.__path__, package.__name__ + ".")
    for _, name, _ in modules:
        logger.debug(f"loading module: {name}")
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            logger.warning(
                f"failed to import module: {name}. ModuleNotFoundError: {error}"
            )


def load_package(package) -> bool:
    """
    Try to load third-party modules, return false if failed.
    """
    logger.debug(f"loading package: {package}")
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as error:
        logger.warning(
            f"failed to import package: {package}. ModuleNotFoundError: {error}"
        )
        return False
    return True
