import logging
import sys

from config.settings import settings

_CONFIGURED = False

def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    _CONFIGURED = True

def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)