"""Central logging configuration for the project."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Returns a logger configured with a single StreamHandler (avoids duplicate
    handlers when called multiple times for the same `name`)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


if __name__ == "__main__":
    log = get_logger("gradf.demo")
    log.info("Logger configured successfully")
