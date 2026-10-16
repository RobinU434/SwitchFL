import logging


class CustomFormatter(logging.Formatter):
    """class copied from: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def format_logger(logger: logging.Logger) -> logging.Logger:
    """Format a logger with a custom formatter.

    Args:
        logger (logging.Logger): The logger to format.

    Returns:
        logging.Logger: The formatted logger.
    """
    logger.propagate = False  # Prevent duplicate logs
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(CustomFormatter())
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int):
    """Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    # Apply comprehensive seeding before creating any generators
    import numpy as np
    import random
    import os
    
    # Set all random seeds aggressively
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set TensorFlow and PyTorch seeds if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass