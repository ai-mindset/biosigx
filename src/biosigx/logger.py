"""Set up and use a logging system with configurable levels and console output."""

# %%
import logging
from typing import Literal

# %% [markdown]
# ## Logging
# [logging levels](https://docs.python.org/3/library/logging.html#logging-levels)
# | Level | Value | Description |
# |---------|--------|-------------|
# | `logging.NOTSET` | 0 | When set on a logger, indicates that ancestor loggers are to be consulted to determine the effective level. If that still resolves to NOTSET, then all events are logged. When set on a handler, all events are handled. |
# | `logging.DEBUG` | 10 | Detailed information, typically only of interest to a developer trying to diagnose a problem. |
# | `logging.INFO` | 20 | Confirmation that things are working as expected. |
# | `logging.WARNING` | 30 | An indication that something unexpected happened, or that a problem might occur in the near future (e.g. 'disk space low'). The software is still working as expected. |
# | `logging.ERROR` | 40 | Due to a more serious problem, the software has not been able to perform some function. |
# | `logging.CRITICAL` | 50 | A serious error, indicating that the program itself may be unable to continue running. |


# %%
def setup_logger(level: str | Literal[0, 10, 20, 30, 40, 50] = "INFO") -> logging.Logger:
    """Set up a logger with specified level and console output.

    Args:
        level (str or int): The logging level as a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') or an integer value (10, 20, 30, 40, 50). Default is 'INFO'.

    Returns:
        logging.Logger: A configured logger object with console handler and specified format.

    Examples:
        >>> setup_logger(level="DEBUG").info("This is a debug message")

    """
    # Create a logger object
    logger = logging.getLogger(__name__)

    # Set the overall logging level of the logger
    logger.setLevel(level)

    # Create a console handler (outputs to terminal)
    ch = logging.StreamHandler()

    # Set the logging level for the handler
    ch.setLevel(level)

    # Create a formatter that specifies the format of log messages
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Attach the formatter to the handler
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger


# %% [markdown]
# ## Main guard

# %%
if __name__ == "__main__":
    # Set up the logger with INFO level
    logger = setup_logger(logging.INFO)

    # Log messages at different levels
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
