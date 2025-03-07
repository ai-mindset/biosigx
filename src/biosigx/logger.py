"""Set up and use a logging system with configurable levels and console output."""

# %%
import inspect
import logging
import os
from typing import Literal

# %% [markdown]
# ## Logging
# [logging levels](https://docs.python.org/3/library/logging.html#logging-levels)
# | Level               | Value  | Description |
# |---------------------|--------|-------------|
# | `logging.NOTSET`    | 0      | When set on a logger, indicates that ancestor loggers are to be consulted to determine the effective level. If that still resolves to NOTSET, then all events are logged. When set on a handler, all events are handled. |
# | `logging.DEBUG`     | 10     | Detailed information, typically only of interest to a developer trying to diagnose a problem. |
# | `logging.INFO`      | 20     | Confirmation that things are working as expected. |
# | `logging.WARNING`   | 30     | An indication that something unexpected happened, or that a problem might occur in the near future (e.g. 'disk space low'). The software is still working as expected. |
# | `logging.ERROR`     | 40     | Due to a more serious problem, the software has not been able to perform some function. |
# | `logging.CRITICAL`  | 50     | A serious error, indicating that the program itself may be unable to continue running. |


# %%
def setup_logger(
    level: str | Literal[0, 10, 20, 30, 40, 50] = "INFO",
    module_name: str | None = None,
) -> logging.Logger:
    """Set up a logger with specified level and console output.

    Args:
        level (str or int): The logging level as a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
                          or an integer value (10, 20, 30, 40, 50). Default is 'INFO'.
        module_name (str | None): Optional module name override. If None, automatically determined from call stack.

    Returns:
        logging.Logger: A configured logger object with console handler and specified format.

    Examples:
        >>> logger = setup_logger()  # Automatically captures module name
        >>> logger.info("This is an info message")

    """
    # Automatically determine the module name if not explicitly provided
    if module_name is None:
        frame = inspect.stack()[1]
        # Get the filename of the calling module
        module_file = os.path.basename(frame[0].f_code.co_filename)
        # Remove the .py extension if present
        module_name = os.path.splitext(module_file)[0]

    # Create a logger object
    logger = logging.getLogger(module_name)

    # Only add handler if the logger doesn't already have handlers
    if not logger.handlers:
        # Set the overall logging level of the logger
        logger.setLevel(level)

        # Create a console handler (outputs to terminal)
        ch = logging.StreamHandler()

        # Set the logging level for the handler
        ch.setLevel(level)

        # Create a formatter that specifies the format of log messages
        formatter = logging.Formatter(
            "%(asctime)s - Module: %(name)s - Level: %(levelname)s - Message: %(message)s"
        )

        # Attach the formatter to the handler
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger


# %% [markdown]
# ## Main guard

# %%
if __name__ == "__main__":
    # Set up the logger with INFO level - module name will be auto-detected
    logger = setup_logger()

    # Log messages at different levels
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
