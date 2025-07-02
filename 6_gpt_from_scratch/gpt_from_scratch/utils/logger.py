"""Module containing logger."""

import logging
from enum import Enum


class LogLevel(Enum):
    """Enumeration of log levels"""

    CRITICAL = "CRITICAL"
    FATAL = "FATAL"
    ERROR = "ERROR"
    WARN = "WARNING"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


class Logger:
    """Class for logging."""

    # Create a logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.propagate = False  # Don't propagate log messages to parent loggers
    # Set the log level to INFO (or any other level you want)
    logger.setLevel(logging.INFO)
    # Create a formatter for the log messages
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s (%(process)d) %(filename)s:%(lineno)d] %(message)s"
    )
    # Create a console handler and add the formatter to it
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # Add the console handler to the logger
    logger.addHandler(console_handler)

    @classmethod
    def set_level(cls, level: int) -> None:
        """
        Set the log level.

        Args:
            level (int): The log level.

        Example:
            >>> from logging import _nameToLevel
            >>> log_level = "INFO"
            >>> Logger.set_level(_nameToLevel[log_level])
        """
        cls.logger.setLevel(level)

    @staticmethod
    def get_logger() -> logging.Logger:
        """
        Return the logger.

        Returns:
            logging.Logger: The logger.
        """
        return Logger.logger
