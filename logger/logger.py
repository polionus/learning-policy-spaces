# logger_setup.py
import sys
from loguru import logger

# Remove any pre-existing handlers (important for notebooks / repeated imports)
logger.remove()

# Console logger
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> "
           "| <level>{level}</level> "
           "| <cyan>{file}</cyan>:<cyan>{line}</cyan> "
           "| <level>{message}</level>",
)

# File logger
logger.add(
    "logger/run.log",
    level="DEBUG",  # file gets everything
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    rotation="10 MB",
    enqueue=True,
)

# Export the configured logger
__all__ = ["logger"]
