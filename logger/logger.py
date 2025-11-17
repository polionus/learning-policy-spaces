# logger_setup.py
import sys
from loguru import logger

def setup_logger(log_file="run.log", level="INFO"):
    logger.remove()
    
    # Console logger
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> "
               "| <level>{level}</level> "
               "| <cyan>{file}</cyan>:<cyan>{line}</cyan> "
               "| <level>{message}</level>",
    )
    
    # File logger
    logger.add(
        log_file,
        level="DEBUG",        # file gets everything
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        rotation="10 MB",
        enqueue=True,
    )
    
    return logger
