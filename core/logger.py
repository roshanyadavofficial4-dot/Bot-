import logging
import os
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Creates a highly professional dual-output logger.
    Logs to both the console and a permanent file (logs/trading_log.txt).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if logger is instantiated multiple times
    if logger.hasHandlers():
        return logger

    # Clean formatting: [YYYY-MM-DD HH:MM:SS] - [LEVEL] - [MODULE] - Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] - [%(levelname)s] - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Permanent Storage)
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/trading_log.txt", mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
