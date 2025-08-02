"""
Logging utilities for the algorithmic trading system.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Union

def setup_logging(
    level: str = "INFO",
    log_file: Union[str, Path] = None,
    rotation: str = "1 day",
    retention: str = "30 days"
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotation: Log rotation setting
        retention: Log retention setting
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

def get_logger(name: str = None):
    """Get logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger

# Setup default logging
setup_logging(
    level="INFO",
    log_file="logs/algotrading.log"
)
