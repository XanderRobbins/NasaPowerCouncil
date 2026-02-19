"""
Centralized logging configuration.
"""
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime


def setup_logging(
    log_file: Path = None,
    log_level: str = "INFO",
    rotation: str = "1 day",
    retention: str = "30 days",
    console_output: bool = True
):
    """
    Configure logging for the entire application.
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate logs (e.g., "1 day", "500 MB")
        retention: How long to keep old logs
        console_output: Whether to output to console
    """
    # Remove default handler
    logger.remove()
    
    # Format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Console handler
    if console_output:
        logger.add(
            sys.stdout,
            format=log_format,
            level=log_level,
            colorize=True
        )
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def get_logger(name: str = None):
    """
    Get logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggingContext:
    """
    Context manager for temporary logging configuration.
    
    Example:
        with LoggingContext(level="DEBUG"):
            # Code here will use DEBUG logging
            pass
    """
    
    def __init__(self, level: str = "DEBUG", enable_console: bool = True):
        self.level = level
        self.enable_console = enable_console
        self.handler_id = None
        
    def __enter__(self):
        log_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )
        
        if self.enable_console:
            self.handler_id = logger.add(
                sys.stdout,
                format=log_format,
                level=self.level,
                colorize=True
            )
        
        return logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler_id is not None:
            logger.remove(self.handler_id)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Example:
        @log_function_call
        def my_function(x, y):
            return x + y
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Example:
        @log_execution_time
        def slow_function():
            time.sleep(2)
    """
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def setup_exception_logging():
    """
    Set up logging for uncaught exceptions.
    """
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = exception_handler


# Performance profiling logger
class PerformanceLogger:
    """
    Log performance metrics.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        
    def log_metric(self, metric_name: str, value: float):
        """Log a performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        logger.debug(f"[{self.name}] {metric_name}: {value:.4f}")
    
    def get_summary(self):
        """Get summary of all metrics."""
        import numpy as np
        
        summary = {}
        for metric_name, values in self.metrics.items():
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary
    
    def print_summary(self):
        """Print summary of metrics."""
        summary = self.get_summary()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Performance Summary: {self.name}")
        logger.info(f"{'='*60}")
        
        for metric_name, stats in summary.items():
            logger.info(f"{metric_name}:")
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Std:  {stats['std']:.4f}")
            logger.info(f"  Min:  {stats['min']:.4f}")
            logger.info(f"  Max:  {stats['max']:.4f}")
        
        logger.info(f"{'='*60}\n")