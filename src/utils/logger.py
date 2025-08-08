"""
Logging utilities for TechAuthor system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

# Global logger instance - singleton pattern
_global_logger = None
_is_configured = False


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with consistent configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    global _global_logger, _is_configured
    
    # Create or get the root logger
    logger = logging.getLogger("TechAuthor")
    
    # Only configure if not already configured OR if we need to add file logging
    needs_configuration = not _is_configured or (log_file and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers))
    
    if not needs_configuration and _global_logger:
        return _global_logger
    
    # Clear existing handlers only if we're doing initial configuration
    if not _is_configured:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Add console handler only if not already configured
    if not _is_configured:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (if specified and not already added)
    if log_file and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        max_bytes = _parse_file_size(max_file_size)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set as global logger and mark as configured
    _global_logger = logger
    _is_configured = True
    
    return logger


def _parse_file_size(size_str: str) -> int:
    """Parse file size string to bytes.
    
    Args:
        size_str: Size string like "10MB", "1GB", "500KB"
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)


def get_logger(name: str = None) -> logging.Logger:
    """Get the global logger instance. All components should use this to ensure consistent logging.
    
    Args:
        name: Optional component name (for backward compatibility, but all use same logger)
        
    Returns:
        The global logger instance
    """
    global _global_logger, _is_configured
    
    # If not configured yet, set up with default configuration
    if not _is_configured or not _global_logger:
        return setup_logger("TechAuthor", level="INFO")
    
    return _global_logger


def configure_logger(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure the global logger. Should be called once at startup.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    return setup_logger("TechAuthor", level=level, log_file=log_file)
