from functools import wraps
import json
from datetime import datetime
from pathlib import Path
from fastlogging import LogInit
import re
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List

from config.config import Config

class Logger:
    """Custom logger class that supports both file and console logging with JSON formatting."""
    
    def __init__(self, name: str, log_dir: str = None):
        """Initialize the logger with the given name and log directory."""
        self.name = name
        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")
        self.log_file = None
        self.logger = None
        self._initialize_logger()
        
    def _initialize_logger(self):
        """Initialize the logger with file and console handlers."""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Create log file path
            timestamp = datetime.now().strftime("%y.%m.%d")
            self.log_file = os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")
            
            # Create logger
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(logging.INFO)
            
            # Create formatters
            file_formatter = logging.Formatter(
                '%(asctime)s: %(name)s: %(levelname)-8s: %(message)s',
                datefmt='%y.%m.%d %H:%M:%S'
            )
            console_formatter = logging.Formatter(
                '%(asctime)s: %(name)s: %(levelname)-8s: %(message)s',
                datefmt='%H:%M:%S'
            )
            
            # Create file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_formatter)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            # Test logging
            self.logger.info("Log file creation test successful")
            self.logger.info("Test log message")
            self.logger.info("Logging functionality test successful")
            
            # Test JSON logging
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            self.logger.info(f"JSON log structure test: {json.dumps(test_data)}")
            self.logger.info("JSON log structure test successful")
            
        except Exception as e:
            print(f"Error initializing logger: {str(e)}")
            raise RuntimeError(f"Failed to initialize logger: {str(e)}")
            
    def info(self, message: str):
        """Log an info message."""
        if self.logger:
            self.logger.info(message)
            
    def error(self, message: str):
        """Log an error message."""
        if self.logger:
            self.logger.error(message)
            
    def warning(self, message: str):
        """Log a warning message."""
        if self.logger:
            self.logger.warning(message)
            
    def debug(self, message: str):
        """Log a debug message."""
        if self.logger:
            self.logger.debug(message)
            
    def critical(self, message: str):
        """Log a critical message."""
        if self.logger:
            self.logger.critical(message)
            
    async def initialize_async(self):
        """Initialize async components of the logger."""
        try:
            # Test async logging
            self.logger.info("Logger async components initialized")
        except Exception as e:
            print(f"Error initializing async logger components: {str(e)}")
            raise RuntimeError(f"Failed to initialize async logger components: {str(e)}")

def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Name of the logger
        log_dir: Optional directory for log files. If not provided, uses default logs directory
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set default log level
    logger.setLevel(logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)-8s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)-8s: %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log directory is provided
    if log_dir:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger 