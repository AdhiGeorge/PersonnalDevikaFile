from utils.logger import Logger
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Class to handle agent errors."""
    
    def __init__(self):
        self._errors: Dict[str, Any] = {}
        self._retry_counts: Dict[str, int] = {}
        self._max_retries = 3
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle an error and determine if it should be retried.
        
        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
            
        Returns:
            bool: True if the error should be retried, False otherwise
        """
        error_id = str(datetime.now().timestamp())
        self._errors[error_id] = {
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        # Log the error
        logger.error(f"Error occurred: {str(error)}")
        if context:
            logger.error(f"Context: {context}")
        
        # Check if we should retry
        if error_id in self._retry_counts:
            self._retry_counts[error_id] += 1
        else:
            self._retry_counts[error_id] = 1
            
        return self._retry_counts[error_id] <= self._max_retries
    
    def get_errors(self) -> Dict[str, Any]:
        """Get all recorded errors."""
        return self._errors.copy()
    
    def clear_errors(self):
        """Clear all recorded errors."""
        self._errors.clear()
        self._retry_counts.clear()
    
    def set_max_retries(self, max_retries: int):
        """Set the maximum number of retries for errors."""
        self._max_retries = max_retries 