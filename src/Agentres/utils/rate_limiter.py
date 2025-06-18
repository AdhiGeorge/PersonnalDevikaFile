import logging
import time
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self._max_requests = max_requests
        self._time_window = time_window
        self._requests: Dict[str, list] = {}
        
    def _cleanup_old_requests(self, key: str) -> None:
        """Remove requests older than the time window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self._time_window)
        
        self._requests[key] = [
            req_time for req_time in self._requests[key]
            if req_time > cutoff
        ]
        
    def can_make_request(self, key: str = "default") -> bool:
        """Check if a request can be made.
        
        Args:
            key: Rate limit key (default: "default")
            
        Returns:
            bool: True if request can be made, False otherwise
        """
        if key not in self._requests:
            self._requests[key] = []
            
        self._cleanup_old_requests(key)
        return len(self._requests[key]) < self._max_requests
        
    def add_request(self, key: str = "default") -> None:
        """Add a request to the rate limiter.
        
        Args:
            key: Rate limit key (default: "default")
        """
        if key not in self._requests:
            self._requests[key] = []
            
        self._requests[key].append(datetime.now())
        
    def wait_if_needed(self, key: str = "default") -> None:
        """Wait if rate limit would be exceeded.
        
        Args:
            key: Rate limit key (default: "default")
        """
        while not self.can_make_request(key):
            time.sleep(1)
            
    def get_remaining_requests(self, key: str = "default") -> int:
        """Get number of remaining requests in current time window.
        
        Args:
            key: Rate limit key (default: "default")
            
        Returns:
            int: Number of remaining requests
        """
        if key not in self._requests:
            return self._max_requests
            
        self._cleanup_old_requests(key)
        return self._max_requests - len(self._requests[key])
        
    def get_time_until_reset(self, key: str = "default") -> Optional[float]:
        """Get time until rate limit resets.
        
        Args:
            key: Rate limit key (default: "default")
            
        Returns:
            float: Seconds until reset, or None if no requests made
        """
        if key not in self._requests or not self._requests[key]:
            return None
            
        oldest_request = min(self._requests[key])
        reset_time = oldest_request + timedelta(seconds=self._time_window)
        return max(0, (reset_time - datetime.now()).total_seconds()) 