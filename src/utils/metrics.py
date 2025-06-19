import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentMetrics:
    """Class to track agent metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {
            'start_time': None,
            'end_time': None,
            'total_tokens': 0,
            'api_calls': 0,
            'errors': 0,
            'warnings': 0,
            'steps_completed': 0,
            'total_steps': 0
        }
        
    def start_execution(self):
        """Start tracking execution metrics."""
        self._metrics['start_time'] = datetime.now().isoformat()
        
    def end_execution(self):
        """End tracking execution metrics."""
        self._metrics['end_time'] = datetime.now().isoformat()
        
    def add_tokens(self, count: int):
        """Add tokens to the total count."""
        if not isinstance(count, int) or count < 0:
            raise ValueError("Token count must be a non-negative integer")
        self._metrics['total_tokens'] += count
        
    def increment_api_calls(self):
        """Increment the API call counter."""
        self._metrics['api_calls'] += 1
        
    def add_error(self):
        """Increment the error counter."""
        self._metrics['errors'] += 1
        
    def add_warning(self):
        """Increment the warning counter."""
        self._metrics['warnings'] += 1
        
    def set_total_steps(self, count: int):
        """Set the total number of steps."""
        if not isinstance(count, int) or count < 0:
            raise ValueError("Step count must be a non-negative integer")
        self._metrics['total_steps'] = count
        
    def increment_steps_completed(self):
        """Increment the completed steps counter."""
        self._metrics['steps_completed'] += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self._metrics.copy()
        
    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if not self._metrics['start_time'] or not self._metrics['end_time']:
            return None
            
        start = datetime.fromisoformat(self._metrics['start_time'])
        end = datetime.fromisoformat(self._metrics['end_time'])
        return (end - start).total_seconds()
        
    def get_completion_percentage(self) -> float:
        """Get percentage of steps completed."""
        if self._metrics['total_steps'] == 0:
            return 0.0
        return (self._metrics['steps_completed'] / self._metrics['total_steps']) * 100 