from typing import Optional, Dict, Any, List
from .workflow_state import WorkflowState
import logging

logger = logging.getLogger(__name__)

class WorkflowContext:
    """Represents the context of a workflow execution."""
    
    def __init__(self, query: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize workflow context.
        
        Args:
            query: Optional initial query for the workflow
            metadata: Optional metadata for the workflow
        """
        self._query = None
        self._metadata = {}
        self._state = WorkflowState.INITIALIZED
        self._results = {}
        self._errors = []
        self._warnings = []
        
        if query is not None:
            self.query = query
        if metadata is not None:
            self.metadata = metadata
            
    @property
    def query(self) -> Optional[str]:
        """Get the workflow query."""
        return self._query
        
    @query.setter
    def query(self, value: str):
        """Set the workflow query.
        
        Args:
            value: Query string
        """
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        if value.strip() == '':
            raise ValueError("query cannot be empty")
        self._query = value
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the workflow metadata."""
        return self._metadata
        
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        """Set the workflow metadata.
        
        Args:
            value: Metadata dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("metadata must be a dictionary")
        self._metadata = value
        
    @property
    def state(self) -> WorkflowState:
        """Get the workflow state."""
        return self._state
        
    @state.setter
    def state(self, value: WorkflowState):
        """Set the workflow state.
        
        Args:
            value: WorkflowState enum value
        """
        if not isinstance(value, WorkflowState):
            raise ValueError("state must be a WorkflowState enum value")
        self._state = value
        
    @property
    def results(self) -> Dict[str, Any]:
        """Get the workflow results."""
        return self._results
        
    @results.setter
    def results(self, value: Dict[str, Any]):
        """Set the workflow results.
        
        Args:
            value: Results dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("results must be a dictionary")
        self._results = value
        
    @property
    def errors(self) -> List[str]:
        """Get the workflow errors."""
        return self._errors
        
    @errors.setter
    def errors(self, value: List[str]):
        """Set the workflow errors.
        
        Args:
            value: List of error messages
        """
        if not isinstance(value, list):
            raise ValueError("errors must be a list")
        if not all(isinstance(e, str) for e in value):
            raise ValueError("all errors must be strings")
        self._errors = value
        
    @property
    def warnings(self) -> List[str]:
        """Get the workflow warnings."""
        return self._warnings
        
    @warnings.setter
    def warnings(self, value: List[str]):
        """Set the workflow warnings.
        
        Args:
            value: List of warning messages
        """
        if not isinstance(value, list):
            raise ValueError("warnings must be a list")
        if not all(isinstance(w, str) for w in value):
            raise ValueError("all warnings must be strings")
        self._warnings = value
        
    def add_error(self, error: str):
        """Add an error message.
        
        Args:
            error: Error message
        """
        if not isinstance(error, str):
            raise ValueError("error must be a string")
        self._errors.append(error)
        
    def add_warning(self, warning: str):
        """Add a warning message.
        
        Args:
            warning: Warning message
        """
        if not isinstance(warning, str):
            raise ValueError("warning must be a string")
        self._warnings.append(warning)
        
    def update_results(self, key: str, value: Any):
        """Update a result value.
        
        Args:
            key: Result key
            value: Result value
        """
        if not isinstance(key, str):
            raise ValueError("key must be a string")
        self._results[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary.
        
        Returns:
            Dictionary representation of context
        """
        return {
            'query': self._query,
            'metadata': self._metadata,
            'state': self._state.value,
            'results': self._results,
            'errors': self._errors,
            'warnings': self._warnings
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowContext':
        """Create context from dictionary.
        
        Args:
            data: Dictionary containing context data
            
        Returns:
            WorkflowContext instance
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")
            
        context = cls()
        
        if 'query' in data:
            context.query = data['query']
            
        if 'metadata' in data:
            context.metadata = data['metadata']
            
        if 'state' in data:
            context.state = WorkflowState(data['state'])
            
        if 'results' in data:
            context.results = data['results']
            
        if 'errors' in data:
            context.errors = data['errors']
            
        if 'warnings' in data:
            context.warnings = data['warnings']
            
        return context 