from enum import Enum
from typing import List, Optional

class WorkflowState(Enum):
    """Represents the possible states of a workflow."""
    
    INITIALIZED = "initialized"
    RUNNING = "running"
    PLANNING = "planning"
    RESEARCHING = "researching"
    GENERATING = "generating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    
    @classmethod
    def get_all_states(cls) -> List['WorkflowState']:
        """Get all possible workflow states.
        
        Returns:
            List of all workflow states
        """
        return list(cls)
        
    @classmethod
    def get_active_states(cls) -> List['WorkflowState']:
        """Get all active workflow states.
        
        Returns:
            List of active workflow states
        """
        return [
            cls.RUNNING,
            cls.PLANNING,
            cls.RESEARCHING,
            cls.GENERATING,
            cls.EXECUTING
        ]
        
    @classmethod
    def get_terminal_states(cls) -> List['WorkflowState']:
        """Get all terminal workflow states.
        
        Returns:
            List of terminal workflow states
        """
        return [
            cls.COMPLETED,
            cls.FAILED,
            cls.ERROR
        ]
        
    @classmethod
    def get_error_states(cls) -> List['WorkflowState']:
        """Get all error workflow states.
        
        Returns:
            List of error workflow states
        """
        return [
            cls.FAILED,
            cls.ERROR
        ]
        
    def is_active(self) -> bool:
        """Check if the state is active.
        
        Returns:
            True if the state is active, False otherwise
        """
        return self in self.get_active_states()
        
    def is_terminal(self) -> bool:
        """Check if the state is terminal.
        
        Returns:
            True if the state is terminal, False otherwise
        """
        return self in self.get_terminal_states()
        
    def is_error(self) -> bool:
        """Check if the state is an error state.
        
        Returns:
            True if the state is an error state, False otherwise
        """
        return self in self.get_error_states()
        
    def can_transition_to(self, new_state: 'WorkflowState') -> bool:
        """Check if transition to new state is valid.
        
        Args:
            new_state: State to transition to
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Can't transition from terminal states
        if self.is_terminal():
            return False
            
        # Can't transition to INITIALIZED
        if new_state == WorkflowState.INITIALIZED:
            return False
            
        # Can transition to any state from ERROR
        if self == WorkflowState.ERROR:
            return True
            
        # Can transition to any state from FAILED
        if self == WorkflowState.FAILED:
            return True
            
        # Can transition to any state from COMPLETED
        if self == WorkflowState.COMPLETED:
            return True
            
        # Can transition to any state from RUNNING
        if self == WorkflowState.RUNNING:
            return True
            
        # Can transition to any state from PLANNING
        if self == WorkflowState.PLANNING:
            return True
            
        # Can transition to any state from RESEARCHING
        if self == WorkflowState.RESEARCHING:
            return True
            
        # Can transition to any state from GENERATING
        if self == WorkflowState.GENERATING:
            return True
            
        # Can transition to any state from EXECUTING
        if self == WorkflowState.EXECUTING:
            return True
            
        return False
        
    def get_next_valid_states(self) -> List['WorkflowState']:
        """Get all valid next states.
        
        Returns:
            List of valid next states
        """
        return [
            state for state in self.get_all_states()
            if self.can_transition_to(state)
        ]
        
    def get_description(self) -> str:
        """Get a description of the state.
        
        Returns:
            Description of the state
        """
        descriptions = {
            WorkflowState.INITIALIZED: "Workflow has been initialized but not started",
            WorkflowState.RUNNING: "Workflow is currently running",
            WorkflowState.PLANNING: "Workflow is in the planning phase",
            WorkflowState.RESEARCHING: "Workflow is in the research phase",
            WorkflowState.GENERATING: "Workflow is in the code generation phase",
            WorkflowState.EXECUTING: "Workflow is in the execution phase",
            WorkflowState.COMPLETED: "Workflow has completed successfully",
            WorkflowState.FAILED: "Workflow has failed",
            WorkflowState.ERROR: "Workflow has encountered an error"
        }
        return descriptions.get(self, "Unknown state") 