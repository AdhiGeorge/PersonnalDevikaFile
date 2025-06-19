"""State management for agents."""

import json
import os
from datetime import datetime
from typing import Optional, Any, Dict, List
from config.config import Config
from database.database import Database
import logging
import asyncio

logger = logging.getLogger(__name__)

class State:
    """State management for agents."""
    
    def __init__(self):
        """Initialize the state."""
        self.logger = logging.getLogger(__name__)
        self._state: Dict[str, Any] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the state."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                self._state = {
                    'current_step': None,
                    'completed_steps': [],
                    'pending_steps': [],
                    'results': {},
                    'errors': [],
                    'metadata': {}
                }
                self._initialized = True
                self.logger.info("State initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize state: {str(e)}")
                raise ValueError(f"Failed to initialize state: {str(e)}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state.
        
        Args:
            key: The key to get
            default: The default value if key is not found
            
        Returns:
            The value for the key, or default if not found
        """
        if not self._initialized:
            raise RuntimeError("State not initialized. Call initialize() first.")
        return self._state.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in the state.
        
        Args:
            key: The key to set
            value: The value to set
        """
        if not self._initialized:
            raise RuntimeError("State not initialized. Call initialize() first.")
        self._state[key] = value
        
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple values in the state.
        
        Args:
            data: Dictionary of key-value pairs to update
        """
        if not self._initialized:
            raise RuntimeError("State not initialized. Call initialize() first.")
        self._state.update(data)
        
    def clear(self) -> None:
        """Clear the state."""
        if not self._initialized:
            raise RuntimeError("State not initialized. Call initialize() first.")
        self._state.clear()
        self.initialize()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary.
        
        Returns:
            Dictionary representation of state
        """
        if not self._initialized:
            raise RuntimeError("State not initialized. Call initialize() first.")
        return self._state.copy()
        
    def to_json(self) -> str:
        """Convert state to JSON string.
        
        Returns:
            JSON string representation of state
        """
        if not self._initialized:
            raise RuntimeError("State not initialized. Call initialize() first.")
        return json.dumps(self._state)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'State':
        """Create state from JSON string.
        
        Args:
            json_str: JSON string representation of state
            
        Returns:
            New State instance
        """
        state = cls()
        state._state = json.loads(json_str)
        state._initialized = True
        return state
        
    def is_initialized(self) -> bool:
        """Check if state is initialized.
        
        Returns:
            True if state is initialized, False otherwise
        """
        return self._initialized

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return
            
        try:
            # Reset state
            self._initialized = False
            self._state.clear()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up: {str(e)}")
            raise ValueError(f"Failed to cleanup: {str(e)}")

class AgentState:
    """Agent state management."""
    
    def __init__(self, config: Config):
        """Initialize agent state with configuration."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        self.config = config
        self.db = None  # Initialize Database lazily
        self._storage = {}  # In-memory storage for temporary state
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
    @property
    def storage(self) -> Dict[str, Any]:
        """Get the storage dictionary."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage
        
    async def initialize(self):
        """Initialize async components."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                # Initialize storage first
                self._storage = {}
                
                # Initialize database
                try:
                    self.db = Database(self.config)
                    await self.db.initialize()
                    logger.info("Database initialized successfully")
                except Exception as e:
                    logger.warning(f"Database initialization warning: {str(e)}")
                    # Continue without database - we'll use in-memory storage only
                    self.db = None
                
                # Simple smoke test for state creation
                try:
                    state = await self.new_state(skip_init_check=True)
                    logger.info("State creation test successful")
                except Exception as e:
                    logger.error(f"State creation test failed: {str(e)}")
                    raise ValueError(f"State initialization failed: {str(e)}")
                
                self._initialized = True
                logger.info("Agent state initialized successfully")
                
            except Exception as e:
                # Clean up on failure
                self._initialized = False
                self._storage = {}
                if hasattr(self, 'db') and self.db:
                    try:
                        await self.db.cleanup()
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup: {str(cleanup_error)}")
                self.db = None
                logger.error(f"Failed to initialize AgentState: {str(e)}")
                raise ValueError(f"AgentState initialization failed: {str(e)}")
        
    async def new_state(self, skip_init_check: bool = False) -> State:
        """Create a new state object.
        
        Args:
            skip_init_check: If True, skips the initialization check. Used during initialization testing.
        """
        if not skip_init_check and not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        state = State()
        await state.initialize()
        return state
        
    async def create_state(self, project: str) -> None:
        """Create a new state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            state = await self.new_state()
            self._storage[project] = state
            await self.db.create_state(project, state.to_dict())
        except Exception as e:
            logger.error(f"Failed to create state for project '{project}': {str(e)}")
            raise ValueError(f"Failed to create state: {str(e)}")
            
    async def get_current_state(self, project: str) -> Optional[State]:
        """Get the current state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project)
        
    async def update_state(self, project: str, state: Dict[str, Any]) -> None:
        """Update the state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                await self.create_state(project)
            self._storage[project].update(state)
            await self.db.update_state(project, state)
        except Exception as e:
            logger.error(f"Failed to update state for project '{project}': {str(e)}")
            raise ValueError(f"Failed to update state: {str(e)}")
            
    async def delete_state(self, project: str) -> None:
        """Delete the state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project in self._storage:
                del self._storage[project]
            await self.db.delete_state(project)
        except Exception as e:
            logger.error(f"Failed to delete state for project '{project}': {str(e)}")
            raise ValueError(f"Failed to delete state: {str(e)}")
            
    async def add_to_current_state(self, project: str, state: Dict[str, Any]) -> None:
        """Add to the current state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                await self.create_state(project)
            current_state = self._storage[project].to_dict()
            current_state.update(state)
            await self.update_state(project, current_state)
        except Exception as e:
            logger.error(f"Failed to add to state for project '{project}': {str(e)}")
            raise ValueError(f"Failed to add to state: {str(e)}")
            
    async def update_latest_state(self, project: str, state: Dict[str, Any]) -> None:
        """Update the latest state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                await self.create_state(project)
            current_state = self._storage[project].to_dict()
            current_state.update(state)
            await self.update_state(project, current_state)
        except Exception as e:
            logger.error(f"Failed to update latest state for project '{project}': {str(e)}")
            raise ValueError(f"Failed to update latest state: {str(e)}")
            
    async def get_latest_state(self, project: str) -> Optional[Dict[str, Any]]:
        """Get the latest state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                return None
            return self._storage[project].to_dict()
        except Exception as e:
            logger.error(f"Failed to get latest state for project '{project}': {str(e)}")
            raise ValueError(f"Failed to get latest state: {str(e)}")
            
    async def set_agent_active(self, project: str, is_active: bool) -> None:
        """Set agent active status for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                await self.create_state(project)
            self._storage[project].set('is_active', is_active)
            await self.db.update_state(project, {'is_active': is_active})
        except Exception as e:
            logger.error(f"Failed to set agent active status for project '{project}': {str(e)}")
            raise ValueError(f"Failed to set agent active status: {str(e)}")
            
    async def is_agent_active(self, project: str) -> bool:
        """Check if agent is active for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project, {}).get('is_active', False)
        
    async def set_agent_completed(self, project: str, is_completed: bool) -> None:
        """Set agent completed status for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                await self.create_state(project)
            self._storage[project].set('is_completed', is_completed)
            await self.db.update_state(project, {'is_completed': is_completed})
        except Exception as e:
            logger.error(f"Failed to set agent completed status for project '{project}': {str(e)}")
            raise ValueError(f"Failed to set agent completed status: {str(e)}")
            
    async def is_agent_completed(self, project: str) -> bool:
        """Check if agent is completed for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project, {}).get('is_completed', False)
        
    async def update_token_usage(self, project: str, token_usage: int) -> None:
        """Update token usage for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                await self.create_state(project)
            current_usage = self._storage[project].get('token_usage', 0)
            new_usage = current_usage + token_usage
            self._storage[project].set('token_usage', new_usage)
            await self.db.update_state(project, {'token_usage': new_usage})
        except Exception as e:
            logger.error(f"Failed to update token usage for project '{project}': {str(e)}")
            raise ValueError(f"Failed to update token usage: {str(e)}")
            
    async def get_latest_token_usage(self, project: str) -> int:
        """Get latest token usage for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project, {}).get('token_usage', 0)
        
    async def add_step_result(self, project: str, step_id: int, result: Any) -> None:
        """Add a step result for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                await self.create_state(project)
            results = self._storage[project].get('results', {})
            results[str(step_id)] = result
            self._storage[project].set('results', results)
            await self.db.update_state(project, {'results': results})
        except Exception as e:
            logger.error(f"Failed to add step result for project '{project}': {str(e)}")
            raise ValueError(f"Failed to add step result: {str(e)}")
            
    async def get_step_result(self, project: str, step_id: int) -> Optional[Any]:
        """Get a step result for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project, {}).get('results', {}).get(str(step_id))
        
    async def is_step_completed(self, project: str, step_id: int) -> bool:
        """Check if a step is completed for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return str(step_id) in self._storage.get(project, {}).get('results', {})
        
    async def get_all_step_results(self, project: str) -> Dict[str, Any]:
        """Get all step results for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project, {}).get('results', {})
        
    def add_agent_message(self, project: str, target_agent: str, message: Dict[str, Any]) -> None:
        """Add a message for an agent."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                raise ValueError(f"Project '{project}' not found")
                
            messages = self._storage[project].get('messages', [])
            messages.append({
                'target_agent': target_agent,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            self._storage[project].set('messages', messages)
        except Exception as e:
            logger.error(f"Failed to add agent message for project '{project}': {str(e)}")
            raise ValueError(f"Failed to add agent message: {str(e)}")
            
    def get_agent_messages(self, project: str) -> List[Dict[str, Any]]:
        """Get all messages for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project, {}).get('messages', [])
        
    def update_agent_state(self, project: str, state: Dict[str, Any]) -> None:
        """Update agent state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
            
        try:
            if project not in self._storage:
                raise ValueError(f"Project '{project}' not found")
                
            current_state = self._storage[project].to_dict()
            current_state.update(state)
            self._storage[project].update(current_state)
        except Exception as e:
            logger.error(f"Failed to update agent state for project '{project}': {str(e)}")
            raise ValueError(f"Failed to update agent state: {str(e)}")
            
    def get_agent_state(self, project: str) -> Dict[str, Any]:
        """Get agent state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent state not initialized. Call initialize() first.")
        return self._storage.get(project, {}).to_dict()
        
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return
            
        try:
            # Cleanup database
            if self.db:
                await self.db.cleanup()
                
            # Reset state
            self._initialized = False
            self._storage.clear()
            
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
            raise ValueError(f"Failed to cleanup: {str(e)}")

async def main():
    """Main function for testing."""
    try:
        # Create config
        config = Config()
        await config.initialize()
        
        # Create agent state
        agent_state = AgentState(config)
        await agent_state.initialize()
        
        # Test state creation
        test_project = "test_project"
        await agent_state.create_state(test_project)
        
        # Test state update
        test_state = {"test_key": "test_value"}
        await agent_state.update_state(test_project, test_state)
        
        # Test state retrieval
        current_state = await agent_state.get_current_state(test_project)
        print(f"Current state: {current_state.to_dict()}")
        
        # Test state deletion
        await agent_state.delete_state(test_project)
        
        # Cleanup
        await agent_state.cleanup()
        await config.cleanup()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())