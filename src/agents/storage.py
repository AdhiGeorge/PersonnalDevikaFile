"""
Storage management for agents.
"""
import json
import logging
import os
from typing import Dict, Any, Optional, List

class Storage:
    """Simple in-memory storage with persistence support."""
    
    def __init__(self, config=None):
        """Initialize the storage.
        
        Args:
            config: Optional configuration object
        """
        self._data = {}
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._initialized = False
        self._storage_file = None
        self._messages = {}  # Store messages by project
        self._states = {}    # Store states by project
        
        if config:
            # Get storage directory from config
            storage_dir = config.get('storage', 'directory')
            if not storage_dir:
                storage_dir = os.path.join(os.getcwd(), 'data', 'storage')
                
            # Ensure storage directory exists
            os.makedirs(storage_dir, exist_ok=True)
            
            self._storage_file = os.path.join(
                storage_dir,
                'agent_storage.json'
            )
    
    async def initialize(self) -> bool:
        """Initialize the storage.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Create storage directory if needed
            if self._storage_file:
                os.makedirs(os.path.dirname(self._storage_file), exist_ok=True)
                
                # Load existing data if available
                if os.path.exists(self._storage_file):
                    try:
                        with open(self._storage_file, 'r') as f:
                            data = json.load(f)
                            self._data = data.get('data', {})
                            self._messages = data.get('messages', {})
                            self._states = data.get('states', {})
                    except Exception as e:
                        self._logger.error(f"Error loading storage file: {e}")
                        self._data = {}
                        self._messages = {}
                        self._states = {}
            
            self._initialized = True
            self._logger.info("Storage initialized")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize storage: {e}")
            self._initialized = False
            return False
    
    def __getitem__(self, key: str) -> Any:
        """Get an item from storage."""
        if not self._initialized:
            raise RuntimeError("Storage not initialized")
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        return self._data.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item in storage."""
        if not self._initialized:
            raise RuntimeError("Storage not initialized")
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        self._data[key] = value
        self._save()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from storage with a default value."""
        if not self._initialized:
            return default
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set an item in storage."""
        self.__setitem__(key, value)
    
    def delete(self, key: str) -> None:
        """Delete an item from storage."""
        if not self._initialized:
            return
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        self._data.pop(key, None)
        self._save()
    
    def clear(self) -> None:
        """Clear all data from storage."""
        if not self._initialized:
            return
        self._data.clear()
        self._messages.clear()
        self._states.clear()
        self._save()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert storage to dictionary."""
        if not self._initialized:
            return {}
        return {
            'data': dict(self._data),
            'messages': dict(self._messages),
            'states': dict(self._states)
        }
    
    async def store_message(self, message: Dict[str, Any], target_agent: str, project_name: str) -> None:
        """Store a message for a project.
        
        Args:
            message: The message to store
            target_agent: The target agent
            project_name: The project name
        """
        if not self._initialized:
            raise RuntimeError("Storage not initialized")
            
        if project_name not in self._messages:
            self._messages[project_name] = []
            
        self._messages[project_name].append({
            'message': message,
            'target_agent': target_agent,
            'timestamp': self._get_timestamp()
        })
        self._save()
    
    async def get_messages(self, project_name: str) -> List[Dict[str, Any]]:
        """Get messages for a project.
        
        Args:
            project_name: The project name
            
        Returns:
            List of messages
        """
        if not self._initialized:
            return []
            
        return self._messages.get(project_name, [])
    
    def update_state(self, project_name: str, state: Dict[str, Any]) -> None:
        """Update state for a project.
        
        Args:
            project_name: The project name
            state: The state to store
        """
        if not self._initialized:
            raise RuntimeError("Storage not initialized")
            
        self._states[project_name] = {
            'state': state,
            'timestamp': self._get_timestamp()
        }
        self._save()
    
    def get_state(self, project_name: str) -> Dict[str, Any]:
        """Get state for a project.
        
        Args:
            project_name: The project name
            
        Returns:
            The project state
        """
        if not self._initialized:
            return {}
            
        state_data = self._states.get(project_name, {})
        return state_data.get('state', {})
    
    def _save(self) -> None:
        """Save data to file."""
        if not self._storage_file:
            return
            
        try:
            with open(self._storage_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            self._logger.error(f"Error saving storage file: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp.
        
        Returns:
            Current timestamp as string
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def cleanup(self) -> None:
        """Cleanup storage resources."""
        if not self._initialized:
            return
            
        try:
            self._save()
            self._initialized = False
            self._logger.info("Storage cleanup completed")
        except Exception as e:
            self._logger.error(f"Error during storage cleanup: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if storage is initialized."""
        return self._initialized
