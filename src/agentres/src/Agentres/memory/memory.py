from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib
from Agentres.database.database import Database
from Agentres.state import AgentState
from Agentres.project import ProjectManager

class Memory:
    def __init__(self):
        self.db = Database()
        self.state = AgentState()
        self.project_manager = ProjectManager()
        self.current_conversation_id = None
        self.current_project = None

    def initialize_conversation(self, query: str, project_name: str) -> int:
        """Initialize a new conversation and project."""
        # Create project if it doesn't exist
        self.project_manager.create_project(project_name)
        self.current_project = project_name
        
        # Create conversation
        conversation_id = self.db.create_conversation(query)
        self.current_conversation_id = conversation_id
        
        # Initialize state
        self.state.create_state(project_name)
        
        # Store initial query
        self.add_message("user", query)
        
        return conversation_id

    def add_message(self, role: str, content: str):
        """Add a message to the current conversation."""
        if not self.current_conversation_id:
            raise ValueError("No active conversation")
            
        # Add to database
        self.db.add_message(self.current_conversation_id, role, content)
        
        # Add to project manager
        if role == "user":
            self.project_manager.add_message_from_user(self.current_project, content)
        else:
            self.project_manager.add_message_from_agent(self.current_project, content)
            
        # Update state
        state_update = {
            "internal_monologue": content if role == "agent" else "",
            "message": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.state.add_to_current_state(self.current_project, state_update)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the complete conversation history."""
        if not self.current_conversation_id:
            return []
        return self.db.get_conversation_history(self.current_conversation_id)

    def store_knowledge(self, text: str, metadata: Dict[str, Any]):
        """Store knowledge in the vector database."""
        self.db.store_knowledge(text, metadata)

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base."""
        return self.db.search_knowledge(query, limit)

    def check_query_cache(self, query: str) -> Optional[int]:
        """Check if a query exists in cache."""
        return self.db.check_query_cache(query)

    def cache_query(self, query: str, embedding: List[float]):
        """Cache a query with its embedding."""
        if not self.current_conversation_id:
            raise ValueError("No active conversation")
        self.db.cache_query(query, self.current_conversation_id, embedding)

    def get_project_state(self) -> Dict[str, Any]:
        """Get the current project state."""
        if not self.current_project:
            return {}
        return self.state.get_latest_state(self.current_project)

    def update_project_state(self, state_update: Dict[str, Any]):
        """Update the project state."""
        if not self.current_project:
            raise ValueError("No active project")
        self.state.update_latest_state(self.current_project, state_update)

    def close(self):
        """Close all connections."""
        self.db.close() 