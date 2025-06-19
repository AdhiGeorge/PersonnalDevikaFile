import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from config.config import Config
from llm.llm import LLM
from state import AgentState
from utils.logger import Logger
from prompts.prompt_manager import PromptManager
from utils.token_tracker import TokenTracker
from .storage import Storage

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class with core functionality."""
    
    def __init__(self, config: Config, model: str = "gpt-4"):
        """Initialize the base agent.
        
        Args:
            config: Configuration instance
            model: Model name to use
        """
        # Initialize logger first
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize other attributes
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._state = None
        self._llm = None
        self._prompt_manager = None
        self._token_tracker = None
        self.config = config
        self.model = model
        self._components = {}
        
    async def initialize(self) -> None:
        """Initialize the base agent."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                self._logger.info("Starting base agent initialization...")
                
                # Initialize state first since other components may depend on it
                self._logger.info("Initializing state...")
                self._state = AgentState(self.config)
                await self._state.initialize()
                self._components['state'] = True
                self._logger.info("Successfully initialized state")
                
                # Initialize prompt manager
                self._logger.info("Initializing prompt_manager...")
                self._prompt_manager = PromptManager(self.config)
                if hasattr(self._prompt_manager, 'initialize') and callable(self._prompt_manager.initialize):
                    if asyncio.iscoroutinefunction(self._prompt_manager.initialize):
                        await self._prompt_manager.initialize()
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, self._prompt_manager.initialize)
                self._components['prompt_manager'] = True
                self._logger.info("Successfully initialized prompt_manager")
                
                # Initialize LLM
                self._logger.info("Initializing llm...")
                self._llm = LLM(self.config, self.model)
                if hasattr(self._llm, 'initialize') and callable(self._llm.initialize):
                    if asyncio.iscoroutinefunction(self._llm.initialize):
                        await self._llm.initialize()
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, self._llm.initialize)
                self._components['llm'] = True
                self._logger.info(f"Successfully initialized llm with model: {self._llm.model}")
                
                # Initialize token tracker
                self._logger.info("Initializing token_tracker...")
                self._token_tracker = TokenTracker(self.config)
                if hasattr(self._token_tracker, 'initialize') and callable(self._token_tracker.initialize):
                    if asyncio.iscoroutinefunction(self._token_tracker.initialize):
                        await self._token_tracker.initialize()
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, self._token_tracker.initialize)
                self._components['token_tracker'] = True
                self._logger.info("Successfully initialized token_tracker")
                
                self._initialized = True
                self._logger.info("Base agent initialized successfully")
                
            except Exception as e:
                self._initialized = False
                self._components = {}
                import traceback
                error_msg = f"Error initializing base agent: {str(e)}\n{traceback.format_exc()}"
                self._logger.error(error_msg)
                raise ValueError(f"Failed to initialize agent: {str(e)}")
            
    @property
    def logger(self) -> logging.Logger:
        """Get the agent's logger.
        
        Returns:
            The agent's logger
        """
        return self._logger
            
    @property
    def state(self) -> AgentState:
        """Get the agent's state.
        
        Returns:
            The agent's state
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._state
        
    @property
    def llm(self) -> LLM:
        """Get the agent's LLM.
        
        Returns:
            The agent's LLM
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._llm
        
    @property
    def prompt_manager(self) -> PromptManager:
        """Get the agent's prompt manager.
        
        Returns:
            The agent's prompt manager
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._prompt_manager
        
    @property
    def token_tracker(self) -> TokenTracker:
        """Get the agent's token tracker.
        
        Returns:
            The agent's token tracker
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._token_tracker
        
    def is_initialized(self) -> bool:
        """Check if the agent is initialized.
        
        Returns:
            True if the agent is initialized, False otherwise
        """
        return self._initialized

    @property
    def knowledge_base(self):
        """Get the knowledge base instance."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._knowledge_base

    @knowledge_base.setter
    def knowledge_base(self, value):
        """Set the knowledge base instance."""
        self._knowledge_base = value

    @property
    def file_manager(self):
        """Get the file manager instance."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._file_manager

    @property
    def storage(self) -> 'Storage':
        """Get the storage instance.
        
        Returns:
            Storage: The storage instance
            
        Note:
            The storage is always initialized when the agent is created.
            This property provides direct access to the storage instance.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._storage

    async def _initialize_component(self, component_name: str, component, init_method: str = 'initialize') -> bool:
        """Initialize a single component with error handling."""
        try:
            if component is None:
                self._logger.warning(f"Component {component_name} is None, skipping initialization")
                return False
                
            if hasattr(component, init_method):
                init_method_callable = getattr(component, init_method)
                if callable(init_method_callable):
                    self._logger.debug(f"Calling {init_method} on {component_name}")
                    if asyncio.iscoroutinefunction(init_method_callable):
                        await init_method_callable()
                    else:
                        init_method_callable()
                    self._logger.info(f"Successfully initialized {component_name}")
                    self._components[component_name] = True
                    return True
                else:
                    self._logger.warning(f"{init_method} is not callable on {component_name}")
            else:
                self._logger.debug(f"No {init_method} method found on {component_name}")
                
            return True  # Not all components need initialization
            
        except Exception as e:
            self._logger.error(f"Error initializing {component_name}: {str(e)}")
            raise ValueError(f"Failed to initialize {component_name}: {str(e)}")

    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute a query.
        
        Args:
            query: The query to execute
            
        Returns:
            The execution result
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        try:
            # Prepare messages
            messages = await self._prepare_messages(query)
            
            # Generate response
            response = await self.llm.chat_completion(messages)
            
            # Validate response
            if not await self._validate_response(response):
                raise ValueError("Invalid response format")
                
            # Parse response
            result = await self._parse_response(response)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error executing query: {str(e)}")
            raise ValueError(f"Failed to execute query: {str(e)}")

    async def cleanup(self):
        """Cleanup resources."""
        if not self._initialized:
            return
            
        try:
            # Cleanup components
            for component_name, component in self._components.items():
                if hasattr(component, 'cleanup'):
                    cleanup_method = getattr(component, 'cleanup')
                    if callable(cleanup_method):
                        if asyncio.iscoroutinefunction(cleanup_method):
                            await cleanup_method()
                        else:
                            cleanup_method()
                            
            # Reset state
            self._initialized = False
            self._components.clear()
            
        except Exception as e:
            self._logger.error(f"Error cleaning up: {str(e)}")
            raise ValueError(f"Failed to cleanup: {str(e)}")

    def _get_token_tracker(self) -> TokenTracker:
        """Get the token tracker instance."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._token_tracker

    def get_prompt(self, task: Dict[str, Any]) -> str:
        """Get a prompt for a task."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.prompt_manager.get_prompt(task)

    async def execute_with_context(self, task: Dict[str, Any], context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute a task with context."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return await self.llm.execute_with_context(task, context)

    def format_prompt(self, prompt: str, **kwargs) -> str:
        """Format a prompt with the given parameters."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.prompt_manager.format_prompt(prompt, **kwargs)

    async def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        """Prepare messages for chat completion."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            return messages
            
        except Exception as e:
            self._logger.error(f"Error preparing messages: {str(e)}")
            raise ValueError(f"Failed to prepare messages: {str(e)}")

    async def _validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate a response."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        try:
            # Check if response has required fields
            required_fields = ["role", "content"]
            if not all(field in response for field in required_fields):
                return False
                
            # Check if role is valid
            valid_roles = ["system", "user", "assistant"]
            if response["role"] not in valid_roles:
                return False
                
            # Check if content is valid
            if not isinstance(response["content"], str):
                return False
                
            return True
            
        except Exception as e:
            self._logger.error(f"Error validating response: {str(e)}")
            return False

    async def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a response."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        try:
            # Validate response
            if not await self._validate_response(response):
                raise ValueError("Invalid response format")
                
            # Parse response
            result = {
                "role": response["role"],
                "content": response["content"]
            }
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error parsing response: {str(e)}")
            raise ValueError(f"Failed to parse response: {str(e)}")

    async def validate_response(self, response: Any) -> Dict[str, Any]:
        """Validate a response."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return await self._validate_response(response)

    async def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a response."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return await self._parse_response(response)

    async def communicate(self, message: Dict[str, Any], target_agent: str, project_name: str) -> None:
        """Communicate with another agent."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        await self.storage.store_message(message, target_agent, project_name)

    async def receive_message(self, project_name: str) -> List[Dict[str, Any]]:
        """Receive messages for a project."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return await self.storage.get_messages(project_name)

    def update_state(self, project_name: str, state: Dict[str, Any]) -> None:
        """Update the state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        self.storage.update_state(project_name, state)

    def get_state(self, project_name: str) -> Dict[str, Any]:
        """Get the state for a project."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.storage.get_state(project_name)

    async def process_input(self, input_data: Any, project_name: str) -> Any:
        """Process input data."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return input_data

    async def generate_output(self, processed_data: Any, project_name: str) -> Any:
        """Generate output data."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return processed_data

    def _track_usage(self, model: str, input_tokens: int, output_tokens: int, project_name: Optional[str] = None) -> None:
        """Track token usage."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        self.token_tracker.track_usage(model, input_tokens, output_tokens, project_name)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.token_tracker.get_usage_summary()

    def _ensure_initialized(self):
        """Ensure the agent is initialized."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.") 