from typing import Any, Dict, Optional, List
from src.llm.llm import LLM
from src.prompts.prompt_manager import PromptManager
from src.logger import Logger
from src.state import AgentState
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, config: Config):
        """Initialize the base agent with configuration."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        self.config = config
        self.llm = LLM(config)
        self.logger = logger
        self.state = AgentState()
        self.prompt_manager = PromptManager()
        self.base_model = config.model_id
        logger.info(f"Base agent initialized with model: {config.model}")

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Get a specific prompt by name."""
        return self.prompt_manager.get_prompt(prompt_name)

    def format_prompt(self, prompt_template: str, **kwargs) -> str:
        """Format the prompt template with the given arguments."""
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing required prompt argument: {str(e)}")
            raise ValueError(f"Missing required prompt argument: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {str(e)}")
            raise ValueError(f"Error formatting prompt: {str(e)}")

    async def execute(self, prompt: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute the agent's task."""
        raise NotImplementedError("Subclasses must implement execute method")

    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        """Prepare messages for LLM."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        return messages

    def _validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the response from LLM."""
        try:
            if not isinstance(response, dict):
                return False
                
            # Add specific validation based on agent type
            return True
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            return False

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response from LLM."""
        try:
            if not self._validate_response(response):
                raise ValueError("Invalid response format")
                
            return response
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise

    def validate_response(self, response: Any) -> Dict[str, Any]:
        """Validate the response format."""
        raise NotImplementedError("Subclasses must implement validate_response()")

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the validated response."""
        raise NotImplementedError("Subclasses must implement parse_response()")

    async def communicate(self, message: Dict[str, Any], target_agent: str, project_name: str) -> None:
        """Communicate with other agents in the swarm."""
        self.state.add_agent_message(project_name, target_agent, message)

    async def receive_message(self, project_name: str) -> List[Dict[str, Any]]:
        """Receive messages from other agents."""
        return self.state.get_agent_messages(project_name)

    def update_state(self, project_name: str, state: Dict[str, Any]) -> None:
        """Update the agent's state."""
        self.state.update_agent_state(project_name, state)

    def get_state(self, project_name: str) -> Dict[str, Any]:
        """Get the agent's current state."""
        return self.state.get_agent_state(project_name)

    async def process_input(self, input_data: Any, project_name: str) -> Any:
        """Process input data before execution."""
        raise NotImplementedError("Subclasses must implement process_input method")

    async def generate_output(self, processed_data: Any, project_name: str) -> Any:
        """Generate output after execution."""
        raise NotImplementedError("Subclasses must implement generate_output method") 