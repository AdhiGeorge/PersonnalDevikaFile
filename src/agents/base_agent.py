from typing import Any, Dict, Optional
from src.llm import LLM
from src.prompts.prompt_manager import PromptManager
from src.logger import Logger

class BaseAgent:
    def __init__(self, base_model: str):
        self.llm = LLM(model_id=base_model)
        self.prompt_manager = PromptManager()
        self.logger = Logger()

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Get a specific prompt by name."""
        return self.prompt_manager.get_prompt(prompt_name)

    def format_prompt(self, prompt_template: str, **kwargs) -> str:
        """Format the prompt template with the provided arguments."""
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing required prompt argument: {str(e)}")
            raise ValueError(f"Missing required prompt argument: {str(e)}")

    def execute(self, prompt: str, project_name: str) -> Any:
        """Execute the agent's main functionality."""
        raise NotImplementedError("Subclasses must implement execute method")

    def validate_response(self, response: Any) -> bool:
        """Validate the response from the LLM."""
        raise NotImplementedError("Subclasses must implement validate_response method") 