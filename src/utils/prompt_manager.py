import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompts for different agents."""
    
    def __init__(self):
        """Initialize prompt manager."""
        try:
            self.prompts = {}
            self.prompt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "prompts.yaml")
            self._load_prompts()
            logger.info("Prompt manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize prompt manager: {str(e)}")
            raise ValueError(f"Prompt manager initialization failed: {str(e)}")

    def _load_prompts(self):
        """Load prompts from YAML file."""
        try:
            if os.path.exists(self.prompt_file):
                with open(self.prompt_file, 'r') as f:
                    self.prompts = yaml.safe_load(f)
            else:
                logger.warning(f"Prompt file not found: {self.prompt_file}")
                self.prompts = {}
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            self.prompts = {}

    def get_prompt(self, name: str) -> Optional[str]:
        """Get a prompt by name."""
        try:
            if not isinstance(name, str):
                raise ValueError("name must be a string")
            return self.prompts.get(name)
        except Exception as e:
            logger.error(f"Error getting prompt: {str(e)}")
            return None

    def format_prompt(self, prompt: str, **kwargs) -> str:
        """Format a prompt with variables."""
        try:
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            return prompt.format(**kwargs)
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return prompt 