import yaml
import os
from typing import Dict, Optional

class PromptManager:
    _instance = None
    _prompts: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_prompts()
        return cls._instance

    def _load_prompts(self):
        """Load prompts from the YAML file."""
        try:
            with open("prompts.yaml", "r", encoding="utf-8") as f:
                self._prompts = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError("prompts.yaml file not found. Please ensure it exists in the project root.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing prompts.yaml: {str(e)}")

    def get_prompt(self, prompt_name: str, query: str = None) -> Optional[str]:
        """Get a specific prompt by name and optionally format it with a query."""
        prompt = self._prompts.get(prompt_name)
        if prompt and query:
            return prompt.format(query=query)
        return prompt

    def get_all_prompts(self) -> Dict[str, str]:
        """Get all prompts."""
        return self._prompts.copy()

    def reload_prompts(self):
        """Reload prompts from the YAML file."""
        self._load_prompts() 