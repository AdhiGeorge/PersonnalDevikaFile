import json
from src.agents.base_agent import BaseAgent
from src.services.utils import retry_wrapper, validate_responses
from src.config import Config

class Action(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
        config = Config()
        self.project_dir = config.get_projects_dir()

    def format_prompt(self, conversation: str) -> str:
        """Format the action prompt with the conversation."""
        prompt_template = self.get_prompt("action")
        if not prompt_template:
            raise ValueError("Action prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, conversation=conversation)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        if "response" not in response and "action" not in response:
            return False
        return response["response"], response["action"]

    @retry_wrapper
    def execute(self, conversation: list, project_name: str) -> str:
        """Execute the action agent."""
        prompt = self.format_prompt(conversation)
        response = self.llm.inference(prompt, project_name)
        return self.validate_response(response)
