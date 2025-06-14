import json
from src.agents.base_agent import BaseAgent
from src.services.utils import retry_wrapper, validate_responses

class InternalMonologue(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, current_prompt: str) -> str:
        """Format the internal monologue prompt with the current prompt."""
        prompt_template = self.get_prompt("internal_monologue")
        if not prompt_template:
            raise ValueError("Internal monologue prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, current_prompt=current_prompt)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            response_json = json.loads(response)
            if "internal_monologue" not in response_json:
                return False
            return response_json["internal_monologue"]
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    def execute(self, current_prompt: str, project_name: str) -> str:
        """Execute the internal monologue agent."""
        prompt = self.format_prompt(current_prompt)
        response = self.llm.inference(prompt, project_name)
        return self.validate_response(response)

