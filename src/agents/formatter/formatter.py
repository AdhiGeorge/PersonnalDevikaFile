import json
from src.agents.base_agent import BaseAgent
from src.services.utils import retry_wrapper, validate_responses

class Formatter(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, code: str, language: str = "python") -> str:
        """Format the formatter prompt with the code and language."""
        prompt_template = self.get_prompt("formatter")
        if not prompt_template:
            raise ValueError("Formatter prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, code=code, language=language)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # The response should be a valid JSON string
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "formatted_code" not in data or not isinstance(data["formatted_code"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    def execute(self, code: str, language: str = "python", project_name: str = "") -> str:
        """Execute the formatter agent."""
        formatted_prompt = self.format_prompt(code, language)
        response = self.llm.inference(formatted_prompt, project_name)
        return self.validate_response(response)

    def parse_response(self, response: str) -> dict:
        """Parse the formatter's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "formatted_code": data.get("formatted_code", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing formatter response: {str(e)}")
            return {
                "formatted_code": "",
                "metadata": {}
            }