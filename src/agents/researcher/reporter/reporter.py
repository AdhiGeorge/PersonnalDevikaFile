import json
from agents.base_agent import BaseAgent
from services.utils import retry_wrapper, validate_responses

class Reporter(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, task: str, results: dict) -> str:
        """Format the reporter prompt with the task and results."""
        prompt_template = self.get_prompt("reporter")
        if not prompt_template:
            raise ValueError("Reporter prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, task=task, results=results)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # The response should be a valid JSON string
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "report" not in data or not isinstance(data["report"], str):
                return False
            if "summary" not in data or not isinstance(data["summary"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    def execute(self, task: str, results: dict, project_name: str = "") -> str:
        """Execute the reporter agent."""
        formatted_prompt = self.format_prompt(task, results)
        response = self.llm.inference(formatted_prompt, project_name)
        return self.validate_response(response)

    def parse_response(self, response: str) -> dict:
        """Parse the reporter's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "report": data.get("report", ""),
                "summary": data.get("summary", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing reporter response: {str(e)}")
            return {
                "report": "",
                "summary": "I apologize, but I encountered an error while generating the report.",
                "metadata": {}
            }

