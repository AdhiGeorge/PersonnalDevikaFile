import json
from src.agents.base_agent import BaseAgent
from src.services.utils import retry_wrapper, validate_responses

class Decision(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, task: str, context: str = "") -> str:
        """Format the decision prompt with the task and context."""
        prompt_template = self.get_prompt("decision")
        if not prompt_template:
            raise ValueError("Decision prompt not found in prompts.yaml")
        return prompt_template

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # First try to parse as JSON
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "decision" not in data or not isinstance(data["decision"], str):
                return False
            if "reasoning" not in data or not isinstance(data["reasoning"], str):
                return False
            return response
        except json.JSONDecodeError:
            # If not JSON, treat as markdown and create a decision structure
            return json.dumps({
                "decision": "answer",
                "reasoning": response,
                "metadata": {
                    "format": "markdown"
                }
            })

    @retry_wrapper
    async def execute(self, task: str, context: str = "", project_name: str = "") -> str:
        """Execute the decision agent."""
        formatted_prompt = self.format_prompt(task, context)
        response = await self.llm.inference(formatted_prompt, project_name)
        validated_response = self.validate_response(response)
        return self.parse_response(validated_response)

    def parse_response(self, response: str) -> dict:
        """Parse the decision agent's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "decision": data.get("decision", "answer"),  # Default to "answer" if not specified
                "reasoning": data.get("reasoning", response),  # Use full response if no reasoning
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing decision response: {str(e)}")
            return {
                "decision": "answer",
                "reasoning": response,
                "metadata": {}
            }