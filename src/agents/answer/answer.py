import json
from src.agents.base_agent import BaseAgent
from src.services.utils import retry_wrapper, validate_responses

class Answer(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, question: str, context: str = "") -> str:
        """Format the answer prompt with the question and context."""
        prompt_template = self.get_prompt("answer")
        if not prompt_template:
            raise ValueError("Answer prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, question=question, context=context)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # Allow decorator to have already parsed JSON into dict
            if isinstance(response, dict):
                data = response
            else:
                data = json.loads(response)
            if not isinstance(data, dict):
                return False

            # If proper keys present, keep them. Otherwise, treat generic "response" key as answer.
            if "answer" not in data:
                # Fallback: whole text becomes answer, explanation empty
                if "response" in data and isinstance(data["response"], str):
                    data = {
                        "answer": data["response"],
                        "explanation": "",
                        "metadata": {}
                    }
                else:
                    return False

            # Ensure explanation key exists
            if "explanation" not in data:
                data["explanation"] = ""

            # Always return a JSON string downstream
            return json.dumps(data)
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    def execute(self, question: str, context: str = "", project_name: str = "") -> str:
        """Execute the answer agent."""
        formatted_prompt = self.format_prompt(question, context)
        response = self.llm.inference(formatted_prompt, project_name)
        return self.validate_response(response)

    def parse_response(self, response: str) -> dict:
        """Parse the answer agent's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "answer": data.get("answer", ""),
                "explanation": data.get("explanation", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing answer response: {str(e)}")
            return {
                "answer": "",
                "explanation": "I apologize, but I encountered an error while generating the answer.",
                "metadata": {}
            }
