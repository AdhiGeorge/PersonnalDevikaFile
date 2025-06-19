import json
import sys
import time
from functools import wraps
import logging
from typing import Any, Dict
from Agentres.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

def retry_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_tries = 5
        tries = 0
        while tries < max_tries:
            result = func(*args, **kwargs)
            if result:
                return result
            logger.warning("Invalid response from the model, trying again...")
            tries += 1
            time.sleep(2)
        logger.error("Maximum 5 attempts reached. Model keeps failing.")
        sys.exit(1)
    return wrapper

class InvalidResponseError(Exception):
    pass

def validate_responses(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        response = args[1]
        response = response.strip()

        try:
            response = json.loads(response)
            args[1] = response
            return func(*args, **kwargs)

        except json.JSONDecodeError:
            pass

        try:
            response = response.split("```")[1]
            if response:
                response = json.loads(response.strip())
                args[1] = response
                return func(*args, **kwargs)

        except (IndexError, json.JSONDecodeError):
            pass

        try:
            start_index = response.find('{')
            end_index = response.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = response[start_index:end_index+1]
                try:
                    response = json.loads(json_str)
                    args[1] = response
                    return func(*args, **kwargs)

                except json.JSONDecodeError:
                    pass
        except json.JSONDecodeError:
            pass

        for line in response.splitlines():
            try:
                response = json.loads(line)
                args[1] = response
                return func(*args, **kwargs)

            except json.JSONDecodeError:
                pass

        raise InvalidResponseError("Failed to parse response as JSON")

    return wrapper

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
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "formatted_code" not in data or not isinstance(data["formatted_code"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    async def execute(self, code: str, language: str = "python", project_name: str = "") -> str:
        """Execute the formatter agent."""
        formatted_prompt = self.format_prompt(code, language)
        response = await self.llm.chat_completion([{"role": "user", "content": formatted_prompt}], self.base_model)
        validated_response = self.validate_response(response.choices[0].message.content)
        return self.parse_response(validated_response)

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