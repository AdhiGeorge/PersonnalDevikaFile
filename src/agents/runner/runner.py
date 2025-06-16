import json
import os
from src.agents.base_agent import BaseAgent
import sys
import time
from functools import wraps
import logging
from typing import Any, Dict

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

class Runner(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, code: str, context: str = "") -> str:
        """Format the runner prompt with the code and context."""
        prompt_template = self.get_prompt("runner")
        if not prompt_template:
            raise ValueError("Runner prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, code=code, context=context)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "commands" not in data or not isinstance(data["commands"], list):
                return False
            for command in data["commands"]:
                if not isinstance(command, str):
                    return False
            return response
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    async def execute(self, code: str, context: str = "", project_name: str = "") -> str:
        """Execute the runner agent."""
        formatted_prompt = self.format_prompt(code, context)
        response = await self.llm.chat_completion([{"role": "user", "content": formatted_prompt}], self.base_model)
        validated_response = self.validate_response(response.choices[0].message.content)
        return self.parse_response(validated_response)

    def parse_response(self, response: str) -> dict:
        """Parse the runner's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "commands": data.get("commands", []),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing runner response: {str(e)}")
            return {
                "commands": [],
                "metadata": {}
            }
