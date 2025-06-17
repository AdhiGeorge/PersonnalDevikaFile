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
    async def execute(self, current_prompt: str, project_name: str) -> str:
        """Execute the internal monologue agent."""
        prompt = self.format_prompt(current_prompt)
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}], self.base_model)
        return self.validate_response(response.choices[0].message.content)

