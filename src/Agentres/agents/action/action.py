import json
import sys
import time
from functools import wraps
import logging
from typing import Any, Dict
from Agentres.agents.base_agent import BaseAgent
from Agentres.config import Config

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
    def validate_response(self, response: dict):
        """Validate the response from the LLM."""
        if not isinstance(response, dict):
            return False
        if "response" not in response and "action" not in response:
            return False
        return response.get("response", None), response.get("action", None)

    @retry_wrapper
    async def execute(self, conversation: list, project_name: str) -> str:
        """Execute the action agent."""
        prompt = self.format_prompt(conversation)
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}], self.base_model)
        return self.validate_response(json.loads(response.choices[0].message.content))
