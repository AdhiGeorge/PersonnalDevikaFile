import logging
import os
import time
import json
from typing import List, Dict, Any, Optional
import sys
from functools import wraps

from Agentres.agents.base_agent import BaseAgent
from Agentres.llm import LLM
from Agentres.project import ProjectManager
from Agentres.prompts.prompt_manager import PromptManager
from Agentres.state import AgentState
from Agentres.services.terminal_runner import TerminalRunner
from Agentres.knowledge_base.knowledge_base import KnowledgeBase

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

class Patcher(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
        self.llm = LLM(base_model)
        self.prompt_manager = PromptManager()
        self.project_manager = ProjectManager()
        self.state = AgentState()
        self.terminal_runner = TerminalRunner()

    def format_prompt(self, conversation: list, code_markdown: str, commands: list, error :str, system_os: str) -> str:
        """Format the patcher prompt with the code and issue."""
        prompt_template = self.get_prompt("patch_code")
        if not prompt_template:
            raise ValueError("Patcher prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template,conversation=conversation, code_markdown=code_markdown, commands=commands, error=error, system_os=system_os)

    @validate_responses
    async def execute(self, conversation: str, code_markdown: str, commands: Optional[list], error: str, system_os: str, project_name: str) -> str:
        self.state.set_agent_state(project_name, self.__class__.__name__)

        prompt_name = "patch_code"
        prompt = self.prompt_manager.get_prompt(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found.")

        prompt_args = {
            "conversation": conversation,
            "code_markdown": code_markdown,
            "commands": "\n".join(commands) if commands else "No commands provided.",
            "error": error,
            "system_os": system_os
        }

        formatted_prompt = self.format_prompt(prompt, **prompt_args)

        logger.info(f"Patcher Agent - Sending prompt to LLM for patching: {formatted_prompt[:200]}...")
        response = await self.llm.chat_completion([{"role": "user", "content": formatted_prompt}], self.base_model)
        logger.info(f"Patcher Agent - Received response from LLM: {response}")

        return response.choices[0].message.content

    def save_code_to_project(self, code: str, project_name: str) -> None:
        """Saves the generated code to the project directory."""
        try:
            self.project_manager.add_code_file(project_name, "patcher_code.py", code)
            logger.info(f"Patcher Agent - Code saved to project: {project_name}")
        except Exception as e:
            logger.error(f"Patcher Agent - Error saving code to project: {e}")
            raise

    def parse_response(self, response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Patcher Agent - Failed to parse response as JSON: {e}")
            raise ValueError("Invalid JSON response from LLM")
