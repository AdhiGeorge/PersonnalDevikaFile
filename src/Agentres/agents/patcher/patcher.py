import logging
import os
import time
import json
from typing import List, Dict, Any, Optional
import sys
from functools import wraps
import platform
import asyncio

from Agentres.agents.base_agent import BaseAgent
from Agentres.llm import LLM
from Agentres.project import ProjectManager
from Agentres.prompts.prompt_manager import PromptManager
from Agentres.state import AgentState
from Agentres.services.terminal_runner import TerminalRunner
from Agentres.config.config import Config

logger = logging.getLogger(__name__)

def retry_wrapper(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        max_tries = 5
        tries = 0
        while tries < max_tries:
            result = await func(*args, **kwargs)
            if result:
                return result
            logger.warning("Invalid response from the model, trying again...")
            tries += 1
            await asyncio.sleep(2)
        logger.error("Maximum 5 attempts reached. Model keeps failing.")
        sys.exit(1)
    return wrapper

class InvalidResponseError(Exception):
    pass

def validate_responses(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        args = list(args)
        response = args[1]
        response = response.strip()

        try:
            response = json.loads(response)
            args[1] = response
            return await func(*args, **kwargs)

        except json.JSONDecodeError:
            pass

        try:
            response = response.split("```")[1]
            if response:
                response = json.loads(response.strip())
                args[1] = response
                return await func(*args, **kwargs)

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
                    return await func(*args, **kwargs)

                except json.JSONDecodeError:
                    pass
        except json.JSONDecodeError:
            pass

        for line in response.splitlines():
            try:
                response = json.loads(line)
                args[1] = response
                return await func(*args, **kwargs)

            except json.JSONDecodeError:
                pass

        raise InvalidResponseError("Failed to parse response as JSON")

    return wrapper

class Patcher(BaseAgent):
    def __init__(self, config: Config):
        """Initialize the patcher with configuration.
        
        Args:
            config: Configuration instance
        """
        super().__init__(config)
        self.llm = LLM(config)
        self.prompt_manager = PromptManager()
        self.project_manager = ProjectManager()
        self.state = AgentState()
        self.terminal_runner = TerminalRunner()

    async def format_prompt(self, conversation: list, code_markdown: str, commands: list, error :str, system_os: str) -> str:
        """Format the patcher prompt with the code and issue."""
        prompt_template = self.get_prompt("patch_code")
        if not prompt_template:
            raise ValueError("Patcher prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template,conversation=conversation, code_markdown=code_markdown, commands=commands, error=error, system_os=system_os)

    @validate_responses
    async def execute(self, code: str) -> str:
        """Execute the patcher agent."""
        self.logger.info("Starting patching phase...")
        
        # Format the prompt
        prompt = await self.format_prompt(
            conversation="",
            code_markdown=code,
            commands=[],
            error="",
            system_os=platform.system()
        )
        
        # Get response from LLM
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        patched_code = response.choices[0].message.content
        
        # Parse and validate response
        try:
            parsed_response = await self.parse_response(patched_code)
            if not parsed_response:
                raise ValueError("Invalid response from LLM")
            
            # Extract the patched code
            if isinstance(parsed_response, dict) and "code" in parsed_response:
                patched_code = parsed_response["code"]
            
            self.logger.info("Patching phase completed")
            return patched_code
            
        except Exception as e:
            self.logger.error(f"Error in patching phase: {str(e)}")
            raise

    async def save_code_to_project(self, code: str, project_name: str) -> None:
        """Saves the generated code to the project directory."""
        try:
            await self.project_manager.add_code_file(project_name, "patcher_code.py", code)
            logger.info(f"Patcher Agent - Code saved to project: {project_name}")
        except Exception as e:
            logger.error(f"Patcher Agent - Error saving code to project: {e}")
            raise

    async def parse_response(self, response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Patcher Agent - Failed to parse response as JSON: {e}")
            raise ValueError("Invalid JSON response from LLM")

    @validate_responses
    async def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "code" not in data or not isinstance(data["code"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False
