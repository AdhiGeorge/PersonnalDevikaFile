import json
import os
from agents.base_agent import BaseAgent
import sys
import time
from functools import wraps
import logging
from typing import Any, Dict, Optional
from config.config import Config
from llm import LLM
from utils.logger import Logger
import asyncio

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

class Runner(BaseAgent):
    def __init__(self, config: Config):
        """Initialize runner with configuration."""
        super().__init__(config)
        self._logger = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.config = config
        self.base_model = config.model

    @property
    def logger(self) -> Logger:
        """Lazy initialization of logger."""
        if self._logger is None:
            self._logger = Logger(self.config)
        return self._logger

    async def initialize(self):
        """Initialize async components."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                # Initialize base agent
                await super().initialize()
                
                # Initialize logger
                await self.logger.initialize()
                
                self._initialized = True
                logger.info("Runner initialized")
            except Exception as e:
                logger.error(f"Failed to initialize async components: {str(e)}")
                raise ValueError(f"Async initialization failed: {str(e)}")

    def _ensure_initialized(self):
        """Ensure runner is initialized before use."""
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")
        if not self._logger:
            raise RuntimeError("Runner components not properly initialized")

    def format_prompt(self, code: str, context: str = "") -> str:
        """Format the runner prompt with the code and context."""
        prompt_template = self.get_prompt("runner")
        if not prompt_template:
            raise ValueError("Runner prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, code=code, context=context)

    @validate_responses
    async def validate_response(self, response: str):
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
        try:
            self._ensure_initialized()
            formatted_prompt = self.format_prompt(code, context)
            response = await self.llm.chat_completion([{"role": "user", "content": formatted_prompt}], self.base_model)
            validated_response = await self.validate_response(response.choices[0].message.content)
            return self.parse_response(validated_response)
        except Exception as e:
            logger.error(f"Error executing runner: {str(e)}")
            raise ValueError(f"Runner execution failed: {str(e)}")

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

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self._initialized:
                # Add cleanup logic here
                pass
        except Exception as e:
            logger.error(f"Error cleaning up runner: {str(e)}")
            raise ValueError(f"Runner cleanup failed: {str(e)}")
