import os
import time
import json
from typing import List, Dict, Any
import sys
from functools import wraps
import logging

from Agentres.config import Config
from Agentres.llm import LLM
from Agentres.state import AgentState
from Agentres.agents.base_agent import BaseAgent
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

class Feature(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
        config = Config()
        self.project_dir = config.get_projects_dir()
        
        self.llm = LLM(model_id=base_model)

    def format_prompt(self, conversation: list, code_markdown: str, system_os: str) -> str:
        """Format the feature prompt with the code and issue."""
        prompt_template = self.get_prompt("feature")
        if not prompt_template:
            raise ValueError("Feature prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template,conversation=conversation, code_markdown=code_markdown, system_os=system_os)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "implementation" not in data or not isinstance(data["implementation"], dict):
                return False
            if "explanation" not in data or not isinstance(data["explanation"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False

    def save_code_to_project(self, response: List[Dict[str, str]], project_name: str):
        file_path_dir = None
        project_name = project_name.lower().replace(" ", "-")

        for file in response:
            file_path = os.path.join(self.project_dir, project_name, file['file'])
            file_path_dir = os.path.dirname(file_path)
            os.makedirs(file_path_dir, exist_ok=True)
    
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file["code"])
        
        return file_path_dir

    def get_project_path(self, project_name: str):
        project_name = project_name.lower().replace(" ", "-")
        return f"{self.project_dir}/{project_name}"

    def response_to_markdown_prompt(self, response: List[Dict[str, str]]) -> str:
        response = "\n".join([f"File: `{file['file']}`:\n```\n{file['code']}\n```" for file in response])
        return f"~~~\n{response}\n~~~"

    async def emulate_code_writing(self, code_set: list, project_name: str):
        files = []
        for file in code_set:
            filename = file["file"]
            code = file["code"]

            new_state = AgentState().new_state()
            new_state["internal_monologue"] = "Writing code..."
            new_state["terminal_session"]["title"] = f"Editing {filename}"
            new_state["terminal_session"]["command"] = f"vim {filename}"
            new_state["terminal_session"]["output"] = code
            files.append({
                "file": filename,
                "code": code,
            })
            AgentState().add_to_current_state(project_name, new_state)
            await asyncio.sleep(1) # Use asyncio.sleep for async functions
        # Removed emit_agent call as it is frontend related

    @retry_wrapper
    async def execute(self, feature_request: str, context: str = "", project_name: str = "") -> str:
        self.state.set_agent_state(project_name, self.__class__.__name__)

        prompt_name = "feature"
        prompt = self.prompt_manager.get_prompt(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found.")

        prompt_args = {
            "feature_request": feature_request,
            "context": context
        }
        formatted_prompt = self.format_prompt(prompt, **prompt_args)

        logger.info(f"Feature Agent - Sending prompt to LLM: {formatted_prompt[:200]}...")
        response = await self.llm.chat_completion([{"role": "user", "content": formatted_prompt}], self.base_model)
        logger.info(f"Feature Agent - Received response from LLM: {response}")

        return response.choices[0].message.content

    def parse_response(self, response: str) -> dict:
        try:
            data = json.loads(response)
            return {
                "implementation": data.get("implementation", {}),
                "explanation": data.get("explanation", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing feature response: {str(e)}")
            return {
                "implementation": {},
                "explanation": "I apologize, but I encountered an error while implementing the feature.",
                "metadata": {}
            }
