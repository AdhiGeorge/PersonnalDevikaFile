import os
import time
import json
from typing import List, Dict, Any
import sys
from functools import wraps
import logging
import asyncio

from Agentres.agents.base_agent import BaseAgent
from Agentres.llm import LLM
from Agentres.config import Config
from Agentres.project import ProjectManager
from Agentres.state import AgentState
from Agentres.prompts.prompt_manager import PromptManager
from Agentres.services.terminal_runner import TerminalRunner
from Agentres.logger import Logger
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

class Coder(BaseAgent):
    def __init__(self, config: Config):
        super().__init__(config)
        self.llm = LLM(config)
        self.logger = Logger()
        self.config = config
        self.project_manager = ProjectManager()
        self.state = AgentState()
        self.prompt_manager = PromptManager()
        self.terminal_runner = TerminalRunner()
        self.knowledge_base = KnowledgeBase()

    def implement(self, query: str, plan: str, research: str) -> str:
        """Implement the given query based on plan and research"""
        self.logger.info(f"Starting implementation phase for query: {query}")
        
        # Generate code
        prompt = self.prompt_manager.get_prompt("coder", query, plan, research)
        code = self.llm.generate(prompt)
        
        # Log the generated code
        self.logger.set_coder_output(code)
        self.logger.set_generated_code(
            code=code,
            language=self._detect_language(code),
            file_path=self._determine_file_path(query)
        )
        
        # Execute the code
        try:
            results = self.terminal_runner.run_code(code)
            self.logger.set_execution_results({
                'success': True,
                'output': results.output,
                'error': results.error,
                'exit_code': results.exit_code
            })
        except Exception as e:
            self.logger.error(f"Error executing code: {str(e)}")
            self.logger.set_execution_results({
                'success': False,
                'error': str(e)
            })
        
        self.logger.info("Implementation phase completed")
        return code

    def _detect_language(self, code: str) -> str:
        """Detect the programming language of the code"""
        # Simple language detection logic
        if code.startswith('def ') or code.startswith('class '):
            return 'python'
        elif code.startswith('function ') or code.startswith('const '):
            return 'javascript'
        # Add more language detection logic as needed
        return 'unknown'

    def _determine_file_path(self, query: str) -> str:
        """Determine the appropriate file path for the code"""
        # Simple file path determination logic
        base_name = query.lower().replace(' ', '_')
        if self._detect_language(query) == 'python':
            return f"{base_name}.py"
        elif self._detect_language(query) == 'javascript':
            return f"{base_name}.js"
        return f"{base_name}.txt"

    def format_prompt(self, step_by_step_plan: str, user_context: str, search_results: dict) -> str:
        """Format the coder prompt with the task and context."""
        prompt_template = self.get_prompt("coder")
        if not prompt_template:
            raise ValueError("Coder prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, step_by_step_plan=step_by_step_plan, user_context=user_context, search_results=search_results)

    def render(
        self, step_by_step_plan: str, user_context: str, search_results: dict
    ) -> str:
        return self.format_prompt(step_by_step_plan, user_context, search_results)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "code" not in data or not isinstance(data["code"], str):
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
        for current_file in code_set:
            file = current_file["file"]
            code = current_file["code"]

            current_state = AgentState().get_latest_state(project_name)
            new_state = AgentState().new_state()
            new_state["browser_session"] = current_state["browser_session"] # keep the browser session
            new_state["internal_monologue"] = "Writing code..."
            new_state["terminal_session"]["title"] = f"Editing {file}"
            new_state["terminal_session"]["command"] = f"vim {file}"
            new_state["terminal_session"]["output"] = code
            files.append({
                "file": file,
                "code": code
            })
            AgentState().add_to_current_state(project_name, new_state)
            await asyncio.sleep(2) # Use asyncio.sleep for async functions
        # Removed emit_agent call as it is frontend related

    @retry_wrapper
    async def execute(self, task: str, context: str = "", project_name: str = "") -> str:
        self.state.set_agent_state(project_name, self.__class__.__name__)

        prompt_name = "coder"
        prompt = self.prompt_manager.get_prompt(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found.")

        prompt_args = {
            "task": task,
            "context": context
        }
        formatted_prompt = self.format_prompt(prompt, **prompt_args)

        self.logger.info(f"Coder Agent - Sending prompt to LLM: {formatted_prompt[:200]}...")
        response = await self.llm.chat_completion([{"role": "user", "content": formatted_prompt}], self.base_model)
        self.logger.info(f"Coder Agent - Received response from LLM: {response}")

        return response.choices[0].message.content

    def parse_response(self, response: str) -> dict:
        """Parse the coder's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "code": data.get("code", ""),
                "explanation": data.get("explanation", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing coder response: {str(e)}")
            return {
                "code": "",
                "explanation": "I apologize, but I encountered an error while generating the code.",
                "metadata": {}
            }
