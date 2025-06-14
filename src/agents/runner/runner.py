import time
import json
import os
from src.services.terminal_runner import TerminalRunner
from src.agents.patcher import Patcher
from src.agents.base_agent import BaseAgent
from src.llm import LLM
from src.services.utils import retry_wrapper, validate_responses
from agent.core.knowledge_base import KnowledgeBase
<<<<<<< HEAD
=======
from src.llm import LLM
from typing import List
>>>>>>> 925f80e (fifth commit)

class Runner(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
        self.base_model = base_model
        self.terminal_runner = TerminalRunner()
        self.llm = LLM(model_id=base_model)

<<<<<<< HEAD
    def format_prompt(self, conversation: str, code_markdown: str, system_os: str, commands: list, error: str) -> str:
        """Format the runner prompt with the code and context."""
        prompt_template = self.get_prompt("runner")
        if not prompt_template:
            raise ValueError("Runner prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, conversation=conversation, code_markdown=code_markdown, system_os=system_os, commands=commands, error=error)
=======
    def format_prompt(self, plan: str) -> str:
        """Format the runner prompt with the plan."""
        prompt_template = self.get_prompt("runner")
        if not prompt_template:
            raise ValueError("Runner prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, plan=plan)

    def format_rerunner_prompt(self, plan: str) -> str:
        """Format the rerunner prompt with the plan."""
        prompt_template = self.get_prompt("rerunner")
        if not prompt_template:
            raise ValueError("Rerunner prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, plan=plan)
>>>>>>> 925f80e (fifth commit)

    def render(
        self,
        conversation: str,
        code_markdown: str,
        system_os: str,
        commands: list,
        error: str
    ) -> str:
        return self.format_prompt(conversation, code_markdown, system_os, commands, error)

    def render_rerunner(
        self,
        conversation: str,
        code_markdown: str,
        system_os: str,
        commands: list,
        error: str
    ):
        return self.format_prompt(conversation, code_markdown, system_os, commands, error)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # The response should be a valid JSON string
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "output" not in data or not isinstance(data["output"], str):
                return False
            if "status" not in data or not isinstance(data["status"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False

    @validate_responses
    def validate_rerunner_response(self, response: str):
        if "action" not in response and "response" not in response:
            return False
        else:
            return response

    @retry_wrapper
    def run_code(self, commands: list, input_text: str = None) -> list:
        results = []
        for command in commands:
            if isinstance(command, str):
                command = command.split()
            result = self.terminal_runner.run(command, input_text=input_text)
            results.append(result)
        return results

    @retry_wrapper
    def execute(self, conversation: str, code_markdown: str, system_os: str, commands: list, error: str, project_name: str = "") -> str:
        """Execute the runner agent."""
        formatted_prompt = self.format_prompt(conversation, code_markdown, system_os, commands, error)
        response = self.llm.inference(formatted_prompt, project_name)
        validated = self.validate_response(response)
        # Store in knowledge base if valid
        if validated:
            kb = KnowledgeBase()
            kb.add_document(
                text=validated,
                metadata={"agent": "runner", "project_name": project_name, "conversation": conversation, "code_markdown": code_markdown, "system_os": system_os, "commands": commands, "error": error}
            )
        return validated

    def parse_response(self, response: str) -> dict:
        """Parse the runner's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "output": data.get("output", ""),
                "status": data.get("status", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing runner response: {str(e)}")
            return {
                "output": "",
                "status": "error",
                "metadata": {}
            }