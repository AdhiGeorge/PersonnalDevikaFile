import os
import time
import json
from typing import List, Dict, Optional

<<<<<<< HEAD
from typing import List, Dict
from src.socket_instance import emit_agent

from src.config import Config
=======
>>>>>>> 925f80e (fifth commit)
from src.llm import LLM
from src.services.utils import retry_wrapper, validate_responses
from src.agents.base_agent import BaseAgent
from agent.core.knowledge_base import KnowledgeBase

class Patcher(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
        self.llm = LLM(model_id=base_model)

<<<<<<< HEAD
    def format_prompt(self, conversation: list, code_markdown: str, commands: list, error :str, system_os: str) -> str:
        """Format the patcher prompt with the code and issue."""
        prompt_template = self.get_prompt("patcher")
        if not prompt_template:
            raise ValueError("Patcher prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template,conversation=conversation, code_markdown=code_markdown, commands=commands, error=error, system_os=system_os)
=======
    def format_prompt(self, plan: str) -> str:
        """Format the patcher prompt with the plan."""
        prompt_template = self.get_prompt("patcher")
        if not prompt_template:
            raise ValueError("Patcher prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, plan=plan)
>>>>>>> 925f80e (fifth commit)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # The response should be a valid JSON string
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "patched_code" not in data or not isinstance(data["patched_code"], str):
                return False
            if "explanation" not in data or not isinstance(data["explanation"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False

    def render(
        self,
        conversation: list,
        code_markdown: str,
        commands: list,
        error :str,
        system_os: str
    ) -> str:
<<<<<<< HEAD
        return self.format_prompt(conversation, code_markdown, commands, error, system_os)
=======
        prompt_template = self.get_prompt("patcher")
        if not prompt_template:
            raise ValueError("Patcher prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, conversation=conversation, code_markdown=code_markdown, commands=commands, error=error, system_os=system_os)
>>>>>>> 925f80e (fifth commit)

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

    def emulate_code_writing(self, code_set: list, project_name: str):
        files = []
        for current_file in code_set:
            file = current_file["file"]
            code = current_file["code"]

            new_state = AgentState().new_state()
            new_state["internal_monologue"] = "Writing code..."
            new_state["terminal_session"]["title"] = f"Editing {file}"
            new_state["terminal_session"]["command"] = f"vim {file}"
            new_state["terminal_session"]["output"] = code
            files.append({
                "file": file,
                "code": code
            })
            AgentState().add_to_current_state(project_name, new_state)
            time.sleep(1)
        emit_agent("code", {
            "files": files,
            "from": "patcher"
        })

    @retry_wrapper
    def execute(self, code: str, issue: str, project_name: str = "") -> str:
        """Execute the patcher agent."""
        formatted_prompt = self.format_prompt(code)
        response = self.llm.inference(formatted_prompt, project_name)
        validated = self.validate_response(response)
        # Store in knowledge base if valid
        if validated:
            kb = KnowledgeBase()
            kb.add_document(
                text=validated,
                metadata={"agent": "patcher", "project_name": project_name, "code": code, "issue": issue}
            )
        return validated

    def parse_response(self, response: str) -> dict:
        """Parse the patcher's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "patched_code": data.get("patched_code", ""),
                "explanation": data.get("explanation", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing patcher response: {str(e)}")
            return {
                "patched_code": "",
                "explanation": "I apologize, but I encountered an error while patching the code.",
                "metadata": {}
            }
