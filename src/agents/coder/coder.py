import os
import time
import json

<<<<<<< HEAD
from typing import List, Dict
=======
from typing import List, Dict, Union

>>>>>>> 925f80e (fifth commit)
from src.config import Config
from src.llm import LLM
from src.state import AgentState
from src.logger import Logger
from src.services.utils import retry_wrapper, validate_responses
from src.socket_instance import emit_agent
from src.agents.base_agent import BaseAgent
from agent.core.knowledge_base import KnowledgeBase

class Coder(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
        config = Config()
        self.project_dir = config.get_projects_dir()
        self.logger = Logger()
        self.llm = LLM(model_id=base_model)

<<<<<<< HEAD
    def format_prompt(self, step_by_step_plan: str, user_context: str, search_results: dict) -> str:
        """Format the coder prompt with the task and context."""
        prompt_template = self.get_prompt("coder")
        if not prompt_template:
            raise ValueError("Coder prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, step_by_step_plan=step_by_step_plan, user_context=user_context, search_results=search_results)
=======
    def format_prompt(self, plan: str) -> str:
        """Format the coder prompt with the plan."""
        prompt_template = self.get_prompt("coder")
        if not prompt_template:
            raise ValueError("Coder prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, plan=plan)
>>>>>>> 925f80e (fifth commit)

    def render(
        self, step_by_step_plan: str, user_context: str, search_results: dict
    ) -> str:
        return self.format_prompt(step_by_step_plan, user_context, search_results)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # The response should be a valid JSON string
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

    def emulate_code_writing(self, code_set: list, project_name: str):
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
            time.sleep(2)
        emit_agent("code", {
            "files": files,
            "from": "coder"
        })

    @retry_wrapper
    def execute(self, task: str, context: str = "", project_name: str = "") -> str:
        """Execute the coder agent."""
        formatted_prompt = self.format_prompt(task, context)
        response = self.llm.inference(formatted_prompt, project_name)
        validated = self.validate_response(response)
        # Store in knowledge base if valid
        if validated:
            kb = KnowledgeBase()
            kb.add_document(
                text=validated,
                metadata={"agent": "coder", "project_name": project_name, "task": task, "context": context}
            )
        return validated

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
