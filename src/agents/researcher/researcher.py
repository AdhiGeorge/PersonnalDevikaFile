import json
from typing import List

from src.llm import LLM
from src.services.utils import retry_wrapper, validate_responses
<<<<<<< HEAD
from src.agents.base_agent import BaseAgent
from agent.core.knowledge_base import KnowledgeBase

class Researcher(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
=======
from src.browser.search import SearchEngine
from src.agents.base_agent import BaseAgent
from agent.core.knowledge_base import KnowledgeBase


class Researcher(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)
        self.search_engine = SearchEngine()
>>>>>>> 925f80e (fifth commit)
        self.llm = LLM(model_id=base_model)

    def format_prompt(self, plan: str) -> str:
        """Format the researcher prompt with the plan."""
        prompt_template = self.get_prompt("researcher")
        if not prompt_template:
            raise ValueError("Researcher prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, plan=plan)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # The response should be a valid JSON string
            data = json.loads(response)
            if not isinstance(data, dict) or "queries" not in data:
                return False
            if not isinstance(data["queries"], list):
                return False
            for query in data["queries"]:
                if not isinstance(query, str):
                    return False
            return response
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    def execute(self, plan: str, project_name: str) -> str:
        """Execute the researcher agent."""
        formatted_prompt = self.format_prompt(plan)
        response = self.llm.inference(formatted_prompt, project_name)
        validated = self.validate_response(response)
        # Store in knowledge base if valid
        if validated:
            kb = KnowledgeBase()
            kb.add_document(
                text=validated,
                metadata={"agent": "researcher", "project_name": project_name, "plan": plan}
            )
        return validated

    def parse_response(self, response: str) -> dict:
        """Parse the researcher's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "queries": data.get("queries", []),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing researcher response: {str(e)}")
            return {
                "queries": [],
                "metadata": {}
            }
