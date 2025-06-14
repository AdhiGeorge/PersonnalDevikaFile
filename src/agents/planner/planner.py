import json
from src.agents.base_agent import BaseAgent
from src.services.utils import retry_wrapper, validate_responses
from agent.core.knowledge_base import KnowledgeBase

class Planner(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, prompt: str) -> str:
        """Format the planner prompt with the user's prompt."""
        prompt_template = self.get_prompt("planner")
        if not prompt_template:
            raise ValueError("Planner prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, prompt=prompt)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            # The response should be in the format specified in the prompt
            # We'll just check if it contains the required sections
            required_sections = ["Project Name:", "Your Reply to the Human Prompter:", "Current Focus:", "Plan:", "Summary:"]
            for section in required_sections:
                if section not in response:
                    return False
            return response
        except Exception:
            return False

    @retry_wrapper
    def execute(self, prompt: str, project_name: str) -> str:
        """Execute the planner agent."""
        formatted_prompt = self.format_prompt(prompt)
        response = self.llm.inference(formatted_prompt, project_name)
        validated = self.validate_response(response)
        # Store in knowledge base if valid
        if validated:
            kb = KnowledgeBase()
            kb.add_document(
                text=validated,
                metadata={"agent": "planner", "project_name": project_name, "prompt": prompt}
            )
        return validated

    def parse_response(self, response: str) -> dict:
        """Parse the planner's response into a structured format."""
        try:
            # Extract sections from the response
            sections = response.split("\n\n")
            result = {}
            
            for section in sections:
                if section.startswith("Project Name:"):
                    result["project_name"] = section.replace("Project Name:", "").strip()
                elif section.startswith("Your Reply to the Human Prompter:"):
                    result["reply"] = section.replace("Your Reply to the Human Prompter:", "").strip()
                elif section.startswith("Current Focus:"):
                    result["focus"] = section.replace("Current Focus:", "").strip()
                elif section.startswith("Plan:"):
                    plan_text = section.replace("Plan:", "").strip()
                    result["plans"] = [step.strip() for step in plan_text.split("\n") if step.strip()]
                elif section.startswith("Summary:"):
                    result["summary"] = section.replace("Summary:", "").strip()
            
            return result
        except Exception as e:
            self.logger.error(f"Error parsing planner response: {str(e)}")
            return {
                "project_name": "",
                "reply": "I apologize, but I encountered an error while creating the plan.",
                "focus": "",
                "plans": [],
                "summary": ""
            }
