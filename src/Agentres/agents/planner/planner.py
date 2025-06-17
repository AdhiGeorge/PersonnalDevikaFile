import json
import logging
import re
from typing import Dict, Any, List
from Agentres.agents.base_agent import BaseAgent
from Agentres.services.utils import retry_wrapper, validate_responses
from Agentres.knowledge_base.knowledge_base import KnowledgeBase
from Agentres.llm import LLM
from Agentres.logger import Logger
from Agentres.config import Config
from Agentres.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class Planner(BaseAgent):
    """Planner agent for creating execution plans."""
    
    def __init__(self, config: Config):
        """Initialize the planner agent."""
        super().__init__(config)
        self.llm = LLM(config)
        self.logger = Logger()
        self.config = config
        self.prompt_manager = PromptManager()
        self.system_prompt = self.get_prompt("planner")
        if not self.system_prompt:
            raise ValueError("Planner prompt not found in prompts.yaml")

    def format_prompt(self, prompt: str) -> str:
        """Format the planner prompt with the user's prompt."""
        return super().format_prompt(self.system_prompt, prompt=prompt)

    def validate_response(self, response: Any) -> Dict[str, Any]:
        """Validate the response format."""
        try:
            # If response is already a dict, use it directly
            if isinstance(response, dict):
                data = response
            else:
                # Try to extract JSON from a markdown code block
                match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response, re.IGNORECASE)
                if match:
                    json_str = match.group(1)
                else:
                    # Fallback: find first { and last }
                    start = response.find('{')
                    end = response.rfind('}')
                    if start != -1 and end != -1:
                        json_str = response[start:end+1]
                    else:
                        raise ValueError("No JSON object found in response")
                data = json.loads(json_str)

            # Check required fields
            if "steps" not in data or not isinstance(data["steps"], list):
                raise ValueError("Response must contain a 'steps' list")
            if "final_answer" not in data or not isinstance(data["final_answer"], dict):
                raise ValueError("Response must contain a 'final_answer' object")
            # Validate steps
            for step in data["steps"]:
                if not isinstance(step, dict):
                    raise ValueError("Each step must be an object")
                if "id" not in step or not isinstance(step["id"], str):
                    raise ValueError("Each step must have an 'id' string")
                if "agent" not in step or step["agent"] not in ["researcher", "developer", "answer"]:
                    raise ValueError("Each step must have a valid 'agent' field")
                if "description" not in step:
                    raise ValueError("Each step must have a 'description' field")
                if "expected_output" not in step:
                    raise ValueError("Each step must have an 'expected_output' field")
                if "queries" in step and not isinstance(step["queries"], list):
                    raise ValueError("Step queries must be a list")
                if "dependencies" in step and not isinstance(step["dependencies"], list):
                    raise ValueError("Step dependencies must be a list")
            # Validate final answer
            final = data["final_answer"]
            if "agent" not in final or final["agent"] != "answer":
                raise ValueError("Final answer must have agent 'answer'")
            if "description" not in final:
                raise ValueError("Final answer must have a description")
            if "required_components" not in final or not isinstance(final["required_components"], list):
                raise ValueError("Final answer must have a required_components list")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response as JSON: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    @retry_wrapper
    async def execute(self, prompt: str) -> Dict[str, Any]:
        """Execute the planning phase."""
        try:
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            # Get response from LLM
            response = await self.llm.chat_completion(messages)
            content = response.choices[0].message.content
            logger.info(f"Raw response content: {content}")

            # Validate and parse the response
            data = self.validate_response(content)

            # Log the plan
            logger.info(f"Generated plan with {len(data['steps'])} steps")
            for step in data['steps']:
                logger.info(f"Step {step['id']}: {step['description']}")

            return data

        except Exception as e:
            logger.error(f"Error in planning phase: {str(e)}")
            raise

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the validated response."""
        return response

    def format_response(self, response: dict) -> str:
        """Format the response from the planner."""
        return json.dumps(response)

    def plan(self, query: str) -> str:
        """Create a plan for implementing the given query"""
        self.logger.info(f"Starting planning phase for query: {query}")
        self.logger.set_query(query)
        
        # Get the planning prompt
        prompt = self.prompt_manager.get_prompt("planner", query)
        
        # Generate the plan
        plan = self.llm.generate(prompt)
        
        # Log the plan
        self.logger.set_planner_output(plan)
        self.logger.add_planner_step({
            "step": "plan_generation",
            "prompt": prompt,
            "output": plan,
            "timestamp": self.logger.json_log["timestamp"]
        })
        
        self.logger.info("Planning phase completed")
        return plan