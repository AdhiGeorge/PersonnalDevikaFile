import json
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from agents.base_agent import BaseAgent
from services.utils import retry_wrapper, validate_responses
from llm.llm import LLM
from utils.logger import Logger
from config.config import Config
from prompts.prompt_manager import PromptManager
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    """Valid query types for the planning system."""
    FACTUAL = "factual"          # For factual information retrieval
    ANALYTICAL = "analytical"      # For analysis and interpretation
    CODE_GENERATION = "code_generation"  # For generating code
    DATA_RETRIEVAL = "data_retrieval"    # For retrieving data
    HOW_TO = "how_to"              # For procedural instructions
    CODE = "code"                  # Alias for code_generation
    RESEARCH = "research"          # For research tasks
    ANSWER = "answer"              # For final answer generation
    DEFINITION = "definition"       # For term definitions
    
    @classmethod
    def get_valid_types(cls) -> List[str]:
        """Return a list of all valid query type values."""
        return [t.value for t in cls]

@dataclass
class SubQuery:
    """Represents a sub-query with its metadata."""
    id: str
    query: str
    query_type: QueryType
    required_data: List[str]
    dependencies: List[str] = None
    min_required_results: int = 3
    is_fulfilled: bool = False
    results: List[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'query': self.query,
            'query_type': self.query_type.value,
            'required_data': self.required_data,
            'dependencies': self.dependencies or [],
            'min_required_results': self.min_required_results,
            'is_fulfilled': self.is_fulfilled,
            'has_results': bool(self.results and len(self.results) >= self.min_required_results)
        }

class Planner(BaseAgent):
    """Planner agent for creating execution plans."""
    
    # Valid agent types
    VALID_AGENT_TYPES = ["researcher", "coder", "answer"]
    VALID_TASK_TYPES = ["code", "research", "answer"]
    
    def __init__(self, config: Config, model: str = "gpt-4"):
        """Initialize the planner agent.
        
        Args:
            config: Configuration instance
            model: The model to use for generating plans
        """
        try:
            # Initialize base agent first
            super().__init__(config, model)
            
            # Cache for sub-queries and their results
            self.sub_queries: Dict[str, SubQuery] = {}
            self.query_dependency_graph: Dict[str, List[str]] = {}
            
            self.logger.info("Planner initialized")
        except Exception as e:
            error_msg = f"Failed to initialize planner: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            else:
                logging.error(error_msg)
            raise ValueError(error_msg)

    async def initialize(self) -> None:
        """Initialize the planner."""
        try:
            # Initialize base agent first
            await super().initialize()
            
            # Verify required prompts exist
            required_prompts = ["planner", "sub_query_generation", "data_validation"]
            for prompt_name in required_prompts:
                try:
                    await self.prompt_manager.get_prompt(prompt_name)
                except Exception as e:
                    raise ValueError(f"Required prompt '{prompt_name}' not found: {str(e)}")
            
            self._initialized = True
            self.logger.info("Planner initialized successfully")
            
        except Exception as e:
            import traceback
            error_msg = f"Error initializing planner: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)


    async def validate_response(self, response: Any) -> Dict[str, Any]:
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
            if "type" not in data:
                raise ValueError("Response must contain a 'type' field")
            if "steps" not in data or not isinstance(data["steps"], list):
                raise ValueError("Response must contain a 'steps' list")
            if "final_answer" not in data or not isinstance(data["final_answer"], dict):
                raise ValueError("Response must contain a 'final_answer' object")

            # Validate type field
            if data["type"] not in self.VALID_TASK_TYPES:
                raise ValueError(f"Invalid type: {data['type']}. Must be one of {self.VALID_TASK_TYPES}")

            # Validate steps
            for i, step in enumerate(data["steps"]):
                if not isinstance(step, dict):
                    raise ValueError(f"Step {i} must be a JSON object")
                if "id" not in step:
                    raise ValueError(f"Step {i} must contain 'id' field")
                if "agent" not in step:
                    raise ValueError(f"Step {i} must contain 'agent' field")
                if "description" not in step:
                    raise ValueError(f"Step {i} must contain 'description' field")
                if "expected_output" not in step:
                    raise ValueError(f"Step {i} must contain 'expected_output' field")
                
                # Validate agent type
                if step["agent"] not in self.VALID_AGENT_TYPES:
                    raise ValueError(
                        f"Step {i} has invalid agent type: {step['agent']}. "
                        f"Must be one of {self.VALID_AGENT_TYPES}. "
                        f"Step details: {json.dumps(step, indent=2)}"
                    )

            # Validate final answer
            if "agent" not in data["final_answer"]:
                raise ValueError("Final answer must contain 'agent' field")
            if "description" not in data["final_answer"]:
                raise ValueError("Final answer must contain 'description' field")
            if "required_components" not in data["final_answer"]:
                raise ValueError("Final answer must contain 'required_components' field")

            # Validate final answer agent type
            if data["final_answer"]["agent"] not in self.VALID_AGENT_TYPES:
                raise ValueError(
                    f"Final answer has invalid agent type: {data['final_answer']['agent']}. "
                    f"Must be one of {self.VALID_AGENT_TYPES}. "
                    f"Final answer details: {json.dumps(data['final_answer'], indent=2)}"
                )

            return data

        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    async def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the validated response."""
        return response

    def format_response(self, response: dict) -> str:
        """Format the response from the planner."""
        return json.dumps(response)

    async def _generate_sub_queries(self, query: str) -> List[SubQuery]:
        """Generate sub-queries from the main query."""
        try:
            # Get the sub-query prompt template
            sub_query_template = await self.prompt_manager.get_prompt("sub_query_generation")
            # Format the sub-query prompt
            prompt = sub_query_template.format(query=query)
            
            # Get LLM response
            response = await self.llm.chat_completion([
                {"role": "system", "content": prompt}
            ])
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # First try direct JSON parsing
                sub_queries_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown code block
                match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response_text, re.IGNORECASE)
                if match:
                    json_str = match.group(1)
                    sub_queries_data = json.loads(json_str)
                else:
                    # Fallback: find first { and last }
                    start = response_text.find('{')
                    end = response_text.rfind('}')
                    if start != -1 and end != -1:
                        json_str = response_text[start:end+1]
                        sub_queries_data = json.loads(json_str)
                    else:
                        raise ValueError("No valid JSON found in response")
            
            # Validate the response structure
            if not isinstance(sub_queries_data, dict):
                raise ValueError("Response must be a JSON object")
            if 'sub_queries' not in sub_queries_data:
                raise ValueError("Response must contain 'sub_queries' field")
                
            # Convert to SubQuery objects
            sub_queries = []
            for i, sq_data in enumerate(sub_queries_data['sub_queries'], 1):
                # Set default type if not provided
                query_type = QueryType.FACTUAL
                if 'type' in sq_data:
                    type_str = str(sq_data['type']).lower().strip()
                    # Map common variations to our enum values
                    type_mapping = {
                        'howto': 'how_to',
                        'code_gen': 'code_generation',
                        'code': 'code_generation',  # Alias for code_generation
                        'data': 'data_retrieval',
                        'analysis': 'analytical',
                        'fact': 'factual',
                        'question': 'factual',
                        'research': 'research',
                        'answer': 'answer',
                        'definition': 'factual',  # Map definition to factual
                        'def': 'factual'           # Common abbreviation
                    }
                    
                    # Apply mapping if needed
                    type_str = type_mapping.get(type_str, type_str)
                    
                    try:
                        query_type = QueryType(type_str)
                    except ValueError:
                        valid_types = QueryType.get_valid_types()
                        logger.warning(
                            f"Invalid query type '{sq_data['type']}'. "
                            f"Valid types are: {', '.join(valid_types)}. Using FACTUAL."
                        )
                        query_type = QueryType.FACTUAL
                
                sub_query = SubQuery(
                    id=f"sq_{i}",
                    query=sq_data['query'],
                    query_type=query_type,
                    required_data=sq_data.get('required_data', []),
                    dependencies=sq_data.get('dependencies', []),
                    min_required_results=sq_data.get('min_required_results', 3)
                )
                sub_queries.append(sub_query)
            
            return sub_queries
            
        except Exception as e:
            error_msg = f"Failed to generate sub-queries: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    async def _validate_data_completeness(self, query_id: str, results: List[Dict]) -> Tuple[bool, str]:
        """Validate if the results for a sub-query are complete."""
        try:
            sub_query = self.sub_queries.get(query_id)
            if not sub_query:
                return False, "Sub-query not found"
                
            # Basic validation
            if not results or len(results) < sub_query.min_required_results:
                return False, f"Insufficient results (need at least {sub_query.min_required_results})"
                
            # Get the validation prompt template
            validation_template = await self.prompt_manager.get_prompt("data_validation")
            # Format the validation prompt
            validation_prompt = validation_template.format(
                query=sub_query.query,
                results=json.dumps(results, indent=2),
                required_data=", ".join(sub_query.required_data)
            )
            
            response = await self.llm.chat_completion([
                {"role": "system", "content": validation_prompt}
            ])
            
            validation_result = json.loads(response.choices[0].message.content)
            return validation_result.get('is_complete', False), validation_result.get('reason', '')
            
        except Exception as e:
            self.logger.error(f"Error validating data completeness: {str(e)}")
            return False, f"Validation error: {str(e)}"

    async def plan(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate a plan for executing the given query."""
        try:
            if not self._initialized:
                raise RuntimeError("Planner not initialized. Call initialize() first.")
            
            # Generate sub-queries
            sub_queries = await self._generate_sub_queries(query)
            
            # Get the planner prompt template
            planner_template = await self.prompt_manager.get_prompt("planner")
            # Format the prompt with the query and sub-queries
            prompt = planner_template.format(
                query=query,
                sub_queries=json.dumps([q.to_dict() for q in sub_queries], indent=2) if sub_queries else ""
            )
            
            # Get LLM response
            response = await self.llm.chat_completion([
                {"role": "system", "content": prompt}
            ])
            
            # Log the raw response
            self.logger.info("Raw response from LLM")
            self.logger.info(response.choices[0].message.content)
            
            # Parse and validate the response
            try:
                data = await self.validate_response(response.choices[0].message.content)
                self.logger.info("Successfully parsed JSON response")
                return data
            except Exception as e:
                self.logger.error(f"Error in planning: {str(e)}")
                raise ValueError(f"Failed to generate a valid plan: {str(e)}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in planning: {str(e)}")
            else:
                logging.error(f"Error in planning: {str(e)}")
            raise ValueError(f"Planning failed: {str(e)}")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a planning task."""
        try:
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")

            query = str(task.get("query", ""))
            context = task.get("context", [])
            
            # Create plan
            plan = await self.plan(query, context)
            
            # Validate the plan structure
            if not isinstance(plan, dict) or "steps" not in plan or "final_answer" not in plan:
                raise ValueError("Invalid plan structure")

            return {
                "query": query,
                "plan": plan,
                "steps": plan["steps"],
                "final_answer": plan["final_answer"]
            }

        except Exception as e:
            logger.error(f"Error executing planning task: {str(e)}")
            raise ValueError(f"Task execution failed: {str(e)}")

    async def execute_with_context(self, task: Dict[str, Any], context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute a planning task with context."""
        try:
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")
            if not isinstance(context, list):
                raise ValueError("context must be a list")

            # Add context to task
            task["context"] = context
            return await self.execute(task)
        except Exception as e:
            logger.error(f"Error executing planning task with context: {str(e)}")
            raise ValueError(f"Task execution with context failed: {str(e)}")

    def get_prompt(self, task: Dict[str, Any]) -> str:
        """Get prompt for the task."""
        try:
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")
            return str(task.get("prompt", ""))
        except Exception as e:
            logger.error(f"Error getting prompt: {str(e)}")
            raise ValueError(f"Failed to get prompt: {str(e)}")

    async def plan_code_generation(self, query: str, research_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan code generation based on query and research results.
        
        Args:
            query: The original query
            research_results: List of research results
            
        Returns:
            Dict containing the code generation plan
        """
        try:
            if not self._initialized:
                await self.initialize()
                
            # Format the code generation prompt
            prompt = self.prompt_manager.get_prompt("code_generation_plan").format(
                query=query,
                research=json.dumps(research_results, indent=2)
            )
            
            # Get LLM response
            response = await self.llm.chat_completion([
                {"role": "system", "content": prompt}
            ])
            
            # Parse and validate the response
            plan = await self.validate_response(response.choices[0].message.content)
            
            # Ensure it's a code generation plan
            if plan["type"] != "code":
                raise ValueError("Generated plan is not a code generation plan")
                
            return plan
            
        except Exception as e:
            error_msg = f"Failed to plan code generation: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)