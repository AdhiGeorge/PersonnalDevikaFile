import json
import logging
import re
from typing import Dict, Any, List, Optional
from Agentres.agents.base_agent import BaseAgent
from Agentres.config.config import Config
from Agentres.llm.llm import LLM

logger = logging.getLogger(__name__)

class Developer(BaseAgent):
    """Developer agent for writing and modifying code."""
    
    def __init__(self, config: Config):
        """Initialize the developer agent."""
        super().__init__(config)
        self.llm = LLM(config)
        self.system_prompt = """You are a developer agent that writes and modifies code.
Your response must be a valid JSON object with the following structure:
{
    "code": {
        "implementation": "The complete code implementation",
        "dependencies": ["package1", "package2"],
        "requirements": "requirements.txt content",
        "setup_instructions": "How to set up and run the code"
    },
    "explanation": {
        "overview": "High-level explanation of the code",
        "key_components": ["component1", "component2"],
        "usage": "How to use the code",
        "examples": ["example1", "example2"]
    },
    "metadata": {
        "language": "python",
        "complexity": "O(n)",
        "quality_score": 0.95,
        "test_coverage": "What parts are tested"
    }
}"""

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
            if "code" not in data or not isinstance(data["code"], dict):
                raise ValueError("Response must contain a 'code' object")
            code = data["code"]
            if "implementation" not in code or not isinstance(code["implementation"], str):
                raise ValueError("Code must contain an 'implementation' string")
            if "dependencies" not in code or not isinstance(code["dependencies"], list):
                raise ValueError("Code must contain a 'dependencies' list")
            if "requirements" not in code or not isinstance(code["requirements"], str):
                raise ValueError("Code must contain a 'requirements' string")
            if "setup_instructions" not in code or not isinstance(code["setup_instructions"], str):
                raise ValueError("Code must contain a 'setup_instructions' string")
            if "explanation" not in data or not isinstance(data["explanation"], dict):
                raise ValueError("Response must contain an 'explanation' object")
            explanation = data["explanation"]
            if "overview" not in explanation or not isinstance(explanation["overview"], str):
                raise ValueError("Explanation must contain an 'overview' string")
            if "key_components" not in explanation or not isinstance(explanation["key_components"], list):
                raise ValueError("Explanation must contain a 'key_components' list")
            if "usage" not in explanation or not isinstance(explanation["usage"], str):
                raise ValueError("Explanation must contain a 'usage' string")
            if "examples" not in explanation or not isinstance(explanation["examples"], list):
                raise ValueError("Explanation must contain an 'examples' list")
            if "metadata" not in data or not isinstance(data["metadata"], dict):
                raise ValueError("Response must contain a 'metadata' object")
            metadata = data["metadata"]
            if "language" not in metadata or not isinstance(metadata["language"], str):
                raise ValueError("Metadata must contain a 'language' string")
            if "complexity" not in metadata or not isinstance(metadata["complexity"], str):
                raise ValueError("Metadata must contain a 'complexity' string")
            if "quality_score" not in metadata or not isinstance(metadata["quality_score"], (int, float)):
                raise ValueError("Metadata must contain a 'quality_score' number")
            if "test_coverage" not in metadata or not isinstance(metadata["test_coverage"], str):
                raise ValueError("Metadata must contain a 'test_coverage' string")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response as JSON: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    def _format_research_prompt(self, research: Dict[str, Any]) -> str:
        """Format research data into a string prompt."""
        prompt = "Based on the following research, write Python code to implement the solution:\n\n"
        
        # Add research findings
        if "research" in research and "findings" in research["research"]:
            prompt += "Research Findings:\n"
            for finding in research["research"]["findings"]:
                prompt += f"- Query: {finding['query']}\n"
                prompt += f"  Content: {finding['content']}\n"
                prompt += f"  Sources: {', '.join(finding['sources'])}\n\n"
        
        # Add synthesis
        if "research" in research and "synthesis" in research["research"]:
            prompt += f"\nSynthesis:\n{research['research']['synthesis']}\n\n"
        
        # Add key points
        if "research" in research and "key_points" in research["research"]:
            prompt += "Key Points:\n"
            for point in research["research"]["key_points"]:
                prompt += f"- {point}\n"
        
        # Add gaps
        if "research" in research and "gaps" in research["research"]:
            prompt += "\nIdentified Gaps:\n"
            for gap in research["research"]["gaps"]:
                prompt += f"- {gap}\n"
        
        return prompt

    async def execute(self, description: str, research_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the development phase."""
        try:
            # Prepare context from research results
            context = "Research Context:\n"
            if research_context:
                for step_id, research in research_context.items():
                    context += f"\nResearch Step {step_id}:\n"
                    if isinstance(research, dict):
                        if 'findings' in research:
                            context += f"Findings: {research['findings']}\n"
                        if 'key_points' in research:
                            context += f"Key Points: {research['key_points']}\n"
                        if 'gaps' in research:
                            context += f"Gaps: {research['gaps']}\n"
                    else:
                        context += f"Research Data: {research}\n"

            # Prepare messages for LLM
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Task: {description}\n\nContext:\n{context}"
                }
            ]

            # Get response from LLM
            response = await self.llm.chat_completion(messages)
            logger.info(f"Raw response content: {response}")

            # Extract content from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
            else:
                content = str(response)

            # Parse and validate response
            try:
                if isinstance(content, str):
                    # Try to extract JSON from markdown code block
                    json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
                    if json_match:
                        response_data = json.loads(json_match.group(1))
                    else:
                        # Try to find first and last curly braces
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start >= 0 and end > start:
                            response_data = json.loads(content[start:end])
                        else:
                            raise ValueError("No JSON object found in response")
                else:
                    response_data = content

                # Validate response structure
                if not isinstance(response_data, dict):
                    raise ValueError("Response must be a dictionary")

                required_fields = ['code', 'explanation', 'metadata']
                for field in required_fields:
                    if field not in response_data:
                        raise ValueError(f"Response must contain a '{field}' field")

                # Validate code field
                if not isinstance(response_data['code'], dict):
                    raise ValueError("'code' field must be a dictionary")
                if 'implementation' not in response_data['code']:
                    raise ValueError("'code' field must contain an 'implementation' key")

                # Validate explanation field
                if not isinstance(response_data['explanation'], dict):
                    raise ValueError("'explanation' field must be a dictionary")
                if 'overview' not in response_data['explanation']:
                    raise ValueError("'explanation' field must contain an 'overview' key")

                # Validate metadata field
                if not isinstance(response_data['metadata'], dict):
                    raise ValueError("'metadata' field must be a dictionary")

                # Log implementation details
                logger.info(f"Generated implementation with {len(response_data['code']['implementation'].splitlines())} lines")
                logger.info(f"Implementation overview: {response_data['explanation']['overview'][:100]}...")

                return response_data

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                raise ValueError(f"Invalid JSON response: {str(e)}")
            except Exception as e:
                logger.error(f"Error validating response: {str(e)}")
                raise ValueError(f"Invalid response format: {str(e)}")

        except Exception as e:
            logger.error(f"Error in development phase: {str(e)}")
            raise 