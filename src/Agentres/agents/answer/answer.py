import json
import sys
import time
from functools import wraps
import logging
from typing import Any, Dict, List, Optional, Union
from Agentres.agents.base_agent import BaseAgent
from Agentres.llm.llm import LLM
from Agentres.utils.retry import retry_wrapper
from Agentres.config import Config
import re

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

class Answer(BaseAgent):
    """Answer agent responsible for generating final responses."""
    
    def __init__(self, config: Config):
        """Initialize the answer agent."""
        super().__init__(config)
        self.llm = LLM(config)
        self.system_prompt = """You are an answer agent that synthesizes research findings and code into a clear, comprehensive response.
Your response must be a valid JSON object with the following structure:
{
    "answer": {
        "summary": "A clear, concise summary of the solution",
        "key_points": ["point1", "point2"],
        "implementation_details": "Detailed explanation of how the solution works"
    },
    "code": {
        "implementation": "The complete, working code",
        "dependencies": ["package1", "package2"],
        "requirements": "requirements.txt content",
        "setup_instructions": "How to set up and run the code"
    },
    "metadata": {
        "sources": ["source1", "source2"],
        "confidence": 0.95,
        "coverage": "How well the solution addresses the requirements"
    }
}"""

    def validate_response(self, response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and parse the response from the LLM."""
        try:
            # If response is already a dictionary, validate its structure
            if isinstance(response, dict):
                if "answer" not in response or not isinstance(response["answer"], str):
                    raise ValueError("Response must contain an 'answer' string")
                if "code" not in response or not isinstance(response["code"], str):
                    raise ValueError("Response must contain a 'code' string")
                if "metadata" not in response or not isinstance(response["metadata"], dict):
                    raise ValueError("Response must contain a 'metadata' dictionary")
                
                # Validate metadata fields
                metadata = response["metadata"]
                if "sources" not in metadata or not isinstance(metadata["sources"], list):
                    raise ValueError("Metadata must contain a 'sources' list")
                if "confidence" not in metadata or not isinstance(metadata["confidence"], (int, float)):
                    raise ValueError("Metadata must contain a numeric 'confidence' score")
                if "explanation" not in metadata or not isinstance(metadata["explanation"], str):
                    raise ValueError("Metadata must contain an 'explanation' string")
                
                return response

            # If response is a string, try to parse it as JSON
            try:
                # Try to extract JSON from markdown code block
                json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response)
                if json_match:
                    response = json_match.group(1)
                
                # Parse the response as JSON
                data = json.loads(response)
                
                # Validate the parsed data
                if not isinstance(data, dict):
                    raise ValueError("Response must be a JSON object")
                
                if "answer" not in data or not isinstance(data["answer"], str):
                    raise ValueError("Response must contain an 'answer' string")
                if "code" not in data or not isinstance(data["code"], str):
                    raise ValueError("Response must contain a 'code' string")
                if "metadata" not in data or not isinstance(data["metadata"], dict):
                    raise ValueError("Response must contain a 'metadata' dictionary")
                
                # Validate metadata fields
                metadata = data["metadata"]
                if "sources" not in metadata or not isinstance(metadata["sources"], list):
                    raise ValueError("Metadata must contain a 'sources' list")
                if "confidence" not in metadata or not isinstance(metadata["confidence"], (int, float)):
                    raise ValueError("Metadata must contain a numeric 'confidence' score")
                if "explanation" not in metadata or not isinstance(metadata["explanation"], str):
                    raise ValueError("Metadata must contain an 'explanation' string")
                
                return data

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}")

        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    @retry_wrapper
    async def execute(self, prompt: str, research_results: Dict[str, Any], development_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the answer generation phase."""
        try:
            # Prepare context from research and development results
            context = "Research Context:\n"
            for step_id, research in research_results.items():
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

            context += "\nDevelopment Results:\n"
            for step_id, development in development_results.items():
                context += f"\nDevelopment Step {step_id}:\n"
                if isinstance(development, dict):
                    if 'code' in development:
                        context += f"Code: {development['code']['implementation']}\n"
                    if 'explanation' in development:
                        context += f"Explanation: {development['explanation']['overview']}\n"
                else:
                    context += f"Development Data: {development}\n"

            # Prepare messages for LLM
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Task: {prompt}\n\nContext:\n{context}"
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

                required_fields = ['answer', 'code', 'metadata']
                for field in required_fields:
                    if field not in response_data:
                        raise ValueError(f"Response must contain a '{field}' field")

                # Convert structured answer to string if needed
                if isinstance(response_data['answer'], dict):
                    answer_parts = []
                    if 'summary' in response_data['answer']:
                        answer_parts.append(response_data['answer']['summary'])
                    if 'key_points' in response_data['answer']:
                        answer_parts.append("\nKey Points:")
                        for point in response_data['answer']['key_points']:
                            answer_parts.append(f"- {point}")
                    if 'implementation_details' in response_data['answer']:
                        answer_parts.append("\nImplementation Details:")
                        answer_parts.append(response_data['answer']['implementation_details'])
                    response_data['answer'] = "\n\n".join(answer_parts)

                # Validate answer field
                if not isinstance(response_data['answer'], str):
                    raise ValueError("'answer' field must be a string")

                # Validate code field
                if not isinstance(response_data['code'], str):
                    if isinstance(response_data['code'], dict) and 'implementation' in response_data['code']:
                        response_data['code'] = response_data['code']['implementation']
                    else:
                        raise ValueError("'code' field must be a string or contain 'implementation'")

                # Validate metadata field
                if not isinstance(response_data['metadata'], dict):
                    raise ValueError("'metadata' field must be a dictionary")
                if 'sources' not in response_data['metadata']:
                    raise ValueError("'metadata' must contain 'sources' field")
                if 'confidence' not in response_data['metadata']:
                    raise ValueError("'metadata' must contain 'confidence' field")
                
                # Handle explanation/coverage field
                if 'explanation' not in response_data['metadata']:
                    if 'coverage' in response_data['metadata']:
                        response_data['metadata']['explanation'] = response_data['metadata']['coverage']
                    else:
                        raise ValueError("'metadata' must contain either 'explanation' or 'coverage' field")

                # Log answer details
                logger.info(f"Generated answer with {len(response_data['answer'])} characters")
                logger.info(f"Generated code with {len(response_data['code'])} characters")
                logger.info(f"Sources: {response_data['metadata']['sources']}")
                logger.info(f"Confidence: {response_data['metadata']['confidence']}")

                return response_data

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                raise ValueError(f"Invalid JSON response: {str(e)}")
            except Exception as e:
                logger.error(f"Error validating response: {str(e)}")
                raise ValueError(f"Invalid response format: {str(e)}")

        except Exception as e:
            logger.error(f"Error in answer execute: {str(e)}")
            raise
