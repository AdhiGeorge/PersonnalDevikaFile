import json
import sys
import time
from functools import wraps
import logging
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
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

class Formatter(BaseAgent):
    def __init__(self, base_model: str):
        super().__init__(base_model)

    def format_prompt(self, code: str, language: str = "python") -> str:
        """Format the formatter prompt with the code and language."""
        prompt_template = self.get_prompt("formatter")
        if not prompt_template:
            raise ValueError("Formatter prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, code=code, language=language)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                return False
            if "formatted_code" not in data or not isinstance(data["formatted_code"], str):
                return False
            return response
        except json.JSONDecodeError:
            return False

    @retry_wrapper
    async def execute(self, code: str, language: str = "python", project_name: str = "") -> str:
        """Execute the formatter agent."""
        formatted_prompt = self.format_prompt(code, language)
        response = await self.llm.chat_completion([{"role": "user", "content": formatted_prompt}], self.base_model)
        validated_response = self.validate_response(response.choices[0].message.content)
        return self.parse_response(validated_response)

    def parse_response(self, response: str) -> dict:
        """Parse the formatter's response into a structured format."""
        try:
            data = json.loads(response)
            return {
                "formatted_code": data.get("formatted_code", ""),
                "metadata": data.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing formatter response: {str(e)}")
            return {
                "formatted_code": "",
                "metadata": {}
            }

    async def format_research(self, research_results: List[Dict[str, Any]]) -> str:
        """Format research results into a structured response.
        
        Args:
            research_results: List of research results
            
        Returns:
            str: Formatted research response
        """
        try:
            self.logger.info("Formatting research results...")
            
            # Get the research synthesis prompt
            research_synthesis_prompt = self.prompt_manager.get_prompt('research_synthesis')
            
            # Format the prompt with research results
            system_prompt = research_synthesis_prompt.format(
                research_findings=json.dumps(research_results, indent=2)
            )
            
            # Get synthesis from LLM
            response = await self.llm.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please synthesize these research findings."}
                ]
            )
            
            if not response:
                raise ValueError("Empty response from LLM")
                
            # Extract the synthesis
            synthesis = response.choices[0].message.content.strip()
            
            # Format the synthesis with markdown
            formatted_synthesis = self._format_markdown(synthesis)
            
            self.logger.info("Research results formatted successfully")
            
            return formatted_synthesis
            
        except Exception as e:
            import traceback
            error_msg = f"Error formatting research: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def format_code(self, code: str, requirements: str) -> str:
        """Format code with documentation and examples.
        
        Args:
            code: The code to format
            requirements: The original requirements
            
        Returns:
            str: Formatted code
        """
        try:
            self.logger.info("Formatting code...")
            
            # Get the code review prompt
            code_review_prompt = self.prompt_manager.get_prompt('code_review')
            
            # Format the prompt with code and requirements
            system_prompt = code_review_prompt.format(
                code=code,
                requirements=requirements
            )
            
            # Get review from LLM
            response = await self.llm.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please review and improve this code."}
                ]
            )
            
            if not response:
                raise ValueError("Empty response from LLM")
                
            # Extract the improved code
            improved_code = response.choices[0].message.content.strip()
            
            # Clean up the response
            if '```' in improved_code:
                # Extract code from markdown code blocks
                code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', improved_code, re.DOTALL)
                if code_blocks:
                    improved_code = code_blocks[0].strip()
            
            # Format the code with markdown
            formatted_code = self._format_markdown(improved_code)
            
            self.logger.info("Code formatted successfully")
            
            return formatted_code
            
        except Exception as e:
            import traceback
            error_msg = f"Error formatting code: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _format_markdown(self, text: str) -> str:
        """Format text with markdown.
        
        Args:
            text: The text to format
            
        Returns:
            str: Formatted text
        """
        try:
            # Add proper spacing around headers
            text = re.sub(r'(?m)^(#+)(.*?)$', r'\n\1\2\n', text)
            
            # Add proper spacing around lists
            text = re.sub(r'(?m)^([*+-]|\d+\.)(.*?)$', r'\n\1\2', text)
            
            # Add proper spacing around code blocks
            text = re.sub(r'```(.*?)```', r'\n```\1```\n', text, flags=re.DOTALL)
            
            # Add proper spacing around inline code
            text = re.sub(r'`(.*?)`', r' `\1` ', text)
            
            # Clean up extra newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Clean up extra spaces
            text = re.sub(r' {2,}', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            import traceback
            error_msg = f"Error formatting markdown: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return text