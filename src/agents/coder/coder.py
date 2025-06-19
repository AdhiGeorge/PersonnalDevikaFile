import os
import time
import json
from typing import List, Dict, Any
import sys
from functools import wraps
import logging
import asyncio
import tempfile
import re

from agents.base_agent import BaseAgent
from llm.llm import LLM
from config.config import Config
from project import ProjectManager
from state import AgentState
from prompts.prompt_manager import PromptManager
from services.terminal_runner import TerminalRunner
from utils.logger import Logger
from knowledge_base.knowledge_base import KnowledgeBase

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

class Coder(BaseAgent):
    """Coder agent for generating and modifying code."""
    
    def __init__(self, config: Config, model: str = "gpt-4o"):
        """Initialize the coder agent with configuration.
        
        Args:
            config: Configuration object
            model: The LLM model to use for code generation
        """
        try:
            # Initialize base agent first
            super().__init__(config, model)
            
            # Initialize components that don't require async
            self._initialized = False
            self._coder_prompt_manager = None
            
            # Code generation state
            self.current_file = None
            self.current_context = {}
            self.code_cache = {}
            
            # Configuration with defaults
            self.max_retries = 3
            self.max_tokens = 4000
            
            # Set default prompts
            self.code_generation_prompt = None
            self.code_review_prompt = None
            
            logging.info("Coder initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize coder: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    async def initialize(self):
        """Initialize async components and validate configuration."""
        try:
            # Skip if already initialized
            if self._initialized:
                return True
                
            # Initialize base agent async components first
            await super().initialize()
            
            # Ensure base components are initialized
            if not hasattr(self, 'logger') or not hasattr(self, 'llm') or not hasattr(self, 'prompt_manager'):
                raise RuntimeError("Base agent components not properly initialized")
            
            # Initialize prompt manager
            try:
                # Use the base agent's prompt manager
                self._coder_prompt_manager = self.prompt_manager
                
                # Load required prompts with error handling
                try:
                    self.code_generation_prompt = self._coder_prompt_manager.get_prompt("code_generation")
                    self.code_review_prompt = self._coder_prompt_manager.get_prompt("code_review")
                except Exception as e:
                    self.logger.warning(f"Failed to load some prompts: {str(e)}")
                
                # Set default prompts if not found
                if not self.code_generation_prompt:
                    self.code_generation_prompt = """Generate code based on the following requirements:
                    {requirements}
                    
                    Instructions:
                    1. Write clean, efficient, and well-documented code
                    2. Include necessary imports and setup
                    3. Add error handling where appropriate
                    4. Follow best practices for the language"""
                    
                if not self.code_review_prompt:
                    self.code_review_prompt = """Review the following code for quality, efficiency, and correctness:
                    
                    {code}
                    
                    Provide feedback on:
                    1. Code quality and readability
                    2. Potential bugs or edge cases
                    3. Performance optimizations
                    4. Security considerations"""
                
                self.logger.info("Coder components initialized successfully")
                self._initialized = True
                return True
                
            except Exception as e:
                error_msg = f"Failed to initialize coder prompts: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"Failed to initialize coder: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg, exc_info=True)
            else:
                logging.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    async def implement(self, query: str, plan: str, research: str) -> str:
        """Implement code based on query, plan and research.
        
        Args:
            query: The original query
            plan: The implementation plan
            research: Research findings
            
        Returns:
            str: Generated code
        """
        try:
            # Ensure initialized
            if not self._initialized:
                await self.initialize()
                
            self.logger.info("Starting code generation...")
            
            # Store current requirements
            self.current_requirements = query
            
            # Get the code generation prompt
            code_generation_prompt = self.prompt_manager.get_prompt('code_generation')
            
            # Format the prompt with requirements, context, and plan
            system_prompt = code_generation_prompt.format(
                requirements=query,
                context=research,
                plan=plan
            )
            
            # Generate code using LLM
            self.logger.debug(f"Sending request to LLM with prompt: {system_prompt[:200]}...")
            response = await self.llm.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please generate code for: {query}"}
                ]
            )
            
            if not response:
                raise ValueError("Empty response from LLM")
                
            # Extract the generated code
            generated_code = response.choices[0].message.content.strip()
            
            # Clean up the response
            if '```' in generated_code:
                # Extract code from markdown code blocks
                import re
                code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', generated_code, re.DOTALL)
                if code_blocks:
                    generated_code = code_blocks[0].strip()
            
            self.logger.info("Code generation completed")
            
            # Review and improve code
            improved_code = await self._review_code(generated_code)
            
            return improved_code
            
        except Exception as e:
            import traceback
            error_msg = f"Error in code generation: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def _review_code(self, code: str) -> str:
        """Review and improve generated code.
        
        Args:
            code: The code to review
            
        Returns:
            str: Improved code
        """
        try:
            self.logger.info("Starting code review...")
            
            # Get the code review prompt
            code_review_prompt = self.prompt_manager.get_prompt('code_review')
            
            # Format the prompt with the code and requirements
            system_prompt = code_review_prompt.format(
                code=code,
                requirements=self.current_requirements
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
                import re
                code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', improved_code, re.DOTALL)
                if code_blocks:
                    improved_code = code_blocks[0].strip()
            
            self.logger.info("Code review completed")
            
            return improved_code
            
        except Exception as e:
            import traceback
            error_msg = f"Error in code review: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _detect_language(self, code: str) -> str:
        """Detect the programming language of the code."""
        try:
            # Check for language-specific patterns
            if re.search(r'^(def|class|import|from|if __name__ == "__main__")', code, re.MULTILINE):
                return 'python'
            elif re.search(r'^(function|const|let|var|class|import|export)', code, re.MULTILINE):
                return 'javascript'
            elif re.search(r'^(<!DOCTYPE|<!doctype|<html|<head|<body)', code, re.MULTILINE):
                return 'html'
            elif re.search(r'^(package|import|class|public|private|protected)', code, re.MULTILINE):
                return 'java'
            elif re.search(r'^(#include|int main|void main|class|namespace)', code, re.MULTILINE):
                return 'cpp'
            elif re.search(r'^(package|import|func|type|interface)', code, re.MULTILINE):
                return 'go'
            elif re.search(r'^(fn|struct|impl|trait|use|mod)', code, re.MULTILINE):
                return 'rust'
            elif re.search(r'^(<?php|function|class|namespace)', code, re.MULTILINE):
                return 'php'
            elif re.search(r'^(import|class|interface|type|enum)', code, re.MULTILINE):
                return 'typescript'
            
            # Default to Python if no specific patterns found
            return 'python'
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return 'python'  # Default to Python on error

    def _determine_file_path(self, query: str) -> str:
        """Determine the appropriate file path for the code"""
        # Simple file path determination logic
        base_name = query.lower().replace(' ', '_')
        if self._detect_language(query) == 'python':
            return f"{base_name}.py"
        elif self._detect_language(query) == 'javascript':
            return f"{base_name}.js"
        return f"{base_name}.txt"

    def format_prompt(self, step_by_step_plan: str, user_context: str, search_results: dict) -> str:
        """Format the coder prompt with the task and context."""
        prompt_template = self.get_prompt("coder")
        if not prompt_template:
            raise ValueError("Coder prompt not found in prompts.yaml")
        return super().format_prompt(prompt_template, step_by_step_plan=step_by_step_plan, user_context=user_context, search_results=search_results)

    def render(
        self, step_by_step_plan: str, user_context: str, search_results: dict
    ) -> str:
        return self.format_prompt(step_by_step_plan, user_context, search_results)

    @validate_responses
    def validate_response(self, response: str):
        """Validate the response from the LLM."""
        try:
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

    async def emulate_code_writing(self, code_set: list, project_name: str):
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
            await asyncio.sleep(2) # Use asyncio.sleep for async functions
        # Removed emit_agent call as it is frontend related

    @retry_wrapper
    async def execute(self, query: str, plan: str, research: str) -> str:
        """Execute the coder agent"""
        self.logger.info(f"Executing coder agent for query: {query}")
        
        # Generate code
        code = await self.implement(query, plan, research)
        
        # Validate the response
        if not await self.validate_response(code):
            raise ValueError("Invalid response from LLM")
        
        # Run the code
        results = await self.run_code(code)
        
        # Log the results
        self.logger.set_execution_results(results)
        
        return code

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

    async def run_code(self, code: str) -> dict:
        """Run the generated code and return results."""
        try:
            if not code:
                raise ValueError("No code to run")
                
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=self._get_file_extension(code), delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Run the code
                results = await self.terminal_runner.run_code(temp_file)
                
                # Log execution results
                self.logger.set_execution_results({
                    'success': True,
                    'output': results.output,
                    'error': results.error,
                    'exit_code': results.exit_code
                })
                
                return {
                    'success': True,
                    'output': results.output,
                    'error': results.error,
                    'exit_code': results.exit_code
                }
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error running code: {error_msg}")
                self.logger.set_execution_results({
                    'success': False,
                    'error': error_msg
                })
                
                return {
                    'success': False,
                    'error': error_msg
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {str(e)}")
                    
        except Exception as e:
            error_msg = f"Failed to run code: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def _get_file_extension(self, code: str) -> str:
        """Get the appropriate file extension for the code."""
        language = self._detect_language(code)
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'html': '.html',
            'css': '.css',
            'java': '.java',
            'c': '.c',
            'cpp': '.cpp',
            'go': '.go',
            'rust': '.rs',
            'ruby': '.rb',
            'php': '.php',
            'swift': '.swift',
            'kotlin': '.kt',
            'typescript': '.ts'
        }
        return extensions.get(language, '.txt')

    async def generate_code(self, task=None, requirements=None, context=None, query=None):
        """
        Generate code based on the workflow's expectations.
        Accepts either (task, requirements, context) or (query, context) for backward compatibility.
        Returns a dict with at least 'code' and optionally 'explanation'.
        """
        self.logger.info(f"Generating code for task: {task or query}")
        try:
            # Determine the main prompt and requirements
            if requirements is None and query is not None:
                requirements = [query]
            if isinstance(requirements, list):
                requirements_str = '\n'.join(requirements)
            else:
                requirements_str = str(requirements) if requirements else ''
            # Compose the prompt
            prompt_template = await self.prompt_manager.get_prompt("code_generation")
            # Only use keys that exist in the template
            prompt_kwargs = {}
            if '{task}' in prompt_template:
                prompt_kwargs['task'] = task or query or ''
            if '{requirements}' in prompt_template:
                prompt_kwargs['requirements'] = requirements_str
            if '{context}' in prompt_template:
                prompt_kwargs['context'] = json.dumps(context) if context else ''
            prompt = prompt_template.format(**prompt_kwargs)
            response = await self.llm.generate(prompt)
            code = self._extract_code(response)
            explanation = None
            # Try to extract an explanation if present
            if isinstance(response, str):
                # Look for an explanation after the code block
                parts = re.split(r"```[a-zA-Z]*\n.*?```", response, flags=re.DOTALL)
                if len(parts) > 1:
                    explanation = parts[-1].strip()
            await self.state.update({"generated_code": code})
            self.logger.info(f"Code generated successfully for task: {task or query}")
            return {"code": code, "explanation": explanation or ""}
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            raise

    def _extract_code(self, response):
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()
