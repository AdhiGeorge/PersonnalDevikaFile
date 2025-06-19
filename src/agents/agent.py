"""Main agent class that coordinates all sub-agents."""

import logging
from typing import Dict, Any, List, Optional

from agents.base_agent import BaseAgent
from config.config import Config
from project import ProjectManager
from memory.memory import Memory
from agents.orchestrator import AgentContext
from utils.metrics import AgentMetrics
from utils.error_handler import ErrorHandler
from utils.rate_limiter import RateLimiter
from utils.cache import Cache
from bert.sentence import SentenceBert
from knowledge_base.knowledge_base import KnowledgeBase
from browser.search import SearchEngine
from browser import Browser, start_interaction
from filesystem import ReadCode
from services.terminal_runner import TerminalRunner

import json
import time
import platform
import tiktoken
import asyncio
import re
import os

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class Agent(BaseAgent):
    """Main agent class that coordinates all sub-agents."""
    
    def __init__(self, config: Config, model: str = "gpt-4"):
        """Initialize the agent with all components.
        
        Args:
            config: Configuration instance
            model: Model name to use
        """
        try:
            # Initialize base agent first
            super().__init__(config, model)
            
            # Initialize components lazily
            self._planner = None
            self._researcher = None
            self._coder = None
            self._project_manager = None
            self._knowledge_base = None
            self._search_engine = None
            
            logging.info("Agent initialized with all components")
        except Exception as e:
            error_msg = f"Failed to initialize components: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    async def initialize(self) -> bool:
        """Initialize the agent and its components.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Initialize base components
            await super().initialize()
            
            # Initialize components lazily
            self._initialized = True
            return True
            
        except Exception as e:
            self._initialized = False
            raise ValueError(f"Async initialization failed: {str(e)}")
            
    @property
    def planner(self):
        """Get the planner component."""
        if not self._planner:
            from .planner import Planner
            self._planner = Planner(config=self.config, model=self.model)
            # Initialize planner
            asyncio.create_task(self._planner.initialize())
        return self._planner
        
    @property
    def researcher(self):
        """Get the researcher component."""
        if not self._researcher:
            from .researcher import Researcher
            self._researcher = Researcher(config=self.config, model=self.model)
            # Initialize researcher
            asyncio.create_task(self._researcher.initialize())
        return self._researcher
        
    @property
    def coder(self):
        """Get the coder component."""
        if not self._coder:
            from .coder import Coder
            self._coder = Coder(config=self.config, model=self.model)
            # Initialize coder
            asyncio.create_task(self._coder.initialize())
        return self._coder
        
    @property
    def project_manager(self):
        """Get the project manager component."""
        if not self._project_manager:
            self._project_manager = ProjectManager()
        return self._project_manager
        
    @property
    def knowledge_base(self):
        """Get the knowledge base component."""
        if not self._knowledge_base:
            self._knowledge_base = KnowledgeBase()
        return self._knowledge_base
        
    @property
    def search_engine(self):
        """Get the search engine component."""
        if not self._search_engine:
            self._search_engine = SearchEngine()
        return self._search_engine

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the agent system."""
        try:
            if not self._initialized:
                raise RuntimeError("Agent not initialized. Call initialize() first.")
                
            logging.info("Starting planning phase...")
            plan = await self.planner.plan(query)
            
            # Validate plan structure
            if not isinstance(plan, dict):
                raise ValueError("Plan must be a dictionary")
                
            steps = plan.get("steps", [])
            if not isinstance(steps, list):
                raise ValueError("Plan must contain a 'steps' array")
                
            final_answer = plan.get("final_answer", {})
            if not isinstance(final_answer, dict):
                raise ValueError("Plan must contain a 'final_answer' object")
                
            # Process each step
            research_results = {}
            code_results = {}
            
            for step in steps:
                if not isinstance(step, dict):
                    raise ValueError(f"Invalid step structure: {step}")
                    
                step_id = step.get("id")
                agent_type = step.get("agent")
                
                if not step_id or not agent_type:
                    raise ValueError(f"Step must contain 'id' and 'agent' fields: {step}")
                    
                if agent_type not in ["researcher", "developer", "answer"]:
                    raise ValueError(f"Invalid agent type: {agent_type}")
                
                if agent_type == "researcher":
                    # Process research queries
                    queries = step.get("queries", [])
                    for query in queries:
                        # Search knowledge base first
                        kb_results = await self.researcher.search_knowledge_base(query)
                        if not kb_results or len(kb_results) < 3:
                            logging.info(f"Insufficient knowledge base results for: {query}")
                            # Perform web search
                            search_results = await self.researcher.search_web(query)
                            if search_results:
                                # Store results in knowledge base
                                await self.researcher.store_research_results(search_results, query)
                                research_results[query] = search_results
                            else:
                                research_results[query] = []
                        else:
                            research_results[query] = kb_results
                            
                    # Synthesize research results
                    if research_results:
                        synthesis = await self.researcher.synthesize_research(query, research_results)
                        research_results["synthesis"] = synthesis
                        
                elif agent_type == "developer":
                    # Generate code based on research
                    if "synthesis" in research_results:
                        code_result = await self.coder.implement(
                            query=query,
                            plan=plan,
                            research=research_results["synthesis"]
                        )
                        code_results[step_id] = code_result
                        
            # Generate final answer
            final_answer = await self.researcher.generate_final_answer(
                query=query,
                research=research_results.get("synthesis", ""),
                code=code_results,
                plan=plan
            )
            
            return {
                "query": query,
                "plan": plan,
                "research": research_results,
                "code": code_results,
                "final_answer": final_answer
            }
            
        except Exception as e:
            logging.error(f"Query processing failed: {str(e)}")
            raise ValueError(f"Query processing failed: {str(e)}")

    async def _save_code(self, code: str, filepath: str) -> None:
        """Save generated code to a file."""
        try:
            if not code or not isinstance(code, str):
                raise ValueError("Invalid code content")
                
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            # Validate file path
            if not os.path.isabs(filepath):
                filepath = os.path.abspath(filepath)
                
            # Check if file is writable
            if os.path.exists(filepath) and not os.access(filepath, os.W_OK):
                raise PermissionError(f"No write permission for file: {filepath}")
                
            # Write code to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
                
            logger.info(f"Code saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving code: {str(e)}")
            raise

    async def _save_response(self, response: str, filepath: str) -> None:
        """Save response to a text file."""
        try:
            if not response or not isinstance(response, str):
                raise ValueError("Invalid response content")
                
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            # Validate file path
            if not os.path.isabs(filepath):
                filepath = os.path.abspath(filepath)
                
            # Check if file is writable
            if os.path.exists(filepath) and not os.access(filepath, os.W_OK):
                raise PermissionError(f"No write permission for file: {filepath}")
                
            # Write response to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response)
                
            logger.info(f"Response saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving response: {str(e)}")
            raise

    def _determine_file_extension(self, code: str) -> str:
        """Determine the appropriate file extension based on code content."""
        try:
            if not code or not isinstance(code, str):
                raise ValueError("Invalid code content")
                
            # Simple heuristic to determine language
            code_lower = code.lower()
            
            if "def " in code_lower or "import " in code_lower or "class " in code_lower:
                return ".py"
            elif "<html" in code_lower or "<!DOCTYPE" in code_lower:
                return ".html"
            elif "function" in code_lower or "const " in code_lower or "let " in code_lower:
                return ".js"
            elif "class" in code_lower and "{" in code_lower:
                return ".java"
            elif "package " in code_lower:
                return ".go"
            elif "fn " in code_lower:
                return ".rs"
            else:
                return ".txt"  # Default to .txt if language can't be determined
                
        except Exception as e:
            logger.error(f"Error determining file extension: {str(e)}")
            return ".txt"  # Default to .txt on error

    async def run_code(self, code_file: str) -> Dict[str, Any]:
        """Run the generated code and return results."""
        try:
            if not os.path.exists(code_file):
                raise FileNotFoundError(f"Code file not found: {code_file}")
                
            # Determine how to run the code based on file extension
            extension = os.path.splitext(code_file)[1].lower()
            
            if extension == '.py':
                result = await self._run_python_code(code_file)
            elif extension == '.js':
                result = await self._run_javascript_code(code_file)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error running code: {str(e)}")
            raise

    async def _run_python_code(self, filepath: str) -> Dict[str, Any]:
        """Run Python code and return results."""
        try:
            process = await asyncio.create_subprocess_exec(
                'python', filepath,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
                "exit_code": process.returncode
            }
            
        except Exception as e:
            logger.error(f"Error running Python code: {str(e)}")
            raise

    async def _run_javascript_code(self, filepath: str) -> Dict[str, Any]:
        """Run JavaScript code and return results."""
        try:
            process = await asyncio.create_subprocess_exec(
                'node', filepath,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
                "exit_code": process.returncode
            }
            
        except Exception as e:
            logger.error(f"Error running JavaScript code: {str(e)}")
            raise

    async def process_query_with_context(self, query: str, context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process a user query with context."""
        try:
            if not isinstance(query, str):
                raise ValueError("query must be a string")
            if not isinstance(context, list):
                raise ValueError("context must be a list")

            # Create task
            task = {
                "query": query,
                "prompt": query,
                "context": context
            }
            
            # Plan the response with context
            plan_result = await self.planner.execute_with_context(task, context)
            plan = plan_result.get("response", "")
            
            # Research the query with context
            research_result = await self.researcher.execute_with_context(task, context)
            research = research_result.get("response", "")
            
            # Combine results
            return {
                "query": query,
                "plan": plan,
                "research": research,
                "context": context
            }
        except Exception as e:
            logger.error(f"Error processing query with context: {str(e)}")
            raise ValueError(f"Query processing failed: {str(e)}")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task."""
        try:
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")

            query = str(task.get("query", ""))
            return await self.process_query(query)
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            raise ValueError(f"Task execution failed: {str(e)}")

    async def execute_with_context(self, task: Dict[str, Any], context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute a task with context."""
        try:
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")
            if not isinstance(context, list):
                raise ValueError("context must be a list")

            query = str(task.get("query", ""))
            return await self.process_query_with_context(query, context)
        except Exception as e:
            logger.error(f"Error executing task with context: {str(e)}")
            raise ValueError(f"Task execution with context failed: {str(e)}")

    async def handle_feedback(self, feedback: str, project_name: str) -> Dict[str, Any]:
        """Handle user feedback and make necessary adjustments."""
        try:
            # Get the current state
            current_state = self.project_manager.get_project_state(project_name)
            
            # If feedback indicates code needs to be fixed or enhanced
            if any(keyword in feedback.lower() for keyword in ["fix", "error", "bug", "issue", "problem"]):
                # Get the current code
                current_code = current_state.get("code", "")
                
                # Ask developer to fix the code
                fixed_code = await self.developer.execute(
                    research={"feedback": feedback, "current_code": current_code},
                    project_name=project_name
                )
                
                # Update the answer with the fixed code
                answer = await self.answer.execute(
                    prompt=f"Based on the following feedback and fixed code, provide an updated answer:\n\nFeedback: {feedback}\n\nCode: {json.dumps(fixed_code)}",
                    project_name=project_name
                )
                
                return {
                    "answer": answer["answer"]["summary"],
                    "explanation": answer["answer"]["implementation_details"],
                    "key_points": answer["answer"]["key_points"],
                    "code": answer["code"]["implementation"],
                    "dependencies": answer["code"]["dependencies"],
                    "requirements": answer["code"]["requirements"],
                    "setup_instructions": answer["code"]["setup_instructions"],
                    "metadata": {
                        "sources": answer["metadata"]["sources"],
                        "confidence": answer["metadata"]["confidence"],
                        "coverage": answer["metadata"]["coverage"]
                    }
                }
            
            # If feedback indicates new features are needed
            elif any(keyword in feedback.lower() for keyword in ["add", "feature", "enhance", "improve"]):
                # Get the current code
                current_code = current_state.get("code", "")
                
                # Ask developer to add new features
                enhanced_code = await self.developer.execute(
                    research={"feedback": feedback, "current_code": current_code},
                    project_name=project_name
                )
                
                # Update the answer with the enhanced code
                answer = await self.answer.execute(
                    prompt=f"Based on the following feedback and enhanced code, provide an updated answer:\n\nFeedback: {feedback}\n\nCode: {json.dumps(enhanced_code)}",
                    project_name=project_name
                )
                
                return {
                    "answer": answer["answer"]["summary"],
                    "explanation": answer["answer"]["implementation_details"],
                    "key_points": answer["answer"]["key_points"],
                    "code": answer["code"]["implementation"],
                    "dependencies": answer["code"]["dependencies"],
                    "requirements": answer["code"]["requirements"],
                    "setup_instructions": answer["code"]["setup_instructions"],
                    "metadata": {
                        "sources": answer["metadata"]["sources"],
                        "confidence": answer["metadata"]["confidence"],
                        "coverage": answer["metadata"]["coverage"]
                    }
                }
            
            # If feedback is just a question or clarification
            else:
                # Update the answer based on the feedback
                answer = await self.answer.execute(
                    prompt=f"Based on the following feedback, provide an updated answer:\n\nFeedback: {feedback}",
                    project_name=project_name
                )
                
                return {
                    "answer": answer["answer"]["summary"],
                    "explanation": answer["answer"]["implementation_details"],
                    "key_points": answer["answer"]["key_points"],
                    "code": current_state.get("code", ""),
                    "dependencies": current_state.get("dependencies", []),
                    "requirements": current_state.get("requirements", ""),
                    "setup_instructions": current_state.get("setup_instructions", ""),
                    "metadata": {
                        "sources": answer["metadata"]["sources"],
                        "confidence": answer["metadata"]["confidence"],
                        "coverage": answer["metadata"]["coverage"]
                    }
                }
            
        except Exception as e:
            logger.error(f"Error handling feedback: {str(e)}")
            raise

    async def run(self, prompt: str) -> Dict[str, Any]:
        """Run the agent with the given prompt."""
        try:
            logger.info(f"Starting agent execution with prompt: {prompt}")
            result = await self.execute({"query": prompt})
            logger.info("Agent execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            raise

    async def search_queries(self, queries: list, project_name: str) -> dict:
        """Search queries using the search engine."""
        with tracer.start_as_current_span("agent_search_queries") as span:
            span.set_attribute("queries", json.dumps(queries))
            results = {}
            knowledge_base = KnowledgeBase()
            web_search = SearchEngine(self.config)  # Pass config to SearchEngine
            self.logger.info(f"\nSearch Engine initialized with Google API: {bool(web_search.google_api_key)}, Tavily API: {bool(web_search.tavily_api_key)}")

            for query in queries:
                query = query.strip().lower()
                self.logger.info(f"\nSearching for: {query}")

                # Perform search
                search_results = web_search.search(query, max_results=3)
                if not search_results:
                    self.logger.warning(f"No results found for query: {query}")
                    continue

                # Process each result
                for result in search_results:
                    link = result.get("link")
                    if not link:
                        continue

                    self.logger.info(f"\nProcessing result: {result.get('title')}")
                    self.logger.info(f"URL: {link}")

                    try:
                        browser, raw, data = await self.open_page(project_name, link)
                        if data:
                            results[query] = data
                            self.logger.info(f"Successfully processed result for query: {query}")
                        else:
                            self.logger.warning(f"No data extracted from URL: {link}")
                    except Exception as e:
                        self.logger.error(f"Error processing URL {link}: {str(e)}")
                        continue

            span.set_status(Status(StatusCode.OK))
            return results

    def update_contextual_keywords(self, sentence: str):
        """Update contextual keywords from a sentence."""
        with tracer.start_as_current_span("update_contextual_keywords") as span:
            span.set_attribute("sentence", sentence)
            keywords = SentenceBert(sentence).extract_keywords()
            for keyword in keywords:
                self.collected_context_keywords.append(keyword[0])
            span.set_status(Status(StatusCode.OK))
            return self.collected_context_keywords

    async def open_page(self, project_name, url):
        """Open a web page and get its content."""
        try:
            browser, raw, data = await start_interaction(url) # This needs to be awaited
            if data:
                self.project_manager.add_document(project_name, url, data)
                return browser, raw, data
            return browser, None, None
        except Exception as e:
            self.logger.error(f"Error opening page {url}: {str(e)}")
            return None, None, None

    def _execute_last_snippet(self, project_name: str):
        """Execute the last code snippet in the project."""
        try:
            # Get the last code snippet
            code = self._extract_code_block(
                self.project_manager.get_last_agent_message(project_name)
            )
            if not code:
                return

            # Execute the code
            runner = TerminalRunner()
            runner.run_code(code, project_name)
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            raise

    def _extract_code_block(self, text: str) -> str:
        """Extract code from a markdown code block."""
        if not text:
            return ""
        
        # Look for code blocks
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        return ""