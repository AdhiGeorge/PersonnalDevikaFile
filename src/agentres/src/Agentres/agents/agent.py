from .planner import Planner
from .researcher import Researcher
from .formatter import Formatter
from .coder import Coder
from .answer import Answer
from .runner import Runner
from .feature import Feature
from .patcher import Patcher
from .reporter import Reporter
from .developer import Developer
from .action import Action

from Agentres.project import ProjectManager
from Agentres.state import AgentState
from Agentres.logger import Logger

from Agentres.bert.sentence import SentenceBert
from Agentres.agents.researcher.knowledge_base import KnowledgeBase
from Agentres.browser.search import SearchEngine
from Agentres.browser import Browser
from Agentres.browser import start_interaction
from Agentres.filesystem import ReadCode

import json
import time
import platform
import tiktoken
import asyncio
import logging
import re
from Agentres.services.terminal_runner import TerminalRunner
from prometheus_client import Counter, Histogram, start_http_server
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from Agentres.llm.llm import LLM
from Agentres.utils.token_tracker import TokenTracker
from Agentres.config import Config
from Agentres.agents.base_agent import BaseAgent

from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Prometheus metrics
LLM_CALLS = Counter('llm_calls_total', 'Total number of LLM calls', ['model'])
SEARCH_CALLS = Counter('search_calls_total', 'Total number of search calls', ['engine'])
LLM_LATENCY = Histogram('llm_latency_seconds', 'LLM call latency in seconds', ['model'])
SEARCH_LATENCY = Histogram('search_latency_seconds', 'Search call latency in seconds', ['engine'])

class Agent(BaseAgent):
    """Main agent class that coordinates all sub-agents."""
    
    def __init__(self, config: Config):
        """Initialize the agent with configuration."""
        super().__init__(config)
        
        self.config = config
        self.logger = Logger()
        
        # Initialize sub-agents
        self.planner = Planner(config)
        self.researcher = Researcher(config)
        self.coder = Coder(config)
        self.patcher = Patcher(config)
        self.action = Action(config)
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase()
        
        # Initialize project manager
        self.project_manager = ProjectManager()
        
        logger.info("Agent initialized with all components")

    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute the agent's workflow."""
        try:
            # Plan
            self.logger.info("Starting planning phase...")
            plan = await self.planner.plan(query)
            self.logger.set_planner_output(str(plan))
            
            # Research
            self.logger.info("Starting research phase...")
            research_data = await self.researcher.research(query, plan)
            self.logger.set_researcher_output(str(research_data))
            
            # Code
            self.logger.info("Starting coding phase...")
            code = await self.coder.code(query, plan, research_data)
            self.logger.set_generated_code(str(code))
            
            # Patch
            self.logger.info("Starting patching phase...")
            patched_code = await self.patcher.patch(code)
            self.logger.set_generated_code(str(patched_code), language="python")
            
            # Execute
            self.logger.info("Starting execution phase...")
            result = await self.action.execute(patched_code)
            self.logger.set_execution_results(result)
            
            return {
                "answer": plan,
                "code": patched_code,
                "metadata": {
                    "research_data": research_data,
                    "execution_results": result
                }
            }
        except Exception as e:
            self.logger.error(f"Error in agent execution: {str(e)}")
            raise

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
            result = await self.execute(prompt)
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