import json
from typing import List
import re

from src.llm import LLM
from src.agents.base_agent import BaseAgent
from src.browser.search import SearchEngine
from src.agents.researcher.knowledge_base import KnowledgeBase
import sys
import time
from functools import wraps
import logging
from typing import Any, Dict, Optional
from src.utils.retry import retry_wrapper
from src.config import Config
from src.browser import Browser
import asyncio

logger = logging.getLogger(__name__)

def retry_wrapper(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        max_tries = 5
        tries = 0
        while tries < max_tries:
            try:
                result = await func(*args, **kwargs)
                if result:
                    return result
                logger.warning("Invalid response from the model, trying again...")
            except Exception as e:
                logger.error(f"Error during retryable function execution: {str(e)}")
            tries += 1
            await asyncio.sleep(2) # Use asyncio.sleep for async functions
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

class Researcher(BaseAgent):
    """Researcher agent for gathering and analyzing information."""
    
    def __init__(self, config: Config):
        """Initialize the researcher with configuration."""
        super().__init__(config)
        self.knowledge_base = KnowledgeBase()
        self.search_engine = SearchEngine(config)
        self.llm = LLM(config)
        logger.info("Researcher initialized")
        self.system_prompt = """You are a research agent that gathers and synthesizes information.
Your response must be a valid JSON object with the following structure:
{
    "research": {
        "findings": [
            {
                "query": "The search query used",
                "sources": ["source1", "source2"],
                "content": "Relevant content from the sources",
                "relevance_score": 0.95
            }
        ],
        "synthesis": "A comprehensive synthesis of all findings",
        "key_points": ["point1", "point2"],
        "gaps": ["gap1", "gap2"]
    },
    "metadata": {
        "sources": ["source1", "source2"],
        "confidence": 0.95,
        "coverage": "How well the research covers the topic"
    }
}"""

    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search the web for information."""
        try:
            # Search using the search engine
            results = self.search_engine.search(query)
            if not results:
                return []

            # Process each result
            processed_results = []
            for result in results:
                try:
                    # Get the URL
                    url = result.get("link")  # Changed from "url" to "link" to match new format
                    if not url:
                        continue

                    # Open the page and extract content
                    browser = Browser()
                    browser.start()
                    success = browser.go_to(url)
                    if not success:
                        browser.close()
                        continue
                        
                    content = browser.extract_text()
                    browser.close()

                    # Add to processed results
                    processed_results.append({
                        "url": url,
                        "title": result.get("title", ""),
                        "content": content,
                        "snippet": result.get("snippet", ""),  # Added snippet from search results
                        "relevance_score": result.get("score", 0.0)
                    })

                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
                    continue

            return processed_results

        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []

    async def search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge base for information."""
        try:
            results = self.knowledge_base.search(query)
            if not results:
                return []

            return [{
                "source": "knowledge_base",
                "content": result.get("content", ""),
                "relevance_score": result.get("score", 0.0)
            } for result in results]

        except Exception as e:
            logger.error(f"Error in knowledge base search: {str(e)}")
            return []

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
            if "research" not in data or not isinstance(data["research"], dict):
                raise ValueError("Response must contain a 'research' object")
            
            research = data["research"]
            if "findings" not in research or not isinstance(research["findings"], list):
                raise ValueError("Research must contain a 'findings' list")
            if "synthesis" not in research or not isinstance(research["synthesis"], str):
                raise ValueError("Research must contain a 'synthesis' string")
            if "key_points" not in research or not isinstance(research["key_points"], list):
                raise ValueError("Research must contain a 'key_points' list")
            if "gaps" not in research or not isinstance(research["gaps"], list):
                raise ValueError("Research must contain a 'gaps' list")

            # Check metadata
            if "metadata" not in data or not isinstance(data["metadata"], dict):
                raise ValueError("Response must contain a 'metadata' object")
            
            metadata = data["metadata"]
            if "sources" not in metadata or not isinstance(metadata["sources"], list):
                raise ValueError("Metadata must contain a 'sources' list")
            if "confidence" not in metadata or not isinstance(metadata["confidence"], (int, float)):
                raise ValueError("Metadata must contain a 'confidence' number")
            if "coverage" not in metadata or not isinstance(metadata["coverage"], str):
                raise ValueError("Metadata must contain a 'coverage' string")

            # Log research findings for debugging
            logger.info(f"Research findings: {len(research['findings'])} items")
            logger.info(f"Research synthesis: {research['synthesis'][:200]}...")
            logger.info(f"Key points: {research['key_points']}")
            logger.info(f"Sources: {metadata['sources']}")

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response as JSON: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    @retry_wrapper
    async def execute(self, queries: List[str]) -> Dict[str, Any]:
        """Execute the research phase."""
        try:
            research_findings = []
            sources = []

            # Execute each query
            for query in queries:
                logger.info(f"Executing research query: {query}")
                
                # Search for information
                search_results = self.search_engine.search(query)
                if search_results:
                    sources.extend(search_results)
                    research_findings.extend(search_results)

            # Synthesize findings
            if research_findings:
                synthesis = await self.synthesize_findings(research_findings)
                key_points = await self.extract_key_points(research_findings)
                gaps = await self.identify_gaps(research_findings)
            else:
                synthesis = "No relevant findings found."
                key_points = []
                gaps = ["No research data available"]

            # Prepare response
            response = {
                "research": {
                    "findings": research_findings,
                    "synthesis": synthesis,
                    "key_points": key_points,
                    "gaps": gaps
                },
                "metadata": {
                    "sources": sources,
                    "confidence": 0.8 if research_findings else 0.5,
                    "coverage": "comprehensive" if research_findings else "limited"
                }
            }

            # Log research results
            logger.info(f"Research completed with {len(research_findings)} findings")
            logger.info(f"Key points: {key_points}")
            logger.info(f"Gaps identified: {gaps}")

            return response

        except Exception as e:
            logger.error(f"Error in research phase: {str(e)}")
            raise 

    async def synthesize_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Synthesize research findings into a coherent summary."""
        try:
            # Prepare findings for synthesis
            research_data = {
                "findings": findings,
                "total_findings": len(findings)
            }

            # Get synthesis from LLM
            messages = [
                {"role": "system", "content": "You are a research synthesis expert. Your task is to synthesize research findings into a clear, coherent summary."},
                {"role": "user", "content": f"Please synthesize the following research findings into a clear summary:\n\n{json.dumps(research_data)}"}
            ]

            response = await self.llm.chat_completion(messages)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error synthesizing findings: {str(e)}")
            return "Failed to synthesize findings."

    async def extract_key_points(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from research findings."""
        try:
            # Prepare findings for key point extraction
            research_data = {
                "findings": findings,
                "total_findings": len(findings)
            }

            # Get key points from LLM
            messages = [
                {"role": "system", "content": "You are a research analyst. Your task is to extract the most important key points from research findings."},
                {"role": "user", "content": f"Please extract the key points from the following research findings:\n\n{json.dumps(research_data)}"}
            ]

            response = await self.llm.chat_completion(messages)
            content = response.choices[0].message.content
            
            # Parse the response into a list of key points
            try:
                # Try to parse as JSON first
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "key_points" in data:
                    return data["key_points"]
            except json.JSONDecodeError:
                # If not JSON, split by newlines or bullet points
                points = [line.strip() for line in content.split('\n') if line.strip()]
                points = [p.lstrip('•-*') for p in points]  # Remove bullet points
                return points

        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return ["Failed to extract key points."]

    async def identify_gaps(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Identify gaps in research findings."""
        try:
            # Prepare findings for gap analysis
            research_data = {
                "findings": findings,
                "total_findings": len(findings)
            }

            # Get gap analysis from LLM
            messages = [
                {"role": "system", "content": "You are a research analyst. Your task is to identify gaps or missing information in the research findings."},
                {"role": "user", "content": f"Please identify any gaps or missing information in the following research findings:\n\n{json.dumps(research_data)}"}
            ]

            response = await self.llm.chat_completion(messages)
            content = response.choices[0].message.content
            
            # Parse the response into a list of gaps
            try:
                # Try to parse as JSON first
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "gaps" in data:
                    return data["gaps"]
            except json.JSONDecodeError:
                # If not JSON, split by newlines or bullet points
                gaps = [line.strip() for line in content.split('\n') if line.strip()]
                gaps = [g.lstrip('•-*') for g in gaps]  # Remove bullet points
                return gaps

        except Exception as e:
            logger.error(f"Error identifying gaps: {str(e)}")
            return ["Failed to identify research gaps."] 