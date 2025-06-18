"""Researcher agent for gathering information."""

import asyncio
import functools
import json
import random
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

T = TypeVar('T')

def retry(max_retries: int = 3, backoff: float = 1.0, exceptions=(Exception,)):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        raise
                    wait = backoff * (2 ** (retries - 1)) + random.uniform(0, 1)
                    await asyncio.sleep(wait)
        return cast(Callable[..., T], wrapper)
    return decorator
import time
import asyncio
import logging
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
import os
import aiohttp
from bs4 import BeautifulSoup

from Agentres.agents.base_agent import BaseAgent
from Agentres.agents.planner.planner import SubQuery, QueryType
from Agentres.utils.retry import retry_wrapper
from Agentres.browser.search import SearchEngine
from Agentres.knowledge_base.knowledge_base import KnowledgeBase
from Agentres.config.config import Config
from Agentres.browser import Browser
from Agentres.utils.logger import Logger
from Agentres.prompts.prompt_manager import PromptManager
from Agentres.state import State

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchResult:
    """Container for research results with metadata."""
    def __init__(self, 
                 query_id: str,
                 content: str,
                 source: str,
                 relevance_score: float,
                 metadata: Optional[Dict] = None):
        self.query_id = query_id
        self.content = content
        self.source = source
        self.relevance_score = relevance_score
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'query_id': self.query_id,
            'content': self.content,
            'source': self.source,
            'relevance_score': self.relevance_score,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

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
    """Researcher agent for gathering information."""
    
    def __init__(self, config: Config, model: str = "gpt-4"):
        """Initialize the researcher.
        
        Args:
            config: Configuration instance
            model: Model name to use
        """
        super().__init__(config, model)
        self._web_search = None
        self._knowledge_base = None
        self._session = None
        
    async def initialize(self) -> None:
        """Initialize the researcher."""
        try:
            # Initialize base agent
            await super().initialize()
            
            # Initialize knowledge base
            self.logger.info("Initializing knowledge base...")
            self._knowledge_base = KnowledgeBase(config=self.config)
            await self._knowledge_base.initialize()
            self.logger.info("Successfully initialized knowledge base")
            
            # Initialize web search
            self.logger.info("Initializing web search...")
            self._web_search = SearchEngine()
            await self._web_search._init_async()
            self.logger.info("Successfully initialized web search")
            
            # Initialize async components
            await self._init_async()
            
            self._initialized = True
            self.logger.info(f"Researcher initialized successfully with model: {self.llm.model}")
            
        except Exception as e:
            import traceback
            error_msg = f"Error initializing researcher: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
    async def _init_async(self) -> None:
        """Initialize async components."""
        try:
            # Initialize aiohttp session
            self._session = aiohttp.ClientSession()
            self.logger.info("Researcher async components initialized")
        except Exception as e:
            import traceback
            error_msg = f"Error initializing async components: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
    @property
    def web_search(self) -> SearchEngine:
        """Get the web search engine.
        
        Returns:
            The web search engine
        """
        if not self._web_search:
            raise ValueError("Researcher not initialized")
        return self._web_search
        
    @property
    def knowledge_base(self) -> KnowledgeBase:
        """Get the knowledge base.
        
        Returns:
            The knowledge base
        """
        if not self._knowledge_base:
            raise ValueError("Researcher not initialized")
        return self._knowledge_base

    def get_prompt(self, task: Dict[str, Any]) -> str:
        """Get prompt for the task."""
        try:
            if not self._initialized:
                raise RuntimeError("Researcher not initialized. Call initialize() first.")
                
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")
            return str(task.get("prompt", ""))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting prompt: {str(e)}")
            else:
                logging.error(f"Error getting prompt: {str(e)}")
            raise ValueError(f"Failed to get prompt: {str(e)}")

    async def _search_knowledge_base(self, query: str, min_results: int = 3) -> Tuple[List[Dict], bool]:
        """Search the knowledge base with result validation.
        
        Args:
            query: The search query
            min_results: Minimum number of relevant results required
            
        Returns:
            Tuple of (results, is_complete) where is_complete indicates if enough results were found
        """
        try:
            if not self._initialized:
                await self.initialize()
                
            self.logger.info(f"Searching knowledge base: {query}")
            results = await self.knowledge_base.search(query, limit=min_results*2)
            
            # Basic validation
            if not results or len(results) < min_results:
                self.logger.warning(f"Insufficient results in KB: {len(results)} < {min_results}")
                return results, False
                
            # Advanced validation using LLM if needed
            validation_prompt = PromptManager().get_prompt("kb_validation").format(
                query=query,
                results=json.dumps(results[:5], indent=2)
            )
            
            validation = await LLM(config=self.config, model=self.model).chat_completion([
                {"role": "system", "content": validation_prompt}
            ])
            
            is_complete = json.loads(validation.choices[0].message.content).get('is_complete', False)
            return results, is_complete
            
        except Exception as e:
            self.logger.error(f"Knowledge base search failed: {str(e)}")
            return [], False

    async def _search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information with fallback between multiple search engines.
        
        Tries search engines in this order:
        1. DuckDuckGo (primary)
        2. Tavily (first fallback)
        3. Google (second fallback)
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with 'url', 'title', 'snippet', and 'source' keys
        """
        if not self._initialized:
            await self.initialize()
            
        self.logger.info(f"Searching web for: {query}")
        
        # Initialize search engine if not already done
        if not hasattr(self, '_search_engine'):
            self._search_engine = SearchEngine()
        
        search_engines = [
            ("DuckDuckGo", self._search_engine._duckduckgo_search),
            ("Tavily", self._search_engine._tavily_search),
            ("Google", self._search_engine._google_search)
        ]
        
        last_error = None
        results = []
        
        for engine_name, search_func in search_engines:
            try:
                self.logger.info(f"Trying {engine_name} search...")
                results = await search_func(query, max_results)
                if results:
                    self.logger.info(f"Successfully retrieved {len(results)} results from {engine_name}")
                    break
            except Exception as e:
                last_error = e
                self.logger.warning(f"{engine_name} search failed: {str(e)}")
                continue
        else:
            # This runs if no search engine succeeded
            error_msg = f"All search engines failed. Last error: {str(last_error)}" if last_error else "All search engines returned no results"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Format results
        formatted_results = []
        for result in results[:max_results]:
            formatted_results.append({
                'url': result.get('url', ''),
                'title': result.get('title', 'No title'),
                'snippet': result.get('snippet', 'No snippet available'),
                'source': self._extract_domain(result.get('url', '')),
                'relevance_score': float(result.get('relevance_score', 0.8))
            })
            
        self.logger.info(f"Found {len(formatted_results)} web results")
        return formatted_results

    async def _scrape_page(self, url: str) -> Optional[str]:
        """Scrape content from a web page with multiple fallback strategies.
        
        Args:
            url: The URL to scrape
            
        Returns:
            str: The scraped content, or None if all scraping attempts fail
        """
        try:
            # Try browser-based scraping first (for JavaScript-heavy sites)
            content = await self._scrape_with_browser(url)
            if self._is_content_sufficient(content):
                return content
                
            # Fallback to direct HTTP request
            content = await self._scrape_with_http(url)
            if self._is_content_sufficient(content):
                return content
                
            # Try with different user agents
            content = await self._scrape_with_alternate_agents(url)
            if self._is_content_sufficient(content):
                return content
                
            self.logger.warning(f"All scraping methods failed for {url}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _scrape_page for {url}: {str(e)}", exc_info=True)
            return None
            
    def _is_content_sufficient(self, content: Optional[str], min_word_count: int = 50) -> bool:
        """Check if the scraped content meets minimum requirements."""
        if not content or not isinstance(content, str):
            return False
        return len(content.split()) >= min_word_count
        
    @retry(max_retries=3, backoff=1.0)
    async def _scrape_with_browser(self, url: str) -> Optional[str]:
        """Scrape content using a headless browser."""
        from selenium.webdriver.chrome.options import Options
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(url)
            
            # Wait for dynamic content to load
            await asyncio.sleep(3)
            
            # Get the main content using common content selectors
            content_selectors = [
                'article', 'main', '.post-content', 
                '#content', '.article-body', 'div[role="main"]',
                'div.content', 'div.main', 'div.entry-content'
            ]
            
            for selector in content_selectors:
                try:
                    element = driver.find_element_by_css_selector(selector)
                    if element:
                        return element.text
                except:
                    continue
                    
            # Fallback to body if no content found
            return driver.find_element_by_tag_name('body').text
            
        except Exception as e:
            self.logger.warning(f"Browser scraping failed for {url}: {str(e)}")
            return None
            
        finally:
            try:
                driver.quit()
            except:
                pass
    
    @retry(max_retries=2, backoff=0.5)
    async def _scrape_with_http(self, url: str) -> Optional[str]:
        """Scrape content using direct HTTP request."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=False) as response:
                    if response.status != 200:
                        return None
                        
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' not in content_type:
                        return None
                        
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                        element.decompose()
                        
                    # Get main content using common selectors
                    main_content = self._extract_main_content(soup)
                    if main_content:
                        return main_content
                        
                    # Fallback to body text
                    return soup.get_text(separator='\n', strip=True)
                    
        except Exception as e:
            self.logger.warning(f"HTTP scraping failed for {url}: {str(e)}")
            return None
            
    async def _scrape_with_alternate_agents(self, url: str) -> Optional[str]:
        """Try scraping with different user agents."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
        ]
        
        for user_agent in user_agents:
            try:
                headers = {'User-Agent': user_agent}
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=headers, ssl=False) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            return self._extract_main_content(soup) or soup.get_text(separator='\n', strip=True)
            except Exception:
                continue
                
        return None
        
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main content using common content patterns."""
        # Try common content selectors
        selectors = [
            'article', 'main', '.post-content', 
            '#content', '.article-body', 'div[role="main"]',
            'div.content', 'div.main', 'div.entry-content',
            'div.post', 'div.article', 'div.story',
            'div#main', 'div#article', 'div#story'
        ]
        
        for selector in selectors:
            try:
                content = soup.select_one(selector)
                if content:
                    # Clean and return content
                    text = content.get_text(separator='\n', strip=True)
                    if len(text.split()) > 50:  # Minimum word count
                        return text
            except Exception:
                continue
                
        # Fallback to body if no content found
        if soup.body:
            return soup.body.get_text(separator='\n', strip=True)
            
        return None

    def _is_high_quality(self, result: Dict[str, Any]) -> bool:
        """Check if a search result is high quality.
        
        Args:
            result: The search result to check
            
        Returns:
            True if the result is high quality, False otherwise
        """
        try:
            # Check if result has required fields
            if not all(key in result for key in ['url', 'title', 'snippet']):
                return False
                
            # Check if URL is valid
            if not self._is_valid_url(result['url']):
                return False
                
            # Check if title and snippet are not empty
            if not result['title'].strip() or not result['snippet'].strip():
                return False
                
            # Check if source is not in excluded domains
            domain = self._extract_domain(result['url'])
            if any(excluded in domain.lower() for excluded in self.web_search.EXCLUDED_DOMAINS):
                return False
                
            # Check relevance score
            if result.get('relevance_score', 0.0) < 0.5:
                return False
                
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Error checking result quality: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return False

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is valid, False otherwise
        """
        try:
            # Basic URL validation
            if not url or not isinstance(url, str):
                return False
                
            # Check if URL starts with http/https
            if not url.startswith(('http://', 'https://')):
                return False
                
            # Check if URL has a valid domain
            domain = self._extract_domain(url)
            if not domain or '.' not in domain:
                return False
                
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Error validating URL: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return False

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.
        
        Args:
            url: The URL to extract domain from
            
        Returns:
            The domain
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception as e:
            import traceback
            error_msg = f"Error extracting domain: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return ""
            
    async def extract_key_points(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from research findings.
        
        Args:
            findings: List of research findings
            
        Returns:
            List of key points
        """
        if not findings:
            return []
            
        try:
            # Get the key points extraction prompt
            prompt = self.prompt_manager.get_prompt("extract_key_points") or """
            Extract the 5-7 most important key points from the following research findings.
            Focus on unique insights, important facts, and actionable information.
            
            Research Findings:
            {findings}
            
            Format your response as a JSON array of strings.
            """
            
            # Format the prompt with the findings
            formatted_prompt = prompt.format(
                findings=json.dumps(findings, indent=2)
            )
            
            # Get key points from the LLM
            response = await self.llm.chat_completion([
                {"role": "system", "content": "You are an expert at extracting key information from research."},
                {"role": "user", "content": formatted_prompt}
            ])
            
            if not response or not response.choices:
                return []
                
            # Parse the response
            try:
                # Try to parse as JSON first
                key_points = json.loads(response.choices[0].message.content)
                if isinstance(key_points, list):
                    return key_points
                    
                # Fallback to extracting from markdown list
                import re
                content = response.choices[0].message.content
                key_points = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', content)
                return key_points or [content.strip()]
                
            except json.JSONDecodeError:
                # If not valid JSON, try to extract key points from text
                content = response.choices[0].message.content
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                return lines[:7]  # Return first 7 non-empty lines as key points
                
        except Exception as e:
            self.logger.error(f"Error extracting key points: {str(e)}")
            return []
            
    async def identify_gaps(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Identify gaps or limitations in the research findings.
        
        Args:
            findings: List of research findings
            
        Returns:
            List of identified gaps
        """
        if not findings:
            return ["No research findings available to identify gaps."]
            
        try:
            # Get the gap analysis prompt
            prompt = self.prompt_manager.get_prompt("identify_gaps") or """
            Analyze the following research findings and identify any gaps, limitations, 
            or areas that need further investigation. Consider:
            - Missing information
            - Contradictions between sources
            - Outdated information
            - Lack of diversity in sources
            - Methodological limitations
            
            Research Findings:
            {findings}
            
            Provide a concise list of the most important gaps.
            """
            
            # Format the prompt with the findings
            formatted_prompt = prompt.format(
                findings=json.dumps(findings, indent=2)
            )
            
            # Get gap analysis from the LLM
            response = await self.llm.chat_completion([
                {"role": "system", "content": "You are an expert research analyst that identifies gaps in information."},
                {"role": "user", "content": formatted_prompt}
            ])
            
            if not response or not response.choices:
                return ["Unable to analyze gaps in the research findings."]
                
            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract list items
            import re
            gaps = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', content)
            
            # If no list items found, split by newlines
            if not gaps:
                gaps = [line.strip() for line in content.split('\n') if line.strip()]
                
            return gaps[:5]  # Return at most 5 gaps
            
        except Exception as e:
            self.logger.error(f"Error identifying gaps: {str(e)}")
            return ["An error occurred while analyzing research gaps."]
            
    def calculate_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for research findings.
        
        Args:
            findings: List of research findings
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not findings:
            return 0.0
            
        try:
            # Calculate base confidence based on number of findings
            num_findings = len(findings)
            base_confidence = min(1.0, num_findings / 5.0)  # Cap at 1.0 for 5+ findings
            
            # Calculate source diversity score
            sources = set()
            for finding in findings:
                if isinstance(finding, dict) and "source" in finding:
                    sources.add(finding["source"])
            
            source_diversity = min(1.0, len(sources) / 3.0)  # Cap at 1.0 for 3+ sources
            
            # Calculate content quality score
            content_scores = []
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                    
                score = 0.0
                
                # Check content length
                content = finding.get("content", "")
                if len(content.split()) >= 50:  # At least 50 words
                    score += 0.3
                    
                # Check for structured content
                if any(marker in content.lower() for marker in ["first", "second", "finally", "in conclusion"]):
                    score += 0.2
                    
                # Check for references/citations
                if any(marker in content.lower() for marker in ["according to", "source:", "reference"]):
                    score += 0.2
                    
                # Check for data/statistics
                if any(marker in content.lower() for marker in ["%"]):
                    score += 0.1
                    
                content_scores.append(min(1.0, score))
            
            avg_content_score = sum(content_scores) / len(content_scores) if content_scores else 0.0
            
            # Calculate final confidence score (weighted average)
            confidence = (
                0.4 * base_confidence + 
                0.3 * source_diversity + 
                0.3 * avg_content_score
            )
            
            return min(1.0, max(0.0, confidence))  # Ensure between 0.0 and 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5  # Default to medium confidence on error
            
    def assess_coverage(self, findings: List[Dict[str, Any]], query: str) -> str:
        """Assess how well the findings cover the query.
        
        Args:
            findings: List of research findings
            query: The original query
            
        Returns:
            Coverage assessment ("low", "medium", or "high")
        """
        if not findings:
            return "low"
            
        try:
            # Extract key terms from query
            query_terms = set(term.lower() for term in re.findall(r'\w+', query) if len(term) > 3)
            
            if not query_terms:
                return "medium"  # Can't assess coverage without meaningful terms
                
            # Combine all content
            all_content = " ".join(
                str(finding.get("content", "")) for finding in findings 
                if isinstance(finding, dict)
            ).lower()
            
            # Calculate term coverage
            covered_terms = sum(1 for term in query_terms if term in all_content)
            coverage_ratio = covered_terms / len(query_terms)
            
            # Determine coverage level
            if coverage_ratio >= 0.8:
                return "high"
            elif coverage_ratio >= 0.5:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Error assessing coverage: {str(e)}")
            return "unknown"
            
    async def store_research_results(self, query: str, findings: List[Dict[str, Any]]) -> bool:
        """Store research results in the knowledge base for future reference.
        
        Args:
            query: The original research query
            findings: List of research findings to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not findings:
            return False
            
        try:
            # Create a structured document with metadata
            document = {
                "id": f"research_{int(time.time())}",
                "query": query,
                "findings": findings,
                "timestamp": datetime.utcnow().isoformat(),
                "sources": list({f.get("source") for f in findings if f.get("source")}),
                "content": "\n\n".join(
                    f"Source: {f.get('source', 'Unknown')}\n{f.get('content', '')}" 
                    for f in findings
                )
            }
            
            # Store in knowledge base if available
            if hasattr(self, 'knowledge_base'):
                await self.knowledge_base.add_document(
                    document_id=document["id"],
                    content=document["content"],
                    metadata={
                        "type": "research",
                        "query": query,
                        "sources": document["sources"],
                        "timestamp": document["timestamp"]
                    }
                )
                self.logger.info(f"Stored research results for query: {query}")
                return True
                
            # Fallback to local storage if knowledge base is not available
            storage_dir = os.path.join(os.getcwd(), "research_data")
            os.makedirs(storage_dir, exist_ok=True)
            
            filename = f"research_{query[:50].replace(' ', '_')}_{int(time.time())}.json"
            filepath = os.path.join(storage_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved research results to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing research results: {str(e)}")
            return False
            
    async def generate_research_report(self, query: str, synthesis: Dict[str, Any], 
                                     format: str = "markdown") -> str:
        """Generate a comprehensive research report from the synthesis.
        
        Args:
            query: The original research query
            synthesis: The research synthesis from synthesize_research()
            format: Output format ("markdown" or "html")
            
        Returns:
            str: Formatted research report
        """
        if not synthesis or not isinstance(synthesis, dict):
            return "Error: Invalid synthesis data"
            
        try:
            # Get the appropriate report template
            template = self.prompt_manager.get_prompt(f"report_template_{format}") or \
                     self.prompt_manager.get_prompt("report_template") or \
                     self._get_default_report_template()
            
            # Prepare data for the template
            report_data = {
                "query": query,
                "summary": synthesis.get("synthesis", "No summary available."),
                "key_points": synthesis.get("key_points", []),
                "gaps": synthesis.get("gaps", []),
                "sources": synthesis.get("metadata", {}).get("sources", []),
                "confidence": synthesis.get("metadata", {}).get("confidence", 0.0),
                "coverage": synthesis.get("metadata", {}).get("coverage", "unknown"),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
            
            # Format the report
            report = template.format(**report_data)
            
            # Post-process based on format
            if format.lower() == "html":
                # Convert markdown to HTML if needed
                try:
                    import markdown
                    from markdown.extensions.tables import TableExtension
                    report = markdown.markdown(
                        report, 
                        extensions=[TableExtension(), 'fenced_code', 'codehilite']
                    )
                    
                    # Add basic HTML structure if not present
                    if not report.strip().lower().startswith('<!doctype html>') and \
                       not report.strip().lower().startswith('<html>'):
                        report = f"""<!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Research Report: {query}</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                                h1, h2, h3 {{ color: #2c3e50; }}
                                .metadata {{ color: #666; font-size: 0.9em; }}
                                .key-point {{ margin-bottom: 10px; }}
                                .source {{ margin-left: 20px; font-style: italic; }}
                                .gap {{ color: #c0392b; }}
                                code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                                pre {{ background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                th {{ background-color: #f2f2f2; }}
                            </style>
                        </head>
                        <body>
                            {report}
                        </body>
                        </html>
                        """.format(report=report)
                except ImportError:
                    self.logger.warning("markdown package not available. HTML formatting limited.")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating research report: {str(e)}")
            return f"Error generating research report: {str(e)}"
    
    def _get_default_report_template(self) -> str:
        """Get the default report template in markdown format."""
        return """# Research Report: {query}

## Summary
{summary}

## Key Findings
{key_points_formatted}

## Research Gaps
{gaps_formatted}

## Metadata
- **Confidence Score**: {confidence:.1%}
- **Coverage**: {coverage}
- **Sources Used**: {sources_count}
- **Report Generated**: {timestamp}

### Sources
{sources_list}
"""
    
    def _format_key_points(self, key_points: List[Any]) -> str:
        """Format key points for the report."""
        if not key_points:
            return "No key points identified."
            
        formatted = []
        for i, point in enumerate(key_points, 1):
            if isinstance(point, dict):
                point_text = point.get('point', str(point))
                score = point.get('relevance', 1.0)
                if isinstance(score, (int, float)):
                    point_text += f" (Relevance: {score:.1f})"
            else:
                point_text = str(point)
                
            formatted.append(f"{i}. {point_text}")
            
        return "\n".join(formatted)
    
    def _format_gaps(self, gaps: List[Any]) -> str:
        """Format gaps for the report."""
        if not gaps:
            return "No significant gaps identified in the research."
            
        formatted = []
        for i, gap in enumerate(gaps, 1):
            if isinstance(gap, dict):
                gap_text = gap.get('gap', str(gap))
                severity = gap.get('severity', 'medium').lower()
                if severity in ['high', 'critical']:
                    gap_text = f"**{gap_text}** (High Priority)"
            else:
                gap_text = str(gap)
                
            formatted.append(f"{i}. {gap_text}")
            
        return "\n".join(formatted)
    
    def _format_sources(self, sources: List[Any]) -> str:
        """Format sources list for the report."""
        if not sources:
            return "No sources available."
            
        formatted = []
        for i, source in enumerate(sources, 1):
            if isinstance(source, str):
                formatted.append(f"{i}. {source}")
            elif isinstance(source, dict):
                title = source.get('title', 'Untitled')
                url = source.get('url', 'No URL')
                formatted.append(f"{i}. [{title}]({url})")
                
        return "\n".join(formatted)

    async def research_sub_query(self, sub_query: SubQuery) -> Dict:
        """Research a single sub-query with fallback to web search."""
        try:
            # Check cache first
            cache_key = f"{sub_query.id}_{hash(sub_query.query)}"
            if cache_key in self.research_cache:
                return self.research_cache[cache_key]
                
            # Search knowledge base first
            kb_results, is_complete = await self._search_knowledge_base(
                sub_query.query, 
                min_results=sub_query.min_required_results
            )
            
            # If KB results are insufficient, search the web
            web_results = []
            if not is_complete:
                web_results = await self._search_web(
                    sub_query.query,
                    max_results=self.max_web_searches
                )
                
            # Combine and deduplicate results
            all_results = await self._combine_results(kb_results, web_results)
            
            # Store in cache
            self.research_cache[cache_key] = {
                'sub_query': sub_query.to_dict(),
                'results': [r.to_dict() for r in all_results],
                'sources': list(set(r.source for r in all_results)),
                'is_complete': is_complete or len(all_results) >= sub_query.min_required_results
            }
            
            return self.research_cache[cache_key]
            
        except Exception as e:
            self.logger.error(f"Research failed for sub-query {sub_query.id}: {str(e)}")
            return {
                'sub_query': sub_query.to_dict(),
                'results': [],
                'sources': [],
                'is_complete': False,
                'error': str(e)
            }
            
    async def synthesize_research(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize research results into a structured format with rich content.
        
        Args:
            query: The original research query
            results: List of research results from various sources
            
        Returns:
            Dict containing structured research synthesis with summary, key points, and sources
        """
        try:
            if not self._initialized:
                await self.initialize()
                
            if not results:
                self.logger.warning("No results to synthesize")
                return {
                    "summary": "No relevant information found.",
                    "key_points": [],
                    "sources": [],
                    "confidence": 0.0,
                    "coverage": "low"
                }
                
            self.logger.info(f"Synthesizing {len(results)} research results for: {query}")
            
            # Group results by source type
            source_groups = self._group_results_by_source(results)
            
            # Generate summaries for each source group
            group_summaries = []
            for source_type, items in source_groups.items():
                if not items:
                    continue
                    
                summary = await self._summarize_source_group(query, source_type, items)
                if summary:
                    group_summaries.append(summary)
            
            # Combine all summaries into a coherent research synthesis
            research_synthesis = await self._combine_research_summaries(query, group_summaries)
            
            # Extract key points and identify gaps
            research_synthesis["key_points"] = await self.extract_key_points(research_synthesis["findings"])
            research_synthesis["gaps"] = await self.identify_gaps(research_synthesis["findings"])
            
            # Calculate confidence and coverage metrics
            research_synthesis["metadata"]["confidence"] = self.calculate_confidence(research_synthesis["findings"])
            research_synthesis["metadata"]["coverage"] = self.assess_coverage(research_synthesis["findings"], query)
            
            # Store the research results for future reference
            await self.store_research_results(query, research_synthesis["findings"])
            
            return research_synthesis
            
        except Exception as e:
            self.logger.error(f"Error in synthesize_research: {str(e)}", exc_info=True)
            return {
                "error": f"Failed to synthesize research: {str(e)}",
                "summary": "An error occurred while processing the research results.",
                "key_points": [],
                "sources": [],
                "confidence": 0.0,
                "coverage": "unknown"
            }
            
    def _group_results_by_source(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group research results by their source type."""
        groups = {"web": [], "knowledge_base": [], "code": [], "documentation": []}
        
        for result in results:
            source_type = result.get("source_type", "web").lower()
            if source_type in groups:
                groups[source_type].append(result)
            else:
                groups["web"].append(result)  # Default to web
                
        return groups
        
    async def _summarize_source_group(self, query: str, source_type: str, 
                                     items: List[Dict[str, Any]]) -> Optional[Dict]:
        """Generate a summary for a group of results from the same source type."""
        if not items:
            return None
            
        try:
            # Get the appropriate prompt template
            prompt_key = f"{source_type}_summary"
            prompt = self.prompt_manager.get_prompt(prompt_key) or \
                    self.prompt_manager.get_prompt("default_summary")
                    
            if not prompt:
                self.logger.warning(f"No prompt found for {source_type}")
                return None
                
            # Format the prompt with query and items
            formatted_prompt = prompt.format(
                query=query,
                items=json.dumps(items, indent=2),
                count=len(items)
            )
            
            # Get the summary from the LLM
            response = await self.llm.chat_completion([
                {"role": "system", "content": "You are a research assistant that summarizes information concisely."},
                {"role": "user", "content": formatted_prompt}
            ])
            
            if not response or not response.choices:
                return None
                
            # Parse the response
            summary = response.choices[0].message.content
            
            return {
                "source_type": source_type,
                "summary": summary,
                "items": items,
                "item_count": len(items)
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing {source_type} group: {str(e)}")
            return None
            
    async def _combine_research_summaries(self, query: str, group_summaries: List[Dict]) -> Dict:
        """Combine multiple source group summaries into a coherent research synthesis."""
        if not group_summaries:
            return {
                "synthesis": "No information could be found on this topic.",
                "findings": [],
                "metadata": {
                    "sources": [],
                    "confidence": 0.0,
                    "coverage": "none"
                }
            }
            
        try:
            # Get the synthesis prompt
            prompt = self.prompt_manager.get_prompt("research_synthesis") or """
            Combine the following research summaries into a coherent synthesis for the query: {query}
            
            Summaries:
            {summaries}
            
            Provide a comprehensive synthesis that addresses the original query.
            Structure your response with clear sections and include key findings.
            """
            
            # Format the summaries for the prompt
            formatted_summaries = "\n\n".join(
                f"## Source: {s['source_type'].upper()}\n{s['summary']}" 
                for s in group_summaries if s and 'summary' in s
            )
            
            # Get the synthesis from the LLM
            response = await self.llm.chat_completion([
                {"role": "system", "content": "You are a research synthesis assistant that combines information from multiple sources into a coherent, well-structured report."},
                {"role": "user", "content": prompt.format(
                    query=query,
                    summaries=formatted_summaries
                )}
            ])
            
            if not response or not response.choices:
                raise ValueError("Failed to generate research synthesis")
                
            # Extract all findings from the source groups
            all_findings = []
            sources = set()
            
            for group in group_summaries:
                if not group:
                    continue
                    
                all_findings.extend(group.get("items", []))
                
                # Extract unique sources
                for item in group.get("items", []):
                    if "source" in item and item["source"]:
                        sources.add(item["source"])
            
            return {
                "synthesis": response.choices[0].message.content,
                "findings": all_findings,
                "metadata": {
                    "sources": list(sources),
                    "confidence": 0.0,  # Will be calculated later
                    "coverage": "unknown"  # Will be calculated later
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error combining research summaries: {str(e)}")
            raise ValueError(f"Failed to combine research summaries: {str(e)}")
            
    # Helper methods
    async def _extract_content(self, url: str) -> Optional[str]:
        """Extract and clean content from a URL.
        
        Args:
            url: The URL to extract content from
            
        Returns:
            Extracted and cleaned content as a string, or None if extraction fails
        """
        if not url or not url.startswith(('http://', 'https://')):
            self.logger.warning(f"Invalid URL for content extraction: {url}")
            return None
            
        try:
            # Set a reasonable timeout
            timeout = aiohttp.ClientTimeout(total=30)
            
            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, allow_redirects=True, ssl=False) as response:
                    if response.status != 200:
                        self.logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
                        
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'text/html' not in content_type:
                        self.logger.warning(f"Unsupported content type {content_type} for {url}")
                        return None
                        
                    html = await response.text()
                    
                    # Parse HTML with BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        script.decompose()
                        
                    # Get text and clean it up
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # Remove excessive whitespace and normalize newlines
                    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
                    
                    # Truncate if too long (to avoid token limits)
                    max_length = 10000
                    if len(text) > max_length:
                        text = text[:max_length] + '... [content truncated]'
                        
                    return text if text.strip() else None
                    
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout while fetching {url}")
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            
        return None
        
    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace('www.', '') if domain else ''
        
    async def _combine_results(self, kb_results: List[Dict], web_results: List[ResearchResult]) -> List[ResearchResult]:
        """Combine and deduplicate results from KB and web."""
        combined = []
        seen_urls = set()
        
        # Add web results first (more recent)
        for result in web_results:
            if result.source not in seen_urls:
                seen_urls.add(result.source)
                combined.append(result)
                
        # Add KB results if not already included
        for result in kb_results:
            url = result.get('metadata', {}).get('source', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(ResearchResult(
                    query_id=result.get('metadata', {}).get('query_id', ''),
                    content=result.get('content', ''),
                    source=url,
                    relevance_score=result.get('score', 0.0),
                    metadata=result.get('metadata', {})
                ))
                
        # Sort by relevance score
        return sorted(combined, key=lambda x: x.relevance_score, reverse=True)

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
            
            return data

        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    async def research(self, query: str) -> List[Dict[str, Any]]:
        """Research a query using knowledge base and web search.
        
        Args:
            query: The query to research
            
        Returns:
            List of research results
        """
        try:
            if not self._initialized:
                await self.initialize()
                
            self.logger.info(f"Starting research for query: {query}")
            
            # First try knowledge base
            kb_results, kb_complete = await self._search_knowledge_base(query, self.min_kb_results)
            
            # If knowledge base results are insufficient, search web
            web_results = []
            if not kb_complete:
                self.logger.info(f"Insufficient knowledge base results for: {query}")
                
                # Generate sub-queries for web search
                sub_queries = await self._generate_sub_queries(query)
                self.logger.info(f"Generated {len(sub_queries)} sub-queries for web search")
                
                # Process each sub-query
                for sub_query in sub_queries:
                    self.logger.info(f"Processing sub-query: {sub_query.query}")
                    
                    # Search web for this sub-query
                    sub_results = await self._search_web(sub_query.query, self.max_web_results)
                    
                    # Extract content from each result
                    for result in sub_results:
                        if 'url' in result:
                            content = await self._extract_content(result['url'])
                            if content:
                                result['content'] = content
                                result['query_id'] = sub_query.id
                                result['query_type'] = sub_query.query_type.value
                                web_results.append(result)
                    
                    # Store results for this sub-query
                    await self.store_research_results(sub_query.query, sub_results)
                    
                    # Small delay between sub-queries to avoid rate limiting
                    await asyncio.sleep(1)
            
            # Combine results
            all_results = await self._combine_results(kb_results, web_results)
            
            # Evaluate results
            if not await self.evaluate_search_results(all_results, query):
                self.logger.warning("Search results evaluation failed")
                return []
            
            # Store final results
            await self.store_research_results(query, all_results)
            
            self.logger.info(f"Research completed with {len(all_results)} results")
            return all_results
            
        except Exception as e:
            error_msg = f"Research failed: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
    async def _generate_sub_queries(self, query: str) -> List[SubQuery]:
        """Generate sub-queries for web search.
        
        Args:
            query: The main query
            
        Returns:
            List of SubQuery objects
        """
        try:
            # Get the sub-query generation prompt
            prompt = self.prompt_manager.get_prompt("sub_query_generation").format(
                query=query
            )
            
            # Generate sub-queries using LLM
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
                # Get query type, defaulting to FACTUAL if not specified
                query_type = QueryType.FACTUAL
                if 'type' in sq_data:
                    try:
                        query_type = QueryType(sq_data['type'].upper())
                    except ValueError:
                        self.logger.warning(f"Invalid query type {sq_data['type']}, using FACTUAL")
                
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
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def evaluate_search_results(self, results: List[Dict[str, Any]], query: str) -> bool:
        """Evaluate if search results are sufficient to answer the query."""
        try:
            if not results:
                logger.info("No search results to evaluate")
                return False
                
            # Extract all text content
            all_content = " ".join([r.get("text", "") for r in results])
            
            # Check if content covers key aspects of query
            query_terms = set(query.lower().split())
            content_terms = set(all_content.lower().split())
            
            # Calculate coverage
            covered_terms = query_terms.intersection(content_terms)
            coverage = len(covered_terms) / len(query_terms) if query_terms else 0
            
            # Check for key concepts
            key_concepts = self._extract_key_concepts(query)
            concept_coverage = sum(1 for concept in key_concepts if concept.lower() in all_content.lower()) / len(key_concepts) if key_concepts else 0
            
            # Log evaluation metrics
            logger.info(f"Search results coverage: {coverage:.2%}")
            logger.info(f"Key concept coverage: {concept_coverage:.2%}")
            
            # Consider results sufficient if both metrics are above threshold
            is_sufficient = coverage >= 0.7 and concept_coverage >= 0.7
            
            if not is_sufficient:
                logger.info("Search results insufficient, will conduct web research")
            
            return is_sufficient
            
        except Exception as e:
            logger.error(f"Error evaluating search results: {str(e)}")
            return False

    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from a query."""
        try:
            # Simple concept extraction - can be enhanced with NLP
            words = query.lower().split()
            # Remove common words
            stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
            concepts = [word for word in words if word not in stop_words]
            return concepts
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
            return []

    async def store_research_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Store research results in the knowledge base."""
        if not isinstance(results, list):
            raise ValueError("results must be a list")
        
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        
        stored_count = 0
        # Process each result
        for result in results:
            try:
                if isinstance(result, str):
                    # Handle string results
                    content = self._clean_text(result)
                    metadata = {
                        "query": query,
                        "source": "search",
                        "timestamp": datetime.now().isoformat()
                    }
                elif isinstance(result, dict):
                    # Handle dictionary results
                    content = self._clean_text(result.get("snippet", result.get("content", "")))
                    metadata = {
                        "query": query,
                        "source": result.get("source", "search"),
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self.logger.warning(f"Skipping invalid result format: {result}")
                    continue
                
                # Store in knowledge base
                await self.knowledge_base.store(content, metadata)
                stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Error storing individual result: {str(e)}")
                continue
        
        self.logger.info(f"Stored {stored_count} research results for query: {query}")

    def _clean_text(self, text: str) -> str:
        """Clean text for storage in knowledge base."""
        try:
            if not text:
                return ""
                
            # Remove extra whitespace
            text = " ".join(text.split())
            
            # Remove special characters but keep important ones
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text

    def calculate_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for research findings."""
        try:
            if not findings or not isinstance(findings, list):
                return 0.0
                
            # Average relevance scores
            scores = [float(f.get("score", 0)) for f in findings if isinstance(f, dict)]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Consider number of findings
            num_factor = min(len(findings) / 5, 1.0)  # Cap at 5 findings
            
            # Consider source diversity
            sources = set(f.get("url", "") for f in findings if isinstance(f, dict))
            diversity_factor = min(len(sources) / 3, 1.0)  # Cap at 3 sources
            
            # Consider content quality
            quality_factor = 0.0
            for finding in findings:
                if isinstance(finding, dict) and "text" in finding:
                    text = finding["text"]
                    # Check for minimum content length
                    if len(text.split()) >= 50:
                        quality_factor += 0.2
                    # Check for structured content
                    if any(marker in text.lower() for marker in ["first", "second", "finally", "in conclusion"]):
                        quality_factor += 0.1
                        
            quality_factor = min(quality_factor, 1.0)
            
            # Combine factors with weights
            confidence = (
                avg_score * 0.4 +  # Relevance score weight
                num_factor * 0.2 +  # Number of findings weight
                diversity_factor * 0.2 +  # Source diversity weight
                quality_factor * 0.2  # Content quality weight
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def assess_coverage(self, findings: List[Dict[str, Any]], query: str) -> str:
        """Assess how well the findings cover the query."""
        try:
            if not findings or not isinstance(findings, list):
                return "none"
                
            if not query or not isinstance(query, str):
                raise ValueError("Invalid query")
                
            # Count unique findings
            unique_findings = set()
            for finding in findings:
                if isinstance(finding, dict) and "text" in finding:
                    unique_findings.add(finding["text"].strip())
                    
            num_findings = len(unique_findings)
            
            # Assess coverage based on number of unique findings
            if num_findings >= 5:
                return "comprehensive"
            elif num_findings >= 3:
                return "substantial"
            elif num_findings >= 1:
                return "partial"
            else:
                return "minimal"
                
        except Exception as e:
            logger.error(f"Error assessing coverage: {str(e)}")
            return "unknown"

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research task."""
        try:
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")

            query = str(task.get("query", ""))
            context = task.get("context", [])
            
            # Conduct research
            research_results = await self.research(query)
            
            return {
                "task": task,
                "research_results": research_results
            }
        except Exception as e:
            logger.error(f"Error executing research task: {str(e)}")
            raise ValueError(f"Task execution failed: {str(e)}")

    async def execute_with_context(self, task: Dict[str, Any], context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute a research task with context."""
        try:
            if not isinstance(task, dict):
                raise ValueError("task must be a dictionary")
            if not isinstance(context, list):
                raise ValueError("context must be a list")

            # Add context to task
            task["context"] = context
            return await self.execute(task)
        except Exception as e:
            logger.error(f"Error executing research task with context: {str(e)}")
            raise ValueError(f"Task execution with context failed: {str(e)}")

    async def synthesize_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Synthesize research findings into a coherent summary."""
        try:
            if not findings:
                return "No findings to synthesize."

            # Prepare context for synthesis
            context = [
                {"role": "system", "content": "You are a research synthesis expert. Synthesize the following findings into a coherent summary."},
                {"role": "user", "content": f"Findings: {json.dumps(findings, indent=2)}"}
            ]

            # Generate synthesis
            response = await LLM(config=self.config, model=self.model).generate_with_context(
                prompt="Synthesize these findings into a coherent summary.",
                context=context
            )

            return response
        except Exception as e:
            logger.error(f"Error synthesizing findings: {str(e)}")
            return "Failed to synthesize findings."

    async def extract_key_points(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from research findings."""
        try:
            if not findings:
                return ["No findings to analyze."]

            # Prepare context for key point extraction
            context = [
                {"role": "system", "content": "You are a research analyst. Extract the key points from these findings."},
                {"role": "user", "content": f"Findings: {json.dumps(findings, indent=2)}"}
            ]

            # Generate key points
            response = await LLM(config=self.config, model=self.model).generate_with_context(
                prompt="Extract the key points from these findings.",
                context=context
            )

            # Parse response into list
            try:
                points = json.loads(response)
                if isinstance(points, list):
                    return points
                return [str(points)]
            except json.JSONDecodeError:
                # If not JSON, split by newlines
                return [point.strip() for point in response.split('\n') if point.strip()]
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return ["Failed to extract key points."]

    async def identify_gaps(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Identify gaps in the research findings."""
        try:
            if not findings:
                return ["No findings to analyze for gaps."]

            # Prepare context for gap analysis
            context = [
                {"role": "system", "content": "You are a research analyst. Identify gaps in these findings."},
                {"role": "user", "content": f"Findings: {json.dumps(findings, indent=2)}"}
            ]

            # Generate gap analysis
            response = await LLM(config=self.config, model=self.model).generate_with_context(
                prompt="Identify gaps in these findings.",
                context=context
            )

            # Parse response into list
            try:
                gaps = json.loads(response)
                if isinstance(gaps, list):
                    return gaps
                return [str(gaps)]
            except json.JSONDecodeError:
                # If not JSON, split by newlines
                return [gap.strip() for gap in response.split('\n') if gap.strip()]
        except Exception as e:
            logger.error(f"Error identifying gaps: {str(e)}")
            return ["Failed to identify research gaps."] 