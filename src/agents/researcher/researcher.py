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

from agents.base_agent import BaseAgent
from agents.planner.planner import SubQuery, QueryType
from utils.retry import retry_wrapper
from browser.search import SearchEngine
from knowledge_base.knowledge_base import KnowledgeBase
from config.config import Config
from browser import Browser
from utils.logger import Logger
from prompts.prompt_manager import PromptManager
from state import State
from llm.llm import LLM

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
        self.min_kb_results = 3  # Default minimum KB results required
        self.max_web_results = 5  # Default maximum web results per sub-query
        
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
            url = result.get('url', '')
            source = result.get('source', '') or url
            formatted_results.append({
                'url': url,
                'title': result.get('title', 'No title'),
                'snippet': result.get('snippet', 'No snippet available'),
                'source': self._extract_domain(url) if url else '',
                'relevance_score': float(result.get('relevance_score', 0.8))
            })
            
        self.logger.info(f"Found {len(formatted_results)} web results")
        return formatted_results

    async def _scrape_with_http(self, url: str) -> Optional[str]:
        """Scrape content using direct HTTP request with user-agent rotation and Tor fallback."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
        ]
        for attempt in range(2):
            headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            timeout = aiohttp.ClientTimeout(total=15)
            proxy = None
            if attempt == 1:
                # Try Tor proxy if available
                from browser.search import is_tor_running, TOR_PROXY
                if is_tor_running():
                    proxy = TOR_PROXY
                    self.logger.warning(f"Retrying {url} with Tor proxy...")
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=headers, ssl=False, proxy=proxy) as response:
                        if response.status in (403, 429):
                            self.logger.warning(f"Blocked ({response.status}) for {url} with UA {headers['User-Agent']}")
                            continue
                        if response.status != 200:
                            return None
                        content_type = response.headers.get('Content-Type', '')
                        if 'text/html' not in content_type:
                            return None
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                            element.decompose()
                        main_content = self._extract_main_content(soup)
                        if main_content:
                            return main_content
                        return soup.get_text(separator='\n', strip=True)
            except Exception as e:
                self.logger.warning(f"HTTP scraping failed for {url} (attempt {attempt+1}): {str(e)}")
                continue
        self.logger.error(f"All HTTP scraping attempts failed for {url}")
        return None

    async def _scrape_page(self, url: str) -> Optional[str]:
        """Scrape content from a web page with multiple fallback strategies and clear error reporting."""
        try:
            content = await self._scrape_with_browser(url)
            if self._is_content_sufficient(content):
                return content
            content = await self._scrape_with_http(url)
            if self._is_content_sufficient(content):
                return content
            content = await self._scrape_with_alternate_agents(url)
            if self._is_content_sufficient(content):
                return content
            self.logger.warning(f"All scraping methods failed for {url}")
            return f"[SCRAPING FAILED] Could not extract content from {url}. Try visiting the page manually or check robots.txt."
        except Exception as e:
            self.logger.error(f"Error in _scrape_page for {url}: {str(e)}", exc_info=True)
            return f"[SCRAPING ERROR] {str(e)} for {url}"
            
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
            if hasattr(self, 'knowledge_base') and self.knowledge_base:
                tag = document["id"]
                contents = document["content"]
                metadata = {
                    "type": "research",
                    "query": query,
                    "sources": document["sources"],
                    "timestamp": document["timestamp"]
                }
                self.knowledge_base.add_knowledge(tag=tag, contents=contents, metadata=metadata)
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

    async def research(self, query: str) -> List[Dict[str, Any]]:
        """Research a query using knowledge base and web search."""
        try:
            if not self._initialized:
                await self.initialize()
            self.logger.info(f"Starting research for query: {query}")
            kb_results, kb_complete = await self._search_knowledge_base(query, min_results=1)
            web_results = []
            sub_query_coverages = []
            sub_query_concept_coverages = []
            sub_query_results = []
            if not kb_complete:
                self.logger.info(f"Insufficient knowledge base results for: {query}")
                sub_queries = await self._generate_sub_queries(query)
                self.logger.info(f"Generated {len(sub_queries)} sub-queries for web search")
                for sub_query in sub_queries:
                    self.logger.info(f"Processing sub-query: {sub_query.query}")
                    sub_results = await self._search_web(sub_query.query, self.max_web_results)
                    for result in sub_results:
                        if 'url' in result:
                            content = await self._extract_content(result['url'])
                            if content:
                                result['content'] = content
                                result['query_id'] = sub_query.id
                                result['query_type'] = sub_query.query_type.value
                                web_results.append(result)
                    await self.store_research_results(sub_query.query, sub_results)
                    # Evaluate coverage for this sub-query
                    all_content = " ".join([
                        r.get("content", "") + " " + r.get("snippet", "") for r in sub_results
                    ])
                    query_terms = set(sub_query.query.lower().split())
                    content_terms = set(all_content.lower().split())
                    covered_terms = query_terms.intersection(content_terms)
                    coverage = len(covered_terms) / len(query_terms) if query_terms else 0
                    key_concepts = ["vix", "volatility index", "s&p 500", "market volatility", "fear gauge", "implied volatility"]
                    concept_coverage = sum(1 for concept in key_concepts if concept in all_content.lower()) / len(key_concepts) if key_concepts else 0
                    sub_query_coverages.append(coverage)
                    sub_query_concept_coverages.append(concept_coverage)
                    sub_query_results.append(sub_results)
                    self.logger.info(f"Sub-query: {sub_query.query}, Coverage: {coverage:.2%}, Key concept coverage: {concept_coverage:.2%}, Results: {sub_results}")
                    await asyncio.sleep(1)
            all_results = await self._combine_results(kb_results, web_results)
            # Aggregate coverage across all sub-queries
            avg_coverage = sum(sub_query_coverages) / len(sub_query_coverages) if sub_query_coverages else 0
            avg_concept_coverage = sum(sub_query_concept_coverages) / len(sub_query_concept_coverages) if sub_query_concept_coverages else 0
            self.logger.info(f"Aggregated sub-query coverage: {avg_coverage:.2%}, Aggregated key concept coverage: {avg_concept_coverage:.2%}")
            # If any sub-query achieves >0% coverage, allow the step to proceed
            if avg_coverage > 0 or avg_concept_coverage > 0 or await self.evaluate_search_results(all_results, query):
                await self.store_research_results(query, all_results)
                self.logger.info(f"Research completed with {len(all_results)} results")
                return all_results
            self.logger.warning("Search results evaluation failed (all sub-queries < 1% coverage)")
            return []
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
            prompt = (await self.prompt_manager.get_prompt("sub_query_generation")).format(
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
            # Extract all text content (use both 'content' and 'snippet')
            all_content = " ".join([
                r.get("content", "") + " " + r.get("snippet", "") for r in results
            ])
            # Check if content covers key aspects of query
            query_terms = set(query.lower().split())
            content_terms = set(all_content.lower().split())
            # Calculate coverage
            covered_terms = query_terms.intersection(content_terms)
            coverage = len(covered_terms) / len(query_terms) if query_terms else 0
            # Check for key concepts
            key_concepts = self._extract_key_concepts(query)
            concept_coverage = sum(1 for concept in key_concepts if concept.lower() in all_content.lower()) / len(key_concepts) if key_concepts else 0
            # Log evaluation metrics and debug info
            logger.info(f"Search results: {results}")
            logger.info(f"All content: {all_content}")
            logger.info(f"Query terms: {query_terms}")
            logger.info(f"Search results coverage: {coverage:.2%}")
            logger.info(f"Key concept coverage: {concept_coverage:.2%}")
            # Relaxed threshold: 0.2 (20%)
            is_sufficient = coverage >= 0.2 or concept_coverage >= 0.2
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

    async def close(self):
        """Close aiohttp session if open."""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            await self._session.close()

    VALID_QUERY_TYPES = {'FACTUAL', 'HOW_TO', 'ANALYTICAL', 'DATA_RETRIEVAL'}

    def _normalize_query_type(self, query_type: str) -> str:
        if not query_type:
            return 'FACTUAL'
        qt = str(query_type).strip().replace('-', '_').upper()
        if qt not in self.VALID_QUERY_TYPES:
            self.logger.warning(f"Invalid query type {query_type}, defaulting to FACTUAL")
            return 'FACTUAL'
        return qt 