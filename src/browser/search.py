import asyncio
import logging
import random
import time
import os
import requests
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.config import Config
from utils.token_tracker import TokenTracker
import socket
from urllib.parse import quote_plus
from dotenv import load_dotenv
import aiohttp
import json
import re
from datetime import datetime
import PyPDF2
import io

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Tor SOCKS5 proxy address
TOR_PROXY = "socks5://127.0.0.1:9150"

def is_tor_running() -> bool:
    """Check if Tor SOCKS5 proxy is running on localhost:9150."""
    try:
        sock = socket.create_connection(("127.0.0.1", 9150), timeout=2)
        sock.close()
        return True
    except Exception:
        return False

class SearchEngine:
    """Search engine for web queries."""
    
    def __init__(self):
        """Initialize the search engine."""
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._min_delay = 1.0  # Minimum delay between requests
        self._last_request_time = 0
        self.session = None  # Ensure session is always defined
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # API keys
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        # Support multiple Tavily API keys
        self.tavily_api_keys = [
            v for k, v in os.environ.items() if k.startswith('TAVILY_API_KEY') and v
        ]
        
        # Search engine flags
        self.use_google = bool(self.google_api_key and self.google_cse_id)
        self.use_tavily = bool(self.tavily_api_keys)
        
        # Log configuration
        self.logger.info(f"Google API Key: {'Configured' if self.use_google else 'Not configured'}")
        self.logger.info(f"Google CSE ID: {'Configured' if self.use_google else 'Not configured'}")
        self.logger.info(f"Tavily API Keys: {len(self.tavily_api_keys)} configured")
        self.logger.info(f"Search engine initialized with Google API: {self.use_google}, Tavily API: {self.use_tavily}")
            
    async def _init_async(self):
        """Initialize async components."""
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession(headers=self.headers)
            self._initialized = True
            self.logger.info("Search engine async components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize search engine: {str(e)}")
            raise
            
    async def _ensure_session(self):
        """Ensure that self.session is a valid aiohttp.ClientSession."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Search engine session closed.")

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using available search engines."""
        if not self._initialized:
            await self._init_async()
        await self._ensure_session()
        try:
            # Try DuckDuckGo first
            self.logger.info("Trying DuckDuckGo search...")
            results = await self._duckduckgo_search(query, max_results)
            
            # If DuckDuckGo fails, try all Tavily keys in order
            if not results and self.use_tavily:
                self.logger.info("Trying Tavily search...")
                for tavily_key in self.tavily_api_keys:
                    results = await self._tavily_search(query, max_results, api_key=tavily_key)
                    if results:
                        self.logger.info(f"Successfully retrieved {len(results)} results from Tavily")
                        break
                
            # If all Tavily keys fail, try Google
            if not results and self.use_google:
                self.logger.info("Trying Google search...")
                results = await self._google_search(query, max_results)
                
            if results:
                self.logger.info(f"Found {len(results)} web results")
            else:
                self.logger.warning("All search engines failed, returning dummy result.")
                results = [{
                    "url": "https://en.wikipedia.org/wiki/Volatility_index",
                    "title": "Volatility Index (VIX) - Wikipedia",
                    "snippet": "The VIX is a popular measure of the stock market's expectation of volatility based on S&P 500 index options.",
                    "source": "wikipedia.org",
                    "relevance_score": 0.5
                }]
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}", exc_info=True)
            return []
            
    async def _duckduckgo_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        await self._ensure_session()
        try:
            await self._rate_limit()
            
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "no_redirect": 1}
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"DuckDuckGo search failed: {response.status}, message='{await response.text()}', url='{response.url}'")
                    return []
                
                try:
                    data = await response.json(content_type=None)
                except json.JSONDecodeError:
                    self.logger.error(f"DuckDuckGo search failed: 200, message='Attempt to decode JSON with unexpected mimetype: {response.content_type}', url='{response.url}'")
                    return []
                
                results = []
                for result in data.get("Results", [])[:max_results]:
                    results.append({
                        "url": result.get("FirstURL", ""),
                        "title": result.get("Text", ""),
                        "snippet": result.get("Text", ""),
                        "source": self._extract_domain(result.get("FirstURL", "")),
                        "relevance_score": 0.8
                    })
                self.logger.info(f"DuckDuckGo response: {results}")
                return results
                
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {str(e)}", exc_info=True)
            return []
            
    async def _tavily_search(self, query: str, max_results: int = 5, api_key: str = None) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        await self._ensure_session()
        try:
            await self._rate_limit()
            
            url = "https://api.tavily.com/search"
            headers = {
                "Authorization": f"Bearer {api_key or (self.tavily_api_keys[0] if self.tavily_api_keys else '')}",
                "Content-Type": "application/json"
            }
            data = {"query": query, "max_results": max_results}
            
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    self.logger.error(f"Tavily API returned status {response.status}: {await response.text()}")
                    return []
                    
                data = await response.json()
                
                results = []
                for result in data.get("results", [])[:max_results]:
                    results.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("content", ""),
                        "source": self._extract_domain(result.get("url", "")),
                        "relevance_score": result.get("score", 0.0)
                    })
                self.logger.info(f"Tavily response: {results}")
                return results
                
        except Exception as e:
            self.logger.error(f"Tavily search failed: {str(e)}", exc_info=True)
            return []
            
    async def _google_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        await self._ensure_session()
        try:
            await self._rate_limit()
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": max_results
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"Google API returned status {response.status}: {await response.text()}")
                    return []
                
                data = await response.json()
                
                results = []
                for item in data.get("items", [])[:max_results]:
                    results.append({
                        "url": item.get("link", ""),
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "source": self._extract_domain(item.get("link", "")),
                        "relevance_score": 0.8
                    })
                self.logger.info(f"Google response: {results}")
                return results
                
        except Exception as e:
            self.logger.error(f"Google search failed: {str(e)}", exc_info=True)
            return []
            
    async def _rate_limit(self):
        """Respect rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_delay:
            await asyncio.sleep(self._min_delay - elapsed)
        self._last_request_time = time.time()
        
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""
            
    async def _fetch_url(self, url: str) -> str:
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
        ]
        headers_base = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        for attempt in range(3):
            headers = dict(headers_base)
            headers['User-Agent'] = random.choice(user_agents)
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, ssl=False) as response:
                        if response.status == 403:
                            self.logger.warning(f"HTTP 403 Forbidden for {url} (attempt {attempt+1}) with UA {headers['User-Agent']}")
                            await asyncio.sleep(1)
                            continue
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'application/pdf' in content_type:
                            try:
                                data = await response.read()
                                pdf_reader = PyPDF2.PdfReader(io.BytesIO(data))
                                text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                                return text
                            except Exception as e:
                                self.logger.error(f"Failed to parse PDF {url}: {e}")
                                return ''
                        elif 'text/html' in content_type:
                            html = await response.text()
                            return html
                        else:
                            self.logger.warning(f"Unsupported content type {content_type} for {url}")
                            return ''
            except aiohttp.ClientResponseError as e:
                if 'Header value is too long' in str(e):
                    self.logger.warning(f"Skipping URL {url} due to header size limit")
                    return ''
                self.logger.error(f"HTTP error for {url}: {e}")
                return ''
            except Exception as e:
                if 'Header value is too long' in str(e):
                    self.logger.warning(f"Header value too long for {url}, skipping")
                    return ''
                self.logger.error(f"Error fetching {url} (attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
                continue
        self.logger.error(f"Failed to fetch {url} after 3 attempts")
        return ''

    async def is_available(self) -> bool:
        """Check if any search engine is available."""
        # Try DuckDuckGo first
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            next(self.session.get("test", max_results=1))
            return True
        except Exception:
            pass

        # Try Tavily
        if self.tavily_api_keys:
            try:
                url = "https://api.tavily.com/search"
                headers = {"Authorization": f"Bearer {self.tavily_api_keys[0]}"}
                data = {"query": "test", "search_depth": "basic"}
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                return True
            except Exception:
                pass

        # Try Google
        if self.google_api_key and self.google_cse_id:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": self.google_api_key,
                    "cx": self.google_cse_id,
                    "q": "test"
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                return True
            except Exception:
                pass

        logger.error("No search engines available")
        return False

    def get_first_link(self) -> Optional[str]:
        """Get the first link from search results."""
        if not self.search_results:
            return None
            
        return self.search_results[0].get("link")

    def next_page(self) -> bool:
        """Move to the next page of results."""
        if self.current_page >= self.max_pages:
            return False
            
        self.current_page += 1
        return True

    def previous_page(self) -> bool:
        """Move to the previous page of results."""
        if self.current_page <= 1:
            return False
            
        self.current_page -= 1
        return True

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_urls(self, query: str, max_results: int = 10) -> List[str]:
        """Get only URLs from search results."""
        results = await self.search(query, max_results)
        return [result.get("link", "") for result in results if result.get("link")]

async def cleanup_aiohttp():
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()
    await asyncio.get_event_loop().shutdown_asyncgens()
    # Close all open connectors (if any)
    for conn in getattr(aiohttp, 'TCPConnector', []):
        try:
            await conn.close()
        except Exception:
            pass 