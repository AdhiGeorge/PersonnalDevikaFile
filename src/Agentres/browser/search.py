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
from Agentres.config.config import Config
from Agentres.utils.token_tracker import TokenTracker
import socket
from urllib.parse import quote_plus
from dotenv import load_dotenv
import aiohttp
import json
import re
from datetime import datetime

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
        
        # API keys
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        
        # Search engine flags
        self.use_google = bool(self.google_api_key and self.google_cse_id)
        self.use_tavily = bool(self.tavily_api_key)
        
        # Log configuration
        self.logger.info(f"Google API Key: {'Configured' if self.use_google else 'Not configured'}")
        self.logger.info(f"Google CSE ID: {'Configured' if self.use_google else 'Not configured'}")
        self.logger.info(f"Tavily API Key: {'Configured' if self.use_tavily else 'Not configured'}")
        self.logger.info(f"Search engine initialized with Google API: {self.use_google}, Tavily API: {self.use_tavily}")
        
        # Initialize async components
        self._init_async()
        
    async def _init_async(self):
        """Initialize async components."""
        try:
            # Initialize aiohttp session
            self.session = aiohttp.ClientSession()
            self._initialized = True
            self.logger.info("Search engine async components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize search engine: {str(e)}")
            raise
            
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using available search engines.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not self._initialized:
            await self._init_async()
            
        try:
            # Try DuckDuckGo first
            results = await self._duckduckgo_search(query, max_results)
            
            # If DuckDuckGo fails, try Tavily
            if not results and self.use_tavily:
                results = await self._tavily_search(query, max_results)
                
            # If Tavily fails, try Google
            if not results and self.use_google:
                results = await self._google_search(query, max_results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []
            
    async def _duckduckgo_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Respect rate limiting
            await self._rate_limit()
            
            # Use DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "no_redirect": 1
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"DuckDuckGo API returned status {response.status}")
                    
                data = await response.json()
                
                # Extract results
                results = []
                for result in data.get("Results", [])[:max_results]:
                    results.append({
                        "url": result.get("FirstURL", ""),
                        "title": result.get("Text", ""),
                        "snippet": result.get("Text", ""),
                        "source": self._extract_domain(result.get("FirstURL", "")),
                        "relevance_score": 0.8  # DuckDuckGo doesn't provide relevance scores
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
            
    async def _tavily_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Tavily API.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Respect rate limiting
            await self._rate_limit()
            
            # Use Tavily API
            url = "https://api.tavily.com/search"
            headers = {
                "Authorization": f"Bearer {self.tavily_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "query": query,
                "max_results": max_results
            }
            
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"Tavily API returned status {response.status}")
                    
                data = await response.json()
                
                # Extract results
                results = []
                for result in data.get("results", [])[:max_results]:
                    results.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("content", ""),
                        "source": self._extract_domain(result.get("url", "")),
                        "relevance_score": result.get("score", 0.0)
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Tavily search failed: {str(e)}")
            return []
            
    async def _google_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Respect rate limiting
            await self._rate_limit()
            
            # Use Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": max_results
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Google API returned status {response.status}")
                    
                data = await response.json()
                
                # Extract results
                results = []
                for item in data.get("items", [])[:max_results]:
                    results.append({
                        "url": item.get("link", ""),
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "source": self._extract_domain(item.get("link", "")),
                        "relevance_score": 0.8  # Google doesn't provide relevance scores
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Google search failed: {str(e)}")
            return []
            
    async def _rate_limit(self):
        """Respect rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_delay:
            await asyncio.sleep(self._min_delay - elapsed)
        self._last_request_time = time.time()
        
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
            return parsed.netloc
        except Exception as e:
            self.logger.error(f"Error extracting domain: {str(e)}")
            return ""
            
    def _get_random_headers(self) -> Dict[str, str]:
        """Get random headers for requests.
        
        Returns:
            Dictionary of headers
        """
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
        ]
        
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

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
        if self.tavily_api_key:
            try:
                url = "https://api.tavily.com/search"
                headers = {"Authorization": f"Bearer {self.tavily_api_key}"}
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