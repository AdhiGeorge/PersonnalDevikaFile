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
from src.config import Config
from src.utils.token_tracker import TokenTracker
import socket
from urllib.parse import quote_plus
from dotenv import load_dotenv
import aiohttp
import json
import re

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
    """Search engine implementation using DuckDuckGo."""
    
    # List of modern browser user agents
    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        # Chrome on Linux
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    
    # Common referrers to make requests look more natural
    REFERRERS = [
        "https://www.google.com/",
        "https://www.bing.com/",
        "https://duckduckgo.com/",
        "https://www.reddit.com/",
        "https://www.wikipedia.org/"
    ]
    
    # List of domains to exclude (video sites, social media, etc.)
    EXCLUDED_DOMAINS = [
        'youtube.com',
        'youtu.be',
        'vimeo.com',
        'dailymotion.com',
        'facebook.com',
        'twitter.com',
        'instagram.com',
        'tiktok.com',
        'linkedin.com',
        'reddit.com',
        'pinterest.com',
        'tumblr.com',
        'flickr.com',
        'imgur.com',
        'slideshare.net',
        'prezi.com',
        'docs.google.com',
        'drive.google.com',
        'dropbox.com',
        'onedrive.live.com',
        'box.com',
        'mega.nz',
        'mediafire.com',
        'wetransfer.com',
        'sendspace.com',
        'rapidshare.com',
        '4shared.com',
        'scribd.com',
        'issuu.com',
        'slides.com',
        'speakerdeck.com',
        'slideshare.net',
        'prezi.com',
        'docs.google.com',
        'drive.google.com',
        'dropbox.com',
        'onedrive.live.com',
        'box.com',
        'mega.nz',
        'mediafire.com',
        'wetransfer.com',
        'sendspace.com',
        'rapidshare.com',
        '4shared.com',
        'scribd.com',
        'issuu.com',
        'slides.com',
        'speakerdeck.com'
    ]
    
    def __init__(self, config: Config):
        """Initialize search engine with configuration."""
        self.config = config
        self.token_tracker = TokenTracker(config)
        self.ddgs = None
        self._last_request_time = 0
        self._min_delay = 2.0
        self._max_delay = 4.0
        self._max_retries = 3
        self._retry_delay = 5.0  # Base delay for retries
        # Get API keys directly from config attributes
        self.google_api_key = config.google_api_key
        self.google_cse_id = config.google_cse_id
        self.tavily_api_key = config.tavily_api_key
        
        # Debug log the actual values (first few characters only)
        logger.info(f"Google API Key: {self.google_api_key[:8]}..." if self.google_api_key else "Not set")
        logger.info(f"Google CSE ID: {self.google_cse_id[:8]}..." if self.google_cse_id else "Not set")
        logger.info(f"Tavily API Key: {self.tavily_api_key[:8]}..." if self.tavily_api_key else "Not set")
        
        logger.info(f"Search engine initialized with Google API: {bool(self.google_api_key)}, Tavily API: {bool(self.tavily_api_key)}")
        if not self.google_api_key:
            logger.warning("Google API key not found. Please set GOOGLE_API_KEY in .env file")
        if not self.google_cse_id:
            logger.warning("Google CSE ID not found. Please set GOOGLE_CSE_ID in .env file")
        if not self.tavily_api_key:
            logger.warning("Tavily API key not found. Please set TAVILY_API_KEY in .env file")
        self._request_count = 0
        self._max_requests_before_delay = 2  # Reduced number of requests before delay
        self._long_delay_interval = 15.0  # Increased delay after multiple requests
        self._session = None
        
    def _get_random_headers(self) -> Dict[str, str]:
        """Generate random headers for each request."""
        user_agent = random.choice(self.USER_AGENTS)
        referrer = random.choice(self.REFERRERS)
        
        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Referer': referrer,
            'Cache-Control': 'max-age=0'
        }
        
    def _wait_before_request(self):
        """Add a random delay between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_delay:
            delay = random.uniform(self._min_delay, self._max_delay)
            logger.info(f"Waiting {delay:.1f} seconds before next request...")
            time.sleep(delay)
        self._last_request_time = time.time()
        
    def _rate_limit(self):
        """Implement smart rate limiting with exponential backoff."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        # Increment request counter
        self._request_count += 1
        
        # If we've made multiple requests, add a longer delay
        if self._request_count >= self._max_requests_before_delay:
            sleep_time = self._long_delay_interval
            self._request_count = 0
        else:
            # Normal rate limiting
            if time_since_last < self._min_delay:
                sleep_time = self._min_delay - time_since_last
            else:
                sleep_time = 0
                
        if sleep_time > 0:
            # Add some randomness to the delay
            sleep_time += random.uniform(1.0, 2.0)
            time.sleep(sleep_time)
            
        self._last_request_time = time.time()
        
    async def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo."""
        try:
            if not self.ddgs:
                self.ddgs = DDGS()
            
            self._wait_before_request()
            results = []
            
            for r in self.ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r['title'],
                    'link': r['link'],
                    'snippet': r['body']
                })
            
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for scraping."""
        try:
            # Extract domain from URL
            domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if not domain:
                return False
            
            domain = domain.group(1).lower()
            
            # Check if domain is in excluded list
            if any(excluded in domain for excluded in self.EXCLUDED_DOMAINS):
                logger.debug(f"Excluding URL from excluded domain: {url}")
                return False
            
            # Check if URL is a file download
            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.7z', '.tar', '.gz']):
                logger.debug(f"Excluding URL as it's a file download: {url}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking URL validity: {str(e)}")
            return False

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using multiple engines and combine results."""
        results = []
        
        # Try Google Custom Search
        if self.google_api_key and self.google_cse_id:
            try:
                logger.info("Attempting Google search...")
                google_results = self._google_search(query, max_results)
                logger.info(f"Google search returned {len(google_results)} results")
                results.extend(google_results)
            except Exception as e:
                logger.error(f"Google search failed: {str(e)}")
        
        # Try Tavily
        if self.tavily_api_key:
            try:
                logger.info("Attempting Tavily search...")
                tavily_results = self._tavily_search(query, max_results)
                logger.info(f"Tavily search returned {len(tavily_results)} results")
                results.extend(tavily_results)
            except Exception as e:
                logger.error(f"Tavily search failed: {str(e)}")
                if "401" in str(e):
                    logger.error("Tavily API key appears to be invalid. Please check your API key.")
        
        if not results:
            logger.warning("No results found from any search engine")
            return []
        
        # Remove duplicates and invalid URLs
        seen_urls = set()
        unique_results = []
        for result in results:
            if result['link'] not in seen_urls and self._is_valid_url(result['link']):
                seen_urls.add(result['link'])
                unique_results.append(result)
        
        logger.info(f"Returning {len(unique_results)} unique and valid results")
        return unique_results[:max_results]

    def _google_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        url = "https://www.googleapis.com/customsearch/v1"
        
        # Debug log the actual values being used
        logger.debug(f"Using Google API Key: {self.google_api_key[:8]}...")
        logger.debug(f"Using Google CSE ID: {self.google_cse_id[:8]}...")
        
        params = {
            'key': self.google_api_key,
            'cx': self.google_cse_id,
            'q': query,
            'num': min(max_results * 2, 10)  # Request more results to account for filtering
        }
        
        logger.debug(f"Google search URL: {url}")
        logger.debug(f"Google search params: {params}")
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'items' not in data:
            logger.warning(f"Google search returned no items. Response: {data}")
            return []
        
        results = []
        for item in data['items']:
            if self._is_valid_url(item.get('link', '')):
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
        return results

    def _tavily_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        url = "https://api.tavily.com/search"
        
        # Debug log the actual values being used
        logger.debug(f"Using Tavily API Key: {self.tavily_api_key[:8]}...")
        
        # Ensure API key is properly formatted
        api_key = self.tavily_api_key.strip()
        if not api_key.startswith('tvly-'):
            logger.warning("Tavily API key should start with 'tvly-'. Please check your API key.")
        
        headers = {
            "X-Api-Key": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results * 2  # Request more results to account for filtering
        }
        
        logger.debug(f"Tavily search URL: {url}")
        logger.debug(f"Tavily search headers: {headers}")
        logger.debug(f"Tavily search data: {data}")
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        data = response.json()
        
        if 'results' not in data:
            logger.warning(f"Tavily search returned no results. Response: {data}")
            return []
        
        results = []
        for result in data['results']:
            if self._is_valid_url(result.get('url', '')):
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('url', ''),
                    'snippet': result.get('content', '')
                })
        return results

    async def is_available(self) -> bool:
        """Check if any search engine is available."""
        # Try DuckDuckGo first
        try:
            if not self.ddgs:
                self.ddgs = DDGS()
            next(self.ddgs.text("test", max_results=1))
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
        if self._session and not self._session.closed:
            await self._session.close() 