import asyncio
import logging
from typing import Dict, List, Optional, Any
import tiktoken
from src.config import Config
from src.utils.token_tracker import TokenTracker
from datetime import datetime, timezone
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, config: Config):
        """Initialize the LLM with configuration."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        self.config = config
        self.token_tracker = TokenTracker(config)
        self._client = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Initialize rate limiting
        self._rate_limit = {}
        self.requests_per_minute = 60
        self.burst_size = 10
        
        # Initialize cache
        self._cache = {}
        
        # Initialize lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"LLM initialized with model: {self.config.model}")

    @property
    def client(self) -> AsyncAzureOpenAI:
        """Get or create the Azure OpenAI client."""
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self.config.azure_api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.azure_endpoint
            )
        return self._client

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        project_name: Optional[str] = None
    ) -> ChatCompletion:
        """Make a chat completion request to Azure OpenAI with retries."""
        # Use provided model or default from config
        model = model or self.config.model
        
        # Validate configuration
        if not self.config.azure_api_key or not self.config.azure_endpoint:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        # Set up parameters
        params = {
            "model": self.config.deployment_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Rate limiting
        async with self._lock:
            now = datetime.now(timezone.utc)
            window = now.replace(second=0, microsecond=0)
            if model not in self._rate_limit:
                self._rate_limit[model] = {}
            if window not in self._rate_limit[model]:
                self._rate_limit[model][window] = 0
            if self._rate_limit[model][window] >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {model}, waiting...")
                await asyncio.sleep(60)
            self._rate_limit[model][window] += 1

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(**params)
                
                # Track token usage
                if hasattr(response, 'usage'):
                    self.token_tracker.track_usage(
                        model=model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        project_name=project_name
                    )
                
                return response

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to get response after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text using Azure OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def get_token_count(self, text: str) -> int:
        """Get token count for text using tiktoken."""
        try:
            return len(TIKTOKEN_ENC.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise 