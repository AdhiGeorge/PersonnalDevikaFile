import logging
import asyncio
from typing import Dict, Any, Optional
from config.config import Config
from utils.logger import Logger
from openai import AzureOpenAI, AsyncAzureOpenAI

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    def __init__(self, config: Config):
        """Initialize Azure OpenAI client with configuration."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        self.config = config
        self._logger = None
        self._client = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def logger(self) -> Logger:
        """Lazy initialization of logger."""
        if self._logger is None:
            self._logger = Logger(self.config)
        return self._logger

    @property
    def client(self) -> AsyncAzureOpenAI:
        """Lazy initialization of Azure OpenAI client."""
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self.config.get_azure_openai_api_key(),
                api_version=self.config.get_openai_api_version(),
                azure_endpoint=self.config.get_azure_openai_endpoint()
            )
        return self._client

    async def initialize(self):
        """Initialize async components."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                # Initialize logger
                await self.logger.initialize()
                
                # Test Azure OpenAI client
                try:
                    client = self.client
                    # Test chat completion with a simple prompt
                    response = await client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "test"}],
                        temperature=0.7,
                        max_tokens=10
                    )
                    logger.info("Azure OpenAI client test successful")
                except Exception as e:
                    logger.error(f"Azure OpenAI client test failed: {str(e)}")
                    raise ValueError(f"Failed to connect to Azure OpenAI: {str(e)}")
                
                self._initialized = True
                logger.info("Azure OpenAI client async components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize async components: {str(e)}")
                raise ValueError(f"Async initialization failed: {str(e)}")

    def _ensure_initialized(self):
        """Ensure client is initialized before use."""
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        if not self._logger or not self._client:
            raise RuntimeError("Client components not properly initialized")

    async def chat_completion(self, messages: list, model: str = "gpt-4") -> Any:
        """Generate a chat completion using Azure OpenAI."""
        try:
            self._ensure_initialized()
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            if "Connection error" in str(e):
                logger.error("Failed to connect to Azure OpenAI. Please check your endpoint and API key configuration.")
            raise

    def _truncate_to_token_limit(self, text: str, max_tokens: int = 8192) -> str:
        # Simple whitespace-based truncation; replace with tokenizer if available
        tokens = text.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens])
        return text

    async def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> list:
        """Get embeddings for text using Azure OpenAI."""
        try:
            self._ensure_initialized()
            truncated_text = self._truncate_to_token_limit(text, 8192)
            response = await self.client.embeddings.create(
                model=model,
                input=truncated_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self._initialized:
                # Add cleanup logic here
                pass
        except Exception as e:
            logger.error(f"Error cleaning up Azure OpenAI client: {str(e)}")
            raise ValueError(f"Azure OpenAI client cleanup failed: {str(e)}") 