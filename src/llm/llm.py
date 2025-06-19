"""LLM class for handling language model interactions."""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union
import tiktoken
from config.config import Config
from utils.token_tracker import TokenTracker
from datetime import datetime, timezone
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
import openai
import json
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential

TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

logger = logging.getLogger(__name__)

class LLM:
    """LLM class for handling language model interactions."""
    
    _instance = None
    _initialized = False
    _init_lock = asyncio.Lock()
    _lock = asyncio.Lock()

    def __new__(cls, config: Config, model: str = "gpt-4o"):
        """Ensure only one instance of LLM exists.
        
        Args:
            config: Configuration instance
            model: Model name (default: gpt-4o)
        """
        if cls._instance is None:
            cls._instance = super(LLM, cls).__new__(cls)
            # Initialize model in the instance
            cls._instance.model = model
        return cls._instance

    def __init__(self, config: Config, model: str = None):
        """Initialize LLM with configuration and model.
        
        Args:
            config: Configuration instance
            model: Optional model name override (default: None, uses config)
        """
        if self._initialized:
            return

        try:
            if not isinstance(config, Config):
                raise ValueError("config must be an instance of Config")
                
            self.config = config
            # Get model from config if not provided
            self.model = model or self.config.get('llm', 'model', 'gpt-4o')
            # Ensure we're using the Azure deployment name for the model
            self.azure_deployment = self.config.get('openai', 'azure_deployment')
            if not self.azure_deployment:
                raise ValueError("Azure OpenAI deployment name not configured")
                
            self._token_tracker = None  # Initialize TokenTracker lazily
            self._client = None
            self._rate_limit = None
            self._last_request_time = time.time()
            self.max_retries = 5  # Increased from 3
            self.retry_delay = 2.0  # Increased from 1.0
            self.timeout = 30.0  # Added timeout configuration
            logger.info(f"LLM instance created with model: {self.model}, Azure deployment: {self.azure_deployment}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise ValueError(f"LLM initialization failed: {str(e)}")

    @property
    async def token_tracker(self) -> TokenTracker:
        """Lazy initialization of TokenTracker."""
        if self._token_tracker is None:
            self._token_tracker = TokenTracker(self.config)
            await self._token_tracker.initialize()
        return self._token_tracker

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True
    )
    async def _create_test_completion(self, deployment_name: str) -> bool:
        """Create a test completion with retry logic."""
        try:
            response = await self._client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                max_tokens=10,
                timeout=self.timeout
            )
            return True
        except Exception as e:
            logger.warning(f"Test completion failed: {str(e)}")
            raise

    async def initialize(self):
        """Initialize async components."""
        async with self._init_lock:
            if self._initialized:
                return

            try:
                # Get API configuration
                api_key = self.config.get('openai', 'azure_api_key')
                api_version = self.config.get('openai', 'azure_api_version_chat')
                endpoint = self.config.get('openai', 'azure_endpoint')
                deployment_name = self.config.get('openai', 'azure_deployment')

                if not all([api_key, api_version, endpoint, deployment_name]):
                    missing = []
                    if not api_key: missing.append("AZURE_OPENAI_API_KEY")
                    if not api_version: missing.append("AZURE_API_VERSION_CHAT")
                    if not endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
                    if not deployment_name: missing.append("AZURE_OPENAI_DEPLOYMENT")
                    raise ValueError(f"Missing required Azure OpenAI configuration: {', '.join(missing)}")

                # Initialize client with timeout
                self._client = AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    timeout=self.timeout,
                    max_retries=self.max_retries
                )
                
                # Test the connection with retries
                try:
                    await self._create_test_completion(deployment_name)
                    logger.info("Azure OpenAI connection test successful")
                except Exception as e:
                    logger.error(f"Failed to connect to Azure OpenAI after retries: {str(e)}")
                    raise ValueError(f"Could not establish connection to Azure OpenAI: {str(e)}")
                
                # Initialize token tracker
                self._token_tracker = TokenTracker(self.config)
                await self._token_tracker.initialize()
                
                self._initialized = True
                logger.info("LLM async components initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize async components: {str(e)}")
                self._initialized = False
                raise ValueError(f"Async initialization failed: {str(e)}")

    def _ensure_initialized(self):
        """Ensure LLM is initialized before use."""
        if not self._initialized:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        if not self._client:
            raise RuntimeError("LLM components not properly initialized")

    @property
    def client(self) -> AsyncAzureOpenAI:
        """Get the Azure OpenAI client."""
        self._ensure_initialized()
        return self._client

    async def _apply_rate_limit(self):
        """Apply rate limiting to API requests."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            # Minimum delay between requests (in seconds)
            min_delay = 0.1
            
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                await asyncio.sleep(sleep_time)
            
            self._last_request_time = time.time()

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                # Execute the function synchronously since AzureOpenAI doesn't support async
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(wait_time)

    async def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> Any:
        """Generate a chat completion using the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate (default: 1000)
            
        Returns:
            The chat completion response
        """
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
                
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Get deployment name from config
            deployment_name = self.config.get('openai', 'azure_deployment')
            if not deployment_name:
                raise ValueError("Azure OpenAI deployment name not configured")
            
            # Truncate each message to fit within context window
            truncated_messages = []
            for msg in messages:
                truncated_content = self._truncate_to_token_limit(msg['content'], 8192, self.azure_deployment)
                truncated_messages.append({**msg, 'content': truncated_content})
            
            # Generate response with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=deployment_name,  # Use deployment name instead of model name
                        messages=truncated_messages,
                        temperature=0.7,
                        max_tokens=max_tokens
                    )
                    
                    # Track token usage
                    if hasattr(response, 'usage'):
                        token_tracker = await self.token_tracker
                        await token_tracker.track_usage(
                            model=self.azure_deployment,  # Use Azure deployment name for tracking
                            input_tokens=response.usage.prompt_tokens,
                            output_tokens=response.usage.completion_tokens
                        )
                    
                    return response
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            if "Connection error" in str(e):
                logger.error("Failed to connect to Azure OpenAI. Please check your endpoint and API key configuration.")
            raise

    def _truncate_to_token_limit(self, text: str, max_tokens: int = 8192, model: str = None) -> str:
        """Truncate text to a maximum number of tokens using tiktoken for accuracy."""
        model_name = model or self.azure_deployment or self.config.get('llm', 'embedding_model')
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            logger.warning(f"Truncating text from {len(tokens)} to {max_tokens} tokens for model {model_name}.")
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        return text

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
            # Apply rate limiting
            await self._apply_rate_limit()
            # Truncate input
            truncated_text = self._truncate_to_token_limit(text, 8192, self.config.get('llm', 'embedding_model'))
            # Get embedding with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.get('llm', 'embedding_model'),
                        input=truncated_text
                    )
                    return response.data[0].embedding
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return len(TIKTOKEN_ENC.encode(text))

    async def generate(self, prompt: str) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Prompt to generate from
            
        Returns:
            Generated text
        """
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
                
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Truncate input
            truncated_prompt = self._truncate_to_token_limit(prompt, 8192, self.azure_deployment)
            
            # Generate response with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.azure_deployment,  # Use Azure deployment name
                        messages=[{"role": "user", "content": truncated_prompt}],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    # Track token usage
                    if hasattr(response, 'usage'):
                        token_tracker = await self.token_tracker
                        await token_tracker.track_usage(
                            model=self.azure_deployment,  # Use Azure deployment name for tracking
                            input_tokens=response.usage.prompt_tokens,
                            output_tokens=response.usage.completion_tokens
                        )
                    
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    async def generate_with_context(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate text from prompt with context.
        
        Args:
            prompt: Prompt to generate from
            context: List of context messages
            
        Returns:
            Generated text
        """
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
                
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Truncate input
            truncated_prompt = self._truncate_to_token_limit(prompt, 8192, self.azure_deployment)
            truncated_context = []
            for ctx in context:
                truncated_content = self._truncate_to_token_limit(ctx["content"], 8192, self.azure_deployment)
                truncated_context.append({"role": ctx["role"], "content": truncated_content})
            
            # Prepare messages
            messages = truncated_context + [{"role": "user", "content": truncated_prompt}]
            
            # Generate response with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.azure_deployment,  # Use Azure deployment name
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    # Track token usage
                    if hasattr(response, 'usage'):
                        token_tracker = await self.token_tracker
                        await token_tracker.track_usage(
                            model=self.azure_deployment,  # Use Azure deployment name for tracking
                            input_tokens=response.usage.prompt_tokens,
                            output_tokens=response.usage.completion_tokens
                        )
                    
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    
        except Exception as e:
            logger.error(f"Error generating text with context: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
            # Apply rate limiting
            await self._apply_rate_limit()
            # Truncate input
            truncated_text = self._truncate_to_token_limit(text, 8192, self.config.get('llm', 'embedding_model'))
            # Generate embedding with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.get('llm', 'embedding_model'),
                        input=truncated_text
                    )
                    return response.data[0].embedding
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def execute_with_context(self, task: Dict[str, Any], context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute task with context.
        
        Args:
            task: Task to execute
            context: List of context messages
            
        Returns:
            Task execution result
        """
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
                
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Truncate context
            truncated_context = []
            for msg in context:
                truncated_content = self._truncate_to_token_limit(msg['content'], 8192, self.azure_deployment)
                truncated_context.append({**msg, 'content': truncated_content})
            
            # Prepare messages
            messages = []
            for ctx in truncated_context:
                messages.append({"role": ctx["role"], "content": json.dumps(ctx)})
            messages.append({"role": "user", "content": json.dumps(task)})
            
            # Generate response with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.azure_deployment,  # Use Azure deployment name
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    # Track token usage
                    if hasattr(response, 'usage'):
                        token_tracker = await self.token_tracker
                        await token_tracker.track_usage(
                            model=self.azure_deployment,  # Use Azure deployment name for tracking
                            input_tokens=response.usage.prompt_tokens,
                            output_tokens=response.usage.completion_tokens
                        )
                    
                    # Parse response
                    result = json.loads(response.choices[0].message.content)
                    return result
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    
        except Exception as e:
            logger.error(f"Error executing task with context: {str(e)}")
            raise

    async def prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for chat completion.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate response.
        
        Args:
            response: Response to validate
            
        Returns:
            True if response is valid, False otherwise
        """
        try:
            if not isinstance(response, dict):
                return False
            if "choices" not in response or not response["choices"]:
                return False
            choice = response["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                return False
            return True
        except Exception:
            return False
