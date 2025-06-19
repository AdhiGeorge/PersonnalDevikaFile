"""Prompt manager for handling system prompts."""

import os
import yaml
import logging
from typing import Dict, Optional, Any
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

class PromptManager:
    """Prompt manager for handling system prompts."""
    
    def __init__(self, config=None):
        """Initialize the prompt manager.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        self.prompts = {}
        self.prompts_file = os.path.join(os.path.dirname(__file__), 'prompts.yaml')
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize async components."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                # Load prompts from YAML file
                try:
                    with open(self.prompts_file, 'r') as f:
                        self.prompts = yaml.safe_load(f)
                    logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
                except Exception as e:
                    logger.error(f"Failed to load prompts: {str(e)}")
                    raise ValueError(f"Failed to load prompts: {str(e)}")
                
                # Test prompt retrieval
                test_prompt = "browser_interaction"
                try:
                    prompt = await self.get_prompt(test_prompt, skip_init_check=True)
                    if not prompt:
                        raise ValueError(f"Empty prompt returned for '{test_prompt}'")
                    logger.info("Prompt retrieval test successful")
                except Exception as e:
                    logger.error(f"Prompt retrieval test failed: {str(e)}")
                    raise ValueError(f"Failed to retrieve prompts: {str(e)}")
                
                self._initialized = True
                logger.info("Prompt manager initialized successfully")
                
            except Exception as e:
                # Clean up on failure
                self._initialized = False
                self.prompts = {}
                logger.error(f"Failed to initialize async components: {str(e)}")
                raise ValueError(f"Async initialization failed: {str(e)}")

    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        try:
            # Get prompts file path
            prompts_file = os.path.join(os.path.dirname(__file__), 'prompts.yaml')
            
            # Load prompts
            with open(prompts_file, 'r') as f:
                prompts = yaml.safe_load(f)
            
            # Convert prompts to strings if needed
            for key, value in prompts.items():
                if not isinstance(value, str):
                    logger.warning(f"Prompt '{key}' is not a string, converting to string")
                    prompts[key] = str(value)
            
            self.prompts = prompts
            logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
            
        except Exception as e:
            logger.error(f"Failed to load prompts: {str(e)}")
            raise ValueError(f"Failed to load prompts: {str(e)}")

    async def get_prompt(self, name: str, skip_init_check: bool = False) -> str:
        """Get a prompt by name.
        
        Args:
            name: The name of the prompt to retrieve
            skip_init_check: If True, skips the initialization check. Used during initialization testing.
            
        Returns:
            The prompt text
            
        Raises:
            ValueError: If the prompt cannot be retrieved
        """
        if not skip_init_check and not self._initialized:
            raise RuntimeError("Prompt manager not initialized. Call initialize() first.")
            
        try:
            if name not in self.prompts:
                raise ValueError(f"Prompt '{name}' not found")
            return str(self.prompts[name])
        except Exception as e:
            raise ValueError(f"Failed to get prompt '{name}': {str(e)}")

    def format_prompt(self, name: str, **kwargs) -> str:
        """Format a prompt with the given parameters.
        
        Args:
            name: The name of the prompt
            **kwargs: The parameters to format the prompt with
            
        Returns:
            str: The formatted prompt
        """
        try:
            if not self._initialized:
                raise RuntimeError("Prompt manager not initialized. Call initialize() first.")
                
            prompt = self.get_prompt(name)
            return prompt.format(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to format prompt '{name}': {str(e)}")
            raise ValueError(f"Failed to format prompt '{name}': {str(e)}")

    def get_all_prompts(self) -> Dict[str, str]:
        """Get all available prompts."""
        if not self._initialized:
            raise RuntimeError("Prompt manager not initialized. Call initialize() first.")
        return self.prompts.copy()

    def reload_prompts(self) -> None:
        """Reload prompts from the YAML file."""
        if not self._initialized:
            raise RuntimeError("Prompt manager not initialized. Call initialize() first.")
        self._load_prompts()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return
            
        try:
            # Reset state
            self._initialized = False
            self.prompts.clear()
            
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
            raise ValueError(f"Failed to cleanup: {str(e)}")

def validate_responses(func):
    """Decorator to validate responses."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await func(self, *args, **kwargs)
                if await self.validate_response(response):
                    return response
                logger.warning(f"Invalid response format on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        raise ValueError("Failed to get valid response after maximum retries")
    return wrapper 