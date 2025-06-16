import functools
import time
import asyncio
from typing import Callable, Any, Union, Awaitable

def retry_wrapper(func: Callable) -> Callable:
    """Decorator that retries a function if it fails. Handles both sync and async functions."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        return None

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        return None

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

def validate_responses(func: Callable) -> Callable:
    """Decorator that validates the response from a function."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        result = await func(*args, **kwargs)
        if result is None or result is False:
            raise ValueError("Invalid response from function")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        result = func(*args, **kwargs)
        if result is None or result is False:
            raise ValueError("Invalid response from function")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper 