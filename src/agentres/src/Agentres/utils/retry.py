import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_wrapper(
    func: Callable = None,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        func: The function to decorate (automatically passed by Python)
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated function that will retry on specified exceptions
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached due to the raise in the loop
            raise last_exception  # type: ignore
            
        return cast(Callable[..., T], wrapper)
        
    if func is None:
        return decorator
    return decorator(func) 