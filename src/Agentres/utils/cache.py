import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Cache:
    """Simple in-memory cache with expiration."""
    
    def __init__(self, default_ttl: int = 3600):
        """Initialize cache with default TTL in seconds."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        if not isinstance(key, str):
            raise ValueError("Cache key must be a string")
            
        ttl = ttl or self._default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            Cached value or default
        """
        if not isinstance(key, str):
            raise ValueError("Cache key must be a string")
            
        if key not in self._cache:
            return default
            
        entry = self._cache[key]
        if datetime.now() > entry['expires_at']:
            del self._cache[key]
            return default
            
        return entry['value']
        
    def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key to delete
        """
        if not isinstance(key, str):
            raise ValueError("Cache key must be a string")
            
        if key in self._cache:
            del self._cache[key]
            
    def clear(self) -> None:
        """Clear all values from the cache."""
        self._cache.clear()
        
    def get_all(self) -> Dict[str, Any]:
        """Get all non-expired values from the cache.
        
        Returns:
            Dictionary of all valid cache entries
        """
        now = datetime.now()
        return {
            k: v['value'] for k, v in self._cache.items()
            if now <= v['expires_at']
        }
        
    def cleanup(self) -> None:
        """Remove all expired entries from the cache."""
        now = datetime.now()
        expired = [
            k for k, v in self._cache.items()
            if now > v['expires_at']
        ]
        for k in expired:
            del self._cache[k] 