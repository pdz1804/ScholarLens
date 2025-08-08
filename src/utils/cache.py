"""
Cache management for TechAuthor system.
"""

import asyncio
import json
import time
import shutil
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import pickle
import hashlib
import logging
from pathlib import Path

from ..core.models import CacheEntry
from ..utils.logger import get_logger


async def clear_cached_data(logger):
    """Clear all cached data and index files."""
    cache_paths = [
        Path("data/embeddings/index"),
        Path("data/cache"),
        Path(".cache")
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache directory: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not clear {cache_path}: {e}")
    
    # Also clear any .pkl, .faiss, .index files in data directory
    data_path = Path("data")
    if data_path.exists():
        for cache_file in data_path.rglob("*.pkl"):
            try:
                cache_file.unlink()
                logger.info(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Could not remove {cache_file}: {e}")
                
        for cache_file in data_path.rglob("*.faiss"):
            try:
                cache_file.unlink() 
                logger.info(f"Removed index file: {cache_file}")
            except Exception as e:
                logger.warning(f"Could not remove {cache_file}: {e}")


class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        async with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self.cache[key]
                return None
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        async with self.lock:
            # Clean up expired entries if cache is full
            if len(self.cache) >= self.max_size:
                await self._cleanup_expired()
                
                # If still full, remove oldest entry
                if len(self.cache) >= self.max_size:
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
                    del self.cache[oldest_key]
            
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl
            )
            self.cache[key] = entry
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    async def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    async def health_check(self) -> Dict[str, Any]:
        """Get cache health information."""
        async with self.lock:
            expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
            return {
                "total_entries": len(self.cache),
                "expired_entries": expired_count,
                "max_size": self.max_size,
                "memory_usage_mb": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            total_size = 0
            for entry in self.cache.values():
                # Rough estimation using pickle
                total_size += len(pickle.dumps(entry.value))
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0


class CacheManager:
    """Main cache manager that can use different backends."""
    
    def __init__(
        self,
        backend: str = "memory",
        max_size: int = 1000,
        default_ttl: int = 3600,
        **kwargs
    ):
        """Initialize cache manager.
        
        Args:
            backend: Cache backend ('memory' or 'redis')
            max_size: Maximum cache size
            default_ttl: Default TTL in seconds
            **kwargs: Additional backend-specific arguments
        """
        self.backend = backend
        self.logger = get_logger()  # Use global logger
        
        if backend == "memory":
            self.cache = MemoryCache(max_size, default_ttl)
        elif backend == "redis":
            # Redis implementation would go here
            raise NotImplementedError("Redis backend not implemented yet")
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        try:
            hashed_key = self._hash_key(key)
            value = await self.cache.get(hashed_key)
            if value is not None:
                self.logger.debug(f"Cache hit for key: {key[:50]}...")
            return value
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        try:
            hashed_key = self._hash_key(key)
            await self.cache.set(hashed_key, value, ttl)
            self.logger.debug(f"Cache set for key: {key[:50]}...")
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed
        """
        try:
            hashed_key = self._hash_key(key)
            result = await self.cache.delete(hashed_key)
            if result:
                self.logger.debug(f"Cache delete for key: {key[:50]}...")
            return result
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            await self.cache.clear()
            self.logger.info("Cache cleared")
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key.
        
        Args:
            key: Original key
            
        Returns:
            Hashed key
        """
        return hashlib.sha256(key.encode('utf-8')).hexdigest()
    
    async def health_check(self) -> Dict[str, Any]:
        """Get cache health information.
        
        Returns:
            Health status information
        """
        try:
            cache_health = await self.cache.health_check()
            return {
                "backend": self.backend,
                "status": "healthy",
                **cache_health
            }
        except Exception as e:
            return {
                "backend": self.backend,
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown cache manager."""
        try:
            if hasattr(self.cache, 'shutdown'):
                await self.cache.shutdown()
            self.logger.info("Cache manager shutdown completed")
        except Exception as e:
            self.logger.error(f"Cache shutdown error: {e}")


# Global cache instance
cache_manager = CacheManager()
