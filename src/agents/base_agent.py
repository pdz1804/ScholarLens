"""
Base agent class for TechAuthor system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from ..core.config import config_manager
from ..utils.logger import get_logger


class BaseAgent(ABC):
    """Base class for all agents in the TechAuthor system."""
    
    def __init__(self, agent_name: str):
        """Initialize base agent.
        
        Args:
            agent_name: Name of the agent
        """
        self.agent_name = agent_name
        self.config = config_manager.config
        self.logger = get_logger(f"{agent_name}Agent")
        self.is_initialized = False
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "last_request_time": None
        }
    
    async def initialize(self) -> None:
        """Initialize the agent. Should be implemented by subclasses."""
        if self.is_initialized:
            return
        
        self.logger.info(f"Initializing {self.agent_name} agent")
        await self._initialize_impl()
        self.is_initialized = True
        self.logger.info(f"{self.agent_name} agent initialized successfully")
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization. Must be implemented by subclasses."""
        pass
    
    async def process(self, *args, **kwargs) -> Any:
        """Process a request with metrics tracking.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Processing result
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        self.metrics["total_requests"] += 1
        self.metrics["last_request_time"] = start_time
        
        try:
            result = await self._process_impl(*args, **kwargs)
            self.metrics["successful_requests"] += 1
            
            # Update average processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            total_time = (
                self.metrics["average_processing_time"] * 
                (self.metrics["successful_requests"] - 1)
            )
            self.metrics["average_processing_time"] = (
                (total_time + processing_time) / self.metrics["successful_requests"]
            )
            
            self.logger.debug(
                f"{self.agent_name} processed request in {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"{self.agent_name} processing failed: {e}")
            raise
    
    @abstractmethod
    async def _process_impl(self, *args, **kwargs) -> Any:
        """Implementation-specific processing. Must be implemented by subclasses.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Processing result
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "agent_name": self.agent_name,
            "is_initialized": self.is_initialized,
            **self.metrics
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "agent_name": self.agent_name,
            "metrics": self.get_metrics()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        self.logger.info(f"Shutting down {self.agent_name} agent")
        await self._shutdown_impl()
        self.is_initialized = False
    
    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown. Can be overridden by subclasses."""
        pass
