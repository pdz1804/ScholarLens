"""
Multi-LLM Provider Manager for TechAuthor System
Supports OpenAI, Google Gemini, and Ollama providers with unified interface.
"""

import os
import yaml
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import logging


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from the LLM."""
        pass
        
    @abstractmethod
    def format_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format messages for the specific provider."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv(config.get('api_key_env', 'OPENAI_API_KEY'))
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable: {config.get('api_key_env')}")
    
    def format_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format messages for OpenAI API."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    async def generate(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini", **kwargs) -> str:
        """Generate response using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Merge default parameters with kwargs
        params = {**self.config.get('parameters', {}), **kwargs}
        
        # Extract timeout for HTTP client (don't send to API)
        timeout = params.pop('timeout', 30)
        
        payload = {
            "model": model,
            "messages": messages,
            **params
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"OpenAI API error {response.status}: {error_text}")
                    raise Exception(f"OpenAI API error: {response.status}")
                
                result = await response.json()
                return result['choices'][0]['message']['content']


class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv(config.get('api_key_env', 'GEMINI_API_KEY'))
        self.base_url = config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')
        
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable: {config.get('api_key_env')}")
    
    def format_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format messages for Gemini API."""
        # Gemini combines system and user prompts
        combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
        return [{"role": "user", "parts": [{"text": combined_prompt}]}]
    
    async def generate(self, messages: List[Dict[str, str]], model: str = "gemini-1.5-flash", **kwargs) -> str:
        """Generate response using Gemini API."""
        # Merge default parameters with kwargs
        params = {**self.config.get('parameters', {}), **kwargs}
        
        # Extract timeout for HTTP client
        timeout = params.pop('timeout', 30)
        
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": params.get('temperature', 0.1),
                "maxOutputTokens": params.get('max_output_tokens', 2000)
            }
        }
        
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Gemini API error {response.status}: {error_text}")
                    raise Exception(f"Gemini API error: {response.status}")
                
                result = await response.json()
                return result['candidates'][0]['content']['parts'][0]['text']


class OllamaProvider(LLMProvider):
    """Ollama local provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
    
    def format_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format messages for Ollama API."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    async def generate(self, messages: List[Dict[str, str]], model: str = "llama3.2:3b", **kwargs) -> str:
        """Generate response using Ollama API."""
        # Merge default parameters with kwargs
        params = {**self.config.get('parameters', {}), **kwargs}
        
        # Extract timeout for HTTP client
        timeout = params.pop('timeout', 60)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": params.get('temperature', 0.1),
                "num_predict": params.get('num_predict', 2000)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Ollama API error {response.status}: {error_text}")
                    raise Exception(f"Ollama API error: {response.status}")
                
                result = await response.json()
                return result['message']['content']


class LLMManager:
    """Unified manager for multiple LLM providers."""
    
    def __init__(self, config_path: str = "config/llm_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.main_config = None  # Will be set by system.py
        self.providers = {}
        self.default_provider = self.config.get('default_provider', 'openai')
        
        # Initialize providers
        self._initialize_providers()
    
    def set_main_config(self, main_config):
        """Set the main application configuration for retrieval settings."""
        self.main_config = main_config
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'default_provider': 'openai',
            'providers': {
                'openai': {
                    'api_key_env': 'OPENAI_API_KEY',
                    'models': {
                        'classification': 'gpt-4o-mini',
                        'retrieval': 'gpt-4o-mini',
                        'analysis': 'gpt-4o-mini',
                        'synthesis': 'gpt-4o-mini',
                        'validation': 'gpt-4o-mini'
                    },
                    'parameters': {'temperature': 0.1, 'max_tokens': 2000}
                }
            }
        }
    
    def _initialize_providers(self):
        """Initialize available providers."""
        provider_classes = {
            'openai': OpenAIProvider,
            'gemini': GeminiProvider,
            'ollama': OllamaProvider
        }
        
        self.logger.info(f"Initializing LLM providers (default: {self.default_provider})")
        
        for provider_name, provider_config in self.config.get('providers', {}).items():
            if provider_name in provider_classes:
                try:
                    self.providers[provider_name] = provider_classes[provider_name](provider_config)
                    models = provider_config.get('models', {})
                    self.logger.info(f"{provider_name.upper()} provider initialized with models: {list(models.keys())}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {provider_name} provider: {e}")
        
        if not self.providers:
            self.logger.error("No LLM providers initialized successfully!")
        else:
            self.logger.info(f"Available providers: {list(self.providers.keys())}")
    
    def get_model_for_agent(self, agent_name: str, provider_name: str = None) -> tuple[str, str]:
        """Get the model and provider for a specific agent."""
        # Use specific provider or default
        provider_name = provider_name or self.default_provider
        
        # Get agent-specific config or use default
        agent_config = self.config.get('agents', {}).get(agent_name, {})
        agent_provider = agent_config.get('provider', 'default')
        agent_model = agent_config.get('model', 'default')
        
        # Resolve 'default' references
        if agent_provider == 'default':
            agent_provider = provider_name
        if agent_model == 'default':
            # Get the model for this agent type from provider config
            provider_config = self.config.get('providers', {}).get(agent_provider, {})
            models = provider_config.get('models', {})
            agent_model = models.get(agent_name.replace('_agent', ''), models.get('classification', 'gpt-4o-mini'))
        
        return agent_provider, agent_model
    
    async def generate(self, system_prompt: str, user_prompt: str, agent_name: str = "general", 
                      provider_name: str = None, model_name: str = None, **kwargs) -> str:
        """Generate response using specified or default provider."""
        # Determine provider and model
        if not provider_name or not model_name:
            provider_name, model_name = self.get_model_for_agent(agent_name, provider_name)
        
        # Get provider instance
        if provider_name not in self.providers:
            self.logger.warning(f"Provider {provider_name} not available, falling back to default")
            provider_name = self.default_provider
            
        if provider_name not in self.providers:
            raise ValueError(f"No providers available")
        
        provider = self.providers[provider_name]
        
        # Format messages and generate
        messages = provider.format_messages(system_prompt, user_prompt)
        
        # Log LLM usage at INFO level for visibility
        self.logger.info(f"{agent_name.upper()}: Using {provider_name.upper()} with model '{model_name}'")
        self.logger.debug(f"Prompt length: {len(system_prompt + user_prompt)} characters")
        
        try:
            response = await provider.generate(messages, model=model_name, **kwargs)
            self.logger.debug(f"Response length: {len(response)} characters")
            return response
        except Exception as e:
            self.logger.error(f"LLM generation failed for {agent_name}: {str(e)}")
            raise
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration from main config file."""
        # Use main config if available, otherwise fall back to llm config
        if self.main_config and hasattr(self.main_config, 'retrieval'):
            retrieval_obj = self.main_config.retrieval
            return {
                'initial_top_k': retrieval_obj.initial_top_k,
                'score_threshold': retrieval_obj.score_threshold,
                'final_top_k': retrieval_obj.final_top_k,
                'hybrid_alpha': retrieval_obj.hybrid_alpha,
                'default_top_k': retrieval_obj.default_top_k,
                'max_top_k': retrieval_obj.max_top_k,
                'min_similarity_threshold': retrieval_obj.min_similarity_threshold,
                'rerank': retrieval_obj.rerank,
                'rerank_top_k': retrieval_obj.rerank_top_k,
                'enable_score_filtering': retrieval_obj.enable_score_filtering,
                'normalize_scores': retrieval_obj.normalize_scores,
            }
        else:
            # Fallback to old method if main config not available
            retrieval_config = self.config.get('retrieval', {})
            return {
                'initial_top_k': retrieval_config.get('initial_top_k', 1000),
                'score_threshold': retrieval_config.get('score_threshold', 0.4),
                'final_top_k': retrieval_config.get('final_top_k', 20),
                'hybrid_alpha': retrieval_config.get('hybrid_alpha', 0.7),
                'default_top_k': retrieval_config.get('default_top_k', 10),
                'max_top_k': retrieval_config.get('max_top_k', 50),
                'min_similarity_threshold': retrieval_config.get('min_similarity_threshold', 0.7),
                'rerank': retrieval_config.get('rerank', True),
                'rerank_top_k': retrieval_config.get('rerank_top_k', 20),
                'enable_score_filtering': retrieval_config.get('enable_score_filtering', True),
                'normalize_scores': retrieval_config.get('normalize_scores', True)
            }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {
            'use_special_characters': False,
            'detailed_retrieval_logs': True,
            'show_paper_details': True,
            'performance_metrics': True
        })
