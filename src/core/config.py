"""
Core configuration management for TechAuthor system.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class SystemConfig(BaseModel):
    """System configuration model."""
    name: str = "TechAuthor"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4


class ParsingConfig(BaseModel):
    """Parsing configuration model."""
    method: str = "regex"  # Options: "regex", "llm"
    llm_fallback: bool = False  # If regex fails, fallback to LLM


class DataConfig(BaseModel):
    """Data configuration model."""
    parsing: ParsingConfig = ParsingConfig()
    datasets: Dict[str, Any] = {}
    processed_data_path: str = "./data/processed/"
    embeddings_cache_path: str = "./data/embeddings/"


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration model."""
    model_config = {"protected_namespaces": ()}
    model_type: str = "sentence-transformers"  # Options: "sentence-transformers", "openai"    
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    cache_embeddings: bool = True


class RetrievalConfig(BaseModel):
    """Retrieval configuration model."""
    # K-value settings
    default_top_k: int = 10          # Default number of final results returned to user
    max_top_k: int = 50              # Maximum allowed k value from CLI
    initial_top_k: int = 1000        # Initial retrieval count before filtering
    final_top_k: int = 20            # Final results passed to analysis agents
    rerank_top_k: int = 20           # Results to consider for reranking
    
    trends_initial_top_k: int = 5000             # Much higher retrieval for trend analysis
    trends_final_top_k: int = 1000               # More papers for comprehensive trend analysis
    trends_score_threshold: float = 0.1          # Much lower threshold for trends (vs 0.5 default)

    # Similarity threshold settings
    min_similarity_threshold: float = 0.7    # Higher threshold for strict filtering
    score_threshold: float = 0.4             # Default score threshold for document filtering
    
    # Hybrid search settings
    hybrid_alpha: float = 0.7                # Weight for dense vs sparse search
    rerank: bool = True                      # Enable result reranking
    
    # Search behavior settings
    enable_score_filtering: bool = True      # Enable filtering by similarity scores
    normalize_scores: bool = True            # Normalize scores across different search types
    
    # Sparse search configuration
    sparse_search: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent configuration model."""
    query_classifier: Dict[str, Any] = {}
    retrieval_agent: Dict[str, Any] = {}
    analysis_agent: Dict[str, Any] = {}
    synthesis_agent: Dict[str, Any] = {}
    validation_agent: Dict[str, Any] = {}


class APIConfig(BaseModel):
    """API configuration model."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    cors_origins: list = ["*"]
    rate_limit: str = "100/minute"


class Config(BaseModel):
    """Main configuration class."""
    system: SystemConfig = SystemConfig()
    data: DataConfig = DataConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    agents: AgentConfig = AgentConfig()
    api: APIConfig = APIConfig()
    scenarios: Dict[str, Any] = {}
    cache: Dict[str, Any] = {}
    logging: Dict[str, Any] = {}


class ConfigManager:
    """Configuration manager for the TechAuthor system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._load_environment_variables()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return str(Path(__file__).parent.parent.parent / "config" / "config.yaml")
    
    def _load_config(self) -> Config:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            return Config(**config_data)
        except FileNotFoundError:
            logging.warning(f"Configuration file not found: {self.config_path}")
            logging.info("Using default configuration")
            return Config()
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            logging.info("Using default configuration")
            return Config()
    
    def _load_environment_variables(self):
        """Load environment variables."""
        # Load .env file if it exists
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Override config with environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Update embedding configuration
        if os.getenv("EMBEDDING_MODEL"):
            self.config.embeddings.model_name = os.getenv("EMBEDDING_MODEL")
        if os.getenv("EMBEDDING_DIMENSION"):
            self.config.embeddings.dimension = int(os.getenv("EMBEDDING_DIMENSION"))
        
        # Update system configuration
        if os.getenv("LOG_LEVEL"):
            self.config.system.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("DEBUG"):
            self.config.system.debug = os.getenv("DEBUG").lower() == "true"
        
        # Update retrieval configuration
        if os.getenv("DEFAULT_TOP_K"):
            self.config.retrieval.default_top_k = int(os.getenv("DEFAULT_TOP_K"))
        if os.getenv("MAX_TOP_K"):
            self.config.retrieval.max_top_k = int(os.getenv("MAX_TOP_K"))
        if os.getenv("INITIAL_TOP_K"):
            self.config.retrieval.initial_top_k = int(os.getenv("INITIAL_TOP_K"))
        if os.getenv("FINAL_TOP_K"):
            self.config.retrieval.final_top_k = int(os.getenv("FINAL_TOP_K"))
        if os.getenv("RERANK_TOP_K"):
            self.config.retrieval.rerank_top_k = int(os.getenv("RERANK_TOP_K"))
        if os.getenv("MIN_SIMILARITY_THRESHOLD"):
            self.config.retrieval.min_similarity_threshold = float(os.getenv("MIN_SIMILARITY_THRESHOLD"))
        if os.getenv("SCORE_THRESHOLD"):
            self.config.retrieval.score_threshold = float(os.getenv("SCORE_THRESHOLD"))
        if os.getenv("HYBRID_ALPHA"):
            self.config.retrieval.hybrid_alpha = float(os.getenv("HYBRID_ALPHA"))
        if os.getenv("ENABLE_RERANK"):
            self.config.retrieval.rerank = os.getenv("ENABLE_RERANK").lower() == "true"
        if os.getenv("ENABLE_SCORE_FILTERING"):
            self.config.retrieval.enable_score_filtering = os.getenv("ENABLE_SCORE_FILTERING").lower() == "true"
        if os.getenv("NORMALIZE_SCORES"):
            self.config.retrieval.normalize_scores = os.getenv("NORMALIZE_SCORES").lower() == "true"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'system.debug')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any):
        """Update configuration value.
        
        Args:
            key: Configuration key in dot notation
            value: New value
        """
        keys = key.split('.')
        config_obj = self.config
        
        for k in keys[:-1]:
            if hasattr(config_obj, k):
                config_obj = getattr(config_obj, k)
            elif isinstance(config_obj, dict):
                if k not in config_obj:
                    config_obj[k] = {}
                config_obj = config_obj[k]
        
        final_key = keys[-1]
        if hasattr(config_obj, final_key):
            setattr(config_obj, final_key, value)
        elif isinstance(config_obj, dict):
            config_obj[final_key] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file.
        
        Args:
            path: Optional path to save configuration
        """
        save_path = path or self.config_path
        config_dict = self.config.dict()
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        # Check required API keys
        if not self.openai_api_key:
            logging.warning("OPENAI_API_KEY not set")
            return False
        
        # Check data paths
        data_path = self.get('data.datasets.arxiv_cs.path')
        if data_path and not Path(data_path).exists():
            logging.warning(f"Dataset path does not exist: {data_path}")
        
        return True


# Global configuration instance
config_manager = ConfigManager()
