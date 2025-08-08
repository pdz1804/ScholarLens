"""
Data models for the TechAuthor system.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class QueryType(str, Enum):
    """Enumeration of supported query types."""
    AUTHOR_EXPERTISE = "author_expertise"
    TECHNOLOGY_TRENDS = "technology_trends"
    AUTHOR_COLLABORATION = "author_collaboration"
    DOMAIN_EVOLUTION = "domain_evolution"
    CROSS_DOMAIN_ANALYSIS = "cross_domain_analysis"
    PAPER_IMPACT = "paper_impact"
    AUTHOR_PRODUCTIVITY = "author_productivity"
    AUTHOR_STATS = "author_stats"
    PAPER_SEARCH = "paper_search"
    UNCLASSIFIED = "unclassified"


class Paper(BaseModel):
    """Research paper model with LLM-based author parsing."""
    title: str
    paper_id: str
    authors: List[str]
    author_institutions: Dict[str, List[str]] = {}  # Author -> [institutions]
    abstract: str
    domain: str
    primary_subject: str
    subjects: List[str]
    date_submitted: datetime
    abstract_url: str
    pdf_url: str
    
    def __init__(self, **data):
        # If authors is a string, parse it based on configuration
        if 'authors' in data and isinstance(data['authors'], str):
            from ..core.config import config_manager
            
            # Check parsing method from config
            parsing_config = config_manager.config.data.parsing
            parsing_method = parsing_config.method
            llm_fallback = parsing_config.llm_fallback
            
            authors = []
            institutions = {}
            
            if parsing_method == "regex":
                from ..utils.regex_parsing import get_regex_parser
                regex_parser = get_regex_parser()
                try:
                    authors, institutions = regex_parser.parse(data['authors'])
                except Exception as e:
                    if llm_fallback:
                        # Fallback to LLM parsing if regex fails and fallback is enabled
                        from ..utils.llm_parsing import get_llm_parser
                        llm_parser = get_llm_parser()
                        authors, institutions = llm_parser.parse_authors_and_institutions_sync(data['authors'])
                    else:
                        # If no fallback, raise the error
                        raise e
            else:
                # Default to LLM parsing
                from ..utils.llm_parsing import get_llm_parser
                llm_parser = get_llm_parser()
                authors, institutions = llm_parser.parse_authors_and_institutions_sync(data['authors'])
            
            data['authors'] = authors
            # Set author_institutions if not already provided
            if 'author_institutions' not in data or not data['author_institutions']:
                data['author_institutions'] = institutions
        super().__init__(**data)
    
    # ========== OLD REGEX-BASED PARSING METHODS (COMMENTED OUT) ==========
    # The following methods used regex patterns for parsing authors and institutions.
    # They have been replaced with LLM-based parsing for better accuracy and robustness.
    # See models_backup.py for the original implementation if needed.
    
    @field_validator('subjects', mode='before')
    @classmethod
    def parse_subjects(cls, v):
        """Parse subjects from string representation of list."""
        if isinstance(v, str):
            try:
                # Handle string representation of list like "['Machine Learning', 'Sound']"
                import ast
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                # If it's not a valid list representation, treat as single subject
                return [v.strip()]
        elif isinstance(v, list):
            return v
        else:
            return [str(v)]

    @field_validator('date_submitted', mode='before')
    @classmethod 
    def parse_date(cls, v):
        """Parse date from various string formats."""
        if isinstance(v, str):
            from dateutil import parser
            try:
                # Handle complex date strings like "28 Nov 2022 (v1), last revised 21 Aug 2023 (this version, v2)"
                # Extract the first date part before any parentheses or version info
                import re
                # Find the first date pattern (handle various formats)
                date_pattern = r'(\d{1,2}\s+[A-Za-z]{3}\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})'
                match = re.search(date_pattern, v)
                if match:
                    date_str = match.group(1)
                    return parser.parse(date_str)
                else:
                    # Try to parse the whole string
                    return parser.parse(v)
            except Exception:
                raise ValueError(f"Unable to parse date: {v}")
        return v


class Author(BaseModel):
    """Author model with aggregated information."""
    name: str
    paper_count: int = 0
    domains: List[str] = []
    primary_subjects: List[str] = []
    collaborators: List[str] = []
    papers: List[str] = []  # Paper IDs
    first_publication: Optional[datetime] = None
    last_publication: Optional[datetime] = None
    h_index: Optional[int] = None   # Can be calculated later (not yet)
    institutions: List[str] = []    # If available (not yet)


class Query(BaseModel):
    """Query model."""
    text: str
    query_type: Optional[QueryType] = None
    parameters: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None


class RetrievalResult(BaseModel):
    """Result from retrieval system."""
    paper: Paper
    score: float
    rank: int
    retrieval_method: str = "semantic"


class AnalysisResult(BaseModel):
    """Result from analysis agent."""
    query: Query
    retrieved_papers: List[RetrievalResult]
    analysis_type: str
    results: Dict[str, Any]
    confidence: float
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class AuthorExpertiseResult(BaseModel):
    """Result for author expertise queries."""
    domain: str
    top_authors: List[Dict[str, Any]]  # Allow any type for flexibility
    total_papers_analyzed: int
    time_range: Optional[Dict[str, str]] = None
    methodology: str
    confidence: float


class TechnologyTrendResult(BaseModel):
    """Result for technology trend queries."""
    domain: str
    trends: List[Dict[str, Any]]
    time_series_data: Optional[Dict[str, Any]] = None
    emerging_technologies: List[str]
    declining_technologies: List[str]
    confidence: float
    insights: Optional[List[str]] = None
    summary: Optional[str] = None
    recommendations: Optional[List[str]] = None


class CollaboratorInfo(BaseModel):
    """Information about a collaborator."""
    collaborator: str
    collaboration_count: int
    shared_papers: int
    common_subjects: List[str]
    centrality: float


class CollaborationResult(BaseModel):
    """Result for collaboration analysis."""
    focal_author: str
    collaborators: List[CollaboratorInfo]
    total_collaborators: int = 0  # Add top-level field for display
    network_size: int = 0  # Add top-level field for display
    collaboration_network: Dict[str, Any]
    collaboration_patterns: Dict[str, Any]
    confidence: float


class SystemResponse(BaseModel):
    """Final system response model."""
    query: Query
    response_type: str
    result: Union[
        AuthorExpertiseResult,
        TechnologyTrendResult,
        CollaborationResult,
        Dict[str, Any]
    ]
    processing_time: float
    agent_chain: List[str]
    confidence: float
    sources: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class CacheEntry(BaseModel):
    """Cache entry model."""
    key: str
    value: Any
    timestamp: datetime = Field(default_factory=datetime.now)
    ttl: int = 3600  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now() - self.timestamp).seconds > self.ttl


class DatasetInfo(BaseModel):
    """Dataset information model."""
    name: str
    path: str
    total_papers: int
    date_range: Dict[str, datetime]
    domains: List[str]
    subjects: List[str]
    last_updated: datetime
    schema_version: str = "1.0"


class ProcessingStats(BaseModel):
    """Processing statistics model."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    most_common_query_types: Dict[str, int] = {}
    uptime_start: datetime = Field(default_factory=datetime.now)




