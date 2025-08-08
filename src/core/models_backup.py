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
    """Research paper model."""
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
        # If authors is a string, parse it using LLM and extract institutions
        if 'authors' in data and isinstance(data['authors'], str):
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
    
    # @classmethod  
    # def _extract_authors_and_institutions(cls, author_string: str) -> tuple:
    #     """Extract both author names and their institutional affiliations."""
    #     import re
    #     
    #     if not author_string or author_string.strip() == '':
    #         return [], {}
    #     
    #     authors = []
    #     author_institutions = {}
    #     
    #     # Step 1: Handle institutional mappings in double parentheses
    #     # Pattern: ((1) Institution A, (2) Institution B, ...)
    #     institution_mapping = {}
    #     double_paren_match = re.search(r'\(\((.+?)\)\)(?:\s*$)', author_string)
    #     if double_paren_match:
    #         mapping_text = double_paren_match.group(1)
    #         # Extract numbered institutions - handle complex patterns
    #         institution_matches = re.findall(r'\((\d+)\)\s*([^(]+?)(?=\s*\(\d+\)|$)', mapping_text)
    #         for number, institution in institution_matches:
    #             clean_inst = re.sub(r'\s+', ' ', institution.strip().rstrip(','))
    #             institution_mapping[number] = clean_inst
    #         
    #         # Remove the mapping from the string
    #         author_string = re.sub(r'\s*\(\(.+?\)\)\s*$', '', author_string)
    #     
    #     # Step 2: Split into individual author entries while preserving parentheses
    #     author_parts = []
    #     current_part = ""
    #     paren_count = 0
    #     
    #     for char in author_string:
    #         if char == '(':
    #             paren_count += 1
    #             current_part += char
    #         elif char == ')':
    #             paren_count -= 1
    #             current_part += char
    #         elif char == ',' and paren_count == 0:
    #             if current_part.strip():
    #                 author_parts.append(current_part.strip())
    #             current_part = ""
    #         else:
    #             current_part += char
    #     
    #     if current_part.strip():
    #         author_parts.append(current_part.strip())
    #     
    #     # Step 3: Process each author part
    #     for part in author_parts:
    #         if not part:
    #             continue
    #         
    #         author_name = None
    #         affiliations = []
    #         
    #         # Case 1: Name(numbers) - numbered affiliations
    #         numbered_match = re.match(r'([^(]+)\(([0-9,\s]+(?:and\s+[0-9,\s]+)*)\)', part)
    #         if numbered_match:
    #             author_name = numbered_match.group(1).strip()
    #             numbers_text = numbered_match.group(2)
    #             numbers = re.findall(r'\d+', numbers_text)
    #             for num in numbers:
    #                 if num in institution_mapping:
    #                     affiliations.append(institution_mapping[num])
    #         
    #         # Case 2: Name(Institution) - direct institution
    #         elif '(' in part and ')' in part:
    #             match = re.match(r'([^(]+)\(([^)]+)\)', part)
    #             if match:
    #                 potential_name = match.group(1).strip()
    #                 potential_affiliation = match.group(2).strip()
    #                 
    #                 if cls._looks_like_institution(potential_affiliation):
    #                     author_name = potential_name
    #                     affiliations.append(potential_affiliation)
    #                 else:
    #                     author_name = potential_name
    #         
    #         # Case 3: Just a name without parentheses
    #         else:
    #             clean_part = part.strip()
    #             if cls._looks_like_person_name(clean_part):
    #                 author_name = clean_part
    #         
    #         # Validate and add the author
    #         if author_name and cls._looks_like_person_name(author_name):
    #             clean_name = cls._clean_author_name(author_name)
    #             if clean_name:
    #                 authors.append(clean_name)
    #                 if affiliations:
    #                     author_institutions[clean_name] = affiliations
    #     
    #     return authors, author_institutions
    # 
    # @classmethod
    # def _looks_like_institution(cls, text: str) -> bool:
    #     """Check if text looks like an institutional affiliation."""
    #     if not text or len(text) < 3:
    #         return False
    #     
    #     text_lower = text.lower()
    #     
    #     # Common institutional keywords
    #     institution_keywords = [
    #         'university', 'institute', 'laboratory', 'lab', 'department', 'school', 
    #         'college', 'center', 'centre', 'hospital', 'medical', 'research',
    #         'academy', 'sciences', 'technology', 'polytechnic', 'nvidia', 'google',
    #         'microsoft', 'meta', 'apple', 'ibm', 'stanford', 'mit', 'harvard',
    #         'berkeley', 'caltech', 'cmu', 'cnrs', 'max planck', 'eth zurich',
    #         'fondazione', 'irccs', 'politecnico', 'national', 'state',
    #         'suny', 'uci', 'ucla', 'usc', 'nyu', 'gsu', 'asu', 'psu',
    #         'crisam', 'marianne', 'wimmics', 'sparks', 'huawei', 'buffalo'
    #     ]
    #     
    #     return any(keyword in text_lower for keyword in institution_keywords)
    # 
    # @classmethod
    # def _looks_like_person_name(cls, text: str) -> bool:
        """Check if text looks like a person's name."""
        if not text or len(text) < 2:
            return False
        
        # Skip if it's clearly institutional
        if cls._looks_like_institution(text):
            return False
        
        # Skip common location names
        locations = [
            'usa', 'uk', 'germany', 'japan', 'china', 'canada', 'france', 'italy', 
            'spain', 'australia', 'korea', 'taiwan', 'singapore', 'netherlands',
            'beijing', 'tokyo', 'london', 'paris', 'berlin', 'milan', 'zurich'
        ]
        
        if text.lower().strip() in locations:
            return False
        
        # Should have alphabetic characters
        if not any(c.isalpha() for c in text):
            return False
        
        # Should not be all uppercase (likely abbreviation) unless short
        if text.isupper() and len(text) > 3:
            return False
        
        # Should have at least one uppercase letter (names are capitalized)
        if not any(c.isupper() for c in text):
            return False
        
        # Check basic name structure
        words = text.split()
        if len(words) >= 1:
            # At least one word that looks like a name part
            return any(len(word) > 1 and word[0].isupper() for word in words)
        
        return False
        """Determine if a token looks like a person's name."""
        import re
        
        if not token or len(token.strip()) < 2:
            return False
            
        # Remove any parenthetical info for analysis
        clean_token = re.sub(r'\([^)]*\)', '', token).strip()
        
        if not clean_token:
            return False
        
        # Basic checks for name-like characteristics
        has_alpha = any(c.isalpha() for c in clean_token)
        has_upper = any(c.isupper() for c in clean_token)
        has_lower = any(c.islower() for c in clean_token)
        
        if not (has_alpha and has_upper):
            return False
        
        # Check for common name patterns
        words = clean_token.split()
        if not words:
            return False
        
        # Single word names (less common but possible)
        if len(words) == 1:
            word = words[0]
            # Must be mixed case and reasonable length
            return len(word) >= 3 and has_upper and has_lower
        
        # Multi-word names
        name_like_score = 0
        
        for word in words:
            if len(word) < 1:
                continue
            
            # Title case words are name-like
            if word[0].isupper() and (len(word) == 1 or word[1:].islower()):
                name_like_score += 2
            # All caps might be initials
            elif word.isupper() and len(word) <= 3:
                name_like_score += 1
            # Mixed case could be names like "McDonald"
            elif any(c.isupper() for c in word) and any(c.islower() for c in word):
                name_like_score += 1
            # Contains periods (initials)
            elif '.' in word:
                name_like_score += 1
        
        # Need at least some name-like characteristics
        return name_like_score >= len(words)
    
    @classmethod
    def _is_institutional_info(cls, token: str) -> bool:
        """Determine if a token is institutional information."""
        import re
        
        if not token:
            return False
            
        token_lower = token.lower().strip()
        
        # Empty after cleaning parentheses
        clean_token = re.sub(r'\([^)]*\)', '', token).strip()
        if not clean_token:
            return True
        
        # Common institutional keywords (flexible matching)
        institutional_indicators = [
            'university', 'college', 'institute', 'laboratory', 'department',
            'school', 'center', 'centre', 'division', 'faculty', 'hospital',
            'academy', 'polytechnic', 'technology', 'sciences', 'research',
            'national', 'state', 'medical', 'engineering', 'computer',
            'nvidia', 'google', 'microsoft', 'ibm', 'mit', 'stanford',
            'harvard', 'berkeley', 'caltech', 'irccs', 'cnrs', 'cas'
        ]
        
        # Check if token contains institutional keywords
        for indicator in institutional_indicators:
            if indicator in token_lower:
                return True
        
        # Check if it's mostly numbers/codes
        if re.match(r'^[\d\s\(\),and]+$', token.strip()):
            return True
        
        # Check if it looks like a location
        location_pattern = r'\b(usa?|uk|china|japan|germany|france|italy|spain|australia)\b'
        if re.search(location_pattern, token_lower):
            return True
        
        return False
    
    @classmethod
    def _could_be_name_continuation(cls, current_name: str, token: str) -> bool:
        """Check if token could be a continuation of a compound name."""
        import re
        
        # This is for cases like "Mary Jane Smith" split across commas
        # Very conservative - only for clearly name-like continuations
        
        if not current_name or not token:
            return False
        
        # If current name looks incomplete (like just a first name)
        current_words = current_name.strip().split()
        if len(current_words) == 1 and cls._is_likely_author_name(token):
            # Check if token could be surname
            token_clean = re.sub(r'\([^)]*\)', '', token).strip()
            token_words = token_clean.split()
            if len(token_words) == 1 and len(token_clean) > 2:
                return True
        
        return False
    
    @classmethod
    def _clean_author_name(cls, author: str) -> str:
        """Clean an author name by removing affiliations and formatting."""
        import re
        
        if not author:
            return ""
        
        # Remove parenthetical information
        cleaned = re.sub(r'\([^)]*\)', '', author)
        
        # Remove leading/trailing numbers and punctuation
        cleaned = re.sub(r'^[\d\s\-,]+', '', cleaned)
        cleaned = re.sub(r'[\d\s\-,]+$', '', cleaned)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    @classmethod
    def _validate_author_name(cls, name: str) -> bool:
        """Final validation for author names."""
        if not name or len(name) < 3:
            return False
        
        # Must contain alphabetic characters
        if not any(c.isalpha() for c in name):
            return False
        
        # Must have some capitalization (names typically do)
        if not any(c.isupper() for c in name):
            return False
        
        # Shouldn't be all caps (likely abbreviation)
        if name.isupper():
            return False
        
        # Shouldn't be mostly numbers
        alpha_chars = sum(1 for c in name if c.isalpha())
        total_chars = len(name.replace(' ', ''))
        if total_chars > 0 and alpha_chars / total_chars < 0.7:
            return False
        
        # Should look like a name pattern (at least have word-like structure)
        words = name.split()
        if not words:
            return False
        
        valid_words = 0
        for word in words:
            if len(word) >= 2 and any(c.isalpha() for c in word):
                valid_words += 1
        
        return valid_words >= 1
    
    @field_validator('subjects', mode='before')
    @classmethod
    def parse_subjects(cls, v):
        """Parse subjects from string or list."""
        if isinstance(v, str):
            # Handle string representation of list
            v = v.strip("[]'\"")
            return [subject.strip().strip("'\"") for subject in v.split(',') if subject.strip()]
        return v
    
    @field_validator('date_submitted', mode='before')
    @classmethod
    def parse_date(cls, v):
        """Parse date from various formats, preferring the original submission date (v1)."""
        if isinstance(v, str):
            # Handle revision info - extract the ORIGINAL submission date (v1)
            if '(v' in v:
                # Examples:
                # "30 Dec 2021 (v1), last revised 3 Oct 2022 (this version, v2)"
                # "24 Jul 2025 (v1)"
                # Extract the first date which is the original submission (v1)
                v1_part = v.split('(v')[0].strip()
                v = v1_part
            elif '(' in v and 'last revised' not in v:
                # Extract date before any parentheses: "31 Jul 2025 (some info)"
                v = v.split('(')[0].strip()
            
            # Try different date formats
            formats = [
                "%d %b %Y",   # 31 Jul 2025, 30 Dec 2021
                "%d-%b-%y",   # 31-Jul-25
                "%d-%b-%Y",   # 31-Jul-2025
                "%Y-%m-%d",   # 2025-07-31
                "%m/%d/%Y",   # 07/31/2025
                "%b %d, %Y",  # Jul 31, 2025
                "%B %d, %Y",  # July 31, 2025
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            
            # If no format matches, try to parse as is
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                # If all else fails, try dateutil parser as fallback
                try:
                    from dateutil import parser
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







