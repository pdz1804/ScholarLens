"""
Retrieval agent for TechAuthor system using hybrid search (sparse + dense).
Responsible for finding relevant papers using both keyword-based and             if final_results:
                # Log search method distribution
                search_types = {}
                for result in final_results:
                    search_type = result.get('search_type', 'unknown')
                    search_types[search_type] = search_types.get(search_type, 0) + 1
                
                self._log_info(f"Search type distribution: {dict(search_types)}")
                self._log_info(f"Score range: {final_results[0]['score']:.3f} to {final_results[-1]['score']:.3f}")
                
                # Log additional score details
                scores = [r['score'] for r in final_results]
                self._log_info(f"Average score: {sum(scores)/len(scores):.3f}")
                self._log_info(f"Results above threshold {self.score_threshold}: {len([s for s in scores if s >= self.score_threshold])}")
            
            return final_resultsapproaches.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from ..core.models import Query, Paper, RetrievalResult, QueryType
from ..core.llm_manager import LLMManager
from ..data.data_manager import DataManager
from .base_agent import BaseAgent
from ..services.extraction_service import ExtractionService


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant papers using hybrid search.
    Combines sparse (keyword-based) and dense (semantic) search for comprehensive results.
    
    The hybrid search approach uses:
    - Sparse search: TF-IDF vectorization for keyword-based matching
    - Dense search: Sentence embeddings for semantic similarity
    - Fusion: Weighted combination of both results for optimal relevance
    """
    
    def __init__(self, data_manager: DataManager, llm_manager: LLMManager = None):
        """
        Initialize retrieval agent with data manager containing hybrid search capabilities.
        
        Args:
            data_manager: Data manager instance with hybrid searcher
            llm_manager: LLM manager for query enhancement
        """
        super().__init__("Retrieval")
        self.data_manager = data_manager
        self.llm_manager = llm_manager or LLMManager()
        
        # Initialize extraction service
        self.extraction_service = ExtractionService(self.llm_manager)
        
        # Get retrieval configuration
        retrieval_config = self.llm_manager.get_retrieval_config()
        self.initial_top_k = retrieval_config.get('initial_top_k', 184)
        self.score_threshold = retrieval_config.get('score_threshold', 0.4)
        self.final_top_k = retrieval_config.get('final_top_k', 20)
        self.hybrid_alpha = retrieval_config.get('hybrid_alpha', 0.7)

        # Log configuration for debugging
        self._log_info(f"Retrieval Agent Config: initial_top_k={self.initial_top_k}, "
                        f"final_top_k={self.final_top_k}, hybrid_alpha={self.hybrid_alpha}, "
                        f"score_threshold={self.score_threshold}")
    
    async def _initialize_impl(self) -> None:
        """
        Initialize the retrieval agent.
        The actual search indices are managed by the data manager's hybrid searcher.
        """
        # Verify data manager is initialized
        if not self.data_manager.is_initialized:
            await self.data_manager.initialize()
            
        self._log_info("Retrieval agent initialized with hybrid search capabilities")
    
    async def _process_impl(self, query: Query) -> RetrievalResult:
        """
        Process a query and retrieve relevant papers using hybrid search.
        
        Args:
            query: Query object with text and parameters
            
        Returns:
            RetrievalResult with relevant papers and metadata
        """
        try:
            self._log_info(f"Processing retrieval query: {query.text}")
            
            # Get retrieval configuration
            retrieval_config = self.llm_manager.get_retrieval_config() if self.llm_manager else {}
            
            # Get search parameters from configuration and query - use different settings for trends
            if query.query_type == QueryType.TECHNOLOGY_TRENDS:
                # For trend analysis, use special high-volume settings
                initial_limit = retrieval_config.get('trends_initial_top_k', 5000)
                final_limit = query.parameters.get('limit', retrieval_config.get('trends_final_top_k', 1000))
                score_threshold = retrieval_config.get('trends_score_threshold', 0.1)
                search_alpha = query.parameters.get('search_alpha', self.hybrid_alpha)
                
                self._log_info(f"TRENDS MODE: Using expanded retrieval settings")
                self._log_info(f"Trends search parameters: initial_k={initial_limit}, final_k={final_limit}, alpha={search_alpha}, threshold={score_threshold}")
                
                # Override score threshold for trends
                original_threshold = self.score_threshold
                self.score_threshold = score_threshold
                
            else:
                # Standard retrieval settings
                initial_limit = self.initial_top_k
                final_limit = query.parameters.get('limit', self.initial_top_k)  # Use initial_top_k instead of final_top_k
                search_alpha = query.parameters.get('search_alpha', self.hybrid_alpha)
                
                self._log_info(f"Search parameters: initial_k={initial_limit}, final_k={final_limit}, alpha={search_alpha}, threshold={self.score_threshold}")
            
            # Perform hybrid search based on query type
            if query.query_type == QueryType.AUTHOR_EXPERTISE:
                self._log_info("Using hybrid search for author expertise query")
                # For author expertise queries, search for papers in the domain/topic mentioned
                results = await self._hybrid_search(query.text, initial_limit, search_alpha)
            elif query.query_type == QueryType.AUTHOR_COLLABORATION:
                self._log_info("Using collaboration search")
                results = await self._search_collaborations(query.text, initial_limit)
            elif query.query_type == QueryType.DOMAIN_EVOLUTION:
                self._log_info("Using domain-specific search")
                results = await self._search_by_domain(query.text, initial_limit)
            elif query.query_type == QueryType.TECHNOLOGY_TRENDS:
                self._log_info("Using EXPANDED hybrid search for technology trends")
                # For trends, use the expanded search with lower thresholds
                results = await self._hybrid_search(query.text, initial_limit, search_alpha)
            else:
                self._log_info("Using general hybrid search")
                # General hybrid search for other query types
                results = await self._hybrid_search(query.text, initial_limit, search_alpha)
            
            # Restore original threshold if it was changed
            if query.query_type == QueryType.TECHNOLOGY_TRENDS:
                self.score_threshold = original_threshold
            
            self._log_info(f"Search completed: {len(results)} documents found after filtering")
            
            # Apply time range filtering if specified
            if 'time_range' in query.parameters and query.parameters['time_range']:
                self._log_info(f"Applying time range filter: {query.parameters['time_range']}")
                results = self._filter_by_time_range(results, query.parameters['time_range'])
                self._log_info(f"After time filtering: {len(results)} documents remain")
            
            # Apply final limit based on query type
            if query.query_type == QueryType.TECHNOLOGY_TRENDS:
                # For trends, use much larger final result set
                final_results = results[:final_limit] if final_limit < len(results) else results
                self._log_info(f"TRENDS MODE: Keeping {len(final_results)} papers for comprehensive analysis")
            else:
                # Standard result limiting
                final_results = results[:self.initial_top_k] if self.initial_top_k < len(results) else results
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            
            for i, result in enumerate(final_results):
                paper = self._result_to_paper(result['document'])
                if paper:
                    retrieval_results.append(RetrievalResult(
                        paper=paper,
                        score=result['score'],
                        rank=i + 1,
                        retrieval_method=result.get('search_type', 'hybrid')
                    ))
                else:
                    self._log_warning(f"Failed to convert document {i+1} to Paper object")
            
            self._log_info(f"Successfully converted {len(retrieval_results)} documents to Paper objects")
            
            if retrieval_results:
                self._log_info("Top retrieved papers:")
                for i, result in enumerate(retrieval_results[:5], 1):
                    self._log_info(f"  {i}. {result.paper.title[:70]}...")
                    self._log_info(f"     Authors: {', '.join(result.paper.authors[:3])}")
                    self._log_info(f"     Score: {result.score:.3f}, Method: {result.retrieval_method}")
                    self._log_info(f"     Paper ID: {result.paper.paper_id}")
            
            return retrieval_results
            
        except Exception as e:
            self._log_error(f"Error in retrieval processing: {str(e)}")
            return []
    
    async def _hybrid_search(self, query_text: str, limit: int, alpha: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search combining sparse and dense approaches.
        
        Args:
            query_text: Search query
            limit: Maximum number of results
            alpha: Weight for dense search (1-alpha for sparse)
            
        Returns:
            List of search results with scores
        """
        try:
            self._log_info(f"Performing hybrid search: '{query_text}'")
            self._log_info(f"Alpha (dense weight): {alpha}, Sparse weight: {1-alpha}")
            self._log_info(f"Using score threshold: {self.score_threshold}")
            
            # Use the improved hybrid search with score filtering
            results = self.data_manager.hybrid_searcher.search(
                query_text, 
                k=limit,  # Use the limit directly (which could be much higher for trends)
                alpha=alpha, 
                score_threshold=self.score_threshold  # Filter by score threshold (lower for trends)
            )
            
            self._log_info(f"Hybrid search returned {len(results)} results after filtering (threshold: {self.score_threshold})")
            if results:
                # Log search method distribution
                search_types = {}
                for result in results:
                    search_type = result.get('search_type', 'unknown')
                    search_types[search_type] = search_types.get(search_type, 0) + 1
                
                self._log_info(f"Search type distribution: {dict(search_types)}")
                self._log_info(f"Score range: {results[0]['score']:.3f} to {results[-1]['score']:.3f}")
                
                # Log additional score details
                scores = [r['score'] for r in results]
                self._log_info(f"Average score: {sum(scores)/len(scores):.3f}")
                self._log_info(f"Results above threshold {self.score_threshold}: {len([s for s in scores if s >= self.score_threshold])}")
            
            return results
        except Exception as e:
            self._log_error(f"Hybrid search failed: {str(e)}")
            return []
    
    async def _search_by_author(self, author_query: str, limit: int) -> List[Dict]:
        """
        Search for papers by specific author.
        
        Args:
            author_query: Author name or query
            limit: Maximum number of results
            
        Returns:
            List of author's papers
        """
        try:
            # Extract author name from query
            author_name = await self._extract_author_name(author_query)
            return self.data_manager.hybrid_searcher.search_by_author(author_name, k=limit)
        except Exception as e:
            self._log_error(f"Author search failed: {str(e)}")
            return []
    
    async def _search_by_domain(self, domain_query: str, limit: int) -> List[Dict]:
        """
        Search for papers in specific domain/category.
        
        Args:
            domain_query: Domain or category query
            limit: Maximum number of results
            
        Returns:
            List of papers in domain
        """
        try:
            # Extract domain/category from query
            domain = await self._extract_domain(domain_query)
            return self.data_manager.hybrid_searcher.search_by_category(domain, k=limit)
        except Exception as e:
            self._log_error(f"Domain search failed: {str(e)}")
            return []
    
    async def _search_collaborations(self, collab_query: str, limit: int) -> List[Dict]:
        """
        Search for collaboration-related papers.
        
        Args:
            collab_query: Collaboration query
            limit: Maximum number of results
            
        Returns:
            List of collaboration papers
        """
        try:
            # Use hybrid search with collaboration-focused terms
            enhanced_query = f"collaboration coauthor research {collab_query}"
            return await self._hybrid_search(enhanced_query, limit, alpha=0.6)
        except Exception as e:
            self._log_error(f"Collaboration search failed: {str(e)}")
            return []
    
    def _result_to_paper(self, document: Dict) -> Optional[Paper]:
        """
        Convert search result document to Paper object.
        
        Args:
            document: Document dictionary from search results
            
        Returns:
            Paper object or None if conversion fails
        """
        try:
            # Handle different date formats
            from datetime import datetime
            date_submitted = None
            if document.get('date'):
                try:
                    date_submitted = datetime.fromisoformat(document['date'].replace('Z', '+00:00'))
                except:
                    date_submitted = None
            
            return Paper(
                title=document.get('title', ''),
                paper_id=document.get('id', ''),
                authors=document.get('authors', '').split(', ') if document.get('authors') else [],
                abstract=document.get('abstract', ''),
                domain=document.get('domain', ''),
                primary_subject=document.get('domain', ''),  # Use domain as primary subject
                subjects=document.get('subjects', '').split(', ') if document.get('subjects') else [],
                date_submitted=date_submitted,
                abstract_url='',
                pdf_url=''
            )
        except Exception as e:
            self._log_warning(f"Failed to convert document to Paper: {str(e)}")
            return None
    
    async def _extract_author_name(self, query: str) -> str:
        """
        Extract author name from query text using LLM-based extraction.
        
        Args:
            query: Query text
            
        Returns:
            Extracted author name
        """
        try:
            extracted = await self.extraction_service.extract_author_name(query)
            return extracted if extracted else query
        except Exception as e:
            self._log_error(f"Failed to extract author name via LLM: {e}")
            # Fallback to simple extraction
            import re
            cleaned = re.sub(r'who are.*?authors?.*?in|top.*?authors?.*?in|papers?.*?by', '', query.lower(), flags=re.IGNORECASE)
            cleaned = re.sub(r'[^\w\s]', '', cleaned).strip()
            return cleaned if cleaned else query
    
    async def _extract_domain(self, query: str) -> str:
        """
        Extract domain/category from query text using LLM-based extraction.
        
        Args:
            query: Query text
            
        Returns:
            Extracted domain
        """
        try:
            extracted = await self.extraction_service.extract_search_term_from_query(query)
            return extracted if extracted else query
        except Exception as e:
            self._log_error(f"Failed to extract domain via LLM: {e}")
            # Fallback to pattern matching
            import re
            
            domains = [
                'machine learning', 'deep learning', 'neural networks',
                'computer vision', 'natural language processing', 'nlp',
                'artificial intelligence', 'ai', 'robotics'
            ]
            
            query_lower = query.lower()
            for domain in domains:
                if domain in query_lower:
                    return domain
            
            # Extract after common patterns
            patterns = [r'research in (.+)', r'papers in (.+)', r'work on (.+)']
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    return match.group(1).strip()
            
            return query
    
    # fix
    # - The regex pattern for year extraction is hardcoded and may need updates for future years.
    def _filter_by_time_range(self, results: List[Dict], time_range: str) -> List[Dict]:
        """
        Filter search results by time range.
        
        Args:
            results: List of search results
            time_range: Time range string (e.g., "2022 to 2025", "2020-2023")
            
        Returns:
            Filtered results list
        """
        import re
        from datetime import datetime
        
        try:
            # Parse time range string
            # Handle formats like "2022 to 2025", "2020-2023", "since 2022", "before 2025"
            time_range_lower = time_range.lower().strip()
            
            # Extract years from time range
            # Fixed: Remove capturing group to get full years
            year_pattern = r'\b(?:19|20)\d{2}\b'
            years = re.findall(year_pattern, time_range)
            
            self._log_info(f"DEBUG: time_range='{time_range}', years={years}")
            
            if len(years) >= 2:
                # Range like "2022 to 2025" - now years will be ['2022', '2025']
                start_year = int(years[0])
                end_year = int(years[-1])
                
                self._log_info(f"DEBUG: Parsed range {start_year} to {end_year}")
                self._log_info(f"Filtering papers from {start_year} to {end_year}")
                
            elif len(years) == 1:
                # Single year or "since/before" patterns - now years[0] is the full year
                year = int(years[0])
                
                if 'since' in time_range_lower or 'from' in time_range_lower:
                    start_year = year
                    end_year = datetime.now().year
                elif 'before' in time_range_lower or 'until' in time_range_lower:
                    start_year = 1990  # Reasonable lower bound
                    end_year = year
                else:
                    # Single year
                    start_year = year
                    end_year = year
                
                self._log_info(f"Filtering papers from {start_year} to {end_year}")
            else:
                self._log_warning(f"Could not parse time range: {time_range}")
                return results
            
            # Filter results based on paper publication date
            filtered_results = []
            for result in results:
                try:
                    doc = result['document']
                    
                    # Try to get date from different possible fields
                    date_submitted = None
                    
                    if isinstance(doc, dict):
                        # Check various date fields
                        date_fields = ['date_submitted', 'date', 'published_date', 'submission_date']
                        for field in date_fields:
                            if field in doc and doc[field]:
                                date_submitted = doc[field]
                                break
                    else:
                        # Check if it's a Paper object
                        if hasattr(doc, 'date_submitted'):
                            date_submitted = doc.date_submitted
                        elif hasattr(doc, 'date'):
                            date_submitted = doc.date
                    
                    if date_submitted:
                        # Extract year from date
                        if isinstance(date_submitted, str):
                            # Parse string date
                            year_match = re.search(r'\b(19|20)\d{2}\b', date_submitted)
                            if year_match:
                                paper_year = int(year_match.group())
                            else:
                                continue
                        elif hasattr(date_submitted, 'year'):
                            # DateTime object
                            paper_year = date_submitted.year
                        else:
                            continue
                        
                        # Check if paper year is within range
                        if start_year <= paper_year <= end_year:
                            filtered_results.append(result)
                        
                except Exception as e:
                    self._log_debug(f"Error filtering paper by date: {e}")
                    # If we can't determine the date, include the paper
                    filtered_results.append(result)
            
            self._log_info(f"Time filtering: {len(results)} -> {len(filtered_results)} papers")
            return filtered_results
            
        except Exception as e:
            self._log_error(f"Error in time range filtering: {e}")
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval agent statistics.
        
        Returns:
            Statistics dictionary
        """
        if hasattr(self.data_manager, 'hybrid_searcher'):
            return self.data_manager.hybrid_searcher.get_search_stats()
        else:
            return {'status': 'not_initialized'}



