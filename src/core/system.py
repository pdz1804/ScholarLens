"""
Main TechAuthor system implementation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .config import config_manager
from .models import (
    Query, QueryType, SystemResponse, ProcessingStats,
    AnalysisResult, AuthorExpertiseResult, TechnologyTrendResult
)

from ..agents.query_classifier import QueryClassifierAgent
from ..agents.retrieval_agent import RetrievalAgent
from ..agents.analysis_agent import AnalysisAgent
from ..agents.synthesis_agent import SynthesisAgent
from ..agents.validation_agent import ValidationAgent
from ..data.data_manager import DataManager
from ..utils.cache import CacheManager
from ..utils.logger import get_logger
from .llm_manager import LLMManager

class QueryResponseAdapter:
    """
    Adapter to make SystemResponse compatible with CLI interface expectations.
    Provides summary, insights, and data attributes from SystemResponse.
    """
    
    def __init__(self, system_response: SystemResponse):
        self.system_response = system_response
        
    @property
    def query(self):
        return self.system_response.query
        
    @property
    def response_type(self):
        return self.system_response.response_type
        
    @property
    def confidence(self):
        return self.system_response.confidence
        
    @property
    def processing_time(self):
        return self.system_response.processing_time
        
    @processing_time.setter 
    def processing_time(self, value):
        self.system_response.processing_time = value
        
    @property
    def agent_chain(self):
        return self.system_response.agent_chain
        
    @property
    def summary(self):
        """Extract summary from the result."""
        # The system result is the actual result (AuthorExpertiseResult, etc.)
        result = self.system_response.result
        
        # Handle specific result types
        if hasattr(result, 'top_authors'):  # AuthorExpertiseResult
            authors_count = len(result.top_authors)
            domain = getattr(result, 'domain', 'unknown domain')
            total_papers = getattr(result, 'total_papers_analyzed', 0)
            unique_authors = getattr(result, 'unique_authors', authors_count)
            return f"Found {authors_count} top authors in {domain} based on {total_papers} papers analyzed from {unique_authors} unique researchers."
            
        elif hasattr(result, 'trends'):  # TechnologyTrendResult
            trends_count = len(result.trends)
            domain = getattr(result, 'domain', 'Unknown')
            return f"Identified {trends_count} technology trends in {domain}."
            
        elif hasattr(result, 'focal_author'):  # CollaborationResult
            return f"Analyzed collaboration network for {result.focal_author}."
            
        elif isinstance(result, dict):
            # Handle our new synthesis result types
            if 'summary' in result:
                return result['summary']
            elif 'author_profile' in result:  # AUTHOR_STATS
                if result['author_profile'].get('found'):
                    author_name = result['author_profile']['author_name']
                    stats = result['author_profile']['stats']
                    papers = stats.get('total_papers', 0)
                    return f"Author profile for {author_name}: {papers} papers"
                else:
                    return result.get('summary', 'Author not found')
            elif 'search_results' in result:  # PAPER_SEARCH
                search_data = result['search_results']
                papers_found = search_data.get('papers_found', 0)
                search_term = search_data.get('search_term', 'unknown')
                return f"Found {papers_found} papers matching '{search_term}'"
            elif 'error_info' in result:  # UNCLASSIFIED or ERROR
                return result.get('summary', result.get('error_info', {}).get('message', 'Error occurred'))
            else:
                return result.get('summary', result.get('error', 'Analysis completed successfully.'))
        
        return 'No summary available'
            
    @property
    def insights(self):
        """Extract insights from the result."""
        # The system result is the actual result (AuthorExpertiseResult, etc.)
        result = self.system_response.result
        
        # Handle specific result types
        if hasattr(result, 'top_authors'):  # AuthorExpertiseResult
            insights = []
            if result.top_authors:
                top_author = result.top_authors[0]
                author_name = top_author.get('author', top_author.get('name', 'Unknown'))
                paper_count = top_author.get('paper_count', 0)
                expertise_score = top_author.get('expertise_score', 0.0)
                insights.append(f"Top author: {author_name} with {paper_count} papers (expertise score: {expertise_score:.1f})")
                
                # Add insight about research areas
                subjects = top_author.get('subjects', [])
                if subjects:
                    insights.append(f"Primary research areas: {', '.join(subjects[:3])}")
            
            # Add methodology insight
            methodology = getattr(result, 'methodology', 'Frequency-based ranking with subject diversity and relevance weighting')
            insights.append(f"Analysis methodology: {methodology}")
            
            # Add data quality insight
            total_papers = getattr(result, 'total_papers_analyzed', 0)
            unique_authors = getattr(result, 'unique_authors', 0)
            if total_papers > 0 and unique_authors > 0:
                avg_papers_per_author = total_papers / unique_authors
                insights.append(f"Data coverage: {avg_papers_per_author:.1f} papers per author on average")
                
            return insights
            
        elif hasattr(result, 'trends'):  # TechnologyTrendResult
            insights = []
            if hasattr(result, 'emerging_technologies') and result.emerging_technologies:
                insights.append(f"Emerging technologies: {', '.join(result.emerging_technologies[:3])}")
            if hasattr(result, 'declining_technologies') and result.declining_technologies:
                insights.append(f"Declining technologies: {', '.join(result.declining_technologies[:3])}")
            return insights
            
        elif isinstance(result, dict):
            # Handle our new synthesis result types
            if 'insights' in result:
                return result['insights']
            else:
                return result.get('insights', [])
        
        return []
            
    @property
    def data(self):
        """Extract data from the result."""
        # The system result is the actual result (AuthorExpertiseResult, etc.)
        result = self.system_response.result
        
        # Handle specific result types
        if hasattr(result, 'top_authors'):  # AuthorExpertiseResult
            return {
                'authors': result.top_authors,
                'domain': getattr(result, 'domain', 'Unknown'),
                'total_papers': getattr(result, 'total_papers_analyzed', 0),
                'confidence': getattr(result, 'confidence', 0.0)
            }
            
        elif hasattr(result, 'trends'):  # TechnologyTrendResult
            return {
                'trends': result.trends,
                'domain': getattr(result, 'domain', 'Unknown'),
                'emerging_technologies': getattr(result, 'emerging_technologies', []),
                'declining_technologies': getattr(result, 'declining_technologies', []),
                'confidence': getattr(result, 'confidence', 0.0)
            }
            
        elif hasattr(result, 'focal_author'):  # CollaborationResult
            return {
                'focal_author': result.focal_author,
                'collaborators': getattr(result, 'collaborators', []),
                'top_collaborators': getattr(result, 'collaborators', []),  # Alias for compatibility
                'total_collaborators': getattr(result, 'total_collaborators', 0),
                'network_size': getattr(result, 'network_size', 0),
                'confidence': getattr(result, 'confidence', 0.0)
            }
            
        elif isinstance(result, dict):
            return result
        
        return {'result': str(result)}

class TechAuthorSystem:
    """Main TechAuthor system orchestrating all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the TechAuthor system.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Initialize configuration
        if config_path:
            global config_manager
            from .config import ConfigManager
            config_manager = ConfigManager(config_path)
        
        self.config = config_manager.config
        
        # Setup logging - use global logger (should already be configured by CLI)
        self.logger = get_logger()
        
        # Initialize LLM manager with main config for retrieval settings
        self.llm_manager = LLMManager()
        # Pass the main config to LLM manager for retrieval configuration
        self.llm_manager.set_main_config(config_manager.config)
        
        # Log LLM configuration
        self._log_llm_configuration()
        
        # Initialize components
        self.data_manager = DataManager()
        self.cache_manager = CacheManager()
        self.stats = ProcessingStats()
        
        # Initialize agents with LLM manager
        self.query_classifier = QueryClassifierAgent(self.llm_manager)
        self.retrieval_agent = RetrievalAgent(self.data_manager, self.llm_manager)
        self.analysis_agent = AnalysisAgent(self.llm_manager, self.data_manager)
        self.synthesis_agent = SynthesisAgent(self.llm_manager)
        self.validation_agent = ValidationAgent(self.llm_manager)
        
        # System state
        self.is_initialized = False
        
        self.logger.info("TechAuthor system initialized")
    
    def _log_llm_configuration(self):
        """Log the current LLM configuration for each agent."""
        self.logger.info("LLM Configuration Summary:")
        
        agents = [
            ('query_classifier', 'Query Classification'),
            ('retrieval_agent', 'Document Retrieval'),
            ('analysis_agent', 'Analysis'),
            ('synthesis_agent', 'Synthesis'),
            ('validation_agent', 'Validation')
        ]
        
        for agent_name, display_name in agents:
            try:
                provider, model = self.llm_manager.get_model_for_agent(agent_name)
                self.logger.info(f"  {display_name}: {provider.upper()} - {model}")
            except Exception as e:
                self.logger.warning(f"  {display_name}: Failed to get config - {e}")

        self.logger.info(f"Default Provider: {self.llm_manager.default_provider.upper()}")
        self.logger.info(f"Available Providers: {list(self.llm_manager.providers.keys())}")
    
    async def initialize(self, indexing_options: Optional[Dict[str, bool]] = None):
        """Initialize the system asynchronously with optional indexing control.
        
        Args:
            indexing_options: Dictionary with indexing control options:
                - force_reindex: Force complete rebuild of all indices
                - update_index: Update indices incrementally 
                - clear_cache: Clear cache before initialization
        """
        if self.is_initialized:
            return
        
        try:
            # Validate configuration
            if not config_manager.validate():
                raise ValueError("Invalid configuration")
            
            # Initialize data manager with indexing options
            await self.data_manager.initialize(indexing_options=indexing_options)
            
            # Initialize agents
            await self.query_classifier.initialize()
            await self.retrieval_agent.initialize()
            await self.analysis_agent.initialize()
            await self.synthesis_agent.initialize()
            await self.validation_agent.initialize()
            
            self.is_initialized = True
            self.logger.info("System initialization completed")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> SystemResponse:
        """Process a query synchronously.
        
        Args:
            query_text: The query text
            params: Optional query parameters
            user_id: Optional user identifier
            
        Returns:
            System response
        """
        return asyncio.run(self.aquery(query_text, params, user_id))
    
    async def aquery(
        self,
        query_text: str,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> SystemResponse:
        """Process a query asynchronously.
        
        Args:
            query_text: The query text
            params: Optional query parameters
            user_id: Optional user identifier
            
        Returns:
            System response
        """
        start_time = datetime.now()
        
        try:
            # Ensure system is initialized
            if not self.is_initialized:
                await self.initialize()
            
            # Create query object
            query = Query(
                text=query_text,
                parameters=params or {},
                user_id=user_id
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_response = await self.cache_manager.get(cache_key)
            if cached_response:
                self.logger.info(f"Cache hit for query: {query_text[:100]}")
                self.stats.cache_hit_rate = (
                    self.stats.cache_hit_rate * self.stats.total_queries + 1
                ) / (self.stats.total_queries + 1)
                return cached_response
            
            # Process query through agent pipeline
            response = await self._process_query_pipeline(query)
            
            # Cache the response
            await self.cache_manager.set(cache_key, response)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            response.processing_time = processing_time
            await self._update_stats(query, processing_time, success=True)
            
            self.logger.info(
                f"Query processed successfully in {processing_time:.2f}s: "
                f"{query_text[:100]}"
            )
            
            # Create adapter and set processing time
            adapter = QueryResponseAdapter(response)
            adapter.processing_time = processing_time
            
            return adapter
            
        except Exception as e:
            # Update statistics for failed query
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_stats(query, processing_time, success=False)
            
            self.logger.error(f"Query processing failed: {e}")
            
            # Return error response
            error_response = SystemResponse(
                query=query,
                response_type="error",
                result={"error": str(e)},
                processing_time=processing_time,
                agent_chain=["error"],
                confidence=0.0,
                sources=[]
            )
            
            return QueryResponseAdapter(error_response)
    
    async def _process_query_pipeline(self, query: Query) -> SystemResponse:
        """Process query through the agent pipeline.
        
        Args:
            query: Query object
            
        Returns:
            System response
        """
        agent_chain = []
        
        self.logger.info("="*50)
        self.logger.info("STARTING MULTI-AGENT PIPELINE")
        self.logger.info("="*50)
        
        # Step 1: Query Classification
        self.logger.info("STEP 1/5: Query Classification")
        self.logger.info(f"Input query: '{query.text}'")
        self.logger.debug("Starting query classification")
        
        classification_result = await self.query_classifier.classify(query)
        query.query_type = classification_result.query_type
        query.parameters.update(classification_result.extracted_params)
        agent_chain.append("query_classifier")
        
        self.logger.info(f"Classified as: {query.query_type.value if query.query_type else 'general'}")
        self.logger.info(f"Extracted parameters: {classification_result.extracted_params}")
        
        # Log classification reasoning if available
        if hasattr(classification_result, 'reasoning') and classification_result.reasoning:
            self.logger.info("Classification reasoning:")
            self.logger.info(f"  {classification_result.reasoning}")
        
        # Check if query was classified as UNCLASSIFIED and terminate early
        if query.query_type == QueryType.UNCLASSIFIED:
            self.logger.info("Query classified as UNCLASSIFIED - terminating pipeline early")
            self.logger.info("Reason: Query could not be classified into any supported category")
            
            # Create a simple unclassified response without going through full pipeline
            unclassified_response = SystemResponse(
                query=query,
                response_type="unclassified",
                result={
                    "error": "Unable to classify your query",
                    "original_query": query.text,
                    "message": "I couldn't understand what you're looking for. Please try rephrasing your query.",
                    "suggestions": [
                        "Ask about specific authors (e.g., 'Tell me about John Smith's research')",
                        "Search for papers about a technology (e.g., 'Show me papers about neural networks')",
                        "Find top experts in a field (e.g., 'Who are the top authors in machine learning?')",
                        "Ask about research trends (e.g., 'What are the emerging trends in AI?')"
                    ]
                },
                processing_time=0.0,
                agent_chain=["query_classifier"],
                confidence=0.0,
                sources=[]
            )
            
            self.logger.info("="*50)
            self.logger.info("PIPELINE TERMINATED - UNCLASSIFIED QUERY")
            self.logger.info("Asking user for clarification")
            self.logger.info("="*50)
            
            return unclassified_response
        
        # Step 2: Retrieval
        self.logger.info("STEP 2/5: Document Retrieval")
        self.logger.info(f"Searching for papers related to: '{query.text}'")
        self.logger.debug("Starting document retrieval")
        
        retrieval_results = await self.retrieval_agent.process(query)
        agent_chain.append("retrieval_agent")
        
        self.logger.info(f"Retrieved {len(retrieval_results)} relevant papers")
        if retrieval_results:
            self.logger.info("Top retrieved papers:")
            for i, result in enumerate(retrieval_results[:5], 1):  # Show top 5 instead of 3
                self.logger.info(f"  {i}. {result.paper.title[:80]}...")
                self.logger.info(f"     Score: {result.score:.3f}, Method: {result.retrieval_method}")
                # Show abstract excerpt
                abstract_excerpt = result.paper.abstract[:300] + "..." if len(result.paper.abstract) > 300 else result.paper.abstract
                self.logger.info(f"     Abstract: {abstract_excerpt}")
                self.logger.info(f"     Authors: {', '.join(result.paper.authors[:3])}{'...' if len(result.paper.authors) > 3 else ''}")
                self.logger.info("")  # Empty line for readability
        
        # Step 3: Analysis
        self.logger.info("STEP 3/5: Analysis")
        query_type_description = {
            QueryType.AUTHOR_EXPERTISE: "author expertise",
            QueryType.TECHNOLOGY_TRENDS: "technology trends",
            QueryType.AUTHOR_COLLABORATION: "author collaboration",
            QueryType.DOMAIN_EVOLUTION: "domain evolution",
            QueryType.CROSS_DOMAIN_ANALYSIS: "cross-domain analysis",
            QueryType.PAPER_IMPACT: "paper impact",
            QueryType.AUTHOR_PRODUCTIVITY: "author productivity",
            QueryType.AUTHOR_STATS: "author statistics",
            QueryType.PAPER_SEARCH: "paper search",
            QueryType.UNCLASSIFIED: "unclassified query"
        }.get(query.query_type, "general analysis")
        
        self.logger.info(f"Analyzing {len(retrieval_results)} papers for {query_type_description}")
        self.logger.debug("Starting analysis")
        
        analysis_result = await self.analysis_agent.analyze(query, retrieval_results)
        agent_chain.append("analysis_agent")
        
        self.logger.info(f"Analysis completed with confidence: {analysis_result.confidence:.2f}")
        self.logger.info(f"Analysis type: {analysis_result.analysis_type}")
        
        # Log detailed analysis reasoning
        if hasattr(analysis_result, 'reasoning') and analysis_result.reasoning:
            self.logger.info("Analysis reasoning:")
            self.logger.info(f"  {analysis_result.reasoning}")
        
        # Log analysis-specific metrics based on query type
        if hasattr(analysis_result.results, 'get'):
            if query.query_type == QueryType.TECHNOLOGY_TRENDS:
                trends_count = len(analysis_result.results.get('trends', []))
                analysis_type = analysis_result.results.get('analysis_type', 'temporal_trends')
                total_papers = analysis_result.results.get('total_papers', 0)
                analysis_period = analysis_result.results.get('analysis_period', 'Unknown')
                
                self.logger.info(f"Found {trends_count} technology areas in {analysis_type} analysis from {total_papers} papers")
                self.logger.info(f"Analysis period: {analysis_period}")
                
                # Log time range filtering if applied
                if 'time_range' in query.parameters and query.parameters['time_range']:
                    self.logger.info(f"Applied time range filter: {query.parameters['time_range']}")
                
                # Explain slope calculation methodology
                self.logger.info("Slope calculation methodology:")
                self.logger.info("  - Slope represents the linear trend in paper count per year")
                self.logger.info("  - Positive slope: increasing research interest over time")
                self.logger.info("  - Negative slope: declining research interest over time")
                self.logger.info("  - Calculated using linear regression: Δ(papers)/Δ(year)")
                self.logger.info("  - Results prioritize increasing trends, then stable, then declining")
                
            elif query.query_type == QueryType.AUTHOR_EXPERTISE:
                # Check for different possible keys for author count
                author_count = (
                    analysis_result.results.get('unique_authors', 0) or 
                    len(analysis_result.results.get('authors', [])) or
                    len(analysis_result.results.get('top_authors', []))
                )
                total_papers = analysis_result.results.get('total_papers_analyzed', 0)
                self.logger.info(f"Found {author_count} unique authors to analyze from {total_papers} papers")
            
            elif query.query_type == QueryType.CROSS_DOMAIN_ANALYSIS:
                # Cross-domain specific logging
                total_authors = analysis_result.results.get('total_authors_analyzed', 0)
                interdisciplinary_authors = analysis_result.results.get('interdisciplinary_authors', 0)
                single_domain_authors = analysis_result.results.get('single_domain_authors', 0)
                avg_domains = analysis_result.results.get('average_domains_per_author', 0)
                
                self.logger.info(f"Cross-domain analysis: {interdisciplinary_authors} interdisciplinary authors out of {total_authors} total")
                self.logger.info(f"Single-domain authors: {single_domain_authors}")
                self.logger.info(f"Average domains per author: {avg_domains}")
                
                # Log domain distribution
                domain_distribution = analysis_result.results.get('domain_distribution', {})
                if domain_distribution:
                    self.logger.info("Domain coverage distribution:")
                    for domain_count in sorted(domain_distribution.keys()):
                        authors_count = domain_distribution[domain_count]
                        self.logger.info(f"  - {domain_count} domains: {authors_count} authors")
                
                # Log top domains
                top_domains = analysis_result.results.get('top_domains', [])
                if top_domains:
                    self.logger.info("Top domains by paper frequency:")
                    for domain_name, paper_count in top_domains[:5]:
                        self.logger.info(f"  - {domain_name}: {paper_count} papers")
            
            else:
                # Generic logging for other query types
                total_papers = analysis_result.results.get('total_papers', 0) or analysis_result.results.get('total_papers_analyzed', 0)
                if total_papers:
                    self.logger.info(f"Analyzed {total_papers} papers for {query.query_type.value if query.query_type else 'general'} analysis")
        
        # Step 4: Synthesis
        self.logger.info("STEP 4/5: Synthesis")
        self.logger.info("Synthesizing analysis results into coherent response")
        self.logger.debug("Starting synthesis")
        
        synthesis_result = await self.synthesis_agent.synthesize(
            query, retrieval_results, analysis_result
        )
        agent_chain.append("synthesis_agent")
        
        self.logger.info(f"Synthesis completed with confidence: {synthesis_result.confidence:.2f}")
        
        # Log detailed synthesis reasoning and summary
        if hasattr(synthesis_result, 'reasoning') and synthesis_result.reasoning:
            self.logger.info("Synthesis reasoning:")
            self.logger.info(f"  {synthesis_result.reasoning}")
        
        if hasattr(synthesis_result.result, 'summary') and synthesis_result.result.summary:
            self.logger.info("Synthesis summary:")
            self.logger.info(f"  {synthesis_result.result.summary}")
        
        # Log synthesis insights
        if hasattr(synthesis_result.result, 'insights') and synthesis_result.result.insights:
            self.logger.info("Key insights from synthesis:")
            for i, insight in enumerate(synthesis_result.result.insights, 1):
                self.logger.info(f"  {i}. {insight}")
        
        if hasattr(synthesis_result.result, 'top_authors'):
            top_authors = synthesis_result.result.top_authors
            self.logger.info(f"Ranked {len(top_authors)} top authors from analysis")
            if top_authors:
                self.logger.info("Top 5 authors with details:")
                for i, author in enumerate(top_authors[:5], 1):  # Show top 5 instead of 3
                    name = author.get('author', 'Unknown')
                    score = author.get('expertise_score', 0)
                    paper_count = author.get('paper_count', 0)
                    subjects = author.get('subjects', [])
                    self.logger.info(f"  {i}. {name} (score: {score:.2f}, papers: {paper_count})")
                    if subjects:
                        self.logger.info(f"     Research areas: {', '.join(subjects)}")
        
        # Log technology trends if applicable
        if hasattr(synthesis_result.result, 'trends'):
            trends = synthesis_result.result.trends
            self.logger.info(f"Technology trend analysis with {len(trends)} technologies:")
            self.logger.info("Top 5 trending technologies:")
            for i, trend in enumerate(trends[:5], 1):
                name = trend.get('technology', 'Unknown')
                papers = trend.get('total_papers', 0)  # Fixed: use correct field name
                slope = trend.get('trend_slope', 0)    # Fixed: use correct field name
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                self.logger.info(f"  {i}. {name}: {papers} papers, slope={slope:.3f} ({trend_direction})")
        
        # Log cross-domain analysis results if applicable
        if hasattr(synthesis_result.result, 'cross_domain_analysis'):
            cross_domain_data = synthesis_result.result.cross_domain_analysis
            cross_domain_authors = cross_domain_data.get('cross_domain_authors', [])
            
            self.logger.info(f"Cross-domain synthesis with {len(cross_domain_authors)} interdisciplinary authors:")
            
            # Log top interdisciplinary authors
            if cross_domain_authors:
                self.logger.info("Top 5 interdisciplinary authors:")
                for i, author_info in enumerate(cross_domain_authors[:5], 1):
                    name = author_info.get('author', 'Unknown')
                    domain_count = author_info.get('domain_count', 0)
                    paper_count = author_info.get('paper_count', 0)
                    score = author_info.get('interdisciplinary_score', 0)
                    domains = author_info.get('domains', [])
                    
                    self.logger.info(f"  {i}. {name}")
                    self.logger.info(f"     - Domains: {domain_count}, Papers: {paper_count}, Score: {score}")
                    self.logger.info(f"     - Fields: {', '.join(domains[:3])}{'...' if len(domains) > 3 else ''}")
            
            # Log interdisciplinary opportunities
            if hasattr(synthesis_result.result, 'interdisciplinary_opportunities'):
                opportunities = synthesis_result.result.interdisciplinary_opportunities
                if opportunities:
                    self.logger.info("Interdisciplinary opportunities identified:")
                    for i, opportunity in enumerate(opportunities[:3], 1):
                        self.logger.info(f"  {i}. {opportunity}")
            
            # Log LLM insights if available
            if hasattr(synthesis_result.result, 'llm_insights'):
                llm_insights = synthesis_result.result.llm_insights
                if llm_insights and "LLM synthesis error" not in llm_insights:
                    self.logger.info("LLM-Enhanced Cross-Domain Insights:")
                    # Log first 200 characters of LLM insights
                    insight_preview = llm_insights[:200] + "..." if len(llm_insights) > 200 else llm_insights
                    self.logger.info(f"  {insight_preview}")
                elif "error" in llm_insights.lower():
                    self.logger.warning(f"LLM synthesis had issues: {llm_insights[:100]}...")
                        
        # Log overall synthesis insights
        if hasattr(synthesis_result.result, 'summary'):
            self.logger.info(f"Synthesis summary: {synthesis_result.result.summary}")
        elif hasattr(synthesis_result.result, 'insights'):
            insights = synthesis_result.result.insights
            if isinstance(insights, list) and insights:
                self.logger.info(f"Key insights: {insights[0]}")
        
        self.logger.info("")  # Empty line for readability
        
        # Step 5: Validation
        self.logger.info("STEP 5/5: Validation")
        self.logger.info("Validating synthesis results")
        self.logger.debug("Starting validation")
        
        validation_result = await self.validation_agent.validate(synthesis_result)
        agent_chain.append("validation_agent")
        
        self.logger.info(f"Validation completed with final confidence: {validation_result.confidence:.2f}")
        
        # Log validation reasoning if available
        if hasattr(validation_result, 'reasoning') and validation_result.reasoning:
            self.logger.info("Validation reasoning:")
            self.logger.info(f"  {validation_result.reasoning}")
        
        # Log validation insights or issues
        if hasattr(validation_result, 'issues') and validation_result.issues:
            self.logger.info("Validation identified issues:")
            for i, issue in enumerate(validation_result.issues[:3], 1):
                self.logger.info(f"  {i}. {issue}")
        
        # Create final response
        response = SystemResponse(
            query=query,
            response_type=query.query_type.value if query.query_type else "general",
            result=synthesis_result.result,
            processing_time=0.0,  # Will be updated by caller
            agent_chain=agent_chain,
            confidence=validation_result.confidence,
            sources=[paper.paper.paper_id for paper in retrieval_results]
        )
        
        self.logger.info("="*50)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"Final confidence: {validation_result.confidence:.2f}")
        self.logger.info(f"Agent chain: {' → '.join(agent_chain)}")
        self.logger.info("="*50)
        
        return response
    
    def batch_query(
        self,
        queries: List[str],
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[SystemResponse]:
        """Process multiple queries in batch.
        
        Args:
            queries: List of query texts
            params: Optional parameters for all queries
            user_id: Optional user identifier
            
        Returns:
            List of system responses
        """
        return asyncio.run(self.abatch_query(queries, params, user_id))
    
    async def abatch_query(
        self,
        queries: List[str],
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[SystemResponse]:
        """Process multiple queries in batch asynchronously.
        
        Args:
            queries: List of query texts
            params: Optional parameters for all queries
            user_id: Optional user identifier
            
        Returns:
            List of system responses
        """
        tasks = [
            self.aquery(query_text, params, user_id)
            for query_text in queries
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch query {i} failed: {response}")
                responses[i] = SystemResponse(
                    query=Query(text=queries[i], parameters=params or {}),
                    response_type="error",
                    result={"error": str(response)},
                    processing_time=0.0,
                    agent_chain=["error"],
                    confidence=0.0,
                    sources=[]
                )
        
        return responses
    
    def _generate_cache_key(self, query: Query) -> str:
        """Generate cache key for query.
        
        Args:
            query: Query object
            
        Returns:
            Cache key string
        """
        import hashlib
        import json
        
        # Create a hashable representation of the query
        query_dict = {
            "text": query.text.lower().strip(),
            "parameters": sorted(query.parameters.items()) if query.parameters else []
        }
        
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def _update_stats(self, query: Query, processing_time: float, success: bool):
        """Update system statistics.
        
        Args:
            query: Query object
            processing_time: Processing time in seconds
            success: Whether query was successful
        """
        self.stats.total_queries += 1
        
        if success:
            self.stats.successful_queries += 1
        else:
            self.stats.failed_queries += 1
        
        # Update average processing time
        total_time = self.stats.average_processing_time * (self.stats.total_queries - 1)
        self.stats.average_processing_time = (total_time + processing_time) / self.stats.total_queries
        
        # Update query type statistics
        if query.query_type:
            query_type = query.query_type.value
            self.stats.most_common_query_types[query_type] = (
                self.stats.most_common_query_types.get(query_type, 0) + 1
            )
    
    def get_stats(self) -> ProcessingStats:
        """Get system processing statistics.
        
        Returns:
            Processing statistics
        """
        return self.stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check.
        
        Returns:
            Health status information
        """
        health = {
            "status": "healthy",
            "initialized": self.is_initialized,
            "components": {},
            "stats": self.stats.model_dump(),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check data manager
            health["components"]["data_manager"] = await self.data_manager.health_check()
            
            # Check agents
            health["components"]["agents"] = {
                "query_classifier": self.query_classifier.is_initialized,
                "retrieval_agent": self.retrieval_agent.is_initialized,
                "analysis_agent": self.analysis_agent.is_initialized,
                "synthesis_agent": self.synthesis_agent.is_initialized,
                "validation_agent": self.validation_agent.is_initialized,
            }
            
            # Check cache
            health["components"]["cache"] = await self.cache_manager.health_check()
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        self.logger.info("Shutting down TechAuthor system")
        
        try:
            # Shutdown agents
            if hasattr(self.query_classifier, 'shutdown'):
                await self.query_classifier.shutdown()
            if hasattr(self.retrieval_agent, 'shutdown'):
                await self.retrieval_agent.shutdown()
            if hasattr(self.analysis_agent, 'shutdown'):
                await self.analysis_agent.shutdown()
            if hasattr(self.synthesis_agent, 'shutdown'):
                await self.synthesis_agent.shutdown()
            if hasattr(self.validation_agent, 'shutdown'):
                await self.validation_agent.shutdown()
            
            # Shutdown data manager
            if hasattr(self.data_manager, 'shutdown'):
                await self.data_manager.shutdown()
            
            # Shutdown cache
            if hasattr(self.cache_manager, 'shutdown'):
                await self.cache_manager.shutdown()
            
            self.is_initialized = False
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Dictionary with health information
        """
        try:
            issues = []
            
            # Check initialization
            if not self.is_initialized:
                issues.append("System not initialized")
            
            # Check data manager
            if not self.data_manager.is_initialized:
                issues.append("Data manager not initialized")
            elif not self.data_manager.papers:
                issues.append("No papers loaded")
            
            # Check agents
            agents = [
                self.query_classifier, self.retrieval_agent, 
                self.analysis_agent, self.synthesis_agent, self.validation_agent
            ]
            
            for agent in agents:
                if not agent.is_initialized:
                    issues.append(f"{agent.agent_name} agent not initialized")
            
            status = "healthy" if not issues else "unhealthy"
            
            return {
                "status": status,
                "issues": issues,
                "initialized": self.is_initialized,
                "papers_loaded": len(self.data_manager.papers) if self.data_manager.papers else 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "issues": [f"Health check failed: {str(e)}"],
                "initialized": False,
                "papers_loaded": 0
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            stats = {
                "total_papers": len(self.data_manager.papers) if self.data_manager.papers else 0,
                "total_authors": len(self.data_manager.authors_index) if hasattr(self.data_manager, 'authors_index') else 0,
                "total_subjects": len(self.data_manager.subject_index) if hasattr(self.data_manager, 'subject_index') else 0,
                "embeddings_ready": hasattr(self.data_manager, 'hybrid_searcher') and self.data_manager.hybrid_searcher is not None,
                "system_initialized": self.is_initialized
            }
            
            # Add search statistics if available
            if hasattr(self.data_manager, 'hybrid_searcher') and self.data_manager.hybrid_searcher:
                search_stats = self.data_manager.hybrid_searcher.get_search_stats()
                stats.update(search_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}


# Convenience function for quick usage
def create_system(config_path: Optional[str] = None) -> TechAuthorSystem:
    """Create and return a TechAuthor system instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        TechAuthor system instance
    """
    return TechAuthorSystem(config_path)



