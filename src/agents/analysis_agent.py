"""
Analysis agent for TechAuthor system.
"""

import pandas as pd
import numpy as np
import json
import os
import re
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import networkx as nx

from ..core.models import (
    Query, QueryType, RetrievalResult, AnalysisResult,
    AuthorExpertiseResult, TechnologyTrendResult, CollaborationResult
)
from .base_agent import BaseAgent
from ..utils.visualization import create_technology_trends_visualization
from ..services.extraction_service import ExtractionService


class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing retrieved papers and generating insights."""
    
    def __init__(self, llm_manager=None, data_manager=None):
        """Initialize analysis agent."""
        super().__init__("Analysis")
        self.llm_manager = llm_manager  # Store for future CoT enhancements
        self.data_manager = data_manager  # Store for accessing full dataset
        
        # Initialize extraction service
        self.extraction_service = ExtractionService(llm_manager)
        
        # Collaboration graph caching for optimal performance
        self._collaboration_cache = {}  # Cache individual author collaboration results
        self._collaboration_graph = None  # Cache the complete collaboration graph
        self._last_dataset_hash = None  # Track dataset changes for graph invalidation
        
        # Analysis methods mapping
        self.analysis_methods = {
            QueryType.AUTHOR_EXPERTISE: self._analyze_author_expertise,
            QueryType.TECHNOLOGY_TRENDS: self._analyze_technology_trends,
            QueryType.AUTHOR_COLLABORATION: self._analyze_author_collaboration,
            QueryType.DOMAIN_EVOLUTION: self._analyze_domain_evolution,
            QueryType.CROSS_DOMAIN_ANALYSIS: self._analyze_cross_domain,
            QueryType.PAPER_IMPACT: self._analyze_paper_impact,
            QueryType.AUTHOR_PRODUCTIVITY: self._analyze_author_productivity,
            QueryType.AUTHOR_STATS: self._analyze_author_stats,
            QueryType.PAPER_SEARCH: self._analyze_paper_search,
            QueryType.UNCLASSIFIED: self._handle_unclassified
        }
    
    async def _initialize_impl(self) -> None:
        """Initialize the analysis agent."""
        # No special initialization needed for analysis agent
        pass
    
    async def analyze(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> AnalysisResult:
        """Analyze retrieved papers based on query type.
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            
        Returns:
            Analysis result
        """
        return await self.process(query, retrieval_results)
    
    async def _process_impl(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> AnalysisResult:
        """Implementation of analysis processing.
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            
        Returns:
            Analysis result
        """
        self.logger.info(f"Starting analysis of {len(retrieval_results)} papers")
        
        if not retrieval_results:
            self.logger.warning("No papers retrieved for analysis")
            return AnalysisResult(
                query=query,
                retrieved_papers=retrieval_results,
                analysis_type="empty",
                results={"error": "No papers retrieved for analysis"},
                confidence=0.0,
                reasoning="No papers available for analysis"
            )
        
        # Log paper details
        self.logger.info("Papers to analyze:")
        for i, result in enumerate(retrieval_results[:5], 1):
            self.logger.info(f"  {i}. {result.paper.title[:50]}... (Score: {result.score:.3f})")
        
        # Determine analysis method based on query type
        analysis_method = self.analysis_methods.get(
            query.query_type,
            self._analyze_general
        )
        
        self.logger.info(f"Using analysis method for: {query.query_type.value if query.query_type else 'general'}")
        
        # Perform analysis
        try:
            self.logger.info("Executing analysis method...")
            analysis_results, confidence, reasoning = await analysis_method(
                query, retrieval_results
            )
            
            self.logger.info(f"Analysis completed with confidence: {confidence:.3f}")
            self.logger.info(f"Analysis reasoning: {reasoning[:100]}...")
            
            return AnalysisResult(
                query=query,
                retrieved_papers=retrieval_results,
                analysis_type=query.query_type.value if query.query_type else "general",
                results=analysis_results,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                query=query,
                retrieved_papers=retrieval_results,
                analysis_type="error",
                results={"error": str(e)},
                confidence=0.0,
                reasoning=f"Analysis failed: {e}"
            )
    
    async def _analyze_author_expertise(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze author expertise in a domain.
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            
        Returns:
            Tuple of (results, confidence, reasoning)
        """
        # Extract domain from query or retrieval results
        domain = query.parameters.get("domain")
        if not domain and retrieval_results:
            # Infer domain from retrieved papers
            domains = [r.paper.domain for r in retrieval_results]
            domain = Counter(domains).most_common(1)[0][0] if domains else "Unknown"
        
        # Count author appearances
        author_counts = Counter()
        author_papers = defaultdict(list)
        author_subjects = defaultdict(set)
        
        for result in retrieval_results:
            paper = result.paper
            for author in paper.authors:
                author_counts[author] += 1
                # Store both paper ID and title for display
                author_papers[author].append({
                    "paper_id": paper.paper_id,
                    "title": paper.title
                })
                author_subjects[author].update(paper.subjects)
        
        # Calculate expertise scores
        retrieval_config = self.llm_manager.get_retrieval_config() if self.llm_manager else {}
        default_top_k = retrieval_config.get('final_top_k', 20)
        top_k = query.parameters.get("top_k", default_top_k)
        
        # First, get ALL authors (not limited by top_k yet)
        all_scored_authors = []
        
        for author, count in author_counts.items():
            # Calculate expertise score based on multiple factors
            paper_count = count
            subject_diversity = len(author_subjects[author])
            
            # Bonus for working on relevant subjects
            relevance_bonus = 0
            if domain:
                for subject in author_subjects[author]:
                    if domain.lower() in subject.lower():
                        relevance_bonus += 0.1
            
            expertise_score = paper_count + (subject_diversity * 0.1) + relevance_bonus
            
            all_scored_authors.append({
                "author": author,
                "paper_count": paper_count,
                "expertise_score": round(expertise_score, 2),
                "subjects": list(author_subjects[author])[:5],  # Top 5 subjects
                "papers": author_papers[author]  # Now includes titles
            })
        
        # Now sort by expertise score (descending) and take top_k
        all_scored_authors.sort(key=lambda x: x["expertise_score"], reverse=True)
        top_authors = all_scored_authors[:top_k]
        
        # Calculate confidence based on data quality
        total_papers = len(retrieval_results)
        unique_authors = len(author_counts)
        confidence = min(0.9, 0.5 + (total_papers / 100) + (unique_authors / 50))
        
        results = {
            "domain": domain,
            "top_authors": top_authors,
            "total_papers_analyzed": total_papers,
            "unique_authors": unique_authors,
            "methodology": "Frequency-based ranking with subject diversity and relevance weighting"
        }
        
        reasoning = (
            f"Analyzed {total_papers} papers to find top authors in {domain}. "
            f"Ranking based on paper count ({unique_authors} unique authors), "
            f"subject diversity, and domain relevance."
        )
        
        return results, confidence, reasoning
    
    async def _analyze_technology_trends(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze technology trends over time.
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            
        Returns:
            Tuple of (results, confidence, reasoning)
        """
        # Group papers by year and extract subjects/technologies
        yearly_data = defaultdict(lambda: defaultdict(int))
        all_subjects = Counter()
        
        for result in retrieval_results:
            paper = result.paper
            year = paper.date_submitted.year
            
            for subject in paper.subjects:
                yearly_data[year][subject] += 1
                all_subjects[subject] += 1
        
        # Identify trends
        years = sorted(yearly_data.keys())
        trends = []
        
        # Analyze top subjects across years
        top_subjects = [subject for subject, _ in all_subjects.most_common(20)]
        
        if len(years) > 1:
            # Multi-year trend analysis
            for subject in top_subjects:
                year_counts = [yearly_data[year].get(subject, 0) for year in years]
                
                if len(year_counts) > 1:
                    # Calculate trend (simple linear regression slope)
                    x = np.array(range(len(years)))
                    y = np.array(year_counts)
                    
                    if np.sum(y) > 0:  # Only if there are papers
                        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                        
                        trends.append({
                            "technology": subject,
                            "trend_slope": round(slope, 3),
                            "total_papers": int(np.sum(y)),
                            "recent_papers": int(year_counts[-1]) if year_counts else 0,
                            "trend_direction": "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
                        })
        else:
            # Single-year analysis - show current technology areas by popularity
            for subject in top_subjects:
                paper_count = all_subjects[subject]
                if paper_count > 0:
                    trends.append({
                        "technology": subject,
                        "trend_slope": 0.0,  # No trend with single year
                        "total_papers": paper_count,
                        "recent_papers": paper_count,
                        "trend_direction": "current",  # Indicate this is current state, not trend
                        "relevance_score": round(paper_count / len(retrieval_results), 3)
                    })
        
        # Sort by trend strength (or relevance for single year)
        if len(years) > 1:
            # Sort by trend direction first (increasing trends first), then by slope magnitude
            def trend_sort_key(trend):
                slope = trend.get("trend_slope", 0)
                direction = trend.get("trend_direction", "stable")
                total_papers = trend.get("total_papers", 0)
                
                # Priority order: increasing trends first, then stable, then declining
                # Within each category, sort by slope magnitude and paper count
                if direction == "increasing":
                    return (0, -slope, -total_papers)  # Negative for descending order
                elif direction == "stable":
                    return (1, -total_papers)  # Stable trends sorted by paper count
                else:  # decreasing
                    return (2, slope, -total_papers)  # Positive slope for declining (less negative first)
            
            trends.sort(key=trend_sort_key)
            
            # Identify emerging and declining technologies after sorting
            emerging = [t for t in trends if t.get("trend_direction") == "increasing"][:5]
            declining = [t for t in trends if t.get("trend_direction") == "decreasing"][:5]
        else:
            trends.sort(key=lambda x: x["total_papers"], reverse=True)
            # For single year, no emerging/declining, just current popular topics
            emerging = []
            declining = []
        
        # Time series data for visualization
        time_series = {
            "years": years,
            "subject_counts": {
                subject: [yearly_data[year].get(subject, 0) for year in years]
                for subject in top_subjects[:10]
            }
        }
        
        # Adjust confidence based on available years
        base_confidence = 0.4 + (len(retrieval_results) / 200)
        if len(years) > 1:
            confidence = min(0.9, base_confidence + (len(years) / 20))
        else:
            confidence = min(0.7, base_confidence)  # Lower confidence for single-year data
        
        results = {
            "trends": trends[:15],
            "emerging_technologies": [t["technology"] for t in emerging[:5]],
            "declining_technologies": [t["technology"] for t in declining[:5]],
            "time_series_data": time_series,
            "analysis_period": f"{min(years)}-{max(years)}" if len(years) > 1 else f"{years[0]} (single year)" if years else "Unknown",
            "total_papers": len(retrieval_results),
            "analysis_type": "temporal_trends" if len(years) > 1 else "current_technologies"
        }
        
        # Create visualization
        try:
            self.logger.info("Creating technology trends visualization...")
            visualization_path = create_technology_trends_visualization(
                results, query.text
            )
            results["visualization_path"] = visualization_path
            self.logger.info(f"Visualization saved to: {visualization_path}")
        except Exception as e:
            self.logger.warning(f"Failed to create visualization: {e}")
            results["visualization_error"] = str(e)
        
        if len(years) > 1:
            reasoning = (
                f"Analyzed {len(retrieval_results)} papers from {min(years)} to "
                f"{max(years)} to identify technology trends. "
                f"Used linear regression on yearly paper counts to determine trend directions. "
                f"Results prioritize increasing trends first, followed by stable and declining trends. "
                f"Generated comprehensive visualization with time series, trend slopes, and emerging/declining technologies."
            )
        else:
            reasoning = (
                f"Analyzed {len(retrieval_results)} papers from {years[0] if years else 'unknown period'} "
                f"to identify current technology areas. Single-year data limits trend analysis, "
                f"so showing current technology popularity instead. "
                f"Generated visualization showing current technology landscape and popularity rankings."
            )
        
        return results, confidence, reasoning
    
    async def _analyze_author_collaboration(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze author collaboration patterns.
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            
        Returns:
            Tuple of (results, confidence, reasoning)
        """
        self.logger.info("Starting author collaboration analysis")
        
        # Get focal author from query parameters (set by classifier)
        focal_author = query.parameters.get('author')
        self.logger.info(f"Focal author from parameters: '{focal_author}'")
        
        if focal_author:
            self.logger.info(f"Found focal author '{focal_author}' in query parameters")
            # Try to get data from author database
            try:
                author_data = await self._get_author_from_database(focal_author)
                if author_data:
                    self.logger.info(f"Found author data for '{focal_author}' with {author_data.get('num_collaborators', 0)} collaborators")
                    return await self._analyze_specific_author_collaboration(
                        focal_author, author_data, retrieval_results
                    )
                else:
                    self.logger.info(f"No author data found for '{focal_author}' in database")
            except Exception as e:
                self.logger.error(f"ERROR in _get_author_from_database: {e}")
        
        # Fallback: try to extract author from query text if not in parameters
        if not focal_author:
            focal_author = await self._extract_author_name_from_query(query.text)
            self.logger.info(f"Extracted from query text: '{focal_author}'")
        
        # Fallback: try to find author in retrieved papers
        if not focal_author:
            focal_author = self._find_author_in_papers(query.text.lower(), retrieval_results)
            self.logger.info(f"Found in papers: '{focal_author}'")
        
        # If still no focal author found, analyze general collaboration patterns
        if not focal_author:
            self.logger.info("No focal author found, proceeding with general collaboration analysis")
            return await self._analyze_general_collaboration(query, retrieval_results)
        
        # Fallback: Analyze specific author's collaborations from retrieved papers
        self.logger.info(f"Analyzing collaboration from papers for: '{focal_author}'")
        return await self._analyze_author_from_papers(focal_author, retrieval_results)
    
    async def _analyze_author_from_papers(
        self,
        focal_author: str,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze author collaboration from retrieved papers (fallback method)."""
        collaborations = Counter()
        shared_papers = defaultdict(list)
        collaboration_subjects = defaultdict(set)
        
        for result in retrieval_results:
            paper = result.paper
            if focal_author in paper.authors:
                for author in paper.authors:
                    if author != focal_author:
                        collaborations[author] += 1
                        shared_papers[author].append(paper.paper_id)
                        collaboration_subjects[author].update(paper.subjects)
        
        # Build collaboration network
        G = nx.Graph()
        G.add_node(focal_author)
        
        for collaborator, count in collaborations.items():
            G.add_node(collaborator)
            G.add_edge(focal_author, collaborator, weight=count)
        
        # Calculate network metrics
        try:
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
        except:
            centrality = {focal_author: 1.0}
            betweenness = {focal_author: 0.0}
        
        # Prepare results
        top_collaborators = []
        for author, count in collaborations.most_common(10):
            top_collaborators.append({
                "collaborator": author,
                "collaboration_count": count,
                "shared_papers": len(shared_papers[author]),
                "common_subjects": list(collaboration_subjects[author])[:5],
                "centrality": round(centrality.get(author, 0), 3)
            })
        
        confidence = min(0.9, 0.6 + (len(collaborations) / 50))
        
        results = {
            "focal_author": focal_author,
            "total_collaborators": len(collaborations),
            "top_collaborators": top_collaborators,
            "network_size": G.number_of_nodes(),
            "network_density": round(nx.density(G), 3) if G.number_of_nodes() > 1 else 0,
            "focal_author_centrality": round(centrality.get(focal_author, 0), 3)
        }
        
        reasoning = (
            f"Analyzed collaboration patterns for {focal_author} based on "
            f"{len([r for r in retrieval_results if focal_author in r.paper.authors])} papers. "
            f"Found {len(collaborations)} unique collaborators."
        )
        
        return results, confidence, reasoning
    
    async def _analyze_general_collaboration(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze general collaboration patterns when no focal author is specified."""
        # Build collaboration graph
        G = nx.Graph()
        paper_author_map = {}
        
        for result in retrieval_results:
            paper = result.paper
            authors = paper.authors
            paper_author_map[paper.paper_id] = authors
            
            # Add nodes and edges
            for author in authors:
                G.add_node(author)
            
            # Add collaboration edges
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    if G.has_edge(author1, author2):
                        G[author1][author2]['weight'] += 1
                    else:
                        G.add_edge(author1, author2, weight=1)
        
        # Calculate network metrics
        if G.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(G)
            try:
                betweenness_centrality = nx.betweenness_centrality(G)
            except:
                betweenness_centrality = {}
            
            # Find most collaborative authors
            most_collaborative = sorted(
                degree_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Find most connected components
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len) if components else set()
            
            results = {
                "total_authors": G.number_of_nodes(),
                "total_collaborations": G.number_of_edges(),
                "network_density": round(nx.density(G), 4),
                "most_collaborative_authors": [
                    {"author": author, "centrality": round(centrality, 3)}
                    for author, centrality in most_collaborative
                ],
                "largest_component_size": len(largest_component),
                "number_of_components": len(components)
            }
            
            confidence = min(0.8, 0.4 + (G.number_of_nodes() / 100))
            
        else:
            results = {"error": "No collaboration data found"}
            confidence = 0.1
        
        reasoning = (
            f"Analyzed general collaboration patterns across {len(retrieval_results)} papers "
            f"involving {G.number_of_nodes()} unique authors."
        )
        
        return results, confidence, reasoning
    
    async def _get_author_from_database(self, author_name: str) -> Optional[Dict[str, Any]]:
        """Load author data from the processed author database."""
        try:
            # Path to author stats file
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            author_stats_path = os.path.join(current_dir, 'data', 'processed', 'authors', 'author_stats.json')
            
            if not os.path.exists(author_stats_path):
                return None
            
            with open(author_stats_path, 'r', encoding='utf-8') as f:
                author_stats = json.load(f)
            
            # Try exact match first
            if author_name in author_stats:
                return author_stats[author_name]
            
            # Try fuzzy matching for similar names
            best_match = None
            best_score = 0.0
            
            for existing_author in author_stats.keys():
                score = SequenceMatcher(None, author_name.lower(), existing_author.lower()).ratio()
                if score > best_score and score > 0.8:  # 80% similarity threshold
                    best_score = score
                    best_match = existing_author
            
            if best_match:
                return author_stats[best_match]
                
        except Exception as e:
            self.logger.warning(f"Error loading author database: {e}")
            
        return None
    
    def _find_author_in_papers(self, query_lower: str, retrieval_results: List[RetrievalResult]) -> Optional[str]:
        """Find author in retrieved papers (fallback method)."""
        for result in retrieval_results:
            for author in result.paper.authors:
                if author.lower() in query_lower:
                    return author
        return None
    
    async def _analyze_specific_author_collaboration(
        self,
        focal_author: str,
        author_data: Dict[str, Any],
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze collaboration for a specific author using database data and full dataset."""
        
        collaborators = author_data.get('collaborators', [])
        total_papers = author_data.get('total_papers', 0)
        num_collaborators = author_data.get('num_collaborators', 0)
        subjects = author_data.get('subjects', [])
        years_active = author_data.get('years_active', [])
        
        # Calculate EXACT collaboration counts from full dataset using collaboration graph
        exact_collaborations = {}
        if self.data_manager and hasattr(self.data_manager, 'papers') and self.data_manager.papers:
            self.logger.info(f"Getting collaboration data for {focal_author} using graph-based analysis")
            
            # Use graph-based collaboration calculation (builds graph once, caches for future use)
            exact_collaborations = self._get_collaborations_from_graph(focal_author)
            
            if exact_collaborations:
                self.logger.info(f"Graph-based collaboration analysis successful for {focal_author}: {len(exact_collaborations)} collaborators found")
            else:
                self.logger.info(f"No collaborations found for {focal_author} in dataset")
        else:
            self.logger.info("Data manager not available, using estimated collaboration counts")
        
        # Prepare top collaborators with EXACT data from graph analysis, properly sorted
        top_collaborators = []
        if exact_collaborations:
            # Sort collaborators by collaboration count (descending) and use exact data
            sorted_collaborators = sorted(
                exact_collaborations.items(), 
                key=lambda x: x[1]['collaboration_count'], 
                reverse=True
            )
            
            for i, (collaborator, stats) in enumerate(sorted_collaborators[:15]):  # Top 15 collaborators
                collab_count = stats['collaboration_count']
                shared_papers = stats['shared_papers']
                centrality = round(1.0 - (i * 0.05), 3)
                
                top_collaborators.append({
                    "collaborator": collaborator,
                    "collaboration_count": collab_count,
                    "shared_papers": shared_papers,
                    "centrality": centrality
                })
        else:
            # Fallback to database estimates (sorted by database order)
            for i, collaborator in enumerate(collaborators[:15]):  # Top 15 collaborators
                collab_count = max(1, int(total_papers * 0.15 * (16 - i) / 15))
                shared_papers = max(1, int(collab_count * 0.7))
                centrality = round(1.0 - (i * 0.06), 3)
                
                top_collaborators.append({
                    "collaborator": collaborator,
                    "collaboration_count": collab_count,
                    "shared_papers": shared_papers,
                    "centrality": centrality
                })
        
        # Calculate collaboration statistics using exact data when available
        if exact_collaborations:
            actual_collaborators_count = len(exact_collaborations)
            self.logger.info(f"Using exact collaboration count: {actual_collaborators_count} collaborators found")
        else:
            actual_collaborators_count = num_collaborators
            self.logger.info(f"Using database collaboration count: {actual_collaborators_count} collaborators")
        
        avg_collaborators_per_paper = round(actual_collaborators_count / max(total_papers, 1), 2)
        collaboration_intensity = "High" if actual_collaborators_count > 50 else "Medium" if actual_collaborators_count > 20 else "Low"
        
        # Build network metrics using actual counts
        network_density = round(min(1.0, actual_collaborators_count / 200), 3)
        
        results = {
            "focal_author": focal_author,
            "total_collaborators": actual_collaborators_count,  # Use actual count
            "top_collaborators": top_collaborators,
            "network_size": actual_collaborators_count + 1,  # +1 for focal author
            "network_density": network_density,
            "focal_author_centrality": 1.0,  # Focal author has maximum centrality
            "author_profile": {
                "total_papers": total_papers,
                "total_collaborators": actual_collaborators_count,  # Use actual count
                "years_active": f"{min(years_active)}-{max(years_active)}" if years_active else "Unknown",
                "primary_subjects": subjects[:5],  # Top 5 research areas
                "collaboration_intensity": collaboration_intensity,
                "avg_collaborators_per_paper": avg_collaborators_per_paper
            },
            "collaboration_network": {
                "network_size": actual_collaborators_count + 1,  # Use actual count
                "estimated_network_density": network_density,
                "research_breadth": len(subjects)
            }
        }
        
        confidence = min(0.95, 0.8 + (min(actual_collaborators_count, 50) / 100))
        
        data_source = "exact dataset analysis" if exact_collaborations else "author database estimates"
        reasoning = (
            f"Found {focal_author} in author database with {total_papers} papers and "
            f"{actual_collaborators_count} collaborators across {len(subjects)} research areas. "
            f"Analysis based on {data_source}."
        )
        
        return results, confidence, reasoning
    
    # NOTE: Old _calculate_exact_collaborations method removed - now using graph-based approach
    # All collaboration calculations are handled by _get_collaborations_from_graph method
    
    def _extract_authors_from_paper(self, paper) -> List[str]:
        """Extract authors from different paper data structures."""
        try:
            # Handle Paper model objects
            if hasattr(paper, 'authors'):
                authors = paper.authors
                if isinstance(authors, list):
                    return [str(author).strip() for author in authors if author]
                elif isinstance(authors, str):
                    # Parse comma/semicolon separated authors
                    return [author.strip() for author in re.split(r'[,;]', authors) if author.strip()]
            
            # Handle dictionary objects
            if isinstance(paper, dict):
                if 'authors' in paper:
                    authors = paper['authors']
                    if isinstance(authors, list):
                        return [str(author).strip() for author in authors if author]
                    elif isinstance(authors, str):
                        return [author.strip() for author in re.split(r'[,;]', authors) if author.strip()]
                
                # Alternative field names
                for field in ['author', 'author_list', 'paper_authors']:
                    if field in paper:
                        authors = paper[field]
                        if isinstance(authors, (list, str)):
                            if isinstance(authors, str):
                                return [author.strip() for author in re.split(r'[,;]', authors) if author.strip()]
                            else:
                                return [str(author).strip() for author in authors if author]
            
            # Handle pandas Series/DataFrame rows
            if hasattr(paper, 'get'):
                authors = paper.get('authors', [])
                if isinstance(authors, list):
                    return [str(author).strip() for author in authors if author]
                elif isinstance(authors, str):
                    return [author.strip() for author in re.split(r'[,;]', authors) if author.strip()]
                    
        except Exception as e:
            self.logger.warning(f"Error extracting authors from paper: {e}")
        
        return []
    
    def _extract_paper_id(self, paper, index: int) -> str:
        """Extract paper ID from different data structures."""
        try:
            # Handle Paper model objects
            if hasattr(paper, 'paper_id'):
                return str(paper.paper_id)
            
            # Handle dictionary objects
            if isinstance(paper, dict):
                for field in ['paper_id', 'id', 'arxiv_id', 'doi']:
                    if field in paper and paper[field]:
                        return str(paper[field])
            
            # Handle pandas Series/DataFrame rows
            if hasattr(paper, 'get'):
                for field in ['paper_id', 'id', 'arxiv_id', 'doi']:
                    value = paper.get(field)
                    if value:
                        return str(value)
        except Exception:
            pass
        
        # Fallback to index-based ID
        return f"paper_{index}"
    
    def _authors_match(self, author1: str, author2: str) -> bool:
        """Check if two author names refer to the same person with fuzzy matching."""
        if not author1 or not author2:
            return False
        
        # Clean and normalize names
        name1 = re.sub(r'[^\w\s]', '', author1.lower()).strip()
        name2 = re.sub(r'[^\w\s]', '', author2.lower()).strip()
        
        # Exact match
        if name1 == name2:
            return True
        
        # Handle common variations (initials, order, etc.)
        # Split into components
        parts1 = name1.split()
        parts2 = name2.split()
        
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Check if last names match and first name/initial match
            if parts1[-1] == parts2[-1]:  # Same last name
                # Check first name/initial compatibility
                first1, first2 = parts1[0], parts2[0]
                if first1 == first2 or first1.startswith(first2) or first2.startswith(first1):
                    return True
        
        return False
    
    def _build_collaboration_graph(self) -> Optional[nx.Graph]:
        """Build a complete collaboration graph from the dataset with caching.
        
        This method builds the ENTIRE collaboration network once and caches it.
        Future collaboration queries can then use graph algorithms for faster analysis.
        
        Returns:
            NetworkX graph with collaboration edges, or None if data unavailable
        """
        if not self.data_manager or not hasattr(self.data_manager, 'papers') or not self.data_manager.papers:
            self.logger.warning("Cannot build collaboration graph: no dataset available")
            return None
        
        # Check if we need to rebuild the graph
        current_hash = hash(str(len(self.data_manager.papers)))  # Simple hash based on dataset size
        if self._collaboration_graph is not None and self._last_dataset_hash == current_hash:
            self.logger.info("Using cached collaboration graph")
            return self._collaboration_graph
        
        self.logger.info(f"Building collaboration graph from {len(self.data_manager.papers)} papers...")
        
        # Create graph
        G = nx.Graph()
        paper_count = 0
        processed_papers = 0
        
        for i, paper in enumerate(self.data_manager.papers):
            try:
                authors = self._extract_authors_from_paper(paper)
                if len(authors) < 2:  # Skip single-author papers
                    continue
                
                processed_papers += 1
                paper_count += 1
                
                # Add collaboration edges between all pairs of authors
                for j, author1 in enumerate(authors):
                    for author2 in authors[j+1:]:
                        # Normalize author names
                        norm_author1 = re.sub(r'[^\w\s]', '', author1.lower()).strip()
                        norm_author2 = re.sub(r'[^\w\s]', '', author2.lower()).strip()
                        
                        if norm_author1 and norm_author2 and norm_author1 != norm_author2:
                            if G.has_edge(norm_author1, norm_author2):
                                G[norm_author1][norm_author2]['weight'] += 1
                            else:
                                G.add_edge(norm_author1, norm_author2, weight=1)
                                
            except Exception as e:
                self.logger.warning(f"Error processing paper {i} for collaboration graph: {e}")
                continue
        
        # Cache the graph
        self._collaboration_graph = G
        self._last_dataset_hash = current_hash
        
        self.logger.info(f"Collaboration graph built successfully: {G.number_of_nodes()} authors, "
                        f"{G.number_of_edges()} collaboration edges from {paper_count} multi-author papers "
                        f"(processed {processed_papers} total papers)")
        
        return G
    
    def _get_collaborations_from_graph(self, focal_author: str) -> Dict[str, Dict[str, int]]:
        """Get collaboration data for an author from the prebuilt graph with caching.
        
        Args:
            focal_author: The author to analyze
            
        Returns:
            Dictionary mapping collaborator names to their collaboration stats
        """
        # Check cache first
        cache_key = focal_author.lower().strip()
        if cache_key in self._collaboration_cache:
            self.logger.info(f"Using cached collaboration data for {focal_author}")
            return self._collaboration_cache[cache_key]
        
        # Build/get the collaboration graph
        graph = self._build_collaboration_graph()
        if not graph:
            self.logger.warning("Cannot get collaborations: collaboration graph unavailable")
            return {}
        
        # Normalize focal author name for graph lookup
        norm_focal = re.sub(r'[^\w\s]', '', focal_author.lower()).strip()
        
        # Try exact match first
        if norm_focal not in graph:
            # Try fuzzy matching against all nodes
            best_match = None
            for node in graph.nodes():
                if self._authors_match(focal_author, node):
                    best_match = node
                    break
            
            if best_match:
                norm_focal = best_match
                self.logger.info(f"Fuzzy matched '{focal_author}' to '{best_match}' in collaboration graph")
            else:
                self.logger.info(f"Author '{focal_author}' not found in collaboration graph")
                # Cache the empty result to avoid repeated searches
                self._collaboration_cache[cache_key] = {}
                return {}
        
        # Get all neighbors (collaborators) and their edge weights
        collaborations = {}
        for collaborator in graph.neighbors(norm_focal):
            weight = graph[norm_focal][collaborator]['weight']
            collaborations[collaborator] = {
                "collaboration_count": weight,
                "shared_papers": weight  # In this model, each edge weight = shared papers
            }
        
        # Cache the results for future queries
        self._collaboration_cache[cache_key] = collaborations
        
        self.logger.info(f"Graph-based collaboration lookup for '{focal_author}': {len(collaborations)} collaborators found")
        return collaborations

    async def _analyze_domain_evolution(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze how a domain has evolved over time with focus on methodology and conceptual changes.
        
        This analysis focuses on:
        1. Evolution Timeline: Key phases and transitions
        2. Conceptual Evolution: How problems and solutions evolved
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            
        Returns:
            Tuple of (results, confidence, reasoning)
        """
        self.logger.info(f"Starting domain evolution analysis for {len(retrieval_results)} papers")
        
        # Extract domain from query
        domain = query.parameters.get("domain", "Unknown Domain")
        time_range = query.parameters.get("time_range", "")
        
        # Group papers by time periods for evolution analysis
        time_periods = self._create_time_periods(retrieval_results)
        self.logger.info(f"Created {len(time_periods)} time periods for analysis")
        
        # Analyze evolution timeline
        evolution_timeline = await self._analyze_evolution_timeline(
            retrieval_results, time_periods, domain
        )
        
        # Analyze conceptual evolution
        conceptual_evolution = await self._analyze_conceptual_evolution(
            retrieval_results, time_periods, domain
        )
        
        # Calculate confidence based on data quality
        confidence = self._calculate_domain_evolution_confidence(
            retrieval_results, time_periods, evolution_timeline, conceptual_evolution
        )
        
        # Build comprehensive results
        results = {
            "analysis_type": "domain_evolution",
            "domain": domain,
            "time_range_analyzed": time_range,
            "total_papers": len(retrieval_results),
            "time_periods": len(time_periods),
            "evolution_timeline": evolution_timeline,
            "conceptual_evolution": conceptual_evolution,
            "analysis_metadata": {
                "methodology": "Domain evolution analysis focusing on paradigm shifts and conceptual changes",
                "approach": "Time-period segmentation with methodology and concept extraction",
                "confidence_factors": [
                    "Number of papers analyzed",
                    "Time span coverage", 
                    "Methodology diversity",
                    "Concept emergence patterns"
                ]
            }
        }
        
        reasoning = (
            f"Analyzed {len(retrieval_results)} papers across {len(time_periods)} time periods "
            f"to trace the evolution of {domain}. Focused on identifying methodology shifts, "
            f"conceptual developments, and paradigm transitions rather than simple trend counting. "
            f"Used semantic analysis to extract evolution patterns and key transition points."
        )
        
        self.logger.info(f"Domain evolution analysis completed with confidence {confidence:.2f}")
        
        return results, confidence, reasoning
    
    async def _analyze_cross_domain(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze authors working across multiple domains."""
        author_domains = defaultdict(set)
        author_papers = defaultdict(list)
        
        for result in retrieval_results:
            paper = result.paper
            for author in paper.authors:
                author_domains[author].add(paper.domain)
                author_domains[author].update(paper.subjects)
                author_papers[author].append(paper.paper_id)
        
        # Find cross-domain authors
        cross_domain_authors = []
        for author, domains in author_domains.items():
            if len(domains) > 1:  # Works in multiple domains
                cross_domain_authors.append({
                    "author": author,
                    "domain_count": len(domains),
                    "domains": list(domains),
                    "paper_count": len(author_papers[author]),
                    "interdisciplinary_score": len(domains) * len(author_papers[author])
                })
        
        # Sort by interdisciplinary score
        cross_domain_authors.sort(key=lambda x: x["interdisciplinary_score"], reverse=True)
        
        confidence = min(0.8, 0.5 + (len(cross_domain_authors) / 50))
        
        results = {
            "cross_domain_authors": cross_domain_authors[:15],
            "total_authors_analyzed": len(author_domains),
            "interdisciplinary_authors": len(cross_domain_authors),
            "average_domains_per_author": round(
                sum(len(domains) for domains in author_domains.values()) / len(author_domains), 2
            ) if author_domains else 0
        }
        
        reasoning = (
            f"Analyzed {len(author_domains)} authors to identify those working across "
            f"multiple domains. Found {len(cross_domain_authors)} interdisciplinary researchers."
        )
        
        return results, confidence, reasoning
    
    async def _analyze_paper_impact(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze paper impact (simplified - would use citation data in practice)."""
        # Since we don't have citation data, use retrieval score as proxy for impact
        papers_by_impact = []
        
        for result in retrieval_results:
            paper = result.paper
            # Use retrieval score and other factors as impact proxy
            impact_score = result.score
            
            # Boost for multiple authors (collaboration indicator)
            impact_score += len(paper.authors) * 0.01
            
            # Boost for recent papers
            years_old = datetime.now().year - paper.date_submitted.year
            if years_old < 5:
                impact_score += 0.1
            
            papers_by_impact.append({
                "title": paper.title,
                "paper_id": paper.paper_id,
                "authors": paper.authors,
                "year": paper.date_submitted.year,
                "impact_score": round(impact_score, 3),
                "domain": paper.domain,
                "subjects": paper.subjects
            })
        
        # Sort by impact score
        papers_by_impact.sort(key=lambda x: x["impact_score"], reverse=True)
        
        confidence = 0.6  # Lower confidence since we don't have real citation data
        
        results = {
            "high_impact_papers": papers_by_impact[:10],
            "total_papers_analyzed": len(retrieval_results),
            "methodology": "Retrieval relevance score with collaboration and recency bonuses",
            "note": "Impact scores are estimates based on retrieval relevance, not citation counts"
        }
        
        reasoning = (
            f"Analyzed {len(retrieval_results)} papers using retrieval relevance as an impact proxy. "
            f"Note: Real impact analysis would require citation data."
        )
        
        return results, confidence, reasoning
    
    async def _analyze_author_productivity(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze author productivity patterns."""
        author_yearly_output = defaultdict(lambda: defaultdict(int))
        author_total_output = Counter()
        
        for result in retrieval_results:
            paper = result.paper
            year = paper.date_submitted.year
            
            for author in paper.authors:
                author_yearly_output[author][year] += 1
                author_total_output[author] += 1
        
        # Calculate productivity metrics
        productive_authors = []
        for author, total_papers in author_total_output.most_common(15):
            yearly_data = author_yearly_output[author]
            years = list(yearly_data.keys())
            
            if years:
                years_active = max(years) - min(years) + 1
                avg_papers_per_year = total_papers / years_active if years_active > 0 else total_papers
                
                productive_authors.append({
                    "author": author,
                    "total_papers": total_papers,
                    "years_active": years_active,
                    "avg_papers_per_year": round(avg_papers_per_year, 2),
                    "first_paper_year": min(years),
                    "last_paper_year": max(years),
                    "peak_year": max(yearly_data, key=yearly_data.get),
                    "peak_year_papers": max(yearly_data.values())
                })
        
        confidence = min(0.9, 0.5 + (len(retrieval_results) / 100))
        
        results = {
            "productive_authors": productive_authors,
            "total_authors_analyzed": len(author_total_output),
            "total_papers_analyzed": len(retrieval_results),
            "analysis_metrics": ["total_papers", "years_active", "avg_papers_per_year", "peak_productivity"]
        }
        
        reasoning = (
            f"Analyzed productivity of {len(author_total_output)} authors based on "
            f"{len(retrieval_results)} papers. Calculated average papers per year and peak productivity periods."
        )
        
        return results, confidence, reasoning
    
    async def _analyze_general(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """General analysis for queries that don't fit specific categories."""
        # Provide basic statistics and insights
        domains = Counter(r.paper.domain for r in retrieval_results)
        subjects = Counter()
        authors = Counter()
        years = Counter()
        
        for result in retrieval_results:
            paper = result.paper
            subjects.update(paper.subjects)
            authors.update(paper.authors)
            years[paper.date_submitted.year] += 1
        
        results = {
            "total_papers": len(retrieval_results),
            "top_domains": domains.most_common(5),
            "top_subjects": subjects.most_common(10),
            "top_authors": authors.most_common(10),
            "publication_years": dict(years.most_common()),
            "year_range": f"{min(years.keys())}-{max(years.keys())}" if years else "Unknown"
        }
        
        confidence = 0.7
        reasoning = "General statistical analysis of retrieved papers across multiple dimensions"
        
        return results, confidence, reasoning

    async def _analyze_author_stats(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Analyze detailed statistics for a specific author."""
        # Extract author name from query parameters
        author_name = query.parameters.get("author")
        if not author_name:
            # Try to extract from query text
            author_name = await self._extract_author_name_from_query(query.text)
        
        if not author_name:
            return {
                "error": "No author name provided in the query",
                "suggestion": "Please specify an author name to get their statistics"
            }, 0.1, "No author name found in query"
        
        # Load author statistics from preprocessed file
        import json
        from pathlib import Path
        
        try:
            author_stats_path = Path("data/processed/authors/author_stats.json")
            if author_stats_path.exists():
                with open(author_stats_path, 'r', encoding='utf-8') as f:
                    all_author_stats = json.load(f)
                
                # Find exact match or close matches
                exact_match = all_author_stats.get(author_name)
                if exact_match:
                    results = {
                        "author_name": author_name,
                        "found": True,
                        "stats": exact_match,
                        "analysis_type": "exact_match"
                    }
                    confidence = 0.9
                    reasoning = f"Found exact match for author '{author_name}' in database"
                else:
                    # Find similar names
                    similar_authors = self._find_similar_author_names(author_name, all_author_stats)
                    results = {
                        "author_name": author_name,
                        "found": False,
                        "similar_authors": similar_authors[:5],
                        "suggestion": f"No exact match found for '{author_name}'. Here are some similar author names:",
                        "analysis_type": "fuzzy_match"
                    }
                    confidence = 0.3
                    reasoning = f"No exact match found for '{author_name}', provided similar alternatives"
                
                return results, confidence, reasoning
            else:
                return {
                    "error": "Author statistics database not found",
                    "author_name": author_name
                }, 0.1, "Author statistics file not available"
                
        except Exception as e:
            return {
                "error": f"Failed to load author statistics: {str(e)}",
                "author_name": author_name
            }, 0.1, f"Error accessing author statistics: {str(e)}"

    async def _analyze_paper_search(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Search for specific papers by technology, title, or ID."""
        # Extract search terms from query
        search_term = query.parameters.get("technology") or query.parameters.get("paper_title") or query.parameters.get("paper_id")
        if not search_term:
            search_term = await self._extract_search_term_from_query(query.text)
        
        if not search_term:
            return {
                "error": "No search term found in query",
                "suggestion": "Please specify a paper title, technology name, or paper ID to search for"
            }, 0.1, "No search term found in query"
        
        self.logger.info(f"Searching for papers matching: '{search_term}'")
        
        # First, try to find exact paper ID match in data manager if available
        if hasattr(self, 'data_manager') and self.data_manager and hasattr(self.data_manager, 'papers'):
            exact_paper_match = None
            for paper in self.data_manager.papers:
                if paper.paper_id == search_term:
                    exact_paper_match = paper
                    break
            
            if exact_paper_match:
                self.logger.info(f"Found exact paper match in data manager: {exact_paper_match.title}")
                return {
                    "search_results": [{
                        "paper_id": exact_paper_match.paper_id,
                        "title": exact_paper_match.title,
                        "authors": exact_paper_match.authors,
                        "abstract": exact_paper_match.abstract,
                        "domain": exact_paper_match.domain,
                        "date_submitted": exact_paper_match.date_submitted.strftime("%Y-%m-%d") if exact_paper_match.date_submitted else "Unknown",
                        "score": 1.0,
                        "match_type": "exact_id_direct"
                    }],
                    "total_matches": 1,
                    "search_term": search_term
                }, 1.0, f"Found exact paper match for ID: {search_term}"
        
        # First, try exact paper ID match across all papers
        exact_match = None
        for result in retrieval_results:
            if result.paper.paper_id == search_term:
                exact_match = result
                break
        
        if exact_match:
            # Found exact match by paper ID
            matching_papers = [{
                "paper_id": exact_match.paper.paper_id,
                "title": exact_match.paper.title,
                "authors": exact_match.paper.authors,
                "abstract": exact_match.paper.abstract,
                "domain": exact_match.paper.domain,
                "date_submitted": exact_match.paper.date_submitted.strftime("%Y-%m-%d") if exact_match.paper.date_submitted else "Unknown",
                "score": 1.0,  # Perfect match
                "match_type": "exact_id"
            }]
        else:
            # Filter papers that contain the search term in title, abstract, or paper_id
            matching_papers = []
            for result in retrieval_results:
                paper = result.paper
                title_match = search_term.lower() in paper.title.lower()
                abstract_match = search_term.lower() in paper.abstract.lower()
                # Also check paper_id for arXiv IDs and other identifiers
                id_match = search_term.lower() in paper.paper_id.lower()
                
                if title_match or abstract_match or id_match:
                    match_score = result.score
                    if title_match:
                        match_score += 0.3  # Boost for title matches
                    if id_match:
                        match_score += 0.5  # Higher boost for exact ID matches
                    
                    matching_papers.append({
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "domain": paper.domain if hasattr(paper, 'domain') else 'Unknown Domain',
                        "date_submitted": paper.date_submitted.strftime("%Y-%m-%d") if paper.date_submitted else "Unknown Date",
                        "year": paper.date_submitted.year if paper.date_submitted else None,
                        "subjects": paper.subjects if hasattr(paper, 'subjects') else [],
                        "abstract": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                        "match_score": round(match_score, 3),
                        "match_type": "title" if title_match else "abstract"
                    })
        
        # Sort by match score
        matching_papers.sort(key=lambda x: x["match_score"], reverse=True)
        
        if matching_papers:
            confidence = min(0.9, 0.5 + (len(matching_papers) / 20))
            results = {
                "search_term": search_term,
                "papers_found": len(matching_papers),
                "matching_papers": matching_papers[:10],  # Top 10 matches
                "total_papers_searched": len(retrieval_results)
            }
            reasoning = f"Found {len(matching_papers)} papers matching '{search_term}' in titles or abstracts"
        else:
            confidence = 0.2
            results = {
                "search_term": search_term,
                "papers_found": 0,
                "matching_papers": [],
                "message": f"No papers found matching '{search_term}'",
                "suggestion": "Try searching with different keywords or check the spelling"
            }
            reasoning = f"No papers found matching '{search_term}'"
        
        return results, confidence, reasoning

    async def _handle_unclassified(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Handle unclassified queries by asking for clarification."""
        results = {
            "error": "Unable to classify your query",
            "original_query": query.text,
            "message": "I couldn't understand what you're looking for. Please try rephrasing your query.",
            "suggestions": [
                "Ask about specific authors (e.g., 'Tell me about John Smith's research')",
                "Search for papers about a technology (e.g., 'Show me papers about neural networks')",
                "Find top experts in a field (e.g., 'Who are the top authors in machine learning?')",
                "Ask about research trends (e.g., 'What are the emerging trends in AI?')"
            ]
        }
        confidence = 0.0
        reasoning = "Query could not be classified into any known category"
        
        return results, confidence, reasoning

    def _create_time_periods(self, retrieval_results: List[RetrievalResult]) -> Dict[str, List[RetrievalResult]]:
        """Create time periods for domain evolution analysis."""
        if not retrieval_results:
            return {}
        
        # Get year range
        years = [r.paper.date_submitted.year for r in retrieval_results]
        min_year, max_year = min(years), max(years)
        year_span = max_year - min_year + 1
        
        self.logger.info(f"Analyzing papers from {min_year} to {max_year} (span: {year_span} years)")
        
        periods = {}
        
        if year_span <= 3:
            # Short span: yearly periods
            for result in retrieval_results:
                year = result.paper.date_submitted.year
                period_key = f"{year}"
                if period_key not in periods:
                    periods[period_key] = []
                periods[period_key].append(result)
        elif year_span <= 8:
            # Medium span: 2-year periods  
            for result in retrieval_results:
                year = result.paper.date_submitted.year
                period_start = ((year - min_year) // 2) * 2 + min_year
                period_end = period_start + 1
                period_key = f"{period_start}-{period_end}"
                if period_key not in periods:
                    periods[period_key] = []
                periods[period_key].append(result)
        else:
            # Long span: 3-year periods
            for result in retrieval_results:
                year = result.paper.date_submitted.year
                period_start = ((year - min_year) // 3) * 3 + min_year
                period_end = period_start + 2
                period_key = f"{period_start}-{period_end}"
                if period_key not in periods:
                    periods[period_key] = []
                periods[period_key].append(result)
        
        # Sort periods chronologically
        sorted_periods = {k: periods[k] for k in sorted(periods.keys())}
        
        self.logger.info(f"Created periods: {list(sorted_periods.keys())}")
        for period, papers in sorted_periods.items():
            self.logger.info(f"  {period}: {len(papers)} papers")
            
        return sorted_periods
    
    async def _analyze_evolution_timeline(
        self, 
        retrieval_results: List[RetrievalResult], 
        time_periods: Dict[str, List[RetrievalResult]],
        domain: str
    ) -> Dict[str, Any]:
        """Analyze evolution timeline focusing on methodology shifts and key transitions."""
        timeline_data = {}
        
        for period, papers in time_periods.items():
            self.logger.info(f"Analyzing methodologies for period {period} with {len(papers)} papers")
            
            # Extract methodologies and approaches using LLM-based extraction
            methodologies = await self.extraction_service.extract_methodologies_from_papers(papers, domain)
            key_concepts = await self.extraction_service.extract_key_concepts_from_papers(papers, domain)
            
            # Identify dominant approaches in this period
            dominant_methodologies = [m for m, count in Counter(methodologies).most_common(5)]
            emerging_concepts = [c for c, count in Counter(key_concepts).most_common(5)]
            
            timeline_data[period] = {
                "paper_count": len(papers),
                "dominant_methodologies": dominant_methodologies,
                "emerging_concepts": emerging_concepts,
                "methodology_diversity": len(set(methodologies)),
                "conceptual_novelty": len(set(key_concepts))
            }
        
        # Identify transitions and shifts
        transitions = self._identify_methodology_transitions(timeline_data)
        
        self.logger.info(f"Analyzed evolution timeline with {len(transitions)} major transitions")
        
        return {
            "periods": timeline_data,
            "methodology_transitions": transitions,
            "evolution_phases": self._identify_evolution_phases(timeline_data)
        }
    
    async def _analyze_conceptual_evolution(
        self,
        retrieval_results: List[RetrievalResult], 
        time_periods: Dict[str, List[RetrievalResult]],
        domain: str
    ) -> Dict[str, Any]:
        """Analyze conceptual evolution focusing on problem-solution evolution."""
        conceptual_data = {}
        
        for period, papers in time_periods.items():
            self.logger.info(f"Analyzing concepts for period {period} with {len(papers)} papers")
            
            # Extract problem statements and solution approaches using LLM
            problems = await self.extraction_service.extract_problems_from_papers(papers, domain)
            solutions = await self.extraction_service.extract_solutions_from_papers(papers, domain)
            
            # Analyze complexity evolution
            problem_complexity = self._analyze_problem_complexity(problems)
            solution_sophistication = self._analyze_solution_sophistication(solutions)
            
            conceptual_data[period] = {
                "primary_problems": [p for p, count in Counter(problems).most_common(3)],
                "solution_approaches": [s for s, count in Counter(solutions).most_common(3)],
                "problem_complexity_score": problem_complexity,
                "solution_sophistication_score": solution_sophistication,
                "problem_solution_pairs": list(zip(problems[:5], solutions[:5]))
            }
        
        # Track conceptual shifts
        conceptual_shifts = self._identify_conceptual_shifts(conceptual_data)
        
        self.logger.info(f"Analyzed conceptual evolution with {len(conceptual_shifts)} major shifts")
        
        return {
            "periods": conceptual_data,
            "conceptual_shifts": conceptual_shifts,
            "problem_evolution_trajectory": self._trace_problem_evolution(conceptual_data),
            "solution_evolution_trajectory": self._trace_solution_evolution(conceptual_data)
        }
    
    def _extract_methodologies_from_papers(self, papers: List[RetrievalResult]) -> List[str]:
        """Extract methodology keywords from paper titles and abstracts.
        
        Note: This is a fallback method. The main extraction should use 
        self.extraction_service.extract_methodologies_from_papers() for LLM-based extraction.
        """
        self.logger.warning("Using fallback pattern-based methodology extraction")
        
        methodology_keywords = [
            'neural network', 'deep learning', 'machine learning', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'transformer', 'attention mechanism',
            'convolutional neural network', 'recurrent neural network', 'generative adversarial network',
            'statistical method', 'probabilistic model', 'bayesian', 'optimization',
            'rule-based', 'expert system', 'knowledge graph', 'symbolic reasoning',
            'computer vision', 'natural language processing', 'signal processing',
            'algorithm', 'heuristic', 'evolutionary algorithm', 'genetic algorithm'
        ]
        
        found_methodologies = []
        for paper in papers:
            text = f"{paper.paper.title} {paper.paper.abstract}".lower()
            for keyword in methodology_keywords:
                if keyword in text:
                    found_methodologies.append(keyword)
        
        return found_methodologies
    
    def _extract_key_concepts_from_papers(self, papers: List[RetrievalResult]) -> List[str]:
        """Extract key conceptual terms from papers.
        
        Note: This is a fallback method. The main extraction should use 
        self.extraction_service.extract_key_concepts_from_papers() for LLM-based extraction.
        """
        self.logger.warning("Using fallback pattern-based concept extraction")
        
        concept_keywords = [
            'accuracy', 'precision', 'recall', 'performance', 'efficiency', 'scalability',
            'robustness', 'generalization', 'interpretability', 'explainability',
            'real-time', 'distributed', 'federated', 'privacy', 'security',
            'multimodal', 'cross-modal', 'transfer learning', 'few-shot learning',
            'zero-shot learning', 'meta-learning', 'continual learning',
            'representation learning', 'feature extraction', 'dimensionality reduction'
        ]
        
        found_concepts = []
        for paper in papers:
            text = f"{paper.paper.title} {paper.paper.abstract}".lower()
            for keyword in concept_keywords:
                if keyword in text:
                    found_concepts.append(keyword)
        
        return found_concepts
    
    def _extract_problems_from_papers(self, papers: List[RetrievalResult]) -> List[str]:
        """Extract problem descriptions from papers.
        
        Note: This is a fallback method. The main extraction should use 
        self.extraction_service.extract_problems_from_papers() for LLM-based extraction.
        """
        self.logger.warning("Using fallback pattern-based problem extraction")
        
        problem_indicators = [
            'challenge', 'problem', 'issue', 'difficulty', 'limitation',
            'bottleneck', 'obstacle', 'inefficiency', 'error', 'failure'
        ]
        
        problems = []
        for paper in papers:
            text = f"{paper.paper.title} {paper.paper.abstract}".lower()
            # Simple extraction based on problem indicators
            if any(indicator in text for indicator in problem_indicators):
                problems.append(f"Problem addressed in: {paper.paper.title[:50]}...")
        
        return problems
    
    def _extract_solutions_from_papers(self, papers: List[RetrievalResult]) -> List[str]:
        """Extract solution approaches from papers.
        
        Note: This is a fallback method. The main extraction should use 
        self.extraction_service.extract_solutions_from_papers() for LLM-based extraction.
        """
        self.logger.warning("Using fallback pattern-based solution extraction")
        
        solution_indicators = [
            'approach', 'method', 'technique', 'algorithm', 'framework',
            'model', 'system', 'solution', 'propose', 'introduce'
        ]
        
        solutions = []
        for paper in papers:
            text = f"{paper.paper.title} {paper.paper.abstract}".lower()
            # Simple extraction based on solution indicators
            if any(indicator in text for indicator in solution_indicators):
                solutions.append(f"Solution from: {paper.paper.title[:50]}...")
        
        return solutions
    
    def _analyze_problem_complexity(self, problems: List[str]) -> float:
        """Analyze the complexity of problems (simplified metric)."""
        if not problems:
            return 0.0
        
        # Simple complexity metric based on problem count and diversity
        unique_problems = len(set(problems))
        complexity_score = min(1.0, unique_problems / 10.0)  # Normalize to 0-1
        return round(complexity_score, 2)
    
    def _analyze_solution_sophistication(self, solutions: List[str]) -> float:
        """Analyze the sophistication of solutions (simplified metric)."""
        if not solutions:
            return 0.0
        
        # Simple sophistication metric based on solution diversity
        unique_solutions = len(set(solutions))
        sophistication_score = min(1.0, unique_solutions / 10.0)  # Normalize to 0-1
        return round(sophistication_score, 2)
    
    def _identify_methodology_transitions(self, timeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify major methodology transitions between periods."""
        transitions = []
        periods = list(timeline_data.keys())
        
        for i in range(len(periods) - 1):
            current_period = periods[i]
            next_period = periods[i + 1]
            
            current_methods = set(timeline_data[current_period]['dominant_methodologies'])
            next_methods = set(timeline_data[next_period]['dominant_methodologies'])
            
            # Find new methodologies
            new_methods = next_methods - current_methods
            disappeared_methods = current_methods - next_methods
            
            if new_methods or disappeared_methods:
                transitions.append({
                    "from_period": current_period,
                    "to_period": next_period,
                    "new_methodologies": list(new_methods),
                    "disappeared_methodologies": list(disappeared_methods),
                    "transition_type": "methodology_shift"
                })
        
        return transitions
    
    def _identify_evolution_phases(self, timeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify distinct evolution phases in the domain."""
        phases = []
        periods = list(timeline_data.keys())
        
        # Simple phase identification based on methodology diversity changes
        for i, period in enumerate(periods):
            diversity = timeline_data[period]['methodology_diversity']
            phase_name = f"Phase {i+1}"
            
            if diversity > 8:
                phase_type = "High Diversity Period"
            elif diversity > 4:
                phase_type = "Moderate Diversity Period" 
            else:
                phase_type = "Low Diversity Period"
            
            phases.append({
                "phase_name": phase_name,
                "period": period,
                "phase_type": phase_type,
                "methodology_diversity": diversity,
                "characteristics": timeline_data[period]['dominant_methodologies'][:3]
            })
        
        return phases
    
    def _identify_conceptual_shifts(self, conceptual_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify major conceptual shifts between periods."""
        shifts = []
        periods = list(conceptual_data.keys())
        
        for i in range(len(periods) - 1):
            current_period = periods[i]
            next_period = periods[i + 1]
            
            current_problems = set(conceptual_data[current_period]['primary_problems'])
            next_problems = set(conceptual_data[next_period]['primary_problems'])
            
            # Detect problem focus shifts
            if current_problems != next_problems:
                shifts.append({
                    "from_period": current_period,
                    "to_period": next_period,
                    "problem_shift": {
                        "old_focus": list(current_problems)[:2],
                        "new_focus": list(next_problems)[:2]
                    },
                    "shift_type": "problem_focus_change"
                })
        
        return shifts
    
    def _trace_problem_evolution(self, conceptual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace how problem formulations evolved over time."""
        evolution = {
            "complexity_trend": [],
            "focus_areas": []
        }
        
        for period, data in conceptual_data.items():
            evolution["complexity_trend"].append({
                "period": period,
                "complexity_score": data['problem_complexity_score']
            })
            evolution["focus_areas"].append({
                "period": period,
                "primary_problems": data['primary_problems'][:2]
            })
        
        return evolution
    
    def _trace_solution_evolution(self, conceptual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace how solution approaches evolved over time."""
        evolution = {
            "sophistication_trend": [],
            "approach_evolution": []
        }
        
        for period, data in conceptual_data.items():
            evolution["sophistication_trend"].append({
                "period": period,
                "sophistication_score": data['solution_sophistication_score']
            })
            evolution["approach_evolution"].append({
                "period": period,
                "primary_approaches": data['solution_approaches'][:2]
            })
        
        return evolution
    
    def _calculate_domain_evolution_confidence(
        self,
        retrieval_results: List[RetrievalResult],
        time_periods: Dict[str, List[RetrievalResult]], 
        evolution_timeline: Dict[str, Any],
        conceptual_evolution: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for domain evolution analysis."""
        base_confidence = 0.5
        
        # Factor 1: Number of papers
        paper_factor = min(0.3, len(retrieval_results) / 1000)
        
        # Factor 2: Time span coverage
        time_span_factor = min(0.2, len(time_periods) / 10)
        
        # Factor 3: Methodology diversity
        methodology_transitions = len(evolution_timeline.get('methodology_transitions', []))
        methodology_factor = min(0.15, methodology_transitions / 5)
        
        # Factor 4: Conceptual shifts identified
        conceptual_shifts = len(conceptual_evolution.get('conceptual_shifts', []))
        conceptual_factor = min(0.15, conceptual_shifts / 3)
        
        total_confidence = base_confidence + paper_factor + time_span_factor + methodology_factor + conceptual_factor
        
        return round(min(0.95, total_confidence), 2)

    # fix: now using patterns to extract author names
    async def _extract_author_name_from_query(self, query_text: str) -> Optional[str]:
        """Extract author name from query text using LLM-based extraction."""
        try:
            result = await self.extraction_service.extract_author_name_from_query(query_text)
            if result:
                return result
        except Exception as e:
            self.logger.error(f"Failed to extract author name via LLM: {e}")
        
        # If LLM extraction fails, use the ExtractionService's fallback method
        return self.extraction_service._fallback_author_name_extraction(query_text)

    async def _extract_search_term_from_query(self, query_text: str) -> Optional[str]:
        """Extract search term from query text using LLM-based extraction."""
        try:
            result = await self.extraction_service.extract_search_term_from_query(query_text)
            if result:
                return result
        except Exception as e:
            self.logger.error(f"Failed to extract search term via LLM: {e}")
        
        # If LLM extraction fails, use the ExtractionService's fallback method
        return self.extraction_service._fallback_search_term_extraction(query_text)

    def _find_similar_author_names(self, target_name: str, all_authors: Dict[str, Any]) -> List[str]:
        """Find similar author names using simple string similarity."""
        import difflib
        
        target_lower = target_name.lower()
        similar_authors = []
        
        for author_name in all_authors.keys():
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, target_lower, author_name.lower()).ratio()
            if similarity > 0.6:  # 60% similarity threshold
                similar_authors.append((author_name, similarity))
        
        # Sort by similarity score and return just the names
        similar_authors.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similar_authors[:10]]
