"""
Synthesis agent for TechAuthor system.
"""

import openai
from typing import Dict, Any, List, Union
from dataclasses import dataclass

from ..core.models import (
    Query, RetrievalResult, AnalysisResult, QueryType,
    AuthorExpertiseResult, TechnologyTrendResult, CollaborationResult, CollaboratorInfo
)
from ..core.config import config_manager
from .base_agent import BaseAgent
from ..prompts.synthesis_prompts import SynthesisPrompts


@dataclass
class SynthesisResult:
    """Result of synthesis processing."""
    result: Union[AuthorExpertiseResult, TechnologyTrendResult, CollaborationResult, Dict[str, Any]]
    confidence: float
    reasoning: str


class SynthesisAgent(BaseAgent):
    """Agent responsible for synthesizing analysis results into coherent responses."""
    
    def __init__(self, llm_manager=None):
        """Initialize synthesis agent."""
        super().__init__("Synthesis")
        self.llm_manager = llm_manager  # Store for future CoT enhancements
        
        # Synthesis templates for different query types
        self.synthesis_templates = {
            QueryType.AUTHOR_EXPERTISE: self._synthesize_author_expertise,
            QueryType.TECHNOLOGY_TRENDS: self._synthesize_technology_trends,
            QueryType.AUTHOR_COLLABORATION: self._synthesize_author_collaboration,
            QueryType.DOMAIN_EVOLUTION: self._synthesize_domain_evolution,
            QueryType.CROSS_DOMAIN_ANALYSIS: self._synthesize_cross_domain,
            QueryType.PAPER_IMPACT: self._synthesize_paper_impact,
            QueryType.AUTHOR_PRODUCTIVITY: self._synthesize_author_productivity,
            QueryType.AUTHOR_STATS: self._synthesize_author_stats,
            QueryType.PAPER_SEARCH: self._synthesize_paper_search,
            QueryType.UNCLASSIFIED: self._synthesize_unclassified
        }
    
    async def _initialize_impl(self) -> None:
        """Initialize the OpenAI client."""
        api_key = config_manager.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = api_key
        self.model = config_manager.openai_model
        
        # Load agent configuration
        agent_config = self.config.agents.synthesis_agent
        self.temperature = agent_config.get('temperature', 0.3)
        self.max_tokens = agent_config.get('max_tokens', 1500)
    
    async def synthesize(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize analysis results into a coherent response.
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            analysis_result: Analysis results
            
        Returns:
            Synthesis result
        """
        return await self.process(query, retrieval_results, analysis_result)
    
    async def _process_impl(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Implementation of synthesis processing.
        
        Args:
            query: Original query
            retrieval_results: Retrieved papers
            analysis_result: Analysis results
            
        Returns:
            Synthesis result
        """
        if analysis_result.results.get("error"):
            return SynthesisResult(
                result=analysis_result.results,
                confidence=0.0,
                reasoning="Analysis failed, cannot synthesize results"
            )
        
        # Determine synthesis method based on query type
        synthesis_method = self.synthesis_templates.get(
            query.query_type,
            self._synthesize_general
        )
        
        try:
            # Perform synthesis
            synthesized_result = await synthesis_method(
                query, retrieval_results, analysis_result
            )
            
            # Enhance with LLM-generated insights if available
            enhanced_result = await self._enhance_with_llm(
                query, synthesized_result, analysis_result
            )
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return SynthesisResult(
                result={"error": f"Synthesis failed: {e}"},
                confidence=0.0,
                reasoning=f"Synthesis processing failed: {e}"
            )
    
    async def _synthesize_author_expertise(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize author expertise analysis."""
        analysis_data = analysis_result.results
        
        # Create structured result
        result = AuthorExpertiseResult(
            domain=analysis_data.get("domain", "Unknown"),
            top_authors=analysis_data.get("top_authors", []),
            total_papers_analyzed=analysis_data.get("total_papers_analyzed", 0),
            time_range=self._extract_time_range(retrieval_results),
            methodology=analysis_data.get("methodology", "Frequency-based analysis"),
            confidence=analysis_result.confidence
        )
        
        # Generate insights
        insights = self._generate_author_expertise_insights(result)
        
        # Add insights to result
        result_dict = result.dict()
        result_dict["insights"] = insights
        result_dict["summary"] = self._create_author_expertise_summary(result)
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning=f"Synthesized author expertise analysis for {result.domain}"
        )
    
    async def _synthesize_technology_trends(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize technology trends analysis."""
        analysis_data = analysis_result.results
        
        # Create structured result
        result = TechnologyTrendResult(
            domain=analysis_data.get("domain", "Technology"),
            trends=analysis_data.get("trends", []),
            time_series_data=analysis_data.get("time_series_data"),
            emerging_technologies=analysis_data.get("emerging_technologies", []),
            declining_technologies=analysis_data.get("declining_technologies", []),
            confidence=analysis_result.confidence
        )
        
        # Generate insights
        insights = self._generate_trend_insights(result)
        
        # Add insights and other data as additional attributes
        # Note: We keep result as TechnologyTrendResult object, not convert to dict
        result.insights = insights
        result.summary = self._create_trend_summary(result)
        result.recommendations = self._generate_trend_recommendations(result)
        
        # Add LLM-generated contextual insights
        llm_insights = await self._generate_llm_insights_about_trends(query, result, analysis_data)
        if llm_insights:
            result.summary = f"{result.summary}\n\nLLM Insights: {llm_insights}"
        
        return SynthesisResult(
            result=result,
            confidence=analysis_result.confidence,
            reasoning="Synthesized technology trend analysis with insights and recommendations"
        )
    
    async def _generate_llm_insights_about_trends(
        self,
        query: Query,
        result: TechnologyTrendResult,
        analysis_data: Dict[str, Any]
    ) -> str:
        """Generate LLM insights about technology trends based on the query and results.
        
        Args:
            query: Original user query
            result: Technology trend result
            analysis_data: Raw analysis data
            
        Returns:
            String containing LLM-generated insights about the trends
        """
        try:
            # Prepare context for LLM
            trends_summary = []
            for trend in result.trends[:5]:  # Top 5 trends
                direction = trend.get('trend_direction', 'stable')
                technology = trend.get('technology', 'Unknown')
                papers = trend.get('total_papers', 0)
                slope = trend.get('trend_slope', 0)
                trends_summary.append(f"{technology}: {direction} trend ({papers} papers, slope: {slope:.3f})")
            
            trends_text = "; ".join(trends_summary)
            total_papers = analysis_data.get("total_papers", 0)
            
            # Create system and user prompts
            system_prompt = SynthesisPrompts.TECHNOLOGY_TRENDS_SYSTEM
            
            user_prompt = f"""Based on the following technology trends analysis results for the query "{query.text}":

Total Papers Analyzed: {total_papers}
Top Trends: {trends_text}
Domain: {result.domain}
Time Range: {query.parameters.get('start_year', 'recent')} to {query.parameters.get('end_year', 'present')}

Please provide your insights about these trends and their contribution to the field. Consider:
1. What do these trends tell us about the current state of the field?
2. How do these trends relate to the broader technological landscape?
3. What implications might these trends have for future research and development?

Keep your response concise (2-3 sentences) and focus on the most significant insights."""

            # Generate LLM response using the correct method signature
            response = await self.llm_manager.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_name="synthesis",
                max_tokens=200
            )
            
            return response.strip() if response else ""
            
        except Exception as e:
            self.logger.warning(f"Failed to generate LLM insights: {e}")
            return ""
    
    async def _synthesize_author_collaboration(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize author collaboration analysis."""
        analysis_data = analysis_result.results
        
        # Check if this is a specific author with database profile
        if "author_profile" in analysis_data:
            # New enhanced author collaboration with database info
            result_dict = self._synthesize_enhanced_author_collaboration(analysis_data)
        elif "focal_author" in analysis_data:
            # Specific author collaboration from papers
            result_dict = self._synthesize_paper_based_collaboration(analysis_data)
        else:
            # General collaboration analysis
            result_dict = {
                "collaboration_overview": analysis_data,
                "insights": self._generate_general_collaboration_insights(analysis_data),
                "summary": "General collaboration network analysis across all authors"
            }
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning="Synthesized collaboration analysis with network insights"
        )
    
    def _synthesize_enhanced_author_collaboration(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize enhanced author collaboration with database profile."""
        focal_author = analysis_data["focal_author"]
        profile = analysis_data["author_profile"]
        collaborators = analysis_data["top_collaborators"]
        
        # Create comprehensive summary
        summary = (
            f"{focal_author} is an active researcher with {profile['total_papers']} papers and "
            f"{profile['total_collaborators']} collaborators spanning {profile['years_active']}. "
            f"Their research focuses on {', '.join(profile['primary_subjects'][:3])} with "
            f"{profile['collaboration_intensity'].lower()} collaboration intensity."
        )
        
        # Generate detailed insights
        insights = []
        
        # Collaboration intensity insight
        if profile['collaboration_intensity'] == 'High':
            insights.append(f"Highly collaborative researcher with {profile['total_collaborators']} unique collaborators")
        elif profile['collaboration_intensity'] == 'Medium':
            insights.append(f"Moderately collaborative with a solid network of {profile['total_collaborators']} collaborators")
        else:
            insights.append(f"Focused collaboration approach with {profile['total_collaborators']} key collaborators")
        
        # Research breadth insight
        breadth = analysis_data["collaboration_network"]["research_breadth"]
        if breadth > 10:
            insights.append(f"Interdisciplinary researcher active in {breadth} research areas")
        elif breadth > 5:
            insights.append(f"Multi-domain expertise spanning {breadth} research areas")
        else:
            insights.append(f"Specialized research focus in {breadth} main areas")
        
        # Top collaborators insight
        if len(collaborators) >= 10:
            insights.append(f"Extensive collaboration network with top partner: {collaborators[0]['collaborator']}")
        elif len(collaborators) >= 5:
            insights.append(f"Strong collaborative relationships with {len(collaborators)} key partners")
        
        return {
            "focal_author": focal_author,
            "author_profile": profile,
            "top_collaborators": collaborators[:10],
            "total_collaborators": analysis_data.get("total_collaborators", profile['total_collaborators']),  # Add top-level field
            "network_size": analysis_data.get("network_size", profile['total_collaborators'] + 1),  # Add top-level field
            "collaboration_network": analysis_data["collaboration_network"],
            "summary": summary,
            "insights": insights
        }
    
    def _synthesize_paper_based_collaboration(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize collaboration analysis based on retrieved papers."""
        # Convert dictionary collaborators to CollaboratorInfo objects
        collaborator_objects = []
        for collab_dict in analysis_data.get("top_collaborators", []):
            collaborator_objects.append(CollaboratorInfo(**collab_dict))
        
        result = CollaborationResult(
            focal_author=analysis_data["focal_author"],
            collaborators=collaborator_objects,
            total_collaborators=analysis_data.get("total_collaborators", len(collaborator_objects)),
            network_size=analysis_data.get("network_size", len(collaborator_objects) + 1),
            collaboration_network={
                "size": analysis_data.get("network_size", 0),
                "density": analysis_data.get("network_density", 0),
                "centrality": analysis_data.get("focal_author_centrality", 0)
            },
            collaboration_patterns=self._extract_collaboration_patterns(analysis_data),
            confidence=analysis_data.get("confidence", 0.5)
        )
        
        result_dict = result.dict()
        result_dict["insights"] = self._generate_collaboration_insights(result)
        result_dict["summary"] = self._create_collaboration_summary(result)
        
        return result_dict
    
    async def _synthesize_domain_evolution(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize domain evolution analysis with focus on paradigm shifts and conceptual changes."""
        analysis_data = analysis_result.results
        
        if analysis_data.get('analysis_type') != 'domain_evolution':
            # Fallback to technology trends if not proper domain evolution analysis
            self.logger.warning("Analysis data is not domain evolution type, falling back to technology trends synthesis")
            return await self._synthesize_technology_trends(query, retrieval_results, analysis_result)
        
        # Use LLM for intelligent synthesis of domain evolution insights
        domain_evolution_prompt = SynthesisPrompts.DOMAIN_EVOLUTION.format(
            domain=analysis_data.get('domain', 'Unknown Domain'),
            periods=analysis_data.get('time_periods', 0),
            total_papers=analysis_data.get('total_papers', 0),
            evolution_timeline=self._format_timeline_data(analysis_data.get('evolution_timeline', {})),
            conceptual_evolution=self._format_conceptual_data(analysis_data.get('conceptual_evolution', {})),
            query_text=query.text
        )
        
        try:
            if self.llm_manager:
                llm_insights = await self.llm_manager.generate(
                    system_prompt=SynthesisPrompts.DOMAIN_EVOLUTION_SYSTEM,
                    user_prompt=domain_evolution_prompt,
                    agent_name="synthesis"
                )
            else:
                llm_insights = "LLM insights unavailable - manager not initialized"
                
        except Exception as e:
            self.logger.warning(f"LLM synthesis failed: {e}")
            llm_insights = f"LLM synthesis error: {str(e)}"
        
        # Create comprehensive domain evolution synthesis
        result_dict = {
            "domain_evolution_analysis": analysis_data,
            "evolution_summary": self._create_domain_evolution_summary(analysis_data),
            "key_insights": self._extract_domain_evolution_insights(analysis_data),
            "paradigm_shifts": self._identify_paradigm_shifts(analysis_data),
            "future_trajectory": self._predict_future_trajectory(analysis_data),
            "llm_insights": llm_insights
        }
        
        # Calculate synthesis confidence
        synthesis_confidence = min(0.95, analysis_result.confidence * 1.05)  # Slight boost for comprehensive synthesis
        
        reasoning = (
            f"Synthesized domain evolution analysis for {analysis_data.get('domain', 'Unknown')} "
            f"covering {analysis_data.get('time_periods', 0)} time periods. "
            f"Focused on methodology transitions, conceptual shifts, and paradigm changes "
            f"rather than simple trend counting. Generated insights about evolution patterns, "
            f"key transition points, and future trajectory predictions."
        )
        
        return SynthesisResult(
            result=result_dict,
            confidence=synthesis_confidence,
            reasoning=reasoning
        )
    
    async def _synthesize_cross_domain(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize cross-domain analysis."""
        analysis_data = analysis_result.results
        
        result_dict = {
            "cross_domain_analysis": analysis_data,
            "insights": self._generate_cross_domain_insights(analysis_data),
            "summary": self._create_cross_domain_summary(analysis_data),
            "interdisciplinary_opportunities": self._identify_interdisciplinary_opportunities(analysis_data)
        }
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning="Synthesized cross-domain analysis with interdisciplinary insights"
        )
    
    async def _synthesize_paper_impact(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize paper impact analysis."""
        analysis_data = analysis_result.results
        
        result_dict = {
            "impact_analysis": analysis_data,
            "insights": self._generate_impact_insights(analysis_data),
            "summary": self._create_impact_summary(analysis_data),
            "methodology_notes": analysis_data.get("methodology", ""),
            "limitations": analysis_data.get("note", "")
        }
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning="Synthesized paper impact analysis with methodology notes"
        )
    
    async def _synthesize_author_productivity(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize author productivity analysis."""
        analysis_data = analysis_result.results
        
        result_dict = {
            "productivity_analysis": analysis_data,
            "insights": self._generate_productivity_insights(analysis_data),
            "summary": self._create_productivity_summary(analysis_data),
            "productivity_patterns": self._identify_productivity_patterns(analysis_data)
        }
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning="Synthesized author productivity analysis with pattern identification"
        )
    
    async def _synthesize_general(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize general analysis."""
        analysis_data = analysis_result.results
        
        result_dict = {
            "general_analysis": analysis_data,
            "insights": self._generate_general_insights(analysis_data),
            "summary": self._create_general_summary(analysis_data, query)
        }
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning="General synthesis of analysis results"
        )
    
    async def _enhance_with_llm(
        self,
        query: Query,
        synthesis_result: SynthesisResult,
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Enhance synthesis result with LLM-generated insights."""
        try:
            # Create a concise summary of the analysis for LLM
            context = {
                "query": query.text,
                "query_type": query.query_type.value if query.query_type else "general",
                "key_findings": self._extract_key_findings(synthesis_result.result),
                "confidence": analysis_result.confidence
            }
            
            # Generate enhanced insights
            enhanced_insights = await self._generate_llm_insights(context)
            
            # Add to result
            if isinstance(synthesis_result.result, dict):
                synthesis_result.result["llm_insights"] = enhanced_insights
                synthesis_result.result["enhanced"] = True
            
            return synthesis_result
            
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed: {e}")
            # Return original result if enhancement fails
            return synthesis_result
    
    async def _generate_llm_insights(self, context: Dict[str, Any]) -> str:
        """Generate insights using LLM."""
        prompt = f"""
        Based on the following research analysis, provide additional insights and interpretations:
        
        Query: {context['query']}
        Analysis Type: {context['query_type']}
        Key Findings: {context['key_findings']}
        Confidence: {context['confidence']}
        
        Please provide:
        1. Key insights from the findings
        2. Potential implications
        3. Suggestions for further research
        4. Any limitations or caveats
        
        Keep the response concise and focused on actionable insights.
        """
        
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SynthesisPrompts.GENERAL_ANALYSIS_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"LLM insight generation failed: {e}")
            return "Additional insights unavailable due to processing limitations."
    
    def _extract_key_findings(self, result: Dict[str, Any]) -> str:
        """Extract key findings from synthesis result for LLM context."""
        key_points = []
        
        if "top_authors" in result:
            top_authors = result["top_authors"][:3]
            authors_str = ", ".join([author["author"] for author in top_authors])
            key_points.append(f"Top authors: {authors_str}")
        
        if "emerging_technologies" in result:
            emerging = result["emerging_technologies"][:3]
            key_points.append(f"Emerging technologies: {', '.join(emerging)}")
        
        if "total_papers_analyzed" in result:
            key_points.append(f"Papers analyzed: {result['total_papers_analyzed']}")
        
        if "domain" in result:
            key_points.append(f"Domain: {result['domain']}")
        
        return "; ".join(key_points) if key_points else "General analysis results"
    
    def _extract_time_range(self, retrieval_results: List[RetrievalResult]) -> Dict[str, str]:
        """Extract time range from retrieval results."""
        if not retrieval_results:
            return {}
        
        years = [r.paper.date_submitted.year for r in retrieval_results]
        return {
            "start": str(min(years)),
            "end": str(max(years))
        }
    
    def _generate_author_expertise_insights(self, result: AuthorExpertiseResult) -> List[str]:
        """Generate insights for author expertise results."""
        insights = []
        
        if result.top_authors:
            top_author = result.top_authors[0]
            insights.append(f"Leading expert: {top_author['author']} with {top_author['paper_count']} papers")
            
            # Analyze expertise distribution
            paper_counts = [author['paper_count'] for author in result.top_authors]
            if len(paper_counts) > 1:
                concentration = paper_counts[0] / sum(paper_counts)
                if concentration > 0.3:
                    insights.append("Research in this domain is highly concentrated among few experts")
                else:
                    insights.append("Research expertise is well-distributed across multiple researchers")
        
        return insights
    
    def _create_author_expertise_summary(self, result: AuthorExpertiseResult) -> str:
        """Create summary for author expertise results."""
        if not result.top_authors:
            return f"No experts found in {result.domain}"
        
        total_authors = len(result.top_authors)
        top_author = result.top_authors[0]
        
        return (
            f"Analysis of {result.domain} expertise reveals {total_authors} leading researchers. "
            f"{top_author['author']} emerges as the top expert with {top_author['paper_count']} publications, "
            f"based on {result.total_papers_analyzed} papers analyzed."
        )
    
    def _generate_trend_insights(self, result: TechnologyTrendResult) -> List[str]:
        """Generate insights for technology trends."""
        insights = []
        
        if result.emerging_technologies:
            insights.append(f"Key emerging areas: {', '.join(result.emerging_technologies[:3])}")
        
        if result.declining_technologies:
            insights.append(f"Declining interest in: {', '.join(result.declining_technologies[:2])}")
        
        if result.trends:
            strong_trends = [t for t in result.trends if abs(t.get("trend_slope", 0)) > 0.5]
            if strong_trends:
                insights.append(f"{len(strong_trends)} technologies show strong trend patterns")
        
        return insights
    
    def _create_trend_summary(self, result: TechnologyTrendResult) -> str:
        """Create summary for technology trends."""
        summary = f"Technology trend analysis reveals {len(result.trends)} tracked technologies. "
        
        if result.emerging_technologies:
            summary += f"Emerging areas include {', '.join(result.emerging_technologies[:3])}. "
        
        if result.declining_technologies:
            summary += f"Declining interest observed in {', '.join(result.declining_technologies[:2])}."
        
        return summary
    
    def _generate_trend_recommendations(self, result: TechnologyTrendResult) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []
        
        if result.emerging_technologies:
            recommendations.append(f"Consider focusing research on emerging areas: {', '.join(result.emerging_technologies[:3])}")
        
        if result.declining_technologies:
            recommendations.append("Explore opportunities to revitalize or pivot research in declining areas")
        
        recommendations.append("Monitor trend changes for strategic research planning")
        
        return recommendations
    
    def _extract_collaboration_patterns(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract collaboration patterns from analysis data."""
        patterns = {}
        
        if "top_collaborators" in analysis_data:
            collaborators = analysis_data["top_collaborators"]
            if collaborators:
                patterns["most_frequent_collaborator"] = collaborators[0]["collaborator"]
                patterns["collaboration_intensity"] = collaborators[0]["collaboration_count"]
        
        if "network_density" in analysis_data:
            density = analysis_data["network_density"]
            if density > 0.5:
                patterns["network_type"] = "highly_connected"
            elif density > 0.2:
                patterns["network_type"] = "moderately_connected"
            else:
                patterns["network_type"] = "sparse"
        
        return patterns
    
    def _generate_collaboration_insights(self, result: CollaborationResult) -> List[str]:
        """Generate insights for collaboration analysis."""
        insights = []
        
        if result.collaborators:
            insights.append(f"{result.focal_author} has {len(result.collaborators)} active collaborators")
            
            top_collab = result.collaborators[0]
            insights.append(f"Strongest collaboration with {top_collab.collaborator} ({top_collab.collaboration_count} papers)")
        
        network_density = result.collaboration_network.get("density", 0)
        if network_density > 0.3:
            insights.append("High collaboration density suggests strong research community")
        
        return insights
    
    def _create_collaboration_summary(self, result: CollaborationResult) -> str:
        """Create summary for collaboration analysis."""
        return (
            f"Collaboration analysis for {result.focal_author} reveals "
            f"{len(result.collaborators)} active collaborators with network density of "
            f"{result.collaboration_network.get('density', 0):.3f}."
        )
    
    def _generate_general_collaboration_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate insights for general collaboration analysis."""
        insights = []
        
        total_authors = analysis_data.get("total_authors", 0)
        total_collaborations = analysis_data.get("total_collaborations", 0)
        
        if total_authors > 0:
            avg_collaborations = total_collaborations / total_authors
            insights.append(f"Average {avg_collaborations:.1f} collaborations per author")
        
        density = analysis_data.get("network_density", 0)
        if density > 0.1:
            insights.append("Research community shows good interconnectedness")
        else:
            insights.append("Research community appears fragmented")
        
        return insights
    
    def _generate_cross_domain_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate insights for cross-domain analysis."""
        insights = []
        
        cross_domain_authors = analysis_data.get("cross_domain_authors", [])
        if cross_domain_authors:
            top_interdisciplinary = cross_domain_authors[0]
            insights.append(
                f"Most interdisciplinary researcher: {top_interdisciplinary['author']} "
                f"({top_interdisciplinary['domain_count']} domains)"
            )
        
        total_authors = analysis_data.get("total_authors_analyzed", 0)
        interdisciplinary_count = len(cross_domain_authors)
        
        if total_authors > 0:
            interdisciplinary_ratio = interdisciplinary_count / total_authors
            if interdisciplinary_ratio > 0.3:
                insights.append("High level of interdisciplinary research activity")
            else:
                insights.append("Most researchers focus on specific domains")
        
        return insights
    
    def _create_cross_domain_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Create summary for cross-domain analysis."""
        interdisciplinary_authors = len(analysis_data.get("cross_domain_authors", []))
        total_authors = analysis_data.get("total_authors_analyzed", 0)
        
        return (
            f"Cross-domain analysis identified {interdisciplinary_authors} interdisciplinary "
            f"researchers out of {total_authors} total authors analyzed."
        )
    
    def _identify_interdisciplinary_opportunities(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify interdisciplinary research opportunities."""
        opportunities = []
        
        cross_domain_authors = analysis_data.get("cross_domain_authors", [])
        if cross_domain_authors:
            # Find common domain combinations
            domain_combinations = {}
            for author_data in cross_domain_authors:
                domains = sorted(author_data.get("domains", []))
                if len(domains) >= 2:
                    combo = tuple(domains[:2])  # Take first two domains
                    domain_combinations[combo] = domain_combinations.get(combo, 0) + 1
            
            # Suggest promising combinations
            for combo, count in sorted(domain_combinations.items(), key=lambda x: x[1], reverse=True)[:3]:
                opportunities.append(f"Promising interdisciplinary area: {' + '.join(combo)} ({count} researchers)")
        
        if not opportunities:
            opportunities.append("Consider exploring connections between established domains")
        
        return opportunities
    
    def _generate_impact_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate insights for paper impact analysis."""
        insights = []
        
        high_impact_papers = analysis_data.get("high_impact_papers", [])
        if high_impact_papers:
            top_paper = high_impact_papers[0]
            insights.append(f"Highest impact paper: '{top_paper['title'][:50]}...' ({top_paper['year']})")
        
        # Analyze impact distribution by year
        if high_impact_papers:
            years = [paper['year'] for paper in high_impact_papers]
            recent_papers = sum(1 for year in years if year >= 2020)
            if recent_papers > len(years) * 0.5:
                insights.append("High-impact research is concentrated in recent years")
            else:
                insights.append("High-impact research spans multiple time periods")
        
        return insights
    
    def _create_impact_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Create summary for impact analysis."""
        total_papers = analysis_data.get("total_papers_analyzed", 0)
        methodology = analysis_data.get("methodology", "relevance-based scoring")
        
        return (
            f"Impact analysis of {total_papers} papers using {methodology}. "
            f"Note: Analysis based on retrieval relevance as proxy for impact."
        )
    
    def _generate_productivity_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate insights for productivity analysis."""
        insights = []
        
        productive_authors = analysis_data.get("productive_authors", [])
        if productive_authors:
            top_author = productive_authors[0]
            insights.append(
                f"Most productive: {top_author['author']} "
                f"({top_author['total_papers']} papers over {top_author['years_active']} years)"
            )
            
            # Analyze productivity patterns
            avg_rates = [author['avg_papers_per_year'] for author in productive_authors[:5]]
            if avg_rates:
                avg_productivity = sum(avg_rates) / len(avg_rates)
                insights.append(f"Top researchers average {avg_productivity:.1f} papers per year")
        
        return insights
    
    def _create_productivity_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Create summary for productivity analysis."""
        total_authors = analysis_data.get("total_authors_analyzed", 0)
        total_papers = analysis_data.get("total_papers_analyzed", 0)
        
        return (
            f"Productivity analysis of {total_authors} authors across {total_papers} papers "
            f"reveals varying publication patterns and career trajectories."
        )
    
    def _identify_productivity_patterns(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify productivity patterns."""
        patterns = []
        
        productive_authors = analysis_data.get("productive_authors", [])
        if productive_authors:
            # Analyze career lengths
            career_lengths = [author['years_active'] for author in productive_authors]
            avg_career = sum(career_lengths) / len(career_lengths)
            patterns.append(f"Average active career span: {avg_career:.1f} years")
            
            # Analyze peak productivity
            peak_years = [author.get('peak_year_papers', 0) for author in productive_authors]
            if peak_years:
                avg_peak = sum(peak_years) / len(peak_years)
                patterns.append(f"Average peak year productivity: {avg_peak:.1f} papers")
        
        return patterns
    
    def _generate_general_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate insights for general analysis."""
        insights = []
        
        total_papers = analysis_data.get("total_papers", 0)
        insights.append(f"Analyzed {total_papers} papers across multiple dimensions")
        
        top_domains = analysis_data.get("top_domains", [])
        if top_domains:
            dominant_domain = top_domains[0]
            insights.append(f"Dominant domain: {dominant_domain[0]} ({dominant_domain[1]} papers)")
        
        year_range = analysis_data.get("year_range", "")
        if year_range:
            insights.append(f"Publications span from {year_range}")
        
        return insights
    
    def _create_general_summary(self, analysis_data: Dict[str, Any], query: Query) -> str:
        """Create summary for general analysis."""
        total_papers = analysis_data.get("total_papers", 0)
        
        return (
            f"General analysis of {total_papers} papers relevant to '{query.text}' "
            f"provides overview across domains, authors, subjects, and time periods."
        )

    async def _synthesize_author_stats(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize author statistics analysis."""
        analysis_data = analysis_result.results
        
        if analysis_data.get("found"):
            # Author found - provide comprehensive stats
            stats = analysis_data.get("stats", {})
            insights = [
                f"Total papers: {stats.get('total_papers', 0)}",
                f"Years active: {', '.join(map(str, stats.get('years_active', [])))}", 
                f"Research areas: {', '.join(stats.get('subjects', [])[:3])}",
                f"Collaborators: {stats.get('num_collaborators', 0)} unique researchers"
            ]
            
            result_dict = {
                "author_profile": analysis_data,
                "insights": insights,
                "summary": f"Detailed profile for {analysis_data.get('author_name')} showing {stats.get('total_papers', 0)} papers across {len(stats.get('years_active', []))} years"
            }
        else:
            # Author not found - provide suggestions
            result_dict = {
                "error_info": analysis_data,
                "insights": [f"No exact match found for '{analysis_data.get('author_name')}'"],
                "summary": f"Author '{analysis_data.get('author_name')}' not found in database",
                "similar_authors": analysis_data.get("similar_authors", [])
            }
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning="Synthesized author statistics lookup"
        )

    async def _synthesize_paper_search(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult], 
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize paper search results."""
        analysis_data = analysis_result.results
        
        papers_found = analysis_data.get("papers_found", 0)
        if papers_found > 0:
            insights = [
                f"Found {papers_found} papers matching '{analysis_data.get('search_term')}'",
                f"Search performed across {analysis_data.get('total_papers_searched', 0)} papers",
                "Results ranked by relevance and match type"
            ]
            
            # Add insights about the found papers
            matching_papers = analysis_data.get("matching_papers", [])
            if matching_papers:
                years = [p.get("year") for p in matching_papers]
                insights.append(f"Papers span from {min(years)} to {max(years)}")
                
                title_matches = sum(1 for p in matching_papers if p.get("match_type") == "title")
                if title_matches > 0:
                    insights.append(f"{title_matches} papers have the search term in their title")
            
            result_dict = {
                "search_results": analysis_data,
                "insights": insights,
                "summary": f"Found {papers_found} papers related to '{analysis_data.get('search_term')}'"
            }
        else:
            insights = [
                f"No papers found matching '{analysis_data.get('search_term')}'",
                analysis_data.get("suggestion", "Try different search terms")
            ]
            
            result_dict = {
                "search_results": analysis_data,
                "insights": insights,
                "summary": f"No papers found for search term '{analysis_data.get('search_term')}'"
            }
        
        return SynthesisResult(
            result=result_dict,
            confidence=analysis_result.confidence,
            reasoning="Synthesized paper search results"
        )

    async def _synthesize_unclassified(
        self,
        query: Query,
        retrieval_results: List[RetrievalResult],
        analysis_result: AnalysisResult
    ) -> SynthesisResult:
        """Synthesize response for unclassified queries."""
        analysis_data = analysis_result.results
        
        result_dict = {
            "error_info": analysis_data,
            "insights": ["Query could not be classified into supported categories"],
            "summary": "Unable to process your query. Please rephrase it.",
            "suggestions": analysis_data.get("suggestions", [])
        }
        
        return SynthesisResult(
            result=result_dict,
            confidence=0.0,
            reasoning="Query unclassified - asking for clarification"
        )

    def _format_timeline_data(self, timeline_data: Dict[str, Any]) -> str:
        """Format evolution timeline data for LLM prompt."""
        if not timeline_data.get('periods'):
            return "No timeline data available."
        
        formatted = "Evolution Timeline Data:\n"
        for period, data in timeline_data.get('periods', {}).items():
            formatted += f"- {period}: {data.get('paper_count', 0)} papers, "
            formatted += f"Methods: {', '.join(data.get('dominant_methodologies', [])[:3])}\n"
        
        transitions = timeline_data.get('methodology_transitions', [])
        if transitions:
            formatted += f"\nMajor Transitions: {len(transitions)} identified\n"
            for trans in transitions[:3]:
                formatted += f"- {trans.get('from_period')} → {trans.get('to_period')}: "
                formatted += f"New methods: {', '.join(trans.get('new_methodologies', [])[:2])}\n"
        
        return formatted
    
    def _format_conceptual_data(self, conceptual_data: Dict[str, Any]) -> str:
        """Format conceptual evolution data for LLM prompt."""
        if not conceptual_data.get('periods'):
            return "No conceptual evolution data available."
        
        formatted = "Conceptual Evolution Data:\n"
        for period, data in conceptual_data.get('periods', {}).items():
            formatted += f"- {period}: Complexity Score: {data.get('problem_complexity_score', 0)}, "
            formatted += f"Sophistication Score: {data.get('solution_sophistication_score', 0)}\n"
        
        shifts = conceptual_data.get('conceptual_shifts', [])
        if shifts:
            formatted += f"\nConceptual Shifts: {len(shifts)} identified\n"
        
        return formatted
    
    def _create_domain_evolution_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Create a summary of domain evolution analysis."""
        domain = analysis_data.get('domain', 'Unknown Domain')
        periods = analysis_data.get('time_periods', 0)
        total_papers = analysis_data.get('total_papers', 0)
        
        evolution_timeline = analysis_data.get('evolution_timeline', {})
        conceptual_evolution = analysis_data.get('conceptual_evolution', {})
        
        transitions = len(evolution_timeline.get('methodology_transitions', []))
        shifts = len(conceptual_evolution.get('conceptual_shifts', []))
        
        summary = (
            f"Domain evolution analysis of {domain} reveals {transitions} methodology transitions "
            f"and {shifts} conceptual shifts across {periods} time periods. "
            f"Analysis based on {total_papers} papers shows clear evolution patterns in both "
            f"research approaches and problem formulations."
        )
        
        return summary
    
    def _extract_domain_evolution_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract key insights from domain evolution analysis."""
        insights = []
        
        evolution_timeline = analysis_data.get('evolution_timeline', {})
        conceptual_evolution = analysis_data.get('conceptual_evolution', {})
        
        # Timeline insights
        transitions = evolution_timeline.get('methodology_transitions', [])
        if transitions:
            insights.append(f"Key methodology transitions identified: {len(transitions)} major shifts in research approaches")
            
            # Identify most significant transition
            if transitions:
                recent_transition = transitions[-1] if transitions else None
                if recent_transition:
                    new_methods = recent_transition.get('new_methodologies', [])
                    if new_methods:
                        insights.append(f"Recent methodological shift: Introduction of {', '.join(new_methods[:2])}")
        
        # Conceptual insights
        shifts = conceptual_evolution.get('conceptual_shifts', [])
        if shifts:
            insights.append(f"Conceptual evolution shows {len(shifts)} major problem focus changes")
        
        # Trajectory insights
        problem_evolution = conceptual_evolution.get('problem_evolution_trajectory', {})
        if problem_evolution:
            complexity_trend = problem_evolution.get('complexity_trend', [])
            if len(complexity_trend) > 1:
                recent_complexity = complexity_trend[-1].get('complexity_score', 0)
                early_complexity = complexity_trend[0].get('complexity_score', 0)
                if recent_complexity > early_complexity:
                    insights.append("Problem complexity has increased over time, indicating more sophisticated challenges")
                elif recent_complexity < early_complexity:
                    insights.append("Problem formulations have become more focused and streamlined")
        
        if not insights:
            insights.append("Limited evolution patterns detected in the analyzed time period")
        
        return insights
    
    def _identify_paradigm_shifts(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify major paradigm shifts from the analysis data."""
        shifts = []
        
        evolution_timeline = analysis_data.get('evolution_timeline', {})
        transitions = evolution_timeline.get('methodology_transitions', [])
        
        for i, transition in enumerate(transitions):
            new_methods = transition.get('new_methodologies', [])
            disappeared_methods = transition.get('disappeared_methodologies', [])
            
            if new_methods and len(new_methods) >= 2:  # Significant methodology change
                shifts.append({
                    "shift_id": f"paradigm_shift_{i+1}",
                    "period": f"{transition.get('from_period')} → {transition.get('to_period')}",
                    "shift_type": "Methodological Paradigm Shift",
                    "description": f"Transition from {', '.join(disappeared_methods[:2])} to {', '.join(new_methods[:2])}",
                    "significance": "High" if len(new_methods) > 2 else "Medium"
                })
        
        # Add conceptual paradigm shifts
        conceptual_evolution = analysis_data.get('conceptual_evolution', {})
        conceptual_shifts = conceptual_evolution.get('conceptual_shifts', [])
        
        for i, shift in enumerate(conceptual_shifts):
            problem_shift = shift.get('problem_shift', {})
            if problem_shift:
                shifts.append({
                    "shift_id": f"conceptual_shift_{i+1}",
                    "period": f"{shift.get('from_period')} → {shift.get('to_period')}",
                    "shift_type": "Conceptual Paradigm Shift",
                    "description": f"Problem focus changed from {', '.join(problem_shift.get('old_focus', [])[:1])} to {', '.join(problem_shift.get('new_focus', [])[:1])}",
                    "significance": "Medium"
                })
        
        return shifts
    
    def _predict_future_trajectory(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future trajectory based on evolution patterns."""
        trajectory = {
            "methodology_predictions": [],
            "conceptual_predictions": [],
            "confidence_level": "Medium"
        }
        
        evolution_timeline = analysis_data.get('evolution_timeline', {})
        conceptual_evolution = analysis_data.get('conceptual_evolution', {})
        
        # Analyze methodology trends
        phases = evolution_timeline.get('evolution_phases', [])
        if len(phases) > 1:
            recent_phase = phases[-1]
            diversity_trend = recent_phase.get('methodology_diversity', 0)
            
            if diversity_trend > 5:
                trajectory["methodology_predictions"].append(
                    "Continued diversification of methodological approaches expected"
                )
            else:
                trajectory["methodology_predictions"].append(
                    "Potential consolidation around dominant methodologies"
                )
        
        # Analyze conceptual trends
        problem_trajectory = conceptual_evolution.get('problem_evolution_trajectory', {})
        solution_trajectory = conceptual_evolution.get('solution_evolution_trajectory', {})
        
        if problem_trajectory:
            complexity_trend = problem_trajectory.get('complexity_trend', [])
            if len(complexity_trend) > 1:
                recent_complexity = complexity_trend[-1].get('complexity_score', 0)
                if recent_complexity > 0.5:
                    trajectory["conceptual_predictions"].append(
                        "Increasing problem complexity suggests focus on more challenging, multidisciplinary issues"
                    )
        
        if not trajectory["methodology_predictions"] and not trajectory["conceptual_predictions"]:
            trajectory["conceptual_predictions"].append(
                "Limited historical data available for reliable trajectory prediction"
            )
            trajectory["confidence_level"] = "Low"
        
        return trajectory



