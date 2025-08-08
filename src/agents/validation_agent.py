"""
Validation agent for TechAuthor system.
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from ..core.models import Query, QueryType
from ..agents.synthesis_agent import SynthesisResult
from .base_agent import BaseAgent


@dataclass
class ValidationResult:
    """Result of validation processing."""
    is_valid: bool
    confidence: float
    issues: List[str]
    recommendations: List[str]
    quality_score: float

# NOTE: This agent is designed to validate synthesis results based on various criteria.
# but currently does not implement advanced CoT techniques.
# Future enhancements may include using LLMs for more sophisticated validation.

class ValidationAgent(BaseAgent):
    """Agent responsible for validating synthesis results for quality and accuracy."""
    
    def __init__(self, llm_manager=None):
        """Initialize validation agent."""
        super().__init__("Validation")
        self.llm_manager = llm_manager  # Store for future CoT enhancements
        
        # Validation criteria for different query types
        self.validation_criteria = {
            QueryType.AUTHOR_EXPERTISE: self._validate_author_expertise,
            QueryType.TECHNOLOGY_TRENDS: self._validate_technology_trends,
            QueryType.AUTHOR_COLLABORATION: self._validate_author_collaboration,
            QueryType.DOMAIN_EVOLUTION: self._validate_domain_evolution,
            QueryType.CROSS_DOMAIN_ANALYSIS: self._validate_cross_domain,
            QueryType.PAPER_IMPACT: self._validate_paper_impact,
            QueryType.AUTHOR_PRODUCTIVITY: self._validate_author_productivity,
            QueryType.AUTHOR_STATS: self._validate_author_stats,
            QueryType.PAPER_SEARCH: self._validate_paper_search,
            QueryType.UNCLASSIFIED: self._validate_unclassified
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_confidence": 0.3,
            "min_data_points": 5,
            "max_missing_fields": 2,
            "min_completeness": 0.7
        }
    
    async def _initialize_impl(self) -> None:
        """Initialize the validation agent."""
        # Load validation configuration
        agent_config = self.config.agents.validation_agent
        self.validation_criteria_config = agent_config.get('validation_criteria', [
            'relevance', 'accuracy', 'completeness'
        ])
    
    async def validate(self, synthesis_result: SynthesisResult) -> ValidationResult:
        """Validate synthesis results.
        
        Args:
            synthesis_result: Result from synthesis agent
            
        Returns:
            Validation result
        """
        return await self.process(synthesis_result)
    
    async def _process_impl(self, synthesis_result: SynthesisResult) -> ValidationResult:
        """Implementation of validation processing.
        
        Args:
            synthesis_result: Result from synthesis agent
            
        Returns:
            Validation result
        """
        issues = []
        recommendations = []
        validation_scores = []
        
        # Basic structure validation
        structure_score = self._validate_structure(synthesis_result)
        validation_scores.append(structure_score)
        
        if structure_score < 0.7:
            issues.append("Result structure is incomplete or malformed")
            recommendations.append("Ensure all required fields are present in the result")
        
        # Data quality validation
        data_quality_score = self._validate_data_quality(synthesis_result)
        validation_scores.append(data_quality_score)
        
        if data_quality_score < 0.6:
            issues.append("Data quality concerns identified")
            recommendations.append("Verify data completeness and accuracy")
        
        # Confidence validation
        confidence_score = self._validate_confidence(synthesis_result)
        validation_scores.append(confidence_score)
        
        if confidence_score < 0.5:
            issues.append("Low confidence in results")
            recommendations.append("Consider gathering more data or refining analysis")
        
        # Content-specific validation
        content_score = await self._validate_content_specific(synthesis_result)
        validation_scores.append(content_score)
        
        if content_score < 0.6:
            issues.append("Content-specific validation concerns")
            recommendations.append("Review analysis methodology and results interpretation")
        
        # Consistency validation
        consistency_score = self._validate_consistency(synthesis_result)
        validation_scores.append(consistency_score)
        
        if consistency_score < 0.7:
            issues.append("Inconsistencies found in results")
            recommendations.append("Check for logical consistency across result components")
        
        # Calculate overall quality score
        quality_score = sum(validation_scores) / len(validation_scores)
        
        # Determine if result is valid
        is_valid = (
            quality_score >= 0.6 and
            synthesis_result.confidence >= self.quality_thresholds["min_confidence"] and
            len(issues) <= self.quality_thresholds["max_missing_fields"]
        )
        
        # Adjust confidence based on validation
        validated_confidence = min(
            synthesis_result.confidence,
            synthesis_result.confidence * quality_score
        )
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=validated_confidence,
            issues=issues,
            recommendations=recommendations,
            quality_score=quality_score
        )
    
    def _validate_structure(self, synthesis_result: SynthesisResult) -> float:
        """Validate the structure of synthesis result.
        
        Args:
            synthesis_result: Result to validate
            
        Returns:
            Structure validation score (0-1)
        """
        score = 1.0
        result = synthesis_result.result
        
        # Handle both dict and object results
        if isinstance(result, dict):
            result_dict = result
        elif hasattr(result, '__dict__'):
            # Convert object to dict for validation
            result_dict = result.__dict__ if hasattr(result, '__dict__') else {}
        elif hasattr(result, 'dict'):
            # Pydantic model with dict() method
            result_dict = result.dict() if callable(getattr(result, 'dict', None)) else {}
        else:
            return 0.0
        
        # Check for error conditions
        if "error" in result_dict:
            return 0.1
        
        # Check for required fields based on result type
        required_fields = self._get_required_fields(result_dict)
        missing_fields = 0
        
        for field in required_fields:
            if field not in result_dict or result_dict[field] is None:
                missing_fields += 1
        
        if required_fields:
            completeness = 1.0 - (missing_fields / len(required_fields))
            score *= completeness
        
        # Check for empty or minimal content
        if not result_dict or len(result_dict) < 2:
            score *= 0.5
        
        return max(0.0, score)
    
    def _validate_data_quality(self, synthesis_result: SynthesisResult) -> float:
        """Validate the quality of data in the result.
        
        Args:
            synthesis_result: Result to validate
            
        Returns:
            Data quality score (0-1)
        """
        score = 1.0
        result = synthesis_result.result
        
        # Handle both dict and object results
        if isinstance(result, dict):
            result_dict = result
        elif hasattr(result, '__dict__'):
            result_dict = result.__dict__ if hasattr(result, '__dict__') else {}
        elif hasattr(result, 'dict'):
            result_dict = result.dict() if callable(getattr(result, 'dict', None)) else {}
        else:
            return 0.0
        
        # Check data completeness
        data_points = self._count_data_points(result_dict)
        if data_points < self.quality_thresholds["min_data_points"]:
            score *= 0.6
        
        # Check for meaningful content
        meaningful_content = self._has_meaningful_content(result_dict)
        if not meaningful_content:
            score *= 0.4
        
        # Check for data consistency
        consistency_issues = self._check_data_consistency(result_dict)
        if consistency_issues > 0:
            score *= max(0.3, 1.0 - (consistency_issues * 0.2))
        
        return max(0.0, score)
    
    def _validate_confidence(self, synthesis_result: SynthesisResult) -> float:
        """Validate confidence levels.
        
        Args:
            synthesis_result: Result to validate
            
        Returns:
            Confidence validation score (0-1)
        """
        confidence = synthesis_result.confidence
        
        # Check confidence range
        if confidence < 0 or confidence > 1:
            return 0.0
        
        # Penalty for very low confidence
        if confidence < self.quality_thresholds["min_confidence"]:
            return confidence / self.quality_thresholds["min_confidence"]
        
        return 1.0
    
    async def _validate_content_specific(self, synthesis_result: SynthesisResult) -> float:
        """Validate content based on specific query type.
        
        Args:
            synthesis_result: Result to validate
            
        Returns:
            Content validation score (0-1)
        """
        result = synthesis_result.result
        
        # Handle both dict and object results
        if isinstance(result, dict):
            result_dict = result
        elif hasattr(result, '__dict__'):
            result_dict = result.__dict__ if hasattr(result, '__dict__') else {}
        elif hasattr(result, 'dict'):
            result_dict = result.dict() if callable(getattr(result, 'dict', None)) else {}
        else:
            return 0.0
        
        # Try to infer query type from result structure
        query_type = self._infer_query_type(result_dict)
        
        if query_type and query_type in self.validation_criteria:
            validator = self.validation_criteria[query_type]
            return await validator(result_dict)
        
        # Default validation for unknown types
        return self._validate_general_content(result_dict)
    
    def _validate_consistency(self, synthesis_result: SynthesisResult) -> float:
        """Validate internal consistency of the result.
        
        Args:
            synthesis_result: Result to validate
            
        Returns:
            Consistency validation score (0-1)
        """
        result = synthesis_result.result
        
        # Handle both dict and object results
        if isinstance(result, dict):
            result_dict = result
        elif hasattr(result, '__dict__'):
            result_dict = result.__dict__ if hasattr(result, '__dict__') else {}
        elif hasattr(result, 'dict'):
            result_dict = result.dict() if callable(getattr(result, 'dict', None)) else {}
        else:
            return 0.0
        
        score = 1.0
        
        # Check numerical consistency
        numerical_consistency = self._check_numerical_consistency(result_dict)
        score *= numerical_consistency
        
        # Check logical consistency
        logical_consistency = self._check_logical_consistency(result_dict)
        score *= logical_consistency
        
        return max(0.0, score)
    
    def _get_required_fields(self, result: Dict[str, Any]) -> List[str]:
        """Get required fields based on result type.
        
        Args:
            result: Result dictionary
            
        Returns:
            List of required field names
        """
        # Infer type and return appropriate required fields
        if "top_authors" in result:
            return ["domain", "top_authors", "total_papers_analyzed"]
        elif "trends" in result:
            return ["trends", "emerging_technologies"]
        elif "collaboration_overview" in result or "focal_author" in result:
            return ["collaborators"] if "focal_author" in result else ["collaboration_overview"]
        elif "cross_domain_authors" in result:
            return ["cross_domain_authors", "total_authors_analyzed"]
        elif "high_impact_papers" in result:
            return ["high_impact_papers", "total_papers_analyzed"]
        elif "productive_authors" in result:
            return ["productive_authors", "total_authors_analyzed"]
        else:
            return ["summary"]  # Generic requirement
    
    def _count_data_points(self, result: Dict[str, Any]) -> int:
        """Count meaningful data points in the result.
        
        Args:
            result: Result dictionary
            
        Returns:
            Number of data points
        """
        data_points = 0
        
        # Count based on result type
        if "top_authors" in result:
            data_points += len(result["top_authors"])
        if "trends" in result:
            data_points += len(result["trends"])
        if "collaborators" in result:
            data_points += len(result["collaborators"])
        if "cross_domain_authors" in result:
            data_points += len(result["cross_domain_authors"])
        if "high_impact_papers" in result:
            data_points += len(result["high_impact_papers"])
        if "productive_authors" in result:
            data_points += len(result["productive_authors"])
        
        # Count other meaningful fields
        for key, value in result.items():
            if isinstance(value, list) and key not in [
                "top_authors", "trends", "collaborators", "cross_domain_authors",
                "high_impact_papers", "productive_authors"
            ]:
                data_points += len(value)
        
        return data_points
    
    def _has_meaningful_content(self, result: Dict[str, Any]) -> bool:
        """Check if result has meaningful content.
        
        Args:
            result: Result dictionary
            
        Returns:
            True if meaningful content exists
        """
        # Check for empty or error results
        if not result or "error" in result:
            return False
        
        # Check for substantial content
        meaningful_keys = [
            "top_authors", "trends", "collaborators", "cross_domain_authors",
            "high_impact_papers", "productive_authors", "insights", "summary"
        ]
        
        has_meaningful = any(
            key in result and result[key] and (
                not isinstance(result[key], list) or len(result[key]) > 0
            )
            for key in meaningful_keys
        )
        
        return has_meaningful
    
    def _check_data_consistency(self, result: Dict[str, Any]) -> int:
        """Check for data consistency issues.
        
        Args:
            result: Result dictionary
            
        Returns:
            Number of consistency issues found
        """
        issues = 0
        
        # Check for count mismatches
        if "total_papers_analyzed" in result and "top_authors" in result:
            total_papers = result["total_papers_analyzed"]
            if isinstance(total_papers, int) and total_papers < len(result["top_authors"]):
                issues += 1
        
        # Check for negative values where they shouldn't exist
        for key, value in result.items():
            if isinstance(value, (int, float)) and value < 0 and key in [
                "total_papers_analyzed", "total_authors_analyzed", "paper_count",
                "collaboration_count", "expertise_score"
            ]:
                issues += 1
        
        # Check list consistency
        for key, value in result.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # Check for required fields in list items
                        if key == "top_authors" and "author" not in item:
                            issues += 1
                        elif key == "trends" and "technology" not in item:
                            issues += 1
        
        return issues
    
    def _infer_query_type(self, result: Dict[str, Any]) -> QueryType:
        """Infer query type from result structure.
        
        Args:
            result: Result dictionary
            
        Returns:
            Inferred query type or None
        """
        if "top_authors" in result:
            return QueryType.AUTHOR_EXPERTISE
        elif "trends" in result or "emerging_technologies" in result:
            return QueryType.TECHNOLOGY_TRENDS
        elif "focal_author" in result or "collaboration_overview" in result:
            return QueryType.AUTHOR_COLLABORATION
        elif "cross_domain_authors" in result:
            return QueryType.CROSS_DOMAIN_ANALYSIS
        elif "high_impact_papers" in result:
            return QueryType.PAPER_IMPACT
        elif "productive_authors" in result:
            return QueryType.AUTHOR_PRODUCTIVITY
        
        return None
    
    def _validate_general_content(self, result: Dict[str, Any]) -> float:
        """General content validation for unknown types.
        
        Args:
            result: Result dictionary
            
        Returns:
            Validation score (0-1)
        """
        score = 1.0
        
        # Check for basic content
        if not result or len(result) < 2:
            score *= 0.5
        
        # Check for summary or insights
        if "summary" not in result and "insights" not in result:
            score *= 0.8
        
        return max(0.0, score)
    
    def _check_numerical_consistency(self, result: Dict[str, Any]) -> float:
        """Check numerical consistency in the result.
        
        Args:
            result: Result dictionary
            
        Returns:
            Consistency score (0-1)
        """
        score = 1.0
        
        # Check for reasonable value ranges
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key.endswith("_count") and value < 0:
                    score *= 0.8
                elif key.endswith("_score") and (value < 0 or value > 10):
                    score *= 0.9
                elif key == "confidence" and (value < 0 or value > 1):
                    score *= 0.7
        
        return max(0.0, score)
    
    def _check_logical_consistency(self, result: Dict[str, Any]) -> float:
        """Check logical consistency in the result.
        
        Args:
            result: Result dictionary
            
        Returns:
            Consistency score (0-1)
        """
        score = 1.0
        
        # Check for logical relationships
        if "total_papers_analyzed" in result:
            total_papers = result["total_papers_analyzed"]
            
            # Check if counts make sense
            if isinstance(total_papers, int) and total_papers > 0:
                for key in ["top_authors", "trends", "high_impact_papers"]:
                    if key in result and isinstance(result[key], list):
                        if len(result[key]) > total_papers:
                            score *= 0.8
        
        return max(0.0, score)
    
    # Query-type specific validation methods
    async def _validate_author_expertise(self, result: Dict[str, Any]) -> float:
        """Validate author expertise specific content."""
        score = 1.0
        
        if "top_authors" not in result:
            return 0.3
        
        top_authors = result["top_authors"]
        if not top_authors:
            return 0.4
        
        # Check author data quality
        for author_data in top_authors[:3]:  # Check top 3
            if not isinstance(author_data, dict):
                score *= 0.8
                continue
            
            if "author" not in author_data or not author_data["author"]:
                score *= 0.8
            
            if "paper_count" not in author_data or author_data["paper_count"] <= 0:
                score *= 0.9
        
        return max(0.0, score)
    
    async def _validate_technology_trends(self, result: Dict[str, Any]) -> float:
        """Validate technology trends specific content."""
        score = 1.0
        
        if "trends" not in result and "emerging_technologies" not in result:
            return 0.3
        
        if "trends" in result:
            trends = result["trends"]
            if not trends:
                score *= 0.7
            else:
                # Check trend data quality
                for trend in trends[:3]:
                    if not isinstance(trend, dict):
                        score *= 0.8
                        continue
                    
                    if "technology" not in trend:
                        score *= 0.9
        
        return max(0.0, score)
    
    async def _validate_author_collaboration(self, result: Dict[str, Any]) -> float:
        """Validate author collaboration specific content."""
        score = 1.0
        
        has_focal = "focal_author" in result
        has_general = "collaboration_overview" in result
        
        if not has_focal and not has_general:
            return 0.3
        
        if has_focal:
            if "collaborators" not in result:
                score *= 0.7
            elif not result["collaborators"]:
                score *= 0.8
        
        return max(0.0, score)
    
    async def _validate_domain_evolution(self, result: Dict[str, Any]) -> float:
        """Validate domain evolution specific content."""
        return await self._validate_technology_trends(result)
    
    async def _validate_cross_domain(self, result: Dict[str, Any]) -> float:
        """Validate cross-domain specific content."""
        score = 1.0
        
        if "cross_domain_authors" not in result:
            return 0.3
        
        cross_domain_authors = result["cross_domain_authors"]
        if not cross_domain_authors:
            return 0.4
        
        # Check cross-domain data quality
        for author_data in cross_domain_authors[:3]:
            if not isinstance(author_data, dict):
                score *= 0.8
                continue
            
            if "author" not in author_data:
                score *= 0.9
            
            if "domain_count" not in author_data or author_data["domain_count"] <= 1:
                score *= 0.9
        
        return max(0.0, score)
    
    async def _validate_paper_impact(self, result: Dict[str, Any]) -> float:
        """Validate paper impact specific content."""
        score = 1.0
        
        if "high_impact_papers" not in result:
            return 0.3
        
        high_impact_papers = result["high_impact_papers"]
        if not high_impact_papers:
            return 0.4
        
        # Check for methodology note (important for impact analysis)
        if "methodology" not in result and "note" not in result:
            score *= 0.8
        
        return max(0.0, score)
    
    async def _validate_author_productivity(self, result: Dict[str, Any]) -> float:
        """Validate author productivity specific content."""
        score = 1.0
        
        if "productive_authors" not in result:
            return 0.3
        
        productive_authors = result["productive_authors"]
        if not productive_authors:
            return 0.4
        
        # Check productivity data quality
        for author_data in productive_authors[:3]:
            if not isinstance(author_data, dict):
                score *= 0.8
                continue
            
            required_fields = ["author", "total_papers", "years_active"]
            for field in required_fields:
                if field not in author_data:
                    score *= 0.9
        
        return max(0.0, score)
    
    async def _validate_author_stats(self, result: Dict[str, Any]) -> float:
        """Validate author statistics content."""
        score = 1.0
        
        if "author_profile" in result:
            # Author found case
            profile = result["author_profile"]
            if not profile.get("found"):
                score *= 0.3
            else:
                stats = profile.get("stats", {})
                required_fields = ["total_papers", "years_active", "subjects", "collaborators"]
                for field in required_fields:
                    if field not in stats:
                        score *= 0.8
        elif "error_info" in result:
            # Author not found case - check if similar authors provided
            error_info = result["error_info"]
            if error_info.get("similar_authors"):
                score = 0.4  # Partial credit for suggestions
            else:
                score = 0.2
        else:
            score = 0.1
        
        return max(0.0, score)
    
    async def _validate_paper_search(self, result: Dict[str, Any]) -> float:
        """Validate paper search content."""
        score = 1.0
        
        if "search_results" in result:
            search_data = result["search_results"]
            papers_found = search_data.get("papers_found", 0)
            
            if papers_found > 0:
                # Papers found - validate structure
                matching_papers = search_data.get("matching_papers", [])
                if not matching_papers:
                    score *= 0.5
                else:
                    # Check if papers have required fields
                    for paper in matching_papers[:3]:  # Check first 3 papers
                        required_fields = ["paper_id", "title", "authors", "year"]
                        for field in required_fields:
                            if field not in paper:
                                score *= 0.9
            else:
                # No papers found - check if helpful message provided
                if search_data.get("suggestion"):
                    score = 0.4
                else:
                    score = 0.3
        else:
            score = 0.1
        
        return max(0.0, score)
    
    async def _validate_unclassified(self, result: Dict[str, Any]) -> float:
        """Validate unclassified query handling."""
        score = 1.0
        
        # Check if error information is provided
        if "error_info" not in result:
            score *= 0.5
        
        # Check if suggestions are provided
        if "suggestions" not in result or not result.get("suggestions"):
            score *= 0.7
        
        # Check if clear message is provided
        if "summary" not in result:
            score *= 0.8
        
        return max(0.0, score)
        
