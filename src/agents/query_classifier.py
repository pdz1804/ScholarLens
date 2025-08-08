"""
Query classification agent for TechAuthor system.
"""

import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..core.models import Query, QueryType
from ..core.config import config_manager
from ..core.llm_manager import LLMManager
from ..prompts.agent_prompts import QUERY_CLASSIFICATION_PROMPT, QUERY_CLASSIFIER_SYSTEM_PROMPT
from .base_agent import BaseAgent


@dataclass
class ClassificationResult:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    extracted_params: Dict[str, Any]
    reasoning: str


class QueryClassifierAgent(BaseAgent):
    """Agent responsible for classifying user queries into appropriate types."""
    
    def __init__(self, llm_manager: LLMManager = None):
        """Initialize query classifier agent."""
        super().__init__("QueryClassifier")
        self.llm_manager = llm_manager or LLMManager()
    
    async def _initialize_impl(self) -> None:
        """Initialize the query classifier agent."""
        # No initialization needed - LLM manager handles configuration
        self.logger.info("Query classifier agent initialized with enhanced CoT prompts")
    
    async def classify(self, query: Query) -> ClassificationResult:
        """Classify a query into appropriate type and extract parameters using enhanced CoT prompts.
        
        Args:
            query: The query object to classify
            
        Returns:
            ClassificationResult with type and extracted parameters
        """
        return await self.process(query)
    
    async def _process_impl(self, query: Query) -> ClassificationResult:
        """Implementation of query classification.
        
        Args:
            query: Query object to classify
            
        Returns:
            Classification result
        """
        # Use LLM for sophisticated classification and parameter extraction
        llm_result = await self._classify_with_llm(query)
        
        # The LLM already extracts all parameters including domain
        # No need for pattern-based parameter extraction
        
        return llm_result
    
    async def _classify_with_llm(self, query: Query) -> ClassificationResult:
        """Classify query using Language Model.
        
        Args:
            query: Query object
            
        Returns:
            Classification result
        """
        try:
            # Format the enhanced prompt
            prompt = QUERY_CLASSIFICATION_PROMPT.format(query=query.text)
            
            # Generate classification using LLM manager
            response = await self.llm_manager.generate(
                system_prompt=QUERY_CLASSIFIER_SYSTEM_PROMPT,
                user_prompt=prompt,
                agent_name="query_classifier"
            )
            
            # Parse JSON response
            try:
                # Try to extract JSON from markdown code blocks first
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL | re.MULTILINE)
                if json_match:
                    json_str = json_match.group(1).strip()
                    self.logger.debug(f"Extracted JSON from markdown: {json_str[:200]}...")
                else:
                    # Try to find JSON object in the response (look for { ... })
                    json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        self.logger.debug(f"Extracted JSON from response: {json_str[:200]}...")
                    else:
                        json_str = response.strip()
                        self.logger.debug(f"Using full response as JSON: {json_str[:200]}...")
                
                # Clean up the JSON string by removing line breaks within string values
                # This fixes issues where the LLM breaks JSON strings across lines
                json_str = re.sub(r'"\s*\n\s*([^"]*)"', r'"\1"', json_str)
                json_str = re.sub(r'\n\s*', ' ', json_str)  # Replace line breaks with spaces
                
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing failed: {e}")
                self.logger.debug(f"Failed to parse: {json_str[:500] if 'json_str' in locals() else response[:500]}")
                # Fallback parsing if JSON is malformed
                result = self._fallback_parse(response)
            
            # Map string query type to enum
            query_type_str = result.get('query_type', 'GENERAL_INQUIRY')
            query_type = self._string_to_query_type(query_type_str)
            
            classification_result = ClassificationResult(
                query_type=query_type,
                confidence=result.get('confidence', 0.5),
                extracted_params=result.get('parameters', {}),
                reasoning=result.get('reasoning', 'No reasoning provided')
            )
            
            self.logger.info(f"Classified query as: {query_type.value} (confidence: {classification_result.confidence:.2f})")
            self.logger.debug(f"Extracted parameters: {classification_result.extracted_params}")
            self.logger.debug(f"Reasoning: {classification_result.reasoning}")
            
            return classification_result
            
        except Exception as e:
            self.logger.error(f"Query classification failed: {str(e)}")
            # Return default classification
            return ClassificationResult(
                query_type=QueryType.AUTHOR_EXPERTISE,
                confidence=0.1,
                extracted_params={},
                reasoning=f"Classification failed: {str(e)}"
            )
    
    def _string_to_query_type(self, query_type_str: str) -> QueryType:
        """Convert string query type to enum."""
        query_type_mapping = {
            'AUTHOR_EXPERTISE': QueryType.AUTHOR_EXPERTISE,
            'TECHNOLOGY_TRENDS': QueryType.TECHNOLOGY_TRENDS,
            'AUTHOR_COLLABORATION': QueryType.AUTHOR_COLLABORATION,
            'DOMAIN_EVOLUTION': QueryType.DOMAIN_EVOLUTION,
            'CROSS_DOMAIN_ANALYSIS': QueryType.CROSS_DOMAIN_ANALYSIS,
            'PAPER_IMPACT': QueryType.PAPER_IMPACT,
            'AUTHOR_PRODUCTIVITY': QueryType.AUTHOR_PRODUCTIVITY,
            'AUTHOR_STATS': QueryType.AUTHOR_STATS,
            'PAPER_SEARCH': QueryType.PAPER_SEARCH,
            'UNCLASSIFIED': QueryType.UNCLASSIFIED
        }
        return query_type_mapping.get(query_type_str, QueryType.UNCLASSIFIED)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing if JSON parsing fails."""
        # Try to extract information using regex patterns
        query_type_match = re.search(r'"query_type":\s*"([^"]+)"', response)
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', response)
        
        return {
            'query_type': query_type_match.group(1) if query_type_match else 'AUTHOR_EXPERTISE',
            'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
            'parameters': {},
            'reasoning': 'Fallback parsing used due to JSON format issues'
        }
    
    def _get_query_type_description(self, query_type: QueryType) -> str:
        """Get description for a query type.
        
        Args:
            query_type: Query type enum
            
        Returns:
            Description string
        """
        descriptions = {
            QueryType.AUTHOR_EXPERTISE: "Find top authors/experts in specific technology domains",
            QueryType.TECHNOLOGY_TRENDS: "Analyze emerging technologies and trends over time",
            QueryType.AUTHOR_COLLABORATION: "Analyze collaboration patterns between authors",
            QueryType.DOMAIN_EVOLUTION: "Track how domains/fields have evolved over time",
            QueryType.CROSS_DOMAIN_ANALYSIS: "Find authors working across multiple domains",
            QueryType.PAPER_IMPACT: "Analyze most influential or impactful papers",
            QueryType.AUTHOR_PRODUCTIVITY: "Analyze author productivity and publication patterns",
            QueryType.AUTHOR_STATS: "Get detailed statistics about a specific author",
            QueryType.PAPER_SEARCH: "Find specific papers by title, ID, or technology",
            QueryType.UNCLASSIFIED: "Query type could not be determined"
        }
        
        return descriptions.get(query_type, "General research query")
