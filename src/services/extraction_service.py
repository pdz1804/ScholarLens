"""
LLM-based extraction service for TechAuthor system.
Provides intelligent extraction of various elements from research papers and queries.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from ..core.models import RetrievalResult
from ..prompts.extraction_prompts import (
    METHODOLOGY_EXTRACTION_PROMPT,
    CONCEPT_EXTRACTION_PROMPT,
    PROBLEM_EXTRACTION_PROMPT,
    SOLUTION_EXTRACTION_PROMPT,
    AUTHOR_NAME_EXTRACTION_PROMPT,
    SEARCH_TERM_EXTRACTION_PROMPT,
    PARADIGM_SHIFT_EXTRACTION_PROMPT
)


class ExtractionService:
    """LLM-based extraction service for intelligent content analysis."""
    
    def __init__(self, llm_manager=None):
        """Initialize extraction service.
        
        Args:
            llm_manager: LLM manager instance for generating completions
        """
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def extract_methodologies_from_papers(
        self, 
        papers: List[RetrievalResult],
        domain: str = "Unknown"
    ) -> List[str]:
        """Extract methodologies from research papers using LLM.
        
        Args:
            papers: List of research papers
            domain: Research domain context
            
        Returns:
            List of extracted methodologies
        """
        if not papers:
            return []
            
        self.logger.info(f"Extracting methodologies from {len(papers)} papers in domain: {domain}")
        
        # Sample papers to avoid token limits
        sample_papers = papers[:10] if len(papers) > 10 else papers
        
        # Prepare paper content for analysis
        paper_content = self._prepare_paper_content_for_extraction(sample_papers)
        
        if not self.llm_manager:
            self.logger.warning("LLM manager not available, using fallback extraction")
            return self._fallback_methodology_extraction(sample_papers)
            
        try:
            # Use LLM for intelligent extraction
            extracted_json = await self.llm_manager.generate(
                system_prompt="You are an expert research methodology analyst.",
                user_prompt=METHODOLOGY_EXTRACTION_PROMPT.format(
                    domain=domain,
                    paper_content=paper_content
                ),
                agent_name="extraction"
            )
            
            # Parse LLM response
            methodologies = self._parse_extraction_response(extracted_json, "methodologies")
            self.logger.info(f"Extracted {len(methodologies)} methodologies using LLM")
            return methodologies
            
        except Exception as e:
            self.logger.warning(f"LLM methodology extraction failed: {e}, using fallback")
            return self._fallback_methodology_extraction(sample_papers)
    
    async def extract_key_concepts_from_papers(
        self,
        papers: List[RetrievalResult],
        domain: str = "Unknown"
    ) -> List[str]:
        """Extract key concepts from research papers using LLM.
        
        Args:
            papers: List of research papers
            domain: Research domain context
            
        Returns:
            List of extracted key concepts
        """
        if not papers:
            return []
            
        self.logger.info(f"Extracting key concepts from {len(papers)} papers in domain: {domain}")
        
        sample_papers = papers[:10] if len(papers) > 10 else papers
        paper_content = self._prepare_paper_content_for_extraction(sample_papers)
        
        if not self.llm_manager:
            self.logger.warning("LLM manager not available, using fallback extraction")
            return self._fallback_concept_extraction(sample_papers)
            
        try:
            extracted_json = await self.llm_manager.generate(
                system_prompt="You are an expert research concept analyst.",
                user_prompt=CONCEPT_EXTRACTION_PROMPT.format(
                    domain=domain,
                    paper_content=paper_content
                ),
                agent_name="extraction"
            )
            
            concepts = self._parse_extraction_response(extracted_json, "concepts")
            self.logger.info(f"Extracted {len(concepts)} key concepts using LLM")
            return concepts
            
        except Exception as e:
            self.logger.warning(f"LLM concept extraction failed: {e}, using fallback")
            return self._fallback_concept_extraction(sample_papers)
    
    async def extract_problems_from_papers(
        self,
        papers: List[RetrievalResult],
        domain: str = "Unknown"
    ) -> List[str]:
        """Extract problem statements from research papers using LLM.
        
        Args:
            papers: List of research papers
            domain: Research domain context
            
        Returns:
            List of extracted problems
        """
        if not papers:
            return []
            
        self.logger.info(f"Extracting problems from {len(papers)} papers in domain: {domain}")
        
        sample_papers = papers[:8] if len(papers) > 8 else papers
        paper_content = self._prepare_paper_content_for_extraction(sample_papers)
        
        if not self.llm_manager:
            self.logger.warning("LLM manager not available, using fallback extraction")
            return self._fallback_problem_extraction(sample_papers)
            
        try:
            extracted_json = await self.llm_manager.generate(
                system_prompt="You are an expert research problem analyst.",
                user_prompt=PROBLEM_EXTRACTION_PROMPT.format(
                    domain=domain,
                    paper_content=paper_content
                ),
                agent_name="extraction"
            )
            
            problems = self._parse_extraction_response(extracted_json, "problems")
            self.logger.info(f"Extracted {len(problems)} problems using LLM")
            return problems
            
        except Exception as e:
            self.logger.warning(f"LLM problem extraction failed: {e}, using fallback")
            return self._fallback_problem_extraction(sample_papers)
    
    async def extract_solutions_from_papers(
        self,
        papers: List[RetrievalResult],
        domain: str = "Unknown"
    ) -> List[str]:
        """Extract solution approaches from research papers using LLM.
        
        Args:
            papers: List of research papers
            domain: Research domain context
            
        Returns:
            List of extracted solutions
        """
        if not papers:
            return []
            
        self.logger.info(f"Extracting solutions from {len(papers)} papers in domain: {domain}")
        
        sample_papers = papers[:8] if len(papers) > 8 else papers
        paper_content = self._prepare_paper_content_for_extraction(sample_papers)
        
        if not self.llm_manager:
            self.logger.warning("LLM manager not available, using fallback extraction")
            return self._fallback_solution_extraction(sample_papers)
            
        try:
            extracted_json = await self.llm_manager.generate(
                system_prompt="You are an expert research solution analyst.",
                user_prompt=SOLUTION_EXTRACTION_PROMPT.format(
                    domain=domain,
                    paper_content=paper_content
                ),
                agent_name="extraction"
            )
            
            solutions = self._parse_extraction_response(extracted_json, "solutions")
            self.logger.info(f"Extracted {len(solutions)} solutions using LLM")
            return solutions
            
        except Exception as e:
            self.logger.warning(f"LLM solution extraction failed: {e}, using fallback")
            return self._fallback_solution_extraction(sample_papers)
    
    async def extract_author_name_from_query(self, query_text: str) -> Optional[str]:
        """Extract author name from query text using LLM.
        
        Args:
            query_text: User query text
            
        Returns:
            Extracted author name or None
        """
        if not query_text.strip():
            return None
            
        self.logger.info(f"Extracting author name from query: {query_text[:50]}...")
        
        if not self.llm_manager:
            self.logger.warning("LLM manager not available, using fallback extraction")
            return self._fallback_author_name_extraction(query_text)
            
        try:
            extracted_json = await self.llm_manager.generate(
                system_prompt="You are an expert at extracting author names from research queries.",
                user_prompt=AUTHOR_NAME_EXTRACTION_PROMPT.format(query=query_text),
                agent_name="extraction"
            )
            
            result = self._parse_extraction_response(extracted_json, "author_name")
            author_name = result[0] if result and isinstance(result, list) else result
            
            if author_name and author_name != "None":
                self.logger.info(f"Extracted author name: {author_name}")
                return author_name
            else:
                self.logger.info("No author name found in query")
                return None
                
        except Exception as e:
            self.logger.warning(f"LLM author name extraction failed: {e}, using fallback")
            return self._fallback_author_name_extraction(query_text)
    
    async def extract_search_term_from_query(self, query_text: str) -> Optional[str]:
        """Extract search term from query text using LLM.
        
        Args:
            query_text: User query text
            
        Returns:
            Extracted search term or None
        """
        if not query_text.strip():
            return None
            
        self.logger.info(f"Extracting search term from query: {query_text[:50]}...")
        
        if not self.llm_manager:
            self.logger.warning("LLM manager not available, using fallback extraction")
            return self._fallback_search_term_extraction(query_text)
            
        try:
            extracted_json = await self.llm_manager.generate(
                system_prompt="You are an expert at extracting search terms from research queries.",
                user_prompt=SEARCH_TERM_EXTRACTION_PROMPT.format(query=query_text),
                agent_name="extraction"
            )
            
            result = self._parse_extraction_response(extracted_json, "search_term")
            search_term = result[0] if result and isinstance(result, list) else result
            
            if search_term and search_term != "None":
                self.logger.info(f"Extracted search term: {search_term}")
                return search_term
            else:
                self.logger.info("No search term found in query")
                return None
                
        except Exception as e:
            self.logger.warning(f"LLM search term extraction failed: {e}, using fallback")
            return self._fallback_search_term_extraction(query_text)
    
    def _prepare_paper_content_for_extraction(self, papers: List[RetrievalResult]) -> str:
        """Prepare paper content for LLM analysis.
        
        Args:
            papers: List of papers to analyze
            
        Returns:
            Formatted paper content string
        """
        content_parts = []
        for i, result in enumerate(papers, 1):
            paper = result.paper
            content_parts.append(f"""
Paper {i}:
Title: {paper.title}
Abstract: {paper.abstract[:500]}...
Subjects: {', '.join(paper.subjects)}
""")
        
        return '\n'.join(content_parts)
    
    def _parse_extraction_response(self, response: str, expected_key: str) -> List[str]:
        """Parse LLM extraction response.
        
        Args:
            response: LLM response string
            expected_key: Expected key in JSON response
            
        Returns:
            List of extracted items
        """
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                data = json.loads(response)
                if expected_key in data:
                    result = data[expected_key]
                    return result if isinstance(result, list) else [result]
            
            # Fallback: extract items from text response
            lines = response.strip().split('\n')
            items = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('*'):
                    # Remove bullet points and numbering
                    clean_line = line.lstrip('- ').lstrip('• ').lstrip('1234567890. ')
                    if clean_line:
                        items.append(clean_line)
            
            return items[:10]  # Limit to reasonable number
            
        except Exception as e:
            self.logger.warning(f"Failed to parse extraction response: {e}")
            return []
    
    # Fallback extraction methods (simplified pattern-based)
    def _fallback_methodology_extraction(self, papers: List[RetrievalResult]) -> List[str]:
        """Fallback methodology extraction using patterns."""
        methodology_keywords = [
            'neural network', 'deep learning', 'machine learning', 'reinforcement learning',
            'transformer', 'attention mechanism', 'convolutional neural network',
            'statistical method', 'bayesian', 'optimization', 'algorithm'
        ]
        
        found_methodologies = []
        for paper in papers:
            text = f"{paper.paper.title} {paper.paper.abstract}".lower()
            for keyword in methodology_keywords:
                if keyword in text:
                    found_methodologies.append(keyword)
        
        return list(set(found_methodologies))[:10]
    
    def _fallback_concept_extraction(self, papers: List[RetrievalResult]) -> List[str]:
        """Fallback concept extraction using patterns."""
        concept_keywords = [
            'accuracy', 'performance', 'efficiency', 'scalability',
            'robustness', 'interpretability', 'real-time',
            'multimodal', 'transfer learning', 'representation learning'
        ]
        
        found_concepts = []
        for paper in papers:
            text = f"{paper.paper.title} {paper.paper.abstract}".lower()
            for keyword in concept_keywords:
                if keyword in text:
                    found_concepts.append(keyword)
        
        return list(set(found_concepts))[:10]
    
    def _fallback_problem_extraction(self, papers: List[RetrievalResult]) -> List[str]:
        """Fallback problem extraction using patterns."""
        problems = []
        for paper in papers:
            if any(word in paper.paper.abstract.lower() for word in ['challenge', 'problem', 'issue']):
                problems.append(f"Research problem from: {paper.paper.title[:50]}...")
        
        return problems[:8]
    
    def _fallback_solution_extraction(self, papers: List[RetrievalResult]) -> List[str]:
        """Fallback solution extraction using patterns."""
        solutions = []
        for paper in papers:
            if any(word in paper.paper.abstract.lower() for word in ['propose', 'method', 'approach']):
                solutions.append(f"Solution approach from: {paper.paper.title[:50]}...")
        
        return solutions[:8]
    
    def _fallback_author_name_extraction(self, query_text: str) -> Optional[str]:
        """Fallback author name extraction using patterns."""
        import re
        
        patterns = [
            r"about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+(?:research|work|profile|stats)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_text)
            if match:
                return match.group(1)
        
        return None
    
    def _fallback_search_term_extraction(self, query_text: str) -> Optional[str]:
        """Fallback search term extraction using patterns."""
        import re
        
        patterns = [
            # ArXiv ID pattern (e.g., 2112.15106, 1234.5678)
            r"(\d{4}\.\d{4,5})",
            # Papers about pattern
            r"papers about\s+([A-Za-z0-9\.\-_\s]+)",
            # Research on pattern  
            r"research on\s+([A-Za-z0-9\.\-_\s]+)",
            # Show me pattern (capture everything after show me)
            r"show me.*?([A-Za-z0-9\.\-_]+)",
            # Paper ID or title pattern
            r"paper\s+([A-Za-z0-9\.\-_\s]+)",
            # Generic word extraction (last resort)
            r"([A-Z][A-Za-z0-9\.\-_]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                # Return the term if it's meaningful (not just whitespace or common words)
                if term and len(term) > 2 and term.lower() not in ['the', 'and', 'for', 'about', 'papers', 'paper', 'show', 'me']:
                    return term
        
        return None
