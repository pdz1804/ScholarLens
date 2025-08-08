#!/usr/bin/env python3
"""
Test script to debug query classification issues.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.models import Query, QueryType
from src.agents.query_classifier import QueryClassifierAgent
from src.core.llm_manager import LLMManager
from src.utils.logger import setup_logger

async def test_classification():
    """Test the query classification with problematic queries."""
    
    logger = setup_logger("ClassificationTest", "INFO")
    
    # Test queries that should be TECHNOLOGY_TRENDS
    test_queries = [
        "What are the latest trends in artificial intelligence?",
        "Show me emerging technologies in computer science",
        "Technology evolution in machine learning over time",
        "What's trending in cybersecurity research?",
        "Current developments in quantum computing"
    ]
    
    logger.info("Initializing query classifier...")
    classifier = QueryClassifierAgent()
    await classifier.initialize()
    
    logger.info("Testing problematic queries:")
    logger.info("=" * 60)
    
    for i, query_text in enumerate(test_queries, 1):
        logger.info(f"Test {i}: {query_text}")
        logger.info("-" * 40)
        
        query = Query(text=query_text)
        result = await classifier.classify(query)
        
        logger.info(f"Classified as: {result.query_type.value}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Reasoning: {result.reasoning}")
        logger.info(f"Parameters: {result.extracted_params}")
        
        # Check if classification is correct
        expected = QueryType.TECHNOLOGY_TRENDS
        is_correct = result.query_type == expected
        logger.info(f"Correct" if is_correct else f"Wrong (expected {expected.value})")

if __name__ == "__main__":
    asyncio.run(test_classification())
