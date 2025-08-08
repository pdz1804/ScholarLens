#!/usr/bin/env python3
"""
Test the actual query classifier to see the issue.
"""

import asyncio
import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.models import Query, QueryType
from src.agents.query_classifier import QueryClassifierAgent
from src.utils.logger import setup_logger

async def test_classifier():
    """Test the actual query classifier."""
    
    logger = setup_logger("ClassifierDebugTest", "INFO")
    
    test_query = "What are the latest trends in artificial intelligence?"
    
    logger.info(f"Testing query: {test_query}")
    logger.info("=" * 60)
    
    # Initialize classifier
    classifier = QueryClassifierAgent()
    await classifier.initialize()
    
    # Test classification
    query = Query(text=test_query)
    result = await classifier.classify(query)
    
    logger.info("Result:")
    logger.info(f"Query Type: {result.query_type.value}")
    logger.info(f"Confidence: {result.confidence}")
    logger.info(f"Reasoning: {result.reasoning}")
    logger.info(f"Parameters: {result.extracted_params}")

if __name__ == "__main__":
    asyncio.run(test_classifier())
