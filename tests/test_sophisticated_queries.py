#!/usr/bin/env python3
"""
Test sophisticated query classification scenarios.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.models import Query, QueryType
from src.agents.query_classifier import QueryClassifierAgent
from src.utils.logger import setup_logger

async def test_sophisticated_classification():
    """Test the query classification with sophisticated and challenging queries."""
    
    logger = setup_logger("SophisticatedQueryTest", "INFO")
    
    # More sophisticated and challenging test queries
    test_cases = [
        {
            "query": "I'm looking for researchers who work at the intersection of computer vision and robotics, particularly those who have made breakthrough contributions",
            "expected": QueryType.AUTHOR_EXPERTISE,
            "description": "Multi-domain expertise with qualitative constraint"
        },
        {
            "query": "What are the recent advances in transformer architectures and how are they being applied beyond NLP?",
            "expected": QueryType.TECHNOLOGY_TRENDS,
            "description": "Technology trends with application scope"
        },
        {
            "query": "Can you map out the collaboration network around Geoffrey Hinton and show how it influenced the deep learning revolution?",
            "expected": QueryType.AUTHOR_COLLABORATION,
            "description": "Collaboration with historical impact context"
        },
        {
            "query": "Compare the research productivity between Stanford, MIT, and CMU in machine learning over the last decade",
            "expected": QueryType.INSTITUTIONAL_ANALYSIS,
            "description": "Multi-institutional comparison with time constraint"
        },
        {
            "query": "Show me the evolution of attention mechanisms from traditional seq2seq to modern transformers",
            "expected": QueryType.DOMAIN_EVOLUTION,
            "description": "Technical evolution with specific technology path"
        },
        {
            "query": "Who are the most cited authors in federated learning, and what makes their work so impactful?",
            "expected": QueryType.AUTHOR_EXPERTISE,
            "description": "Mixed author expertise and impact query"
        },
        {
            "query": "I want to understand how AI research differs between industry labs like Google DeepMind and academic institutions",
            "expected": QueryType.CROSS_DOMAIN_ANALYSIS,
            "description": "Industry vs academic comparison"
        },
        {
            "query": "What papers introduced the key concepts that led to the current generative AI boom?",
            "expected": QueryType.PAPER_IMPACT,
            "description": "Historical paper impact with trend context"
        }
    ]
    
    logger.info("Initializing query classifier...")
    classifier = QueryClassifierAgent()
    await classifier.initialize()
    
    logger.info("Testing sophisticated queries:")
    logger.info("=" * 80)
    
    correct = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test {i}: {test_case['description']}")
        logger.info(f"Query: \"{test_case['query']}\"")
        logger.info("-" * 60)
        
        query = Query(text=test_case['query'])
        result = await classifier.classify(query)
        
        logger.info(f"Classified as: {result.query_type.value}")
        logger.info(f"Expected: {test_case['expected'].value}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Parameters: {result.extracted_params}")
        logger.info(f"Reasoning: {result.reasoning}")
        
        # Check if classification is correct
        is_correct = result.query_type == test_case['expected']
        if is_correct:
            correct += 1
            logger.info("Correct")
        else:
            logger.info("Wrong")
    
    logger.info("=" * 80)
    logger.info(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(test_sophisticated_classification())
