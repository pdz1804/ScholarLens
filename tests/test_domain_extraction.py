#!/usr/bin/env python3
"""
Test domain extraction patterns.
"""

import re
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logger

logger = setup_logger("DomainExtractionTest", "INFO")

test_queries = [
    "What are the latest trends in artificial intelligence?",
    "Show me emerging technologies in computer science",
    "Technology evolution in machine learning over time",
    "What's trending in cybersecurity research?",
    "Current developments in quantum computing"
]

domain_patterns = [
    # Specific technology domains (most specific)
    r"(computer\s+vision|natural\s+language\s+processing|machine\s+learning|artificial\s+intelligence|deep\s+learning|neural\s+networks|cybersecurity|quantum\s+computing|computer\s+science|data\s+science|software\s+engineering)",
    # Pattern: trends in X, emerging technologies in X, etc.
    r"(?:trends?|emerging\s+technologies?|developments?)\s+in\s+([a-zA-Z\s]+?)(?:\s+(?:research|field|domain|area))?$",
    r"evolution\s+in\s+([a-zA-Z\s]+)\s+over\s+time",
    # Pattern: in/for/about/on X
    r"(?:in|for|about|on)\s+([a-zA-Z\s]+?)(?:\s+(?:research|field|domain|area|over\s+time))",
    # Acronyms (least greedy)
    r"\b([A-Z]{2,4})\b"  # Match only 2-4 letter acronyms
]

for i, query in enumerate(test_queries, 1):
    logger.info(f"Test {i}: {query}")
    logger.info("-" * 40)
    
    query_lower = query.lower()
    domains = []
    
    for j, pattern in enumerate(domain_patterns):
        matches = re.findall(pattern, query_lower, re.IGNORECASE)
        if matches:
            logger.info(f"Pattern {j+1}: {pattern}")
            logger.info(f"Matches: {matches}")
            
            for match in matches:
                if isinstance(match, tuple):
                    # Take the last non-empty part of the tuple
                    domain = next((part for part in reversed(match) if part and part.strip()), None)
                else:
                    domain = match
                if domain and len(domain.strip()) > 2:
                    domain = domain.strip().lower()
                    domain = re.sub(r'\s+', ' ', domain)
                    domains.append(domain)
                    logger.info(f"Extracted domain: '{domain}'")
    
    if domains:
        best_domain = max(domains, key=len)
        logger.info(f"Best domain: '{best_domain}'")
    else:
        logger.info("No domain extracted")
