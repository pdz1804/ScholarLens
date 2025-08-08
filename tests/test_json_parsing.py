#!/usr/bin/env python3
"""
Simple JSON extraction test.
"""

import re
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logger

logger = setup_logger("JsonParsingTest", "INFO")

response = '''Let's analyze the query step by step:
1. **Query Analysis**: The user is asking about "latest trends" in a specific field, which is "artificial intelligence." T
This indicates they are interested in identifying current patterns or developments within that domain.
2. **Pattern Recognition**: The phrase "latest trends in artificial intelligence" fits the pattern for identifying trendin
ng technologies and patterns, which corresponds to the query type TECHNOLOGY_TRENDS.
3. **Parameter Extraction**:
   - Domain: "artificial intelligence"
   - Time range: The term "latest" suggests a focus on recent trends, but it does not specify an exact time frame. We can 
 interpret this as a general interest in the most current trends without a specific year mentioned.
Now, we can compile this information into the required JSON format:
```json
{
  "query_type": "TECHNOLOGY_TRENDS",
  "confidence": 0.95,
  "parameters": {
    "domain": "artificial intelligence",
    "author": null,
    "top_k": null,
    "time_range": null,
    "technologies": null
  },
  "reasoning": "The query asks for the latest trends in artificial intelligence, which indicates a focus on current develo
opments in that domain, classifying it as TECHNOLOGY_TRENDS."
}
```'''

logger.info("Raw response:")
logger.info(repr(response))
logger.info("=" * 60)

# Try to extract JSON from markdown code blocks first
json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL | re.MULTILINE)
if json_match:
    json_str = json_match.group(1).strip()
    logger.info("Extracted JSON from markdown:")
    logger.info(repr(json_str))
    
    # Clean up the JSON string by removing line breaks within string values
    json_str = re.sub(r'"\s*\n\s*([^"]*)"', r'"\1"', json_str)
    json_str = re.sub(r'\n\s*', ' ', json_str)  # Replace line breaks with spaces
    
    logger.info("Cleaned JSON:")
    logger.info(repr(json_str))
    
    try:
        result = json.loads(json_str)
        logger.info("Successfully parsed JSON:")
        logger.info(json.dumps(result, indent=2))
    except json.JSONDecodeError as e:
        logger.info(f"JSON parsing failed: {e}")
else:
    logger.info("No JSON found in markdown blocks")
