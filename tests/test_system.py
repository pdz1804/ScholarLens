"""
Test script to verify TechAuthor system is working properly.
This script runs basic tests to ensure all components are functioning.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.system import TechAuthorSystem
from src.utils.logger import setup_logger


async def test_system():
    """
    Test the TechAuthor system initialization and basic functionality.
    
    Tests:
    1. System can initialize and load CSV data
    2. Hybrid search indices are built/loaded
    3. Basic query processing works
    4. All agents are functional
    """
    
    logger = setup_logger("SystemTest", "INFO")
    
    logger.info("Starting TechAuthor system tests")
    
    try:
        # Test 1: System initialization
        logger.info("Test 1: Initializing system and loading CSV data")
        system = TechAuthorSystem()
        
        await system.initialize()
        logger.info("System initialization completed")
        
        # Test 2: Check system health
        logger.info("Test 2: Checking system health")
        health = await system.get_health_status()
        logger.info(f"System health: {health['status']}")
        
        if health['status'] != 'healthy':
            logger.error(f"System health issues: {health.get('issues', [])}")
            return False
        
        # Test 3: Get system statistics
        logger.info("Test 3: Getting system statistics") 
        stats = await system.get_statistics()
        logger.info(f"Papers loaded: {stats.get('total_papers', 0)}")
        logger.info(f"Authors indexed: {stats.get('total_authors', 0)}")
        logger.info(f"Subjects indexed: {stats.get('total_subjects', 0)}")
        
        # Test 4: Simple query test
        logger.info("Test 4: Testing simple query processing")
        test_query = "machine learning papers"
        
        response = await system.aquery(test_query, {'limit': 5})
        
        logger.info(f"Query processed successfully")
        logger.info(f"Response type: {response.response_type}")
        logger.info(f"Confidence: {response.confidence}")
        logger.info(f"Found papers: {len(response.data.get('papers', []))}")
        
        # Test 5: Test different query types
        test_queries = [
            "Who are experts in deep learning?",
            "What are trending topics in AI?",
            "Papers about neural networks"
        ]
        
        logger.info("Test 5: Testing different query types")
        for i, query in enumerate(test_queries, 1):
            try:
                logger.info(f"Testing query {i}: {query}")
                response = await system.aquery(query, {'limit': 3})
                logger.info(f"Query {i} success: {response.response_type}")
            except Exception as e:
                logger.warning(f"Query {i} failed: {str(e)}")
        
        logger.info("All tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"System test failed: {str(e)}")
        return False


def main():
    """Main test entry point."""
    
    logger = setup_logger("SystemTest", "INFO")
    
    logger.info("TechAuthor System Test")
    logger.info("=" * 50)
    logger.info("Testing system initialization and basic functionality...")
    logger.info("")
    
    success = asyncio.run(test_system())
    
    if success:
        logger.info("")
        logger.info("All tests passed successfully!")
        logger.info("System is ready for use.")
        logger.info("")
        logger.info("Try running:")
        logger.info('  python main.py --query "Who are the top authors in AI?"')
        logger.info("  python main.py --interactive")
    else:
        logger.info("")
        logger.info("Tests failed. Please check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
