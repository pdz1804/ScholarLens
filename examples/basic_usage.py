"""
Basic usage example for TechAuthor system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent  
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

from src import create_system
from src.utils.logger import setup_logger


async def main():
    """Basic usage example."""
    
    logger = setup_logger("BasicUsage", "INFO")
    
    # Create and initialize system
    logger.info("Initializing TechAuthor system...")
    system = create_system()
    
    try:
        await system.initialize()
        logger.info("System initialized successfully!")
        
        # Example queries
        queries = [
            "Who are the top authors in Computer Vision?",
            "What are the emerging technologies in Machine Learning?",
            "Show me collaboration patterns in Natural Language Processing",
            "How has Deep Learning evolved over time?",
            "Which authors work across multiple AI domains?"
        ]
        
        logger.info("Running example queries...")
        logger.info("=" * 60)
        
        for i, query in enumerate(queries, 1):
            logger.info(f"{i}. Query: {query}")
            logger.info("-" * len(query))
            
            try:
                # Process query
                response = await system.aquery(query)
                
                logger.info(f"Type: {response.response_type}")
                logger.info(f"Confidence: {response.confidence:.2f}")
                logger.info(f"Processing Time: {response.processing_time:.2f}s")
                
                # Show key results
                result = response.result
                if isinstance(result, dict):
                    if 'summary' in result:
                        logger.info(f"Summary: {result['summary']}")
                    
                    if 'top_authors' in result:
                        authors = result['top_authors'][:3]
                        logger.info(f"Top Authors: {', '.join(a['author'] for a in authors)}")
                    
                    if 'emerging_technologies' in result:
                        emerging = result['emerging_technologies'][:3]
                        logger.info(f"Emerging Tech: {', '.join(emerging)}")
                    
                    if 'insights' in result and isinstance(result['insights'], list):
                        insights = result['insights'][:2]
                        for insight in insights:
                            logger.info(f"  • {insight}")
                
            except Exception as e:
                logger.info(f"Error processing query: {e}")
            
            if i < len(queries):
                logger.info("")
        
        logger.info("=" * 60)
        logger.info("Example completed successfully!")
        
        # Show system statistics
        stats = system.get_stats()
        logger.info("System Statistics:")
        logger.info(f"  Total Queries: {stats.total_queries}")
        logger.info(f"  Successful: {stats.successful_queries}")
        logger.info(f"  Average Time: {stats.average_processing_time:.2f}s")
        
    except Exception as e:
        logger.info(f"System error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await system.shutdown()
        logger.info("System shutdown completed.")


if __name__ == "__main__":
    asyncio.run(main())
