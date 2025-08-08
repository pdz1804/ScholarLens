"""
Batch query processing example for TechAuthor system.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src import create_system
from src.utils.logger import setup_logger


async def main():
    """Batch query processing example."""
    
    logger = setup_logger("BatchProcessing", "INFO")
    
    # Create and initialize system
    logger.info("Initializing TechAuthor system for batch processing...")
    system = create_system()
    
    try:
        await system.initialize()
        logger.info("System initialized successfully!")
        
        # Define batch queries
        batch_queries = [
            "Top 15 authors in Computer Vision",
            "Top 10 authors in Natural Language Processing", 
            "Top 12 authors in Machine Learning",
            "Top 8 authors in Robotics",
            "Top 20 authors in Artificial Intelligence",
            "Emerging trends in Deep Learning",
            "Technology evolution in Computer Graphics",
            "Cross-domain researchers in AI and Biology",
            "Most productive authors in recent years",
            "Collaboration patterns in Neural Networks research"
        ]
        
        logger.info(f"Processing {len(batch_queries)} queries in batch...")
        logger.info("=" * 70)
        
        # Process all queries in batch
        start_time = asyncio.get_event_loop().time()
        responses = await system.abatch_query(batch_queries)
        end_time = asyncio.get_event_loop().time()
        
        batch_time = end_time - start_time
        logger.info(f"Batch processing completed in {batch_time:.2f} seconds")
        
        # Analyze results
        successful_queries = 0
        total_confidence = 0
        results_summary = []
        
        logger.info("Batch Results Summary:")
        logger.info("-" * 70)
        
        for i, response in enumerate(responses):
            query = batch_queries[i]
            
            if 'error' not in response.result:
                successful_queries += 1
                total_confidence += response.confidence
                
                # Extract key information
                result_info = {
                    "query": query,
                    "type": response.response_type,
                    "confidence": response.confidence,
                    "processing_time": response.processing_time
                }
                
                # Add specific results based on type
                result = response.result
                if isinstance(result, dict):
                    if 'top_authors' in result:
                        authors = result['top_authors'][:3]
                        result_info["top_authors"] = [a['author'] for a in authors]
                    
                    if 'emerging_technologies' in result:
                        result_info["emerging_tech"] = result['emerging_technologies'][:3]
                    
                    if 'summary' in result:
                        result_info["summary"] = result['summary'][:100] + "..." if len(result['summary']) > 100 else result['summary']
                
                results_summary.append(result_info)
                
                logger.info(f"{i+1:2d}. {query[:50]}{'...' if len(query) > 50 else ''}")
                logger.info(f"    Type: {response.response_type}, Confidence: {response.confidence:.2f}")
                
                if 'top_authors' in result_info:
                    authors_str = ', '.join(result_info['top_authors'])
                    logger.info(f"    Top Authors: {authors_str}")
                
                if 'emerging_tech' in result_info:
                    tech_str = ', '.join(result_info['emerging_tech'])
                    logger.info(f"    Emerging Tech: {tech_str}")
            
            else:
                logger.info(f"{i+1:2d}. {query[:50]}{'...' if len(query) > 50 else ''}")
                logger.info(f"    ERROR: {response.result.get('error', 'Unknown error')}")
        
        # Overall statistics
        logger.info("=" * 70)
        logger.info("BATCH PROCESSING STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total Queries: {len(batch_queries)}")
        logger.info(f"Successful: {successful_queries}")
        logger.info(f"Failed: {len(batch_queries) - successful_queries}")
        logger.info(f"Success Rate: {successful_queries/len(batch_queries)*100:.1f}%")
        logger.info(f"Average Confidence: {total_confidence/successful_queries:.2f}" if successful_queries > 0 else "N/A")
        logger.info(f"Total Processing Time: {batch_time:.2f}s")
        logger.info(f"Average Time per Query: {batch_time/len(batch_queries):.2f}s")
        
        # Save results to file
        output_file = Path(__file__).parent / "batch_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {output_file}")
        
        # Show system statistics
        system_stats = system.get_stats()
        logger.info("System Statistics:")
        logger.info(f"  Cache Hit Rate: {system_stats.cache_hit_rate:.2f}")
        logger.info(f"  Total System Queries: {system_stats.total_queries}")
        
    except Exception as e:
        logger.info(f"Batch processing error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await system.shutdown()
        logger.info("System shutdown completed.")


if __name__ == "__main__":
    asyncio.run(main())
