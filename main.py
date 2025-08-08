"""
TechAuthor - Research Paper Analysis System
Main entry point for the TechAuthor research paper analysis system.

This system uses a RAG + Multi-agents architecture to analyze relationships between authors 
and technology domains using ArXiv papers as bridges between them.

Key Features:
- Hybrid Search: Combines sparse (TF-IDF keyword) and dense (semantic embedding) search
- Multi-Agent Pipeline: Query classification, retrieval, analysis, synthesis, validation
- Incremental Indexing: Avoids rebuilding indices unless data changes
- 8 Query Scenarios: Author expertise, trends, collaborations, domains, etc.

Data Flow:
1. CSV data (data/arxiv_cs.csv) is loaded and processed into Paper objects
2. Papers are indexed with both TF-IDF (sparse) and sentence embeddings (dense)
3. Hybrid search combines both approaches for comprehensive results
4. Multi-agent pipeline processes queries and generates insights

Usage:
    python main.py --query "Who are the top authors in AI Agents?"
    python main.py --interactive
    python main.py --batch-file queries.txt --output results.json
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.cli import setup_argument_parser, validate_arguments, setup_logging, prepare_query_parameters
from src.utils.query_processor import process_single_query, process_batch_queries, interactive_mode
from src.utils.output_handler import save_results_json, save_results_csv, print_summary_stats, generate_report
from src.utils.cache import clear_cached_data
from src.utils.system_utils import get_index_statistics, check_index_status
from src.utils.index_builder import build_indices_smart, build_sparse_index, build_embeddings_index
from src.core.system import TechAuthorSystem


async def main():
    """
    Main entry point for TechAuthor system.
    
    Handles CLI argument parsing, system initialization, query processing, and output formatting.
    The system automatically builds or loads hybrid search indices (sparse + dense) as needed.
    """
    
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup logging with unique filename (no special characters)
    logger = setup_logging(args)
    
    logger.info("Starting TechAuthor system")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Handle cache/index management options
        if args.clear_cache:
            logger.info("Clearing all cached data and indices...")
            await clear_cached_data(logger)
        
        # Handle index checking first (before initialization)
        if args.check_index:
            logger.info("Checking index status...")
            # Initialize system to check current state
            system = TechAuthorSystem()
            
            # Minimal initialization to load data
            indexing_options = {'force_reindex': False, 'update_index': False, 'clear_cache': False, 'test_mode': args.test_mode}
            await system.initialize(indexing_options=indexing_options)
            
            await check_index_status(system, logger)
            
            # If only checking index status, exit here
            index_ops_only = not (args.query or args.batch or args.interactive or 
                                args.build_index or args.build_sparse or args.build_embeddings or 
                                args.force_reindex or args.update_index)
            if index_ops_only:
                return
        
        # Initialize the system
        # This will load CSV data from data/arxiv_cs.csv and build/load hybrid search indices
        logger.info("Initializing TechAuthor system...")
        
        if not 'system' in locals():
            system = TechAuthorSystem()
        
        # Determine indexing strategy
        if args.force_reindex:
            logger.info("Force reindexing enabled - will rebuild all search indices from scratch")
            logger.info("This includes sparse indices and semantic embedding indices")
        elif args.update_index:
            logger.info("Update indexing enabled - will incrementally update indices with new data")
        elif args.build_index or args.build_sparse or args.build_embeddings:
            logger.info("Smart index building enabled - will build only missing or outdated indices")
        else:
            logger.info("Loading CSV data and building hybrid search indices (sparse + dense embeddings)")
        
        # Initialize with dataset - pass indexing options
        indexing_options = {
            'force_reindex': args.force_reindex,
            'update_index': args.update_index,
            'clear_cache': args.clear_cache,
            'test_mode': args.test_mode
        }
        
        await system.initialize(indexing_options=indexing_options)
        logger.info("System initialized successfully with hybrid search capabilities")
        
        # Handle specific index building operations
        if args.build_index:
            success = await build_indices_smart(system, logger, "all", force=False)
            if not success:
                logger.error("Failed to build indices")
                sys.exit(1)
        
        if args.build_sparse:
            success = await build_indices_smart(system, logger, "sparse", force=False)
            if not success:
                logger.error("Failed to build sparse index")
                sys.exit(1)
        
        if args.build_embeddings:
            success = await build_indices_smart(system, logger, "embeddings", force=False)
            if not success:
                logger.error("Failed to build embeddings index")
                sys.exit(1)
        
        # Log index statistics after initialization
        if hasattr(system, 'data_manager') and system.data_manager:
            stats = await get_index_statistics(system.data_manager, logger)
            logger.info(f"Index statistics: {stats}")
        
        # Check if only index management operations were requested
        index_ops_only = ((args.force_reindex or args.update_index or args.clear_cache or 
                          args.build_index or args.build_sparse or args.build_embeddings or args.check_index or
                          args.test_mode) 
                         and not (args.query or args.batch or args.interactive))
        
        if index_ops_only:
            logger.info("Index management operations completed successfully!")
            if args.force_reindex:
                logger.info("All search indices have been rebuilt from scratch")
            if args.update_index:
                logger.info("Indices have been updated with new data")
            if args.clear_cache:
                logger.info("All cached data has been cleared")
            if args.build_index:
                logger.info("All missing indices have been built")
            if args.build_sparse:
                logger.info("Sparse search index has been built")
            if args.build_embeddings:
                logger.info("Embeddings index has been built")
            if args.check_index:
                logger.info("Index status check completed")
            if args.test_mode:
                logger.info("Test mode initialization completed with 1% dataset sample")
            return
        
        # Prepare query parameters
        params = prepare_query_parameters(args)
        
        # Process queries based on mode
        results = []
        
        if args.interactive:
            # Interactive mode - user enters queries in real-time
            await interactive_mode(system, params, logger)
            return
            
        elif args.batch:
            # Batch processing mode - read queries from file
            results = await process_batch_queries(system, args.batch, params, logger)
            
        elif args.query:
            # Single query mode - process one query
            response = await process_single_query(system, args.query, params, logger)
            results = [response]
            
        else:
            # Default: show help and run example to demonstrate system
            logger.info("No query specified, running example query to demonstrate system")
            
            example_query = "Who are the top authors in AI Agents?"
            logger.info(f"Running example query: {example_query}")
            response = await process_single_query(system, example_query, params, logger)
            results = [response]
        
        # Handle output formatting and saving
        if results:
            # Print results to console using logger-friendly output
            from src.utils.query_processor import print_query_response
            
            if len(results) == 1:
                print_query_response(results[0], args.format, logger)
            else:
                # Multiple results - print summary statistics
                print_summary_stats(results, logger)
                if args.format == "json":
                    import json
                    output_data = []
                    for response in results:
                        result_dict = {
                            "query": response.query.text,
                            "type": response.response_type,
                            "confidence": response.confidence,
                            "summary": response.summary,
                            "insights": response.insights,
                            "data": response.data
                        }
                        output_data.append(result_dict)
                    logger.info(json.dumps(output_data, indent=2, ensure_ascii=False))
            
            # Save to files if requested
            if args.output:
                if args.output.endswith('.json'):
                    save_results_json(results, args.output, logger)
                elif args.output.endswith('.csv'):
                    save_results_csv(results, args.output, logger)
                else:
                    generate_report(results, args.output, logger)
                
                logger.info(f"Results saved to: {args.output}")
        
        logger.info("TechAuthor session completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
        
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
