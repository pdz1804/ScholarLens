"""
Command line interface utilities for TechAuthor system.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from .logger import setup_logger


def create_unique_log_file() -> str:
    """Create unique log file name based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return str(log_dir / f"techauthor_{timestamp}.log")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="TechAuthor Research Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Who are the top authors in AI Agents?"
  python main.py "Technology trends in NLP" --top-k 20 --format json
  python main.py --batch queries.txt --output results.json
        """
    )
    
    parser.add_argument(
        "query", 
        nargs="?",
        help="Research query to analyze"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=10, 
        help="Number of top results to return (default: 10)"
    )
    
    parser.add_argument(
        "--domain", 
        help="Specific domain to focus on (e.g., 'Computer Vision')"
    )
    
    parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text", 
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output with detailed logging"
    )
    
    parser.add_argument(
        "--batch", 
        help="Path to file with multiple queries (one per line)"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file path (default: console output)"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive mode"
    )
    
    # Index management arguments
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build all search indices (sparse and embeddings) with smart change detection"
    )
    
    parser.add_argument(
        "--build-sparse",
        action="store_true", 
        help="Build only sparse search index (TF-IDF or BM25)"
    )
    
    parser.add_argument(
        "--build-embeddings",
        action="store_true",
        help="Build only semantic embeddings index"
    )
    
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force complete rebuild of all search indices (useful after data changes or parsing improvements)"
    )
    
    parser.add_argument(
        "--update-index",
        action="store_true", 
        help="Update indices incrementally with new data (faster than full rebuild)"
    )
    
    parser.add_argument(
        "--check-index",
        action="store_true",
        help="Check status of all indices and display statistics"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached data and indices before processing"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use only 1%% of the dataset for testing (useful for development and debugging)"
    )
    
    return parser


def validate_arguments(args) -> bool:
    """Validate command line arguments."""
    # Allow index management operations without requiring query/batch/interactive
    has_index_ops = (args.force_reindex or args.update_index or args.clear_cache or 
                     args.build_index or args.build_sparse or args.build_embeddings or args.check_index or
                     args.test_mode)
    has_query_ops = args.query or args.batch or args.interactive
    
    if not has_query_ops and not has_index_ops:
        # For validation errors, we'll use print since this is CLI interface
        print("Error: Must provide either a query, --batch file, --interactive mode, or index management options")
        print("Index management options: --build-index, --build-sparse, --build-embeddings, --check-index, --force-reindex, --update-index, --clear-cache, --test-mode")
        return False
    
    if args.batch and not Path(args.batch).exists():
        # For validation errors, we'll use print since this is CLI interface
        print(f"Error: Batch file not found: {args.batch}")
        return False
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return True


def setup_logging(args) -> object:
    """Setup logging for the session."""
    log_file = create_unique_log_file()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("TechAuthor-CLI", level=log_level, log_file=log_file)
    
    logger.info("Starting TechAuthor Research Analysis System")
    logger.info(f"Log file: {log_file}")
    if args.query:
        logger.info(f"Query: {args.query}")
    elif args.batch:
        logger.info(f"Batch file: {args.batch}")
    elif args.interactive:
        logger.info("Interactive mode")
    
    return logger


def prepare_query_parameters(args) -> Dict[str, Any]:
    """Prepare query parameters from arguments."""
    params = {"top_k": args.top_k}
    
    if args.domain:
        params["domain"] = args.domain
    
    return params
