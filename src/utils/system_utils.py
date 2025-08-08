"""
System utilities for TechAuthor system.
Provides functions for checking system status, getting statistics, and system diagnostics.
"""

import logging
from typing import Dict, Any
from pathlib import Path


async def get_index_statistics(data_manager, logger):
    """Get statistics about the current indices."""
    try:
        stats = {}
        
        if hasattr(data_manager, 'papers') and data_manager.papers:
            stats['total_papers'] = len(data_manager.papers)
        
        if hasattr(data_manager, 'authors_index') and data_manager.authors_index:
            stats['total_authors'] = len(data_manager.authors_index)
        
        # Vector database stats
        if hasattr(data_manager, 'vector_db_manager') and data_manager.vector_db_manager:
            vector_stats = data_manager.vector_db_manager.get_stats()
            stats.update(vector_stats)
            
        # Hybrid searcher stats (for backward compatibility)
        if hasattr(data_manager, 'hybrid_searcher') and data_manager.hybrid_searcher:
            hybrid_stats = data_manager.hybrid_searcher.get_search_stats()
            if hybrid_stats:
                stats['hybrid_search'] = hybrid_stats
                
        return stats
        
    except Exception as e:
        logger.warning(f"Could not get index statistics: {e}")
        return {"error": str(e)}


async def check_index_status(system, logger):
    """Check and display the status of all indices."""
    logger.info("Checking index status...")
    
    if not hasattr(system, 'data_manager') or not system.data_manager:
        logger.error("Data manager not initialized")
        return
    
    data_manager = system.data_manager
    config = data_manager.config
    
    # Check data freshness
    logger.info("=" * 50)
    logger.info("DATA STATUS")
    logger.info("=" * 50)
    
    total_papers = len(data_manager.papers) if data_manager.papers else 0
    logger.info(f"Total papers loaded: {total_papers:,}")
    
    if hasattr(data_manager, 'authors_index'):
        logger.info(f"Total authors indexed: {len(data_manager.authors_index):,}")
    if hasattr(data_manager, 'domain_index'):
        logger.info(f"Total domains indexed: {len(data_manager.domain_index):,}")
    
    # Check vector database status
    logger.info("\n" + "=" * 50)
    logger.info("VECTOR DATABASE STATUS")
    logger.info("=" * 50)
    
    if hasattr(data_manager, 'vector_db_manager') and data_manager.vector_db_manager:
        vector_db = data_manager.vector_db_manager
        if vector_db.is_initialized():
            stats = vector_db.get_stats()
            backend = getattr(config.vector_db, 'backend', 'Unknown')
            logger.info(f"Backend: {backend}")
            logger.info(f"Status: Initialized")
            num_docs = stats.get('num_documents', 'Unknown')
            if isinstance(num_docs, (int, float)):
                logger.info(f"Documents: {num_docs:,}")
            else:
                logger.info(f"Documents: {num_docs}")
            if 'index_size_mb' in stats:
                logger.info(f"Index size: {stats['index_size_mb']:.2f} MB")
        else:
            logger.info("Status: Not initialized")
    
    # Check sparse search status
    logger.info("\n" + "=" * 50)
    logger.info("SPARSE SEARCH STATUS")
    logger.info("=" * 50)
    
    sparse_config = config.retrieval.sparse_search or {}
    algorithm = sparse_config.get('algorithm', 'tfidf')
    logger.info(f"Algorithm: {algorithm.upper()}")
    
    # Check if sparse index exists
    cache_dir = sparse_config.get('cache_dir', './data/sparse_index/')
    algorithm_config = sparse_config.get(algorithm, {})
    index_file = algorithm_config.get('index_file', f'{algorithm}_index.pkl')
    index_path = Path(cache_dir) / index_file
    
    if index_path.exists():
        import os
        size_mb = os.path.getsize(index_path) / (1024 * 1024)
        logger.info(f"Status: Index file exists")
        logger.info(f"Index file: {index_path}")
        logger.info(f"Index size: {size_mb:.2f} MB")
        
        # Try to get more detailed stats from hybrid searcher
        if (hasattr(data_manager, 'hybrid_searcher') and 
            hasattr(data_manager.hybrid_searcher, 'sparse_searcher')):
            sparse_stats = data_manager.hybrid_searcher.sparse_searcher.get_stats()
            if sparse_stats and 'num_documents' in sparse_stats:
                num_docs = sparse_stats.get('num_documents', 'Unknown')
                num_features = sparse_stats.get('num_features', 'Unknown')
                if isinstance(num_docs, (int, float)):
                    logger.info(f"Documents: {num_docs:,}")
                else:
                    logger.info(f"Documents: {num_docs}")
                if isinstance(num_features, (int, float)):
                    logger.info(f"Features: {num_features:,}")
                else:
                    logger.info(f"Features: {num_features}")
    else:
        logger.info("Status: Index file not found")
        logger.info(f"Expected location: {index_path}")
    
    # Check semantic embeddings status
    logger.info("\n" + "=" * 50)
    logger.info("SEMANTIC EMBEDDINGS STATUS")
    logger.info("=" * 50)
    
    embeddings_config = config.embeddings
    model_name = getattr(embeddings_config, 'model_name', 'Unknown')
    dimension = getattr(embeddings_config, 'dimension', 'Unknown')
    logger.info(f"Model: {model_name}")
    logger.info(f"Dimension: {dimension}")
    
    # Check if embeddings cache exists
    cache_path = Path(getattr(config.data, 'embeddings_cache_path', './data/embeddings/'))
    embedding_files = []
    if cache_path.exists():
        # Look for embedding files
        embedding_files = list(cache_path.rglob("*.npy")) + list(cache_path.rglob("*.pkl"))
        if embedding_files:
            logger.info("Status: Cached embeddings found")
            total_size_mb = sum(f.stat().st_size for f in embedding_files) / (1024 * 1024)
            logger.info(f"Cache files: {len(embedding_files)}")
            logger.info(f"Total cache size: {total_size_mb:.2f} MB")
        else:
            logger.info("Status: No cached embeddings found")
    else:
        logger.info("Status: Embeddings cache directory not found")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    
    # Determine what needs to be built
    needs_building = []
    
    if not (hasattr(data_manager, 'vector_db_manager') and 
            data_manager.vector_db_manager and 
            data_manager.vector_db_manager.is_initialized()):
        needs_building.append("Vector database")
    
    if not index_path.exists():
        needs_building.append(f"Sparse index ({algorithm.upper()})")
    
    if not embedding_files:
        needs_building.append("Embeddings cache")
    
    if needs_building:
        logger.info("⚠️  The following indices need to be built:")
        for item in needs_building:
            logger.info(f"   - {item}")
        logger.info("\nRun with --build-index to build all missing indices")
    else:
        logger.info("✅ All indices are available and ready to use")


async def get_legacy_index_statistics(data_manager, logger):
    """Get statistics about the current indices (legacy version for backward compatibility)."""
    try:
        stats = {}
        
        if hasattr(data_manager, 'papers') and data_manager.papers:
            stats['total_papers'] = len(data_manager.papers)
        
        if hasattr(data_manager, 'author_index') and data_manager.author_index:
            stats['total_authors'] = len(data_manager.author_index)
            
        if hasattr(data_manager, 'hybrid_searcher'):
            if hasattr(data_manager.hybrid_searcher, 'sparse_searcher'):
                sparse = data_manager.hybrid_searcher.sparse_searcher
                if hasattr(sparse, 'tfidf_matrix') and sparse.tfidf_matrix is not None:
                    stats['sparse_features'] = sparse.tfidf_matrix.shape[1]
                    stats['sparse_documents'] = sparse.tfidf_matrix.shape[0]
            
            if hasattr(data_manager.hybrid_searcher, 'semantic_searcher'):
                semantic = data_manager.hybrid_searcher.semantic_searcher
                if hasattr(semantic, 'vector_store') and semantic.vector_store:
                    if hasattr(semantic.vector_store, 'index') and semantic.vector_store.index:
                        stats['semantic_documents'] = semantic.vector_store.index.ntotal
                        
        return stats
        
    except Exception as e:
        logger.warning(f"Could not get index statistics: {e}")
        return {"error": str(e)}
