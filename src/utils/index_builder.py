"""
Index building utilities for TechAuthor system.
Provides functions for building and managing search indices with smart change detection.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional


async def build_indices_smart(system, logger, index_type="all", force=False):
    """Build indices with smart change detection."""
    if not hasattr(system, 'data_manager') or not system.data_manager:
        logger.error("Data manager not initialized")
        return False
    
    data_manager = system.data_manager
    
    logger.info(f"Building {index_type} indices...")
    
    if index_type in ["all", "sparse"]:
        await build_sparse_index(data_manager, logger, force)
    
    if index_type in ["all", "embeddings"]:
        await build_embeddings_index(data_manager, logger, force)
    
    logger.info(f"✅ {index_type.title()} indices built successfully!")
    return True


async def build_sparse_index(data_manager, logger, force=False):
    """Build sparse search index with change detection."""
    logger.info("Building sparse search index...")
    
    config = data_manager.config
    sparse_config = config.retrieval.sparse_search or {}
    algorithm = sparse_config.get('algorithm', 'tfidf')
    
    # Check if index already exists and is current
    cache_dir = Path(sparse_config.get('cache_dir', './data/sparse_index/'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    algorithm_config = sparse_config.get(algorithm, {})
    index_file = algorithm_config.get('index_file', f'{algorithm}_index.pkl')
    index_path = cache_dir / index_file
    
    # Check if we need to rebuild
    needs_rebuild = force or not index_path.exists()
    
    if not needs_rebuild:
        # Check if data has changed
        try:
            # Create hash of current papers data
            papers_data = str([(p.paper_id, p.title, p.abstract) for p in data_manager.papers[:100]])  # Sample for speed
            current_hash = hashlib.md5(papers_data.encode()).hexdigest()
            
            # Check stored hash
            hash_file = cache_dir / f"{algorithm}_hash.txt"
            if hash_file.exists():
                stored_hash = hash_file.read_text().strip()
                if stored_hash != current_hash:
                    needs_rebuild = True
                    logger.info("Data has changed, rebuilding sparse index...")
            else:
                needs_rebuild = True
                
        except Exception as e:
            logger.warning(f"Could not check data hash: {e}, rebuilding index")
            needs_rebuild = True
    
    if not needs_rebuild:
        logger.info(f"✅ {algorithm.upper()} sparse index is up to date")
        return
    
    # Build the index
    logger.info(f"Building {algorithm.upper()} sparse index for {len(data_manager.papers)} documents...")
    
    # Convert papers to documents format
    documents = []
    for paper in data_manager.papers:
        doc = {
            'id': paper.paper_id,
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': ', '.join(paper.authors) if isinstance(paper.authors, list) else paper.authors,
            'domain': paper.domain,
            'subjects': ', '.join(paper.subjects) if isinstance(paper.subjects, list) else paper.subjects
        }
        documents.append(doc)
    
    # Create sparse searcher and build index
    from src.retrieval.sparse_search import SparseSearcher
    
    sparse_searcher = SparseSearcher(sparse_config)
    sparse_searcher.build_index(documents)
    
    # Save index
    sparse_searcher.save_index(str(cache_dir))
    
    # Save data hash for future change detection
    papers_data = str([(p.paper_id, p.title, p.abstract) for p in data_manager.papers[:100]])
    current_hash = hashlib.md5(papers_data.encode()).hexdigest()
    hash_file = cache_dir / f"{algorithm}_hash.txt"
    hash_file.write_text(current_hash)
    
    # Get stats
    stats = sparse_searcher.get_stats()
    logger.info(f"{algorithm.upper()} index built: {stats.get('num_documents', 0):,} docs, {stats.get('num_features', 0):,} features")


async def build_embeddings_index(data_manager, logger, force=False):
    """Build embeddings index with change detection."""
    logger.info("Building embeddings index...")
    
    config = data_manager.config
    
    # Check if embeddings already exist and are current
    cache_path = Path(getattr(config.data, 'embeddings_cache_path', './data/embeddings/'))
    
    needs_rebuild = force
    
    if not needs_rebuild:
        # Check if vector database is already initialized
        if (hasattr(data_manager, 'vector_db_manager') and 
            data_manager.vector_db_manager and 
            data_manager.vector_db_manager.is_initialized()):
            stats = data_manager.vector_db_manager.get_stats()
            current_docs = len(data_manager.papers)
            indexed_docs = stats.get('num_documents', 0)
            
            if indexed_docs == current_docs:
                logger.info(f"Embeddings index is up to date ({current_docs:,} documents)")
                return
            else:
                logger.info(f"Document count mismatch: {indexed_docs} indexed vs {current_docs} current")
                needs_rebuild = True
        else:
            needs_rebuild = True
    
    if needs_rebuild:
        logger.info(f"Building embeddings for {len(data_manager.papers)} documents...")
        
        # Force rebuild embeddings
        await data_manager._build_embeddings(force_rebuild=True)
        
        logger.info("Embeddings index built successfully")



