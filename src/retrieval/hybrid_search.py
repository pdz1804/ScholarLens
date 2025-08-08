"""
Hybrid search implementation combining sparse (keyword-based) and dense (semantic) search.
This module provides unified search interface that fuses results from both approaches.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from .semantic_search import SemanticSearcher
from .sparse_search import SparseSearcher
from .embeddings_manager import EmbeddingsManager


class HybridSearcher:
    """
    Hybrid search combining sparse (keyword) and dense (semantic) search with result fusion.
    Provides comprehensive search by combining both approaches and ranking results.
    """
    
    def __init__(self, embeddings_manager: EmbeddingsManager = None, sparse_config: Dict[str, Any] = None):
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()
        self.semantic_searcher = SemanticSearcher(self.embeddings_manager)
        self.sparse_searcher = SparseSearcher(sparse_config or {})
        self.logger = logging.getLogger(__name__)
        
    def build_index(self, documents: List[Dict], text_fields: List[str] = None):
        """
        Build both sparse and dense search indices from documents.
        
        Args:
            documents: List of document dictionaries
            text_fields: Fields to use for dense embedding (default: ['title', 'abstract'])
        """
        self.logger.info(f"Building hybrid search index for {len(documents)} documents")
        
        # Build dense (semantic) index
        self.semantic_searcher.build_index(documents, text_fields)
        
        # Build sparse (keyword) index
        self.sparse_searcher.build_index(documents)
        
        self.logger.info("Hybrid search index built successfully")
        
    def search(self, query: str, k: int = 10, alpha: float = 0.7, score_threshold: float = 0.0) -> List[Dict]:
        """
        Perform hybrid search combining sparse and dense results.
        
        Args:
            query: Search query string
            k: Number of results to return after filtering
            alpha: Weight for dense search (1-alpha for sparse). Higher alpha favors semantic search.
            score_threshold: Minimum score threshold to include results
            
        Returns:
            List of fused and ranked documents above score threshold
        """
        self.logger.info(f"Performing hybrid search for: {query}")
        self.logger.info(f"Search parameters: k={k}, alpha={alpha}, score_threshold={score_threshold}")
        
        # Get more results initially for better fusion, then filter
        initial_k = max(k * 3, 100)  # Get 3x more results for fusion
        
        # Get results from both search methods
        dense_results = self.semantic_searcher.search(query, k=initial_k)
        sparse_results = self.sparse_searcher.search(query, k=initial_k)
        
        # Create document ID to result mapping for fusion
        dense_scores = {}
        sparse_scores = {}
        all_docs = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = self._get_doc_id(result['document'])
            dense_scores[doc_id] = result['score']
            all_docs[doc_id] = result
            
        # Process sparse results
        for result in sparse_results:
            doc_id = self._get_doc_id(result['document'])
            sparse_scores[doc_id] = result['score']
            if doc_id not in all_docs:
                all_docs[doc_id] = result
        
        # Normalize scores to [0, 1] range
        if dense_scores:
            max_dense = max(dense_scores.values())
            if max_dense > 0:
                dense_scores = {k: v/max_dense for k, v in dense_scores.items()}
                
        if sparse_scores:
            max_sparse = max(sparse_scores.values())
            if max_sparse > 0:
                sparse_scores = {k: v/max_sparse for k, v in sparse_scores.items()}
        
        # Compute hybrid scores using weighted combination
        hybrid_results = []
        for doc_id, doc_result in all_docs.items():
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)
            
            # Weighted combination of scores
            hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
            
            # Determine search type that contributed most
            search_type = 'dense' if dense_score > sparse_score else 'sparse'
            if dense_score > 0 and sparse_score > 0:
                search_type = 'hybrid'
            
            hybrid_results.append({
                'document': doc_result['document'],
                'score': hybrid_score,
                'dense_score': dense_score,
                'sparse_score': sparse_score,
                'search_type': search_type,
                'index': doc_result.get('index', 0)
            })
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter by score threshold
        if score_threshold > 0:
            filtered_results = [r for r in hybrid_results if r['score'] >= score_threshold]
            self.logger.info(f"Filtered {len(hybrid_results)} results to {len(filtered_results)} above threshold {score_threshold}")
        else:
            filtered_results = hybrid_results
        
        # Return top k results
        final_results = filtered_results[:k]
        
        # Log search statistics
        if final_results:
            self.logger.info(f"Hybrid search returned {len(final_results)} results")
            score_stats = {
                'min_score': min(r['score'] for r in final_results),
                'max_score': max(r['score'] for r in final_results),
                'avg_score': sum(r['score'] for r in final_results) / len(final_results)
            }
            self.logger.info(f"Score range: {score_stats['min_score']:.3f} - {score_stats['max_score']:.3f} (avg: {score_stats['avg_score']:.3f})")
            
            # Count search types
            search_types = {}
            for result in final_results:
                stype = result['search_type']
                search_types[stype] = search_types.get(stype, 0) + 1
            self.logger.info(f"Search type distribution: {search_types}")
        else:
            self.logger.warning(f"No results found above score threshold {score_threshold}")
        
        return final_results
        
    def _get_doc_id(self, document: Dict) -> str:
        """
        Get unique identifier for a document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Unique document identifier
        """
        # Use paper_id if available, otherwise use title
        return document.get('id') or document.get('paper_id') or document.get('title', '')
        
    def search_by_author(self, author_name: str, k: int = 10) -> List[Dict]:
        """
        Search for papers by specific author using hybrid approach.
        
        Args:
            author_name: Name of the author
            k: Number of results to return
            
        Returns:
            List of papers by the author
        """
        # Use semantic searcher for author-based search as it handles this well
        return self.semantic_searcher.search_by_author(author_name, k)
        
    def search_by_category(self, category: str, k: int = 10) -> List[Dict]:
        """
        Search for papers in specific category using hybrid approach.
        
        Args:
            category: Research category/subject
            k: Number of results to return
            
        Returns:
            List of papers in the category
        """
        # Combine both approaches for category search
        query = f"research in {category} {category} papers"
        return self.search(query, k=k, alpha=0.5)  # Equal weight for category search
        
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid search system.
        
        Returns:
            Dictionary with search system statistics
        """
        semantic_stats = self.semantic_searcher.get_search_stats()
        sparse_stats = self.sparse_searcher.get_stats()
        
        return {
            **semantic_stats,
            **sparse_stats,
            'search_type': 'hybrid'
        }
        
    def save_index(self, save_path: str):
        """
        Save hybrid search indices to disk.
        
        Args:
            save_path: Directory to save indices
        """
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save semantic index
        self.semantic_searcher.save_index(str(save_path / "semantic"))
        
        # Save sparse index using new sparse searcher
        self.sparse_searcher.save_index(str(save_path / "sparse"))
        
        self.logger.info(f"Hybrid search index saved to: {save_path}")
        
    def load_index(self, load_path: str) -> bool:
        """
        Load hybrid search indices from disk.
        
        Args:
            load_path: Directory containing saved indices
            
        Returns:
            True if loaded successfully
        """
        from pathlib import Path
        
        load_path = Path(load_path)
        
        try:
            # Load semantic index
            semantic_success = self.semantic_searcher.load_index(str(load_path / "semantic"))
            
            # Load sparse index using new sparse searcher
            sparse_success = self.sparse_searcher.load_index(str(load_path / "sparse"))
            
            success = semantic_success and sparse_success
            
            if success:
                self.logger.info(f"Hybrid search index loaded from: {load_path}")
            else:
                self.logger.warning("Failed to load complete hybrid index")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to load hybrid index: {str(e)}")
            return False
