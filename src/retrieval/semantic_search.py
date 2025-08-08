"""
Semantic search implementation using vector similarity.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from .faiss_store import FaissStore
from .embeddings_manager import EmbeddingsManager


class SemanticSearcher:
    """Semantic search using vector similarity."""
    
    def __init__(self, embeddings_manager: EmbeddingsManager = None, vector_store: FaissStore = None):
        """Initialize semantic searcher.
        
        Args:
            embeddings_manager: Manager for creating embeddings
            vector_store: Vector store for similarity search
        """
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()
        self.vector_store = vector_store or FaissStore()
        self.logger = logging.getLogger(__name__)
        
    def build_index(self, documents: List[Dict], text_fields: List[str] = None):
        """Build search index from documents.
        
        Args:
            documents: List of document dictionaries
            text_fields: Fields to use for embedding
        """
        self.logger.info("Building semantic search index")
        
        # Create embeddings
        embeddings, processed_docs = self.embeddings_manager.process_documents(documents, text_fields)
        
        # Build vector store
        self.vector_store.build_index(embeddings, processed_docs)
        
        self.logger.info("Semantic search index built successfully")
        
    def search(self, query: str, k: int = 10, threshold: float = 0.1) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching documents with scores
        """
        self.logger.info(f"Searching for: {query}")
        
        # Create query embedding
        query_embedding = self.embeddings_manager.create_query_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k, threshold)
        
        self.logger.info(f"Found {len(results)} matching documents")
        return results
        
    def search_by_author(self, author_name: str, k: int = 10) -> List[Dict]:
        """Search for papers by a specific author.
        
        Args:
            author_name: Name of the author
            k: Number of results to return
            
        Returns:
            List of papers by the author
        """
        results = []
        author_lower = author_name.lower()
        
        for doc in self.vector_store.documents:
            authors = doc.get('authors', '')
            if isinstance(authors, str) and author_lower in authors.lower():
                results.append({
                    'document': doc,
                    'score': 1.0,  # Exact match
                    'match_type': 'author'
                })
                
        # Sort by date if available
        results.sort(key=lambda x: x['document'].get('date', ''), reverse=True)
        
        self.logger.info(f"Found {len(results)} papers by author: {author_name}")
        return results[:k]
        
    def search_by_category(self, category: str, k: int = 10) -> List[Dict]:
        """Search for papers in a specific category.
        
        Args:
            category: Research category/subject
            k: Number of results to return
            
        Returns:
            List of papers in the category
        """
        results = []
        category_lower = category.lower()
        
        for doc in self.vector_store.documents:
            subjects = doc.get('subjects', '')
            if isinstance(subjects, str) and category_lower in subjects.lower():
                results.append({
                    'document': doc,
                    'score': 1.0,  # Category match
                    'match_type': 'category'
                })
                
        self.logger.info(f"Found {len(results)} papers in category: {category}")
        return results[:k]
        
    def search_similar_papers(self, paper_id: str, k: int = 10) -> List[Dict]:
        """Find papers similar to a given paper.
        
        Args:
            paper_id: ID of the reference paper
            k: Number of results to return
            
        Returns:
            List of similar papers
        """
        # Find the reference paper
        ref_doc = None
        ref_index = None
        
        for i, doc in enumerate(self.vector_store.documents):
            if doc.get('id') == paper_id or doc.get('title') == paper_id:
                ref_doc = doc
                ref_index = i
                break
                
        if ref_doc is None:
            self.logger.warning(f"Reference paper not found: {paper_id}")
            return []
            
        # Use the paper's embedding for similarity search
        if hasattr(self.vector_store, 'index') and self.vector_store.index is not None:
            # Get the embedding from the index
            ref_embedding = self.vector_store.index.reconstruct(ref_index).reshape(1, -1)
            results = self.vector_store.search(ref_embedding, k + 1)  # +1 to exclude self
            
            # Filter out the reference paper itself
            results = [r for r in results if r['index'] != ref_index]
            
            self.logger.info(f"Found {len(results)} similar papers to: {paper_id}")
            return results[:k]
        else:
            self.logger.error("Vector store index not available")
            return []
            
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index.
        
        Returns:
            Dictionary with search statistics
        """
        return self.vector_store.get_stats()
        
    def save_index(self, save_path: str, save_raw_embeddings: bool = True):
        """Save search index to disk.
        
        Args:
            save_path: Directory to save index
            save_raw_embeddings: Whether to save raw embeddings as .npy file
        """
        # Save FAISS index and documents (with optional raw embeddings)
        self.vector_store.save(save_path, save_raw_embeddings=save_raw_embeddings)
        
        self.logger.info(f"Search index saved to: {save_path}")
        if save_raw_embeddings:
            self.logger.info(f"Raw embeddings saved to: {save_path}/embeddings.npy")
        
    def load_index(self, load_path: str) -> bool:
        """Load search index from disk.
        
        Args:
            load_path: Directory containing saved index
            
        Returns:
            True if loaded successfully
        """
        success = self.vector_store.load(load_path)
        if success:
            self.logger.info(f"Search index loaded from: {load_path}")
        return success



