"""
Sparse search implementations supporting both TF-IDF and BM25 algorithms.
This module provides keyword-based search using different scoring methods.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from abc import ABC, abstractmethod

# Try to import rank_bm25, fallback to custom implementation if not available
try:
    from rank_bm25 import BM25Okapi, BM25L, BM25Plus
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False


class BaseSparseSearcher(ABC):
    """Abstract base class for sparse search implementations."""
    
    def __init__(self):
        self.documents = []
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def build_index(self, documents: List[Dict]):
        """Build search index from documents."""
        pass
        
    @abstractmethod
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """Search documents and return ranked results."""
        pass
        
    def _extract_text_content(self, documents: List[Dict]) -> List[str]:
        """Extract and combine text content from documents."""
        texts = []
        for doc in documents:
            text_parts = []
            if 'title' in doc and doc['title']:
                text_parts.append(doc['title'])
            if 'abstract' in doc and doc['abstract']:
                text_parts.append(doc['abstract'])
            if 'authors' in doc and doc['authors']:
                text_parts.append(doc['authors'])
            if 'subjects' in doc and doc['subjects']:
                text_parts.append(doc['subjects'])
                
            combined_text = ' '.join(text_parts)
            texts.append(combined_text)
        return texts


class TFIDFSearcher(BaseSparseSearcher):
    """
    TF-IDF based sparse search using scikit-learn's TfidfVectorizer.
    Searches documents based on term frequency-inverse document frequency.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def build_index(self, documents: List[Dict]):
        """
        Build TF-IDF index from documents for keyword-based search.
        
        Args:
            documents: List of document dictionaries with text fields
        """
        self.logger.info(f"Building TF-IDF sparse search index for {len(documents)} documents")
        
        # Extract text content
        texts = self._extract_text_content(documents)
        
        # Build TF-IDF vectorizer and matrix with configurable parameters
        vectorizer_params = {
            'max_features': self.config.get('max_features', 10000),
            'stop_words': self.config.get('stop_words', 'english'),
            'ngram_range': tuple(self.config.get('ngram_range', [1, 2])),
            'min_df': self.config.get('min_df', 2),
            'max_df': self.config.get('max_df', 0.8)
        }
        
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.documents = documents
        
        self.logger.info(f"TF-IDF index built with {self.tfidf_matrix.shape[1]} features")
        
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """
        Search documents using TF-IDF cosine similarity.
        
        Args:
            query: Search query string
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching documents with TF-IDF scores
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            self.logger.error("TF-IDF index not built")
            return []
            
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= k:
                break
            score = similarities[idx]
            if score >= threshold:
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'search_type': 'tfidf',
                    'index': int(idx)
                })
        
        self.logger.info(f"TF-IDF search found {len(results)} results above threshold {threshold}")
        return results


class BM25Searcher(BaseSparseSearcher):
    """
    BM25 based sparse search implementation using the rank_bm25 library.
    Falls back to custom implementation if rank_bm25 is not available.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.bm25 = None
        self.processed_docs = []
        
        # BM25 parameters
        self.k1 = self.config.get('k1', 1.2)
        self.b = self.config.get('b', 0.75)
        
        # BM25 variant selection
        self.variant = self.config.get('variant', 'okapi')  # okapi, l, plus
        
        if not RANK_BM25_AVAILABLE:
            self.logger.warning("rank_bm25 library not available, using basic implementation")
        
    def build_index(self, documents: List[Dict]):
        """
        Build BM25 index from documents using rank_bm25 library.
        
        Args:
            documents: List of document dictionaries with text fields
        """
        self.logger.info(f"Building BM25 sparse search index for {len(documents)} documents")
        self.logger.info(f"BM25 parameters: k1={self.k1}, b={self.b}, variant={self.variant}")
        
        # Extract and process text content
        self.documents = documents
        texts = self._extract_text_content(documents)
        
        if RANK_BM25_AVAILABLE:
            # Use rank_bm25 library for efficient implementation
            import re
            import string
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            # Try to download NLTK data if needed
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                use_nltk = True
            except:
                use_nltk = False
                self.logger.warning("NLTK not available, using basic tokenization")
            
            # Tokenize and preprocess documents
            self.processed_docs = []
            stop_words = set(stopwords.words('english')) if use_nltk else set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            
            for text in texts:
                # Basic preprocessing
                text = text.lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                
                if use_nltk:
                    tokens = word_tokenize(text)
                else:
                    tokens = text.split()
                
                # Remove stop words and short tokens
                tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
                
                self.processed_docs.append(tokens)
            
            # Initialize BM25 model based on variant
            if self.variant == 'l':
                self.bm25 = BM25L(self.processed_docs, k1=self.k1, b=self.b)
            elif self.variant == 'plus':
                self.bm25 = BM25Plus(self.processed_docs, k1=self.k1, b=self.b)
            else:  # okapi (default)
                self.bm25 = BM25Okapi(self.processed_docs, k1=self.k1, b=self.b)
            
            self.logger.info(f"BM25 index built with {len(self.processed_docs)} documents using rank_bm25")
            
        else:
            # Fallback to basic implementation (for compatibility)
            self._build_basic_bm25_index(texts)
            
    def _build_basic_bm25_index(self, texts: List[str]):
        """Fallback BM25 implementation when rank_bm25 is not available."""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Build count vectorizer for BM25
        vectorizer_params = {
            'max_features': self.config.get('max_features', 10000),
            'stop_words': self.config.get('stop_words', 'english'),
            'ngram_range': tuple(self.config.get('ngram_range', [1, 2])),
            'min_df': self.config.get('min_df', 2),
            'max_df': self.config.get('max_df', 0.8),
            'lowercase': True,
            'token_pattern': r'\b[a-zA-Z][a-zA-Z]+\b'
        }
        
        self.vectorizer = CountVectorizer(**vectorizer_params)
        self.term_matrix = self.vectorizer.fit_transform(texts)
        self.vocabulary = self.vectorizer.vocabulary_
        
        # Compute IDF values
        N = self.term_matrix.shape[0]
        df = np.array(self.term_matrix.sum(axis=0)).flatten()
        self.idf = np.log((N - df + 0.5) / (df + 0.5))
        
        # Compute document lengths
        self.doc_lengths = np.array(self.term_matrix.sum(axis=1)).flatten()
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        self.logger.info(f"Basic BM25 index built with {self.term_matrix.shape[1]} features")
        
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """
        Search documents using BM25 scoring.
        
        Args:
            query: Search query string
            k: Number of results to return
            threshold: Minimum BM25 score threshold
            
        Returns:
            List of matching documents with BM25 scores
        """
        if RANK_BM25_AVAILABLE and self.bm25 is not None:
            return self._search_with_rank_bm25(query, k, threshold)
        elif hasattr(self, 'vectorizer') and self.vectorizer is not None:
            return self._search_with_basic_bm25(query, k, threshold)
        else:
            self.logger.error("BM25 index not built")
            return []
            
    def _search_with_rank_bm25(self, query: str, k: int, threshold: float) -> List[Dict]:
        """Search using rank_bm25 library."""
        import string
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        try:
            use_nltk = True
            stop_words = set(stopwords.words('english'))
        except:
            use_nltk = False
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Preprocess query the same way as documents
        query = query.lower()
        query = query.translate(str.maketrans('', '', string.punctuation))
        
        if use_nltk:
            query_tokens = word_tokenize(query)
        else:
            query_tokens = query.split()
            
        query_tokens = [token for token in query_tokens if token not in stop_words and len(token) > 2]
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= k:
                break
            score = scores[idx]
            if score >= threshold:
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'search_type': 'bm25',
                    'index': int(idx)
                })
        
        self.logger.info(f"BM25 search found {len(results)} results above threshold {threshold}")
        return results
        
    def _search_with_basic_bm25(self, query: str, k: int, threshold: float) -> List[Dict]:
        """Search using basic BM25 implementation."""
        # Transform query to count vector
        query_vector = self.vectorizer.transform([query])
        query_terms = query_vector.toarray()[0]
        
        # Calculate BM25 scores for all documents
        scores = np.zeros(self.term_matrix.shape[0])
        
        for term_idx, query_term_freq in enumerate(query_terms):
            if query_term_freq > 0:
                term_freqs = np.array(self.term_matrix[:, term_idx]).flatten()
                idf_component = self.idf[term_idx]
                tf_component = (term_freqs * (self.k1 + 1)) / (
                    term_freqs + self.k1 * (1 - self.b + self.b * (self.doc_lengths / self.avg_doc_length))
                )
                scores += idf_component * tf_component * query_term_freq
        
        # Get top results
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= k:
                break
            score = scores[idx]
            if score >= threshold:
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'search_type': 'bm25',
                    'index': int(idx)
                })
        
        self.logger.info(f"Basic BM25 search found {len(results)} results above threshold {threshold}")
        return results


class SparseSearcher:
    """
    Unified interface for sparse search supporting both TF-IDF and BM25.
    Automatically selects the appropriate searcher based on configuration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.algorithm = self.config.get('algorithm', 'tfidf').lower()
        self.logger = logging.getLogger(__name__)
        
        # Initialize the appropriate searcher
        if self.algorithm == 'bm25':
            self.searcher = BM25Searcher(self.config.get('bm25', {}))
            self.logger.info("Initialized BM25 sparse searcher")
        elif self.algorithm == 'tfidf':
            self.searcher = TFIDFSearcher(self.config.get('tfidf', {}))
            self.logger.info("Initialized TF-IDF sparse searcher")
        else:
            self.logger.warning(f"Unknown algorithm '{self.algorithm}', defaulting to TF-IDF")
            self.searcher = TFIDFSearcher(self.config.get('tfidf', {}))
            self.algorithm = 'tfidf'
    
    def build_index(self, documents: List[Dict]):
        """Build search index using the configured algorithm."""
        self.searcher.build_index(documents)
        
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """Search documents using the configured algorithm."""
        return self.searcher.search(query, k, threshold)
        
    def save_index(self, save_path: str):
        """Save the sparse search index to disk."""
        import os
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the searcher type and algorithm info
        with open(save_path / "sparse_info.pkl", 'wb') as f:
            pickle.dump({
                'algorithm': self.algorithm,
                'config': self.config
            }, f)
        
        # Save algorithm-specific components
        if self.algorithm == 'bm25':
            if RANK_BM25_AVAILABLE and hasattr(self.searcher, 'bm25') and self.searcher.bm25:
                # Save rank_bm25 model
                with open(save_path / "bm25_model.pkl", 'wb') as f:
                    pickle.dump(self.searcher.bm25, f)
                with open(save_path / "bm25_processed_docs.pkl", 'wb') as f:
                    pickle.dump(self.searcher.processed_docs, f)
                with open(save_path / "bm25_variant.pkl", 'wb') as f:
                    pickle.dump(self.searcher.variant, f)
            else:
                # Save basic BM25 components
                if hasattr(self.searcher, 'vectorizer'):
                    with open(save_path / "bm25_vectorizer.pkl", 'wb') as f:
                        pickle.dump(self.searcher.vectorizer, f)
                    with open(save_path / "bm25_term_matrix.pkl", 'wb') as f:
                        pickle.dump(self.searcher.term_matrix, f)
                    with open(save_path / "bm25_idf.pkl", 'wb') as f:
                        pickle.dump(self.searcher.idf, f)
                    with open(save_path / "bm25_doc_lengths.pkl", 'wb') as f:
                        pickle.dump(self.searcher.doc_lengths, f)
                    with open(save_path / "bm25_avg_doc_length.pkl", 'wb') as f:
                        pickle.dump(self.searcher.avg_doc_length, f)
        else:  # tfidf
            with open(save_path / "tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.searcher.vectorizer, f)
            with open(save_path / "tfidf_matrix.pkl", 'wb') as f:
                pickle.dump(self.searcher.tfidf_matrix, f)
                
        with open(save_path / "sparse_documents.pkl", 'wb') as f:
            pickle.dump(self.searcher.documents, f)
            
        self.logger.info(f"Sparse search index ({self.algorithm}) saved to: {save_path}")
        
    def load_index(self, load_path: str) -> bool:
        """Load sparse search index from disk."""
        from pathlib import Path
        
        load_path = Path(load_path)
        
        try:
            # Load sparse info
            with open(load_path / "sparse_info.pkl", 'rb') as f:
                info = pickle.load(f)
                loaded_algorithm = info['algorithm']
                
            # Verify algorithm matches
            if loaded_algorithm != self.algorithm:
                self.logger.warning(f"Algorithm mismatch: expected {self.algorithm}, found {loaded_algorithm}")
                return False
            
            # Load documents
            with open(load_path / "sparse_documents.pkl", 'rb') as f:
                self.searcher.documents = pickle.load(f)
            
            # Load algorithm-specific components
            if self.algorithm == 'bm25':
                # Try to load rank_bm25 model first
                if (load_path / "bm25_model.pkl").exists() and RANK_BM25_AVAILABLE:
                    with open(load_path / "bm25_model.pkl", 'rb') as f:
                        self.searcher.bm25 = pickle.load(f)
                    with open(load_path / "bm25_processed_docs.pkl", 'rb') as f:
                        self.searcher.processed_docs = pickle.load(f)
                    with open(load_path / "bm25_variant.pkl", 'rb') as f:
                        self.searcher.variant = pickle.load(f)
                # Fallback to basic BM25 components
                elif (load_path / "bm25_vectorizer.pkl").exists():
                    with open(load_path / "bm25_vectorizer.pkl", 'rb') as f:
                        self.searcher.vectorizer = pickle.load(f)
                    with open(load_path / "bm25_term_matrix.pkl", 'rb') as f:
                        self.searcher.term_matrix = pickle.load(f)
                    with open(load_path / "bm25_idf.pkl", 'rb') as f:
                        self.searcher.idf = pickle.load(f)
                    with open(load_path / "bm25_doc_lengths.pkl", 'rb') as f:
                        self.searcher.doc_lengths = pickle.load(f)
                    with open(load_path / "bm25_avg_doc_length.pkl", 'rb') as f:
                        self.searcher.avg_doc_length = pickle.load(f)
            else:  # tfidf
                with open(load_path / "tfidf_vectorizer.pkl", 'rb') as f:
                    self.searcher.vectorizer = pickle.load(f)
                with open(load_path / "tfidf_matrix.pkl", 'rb') as f:
                    self.searcher.tfidf_matrix = pickle.load(f)
                    
            self.logger.info(f"Sparse search index ({self.algorithm}) loaded from: {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load sparse index: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the sparse search index."""
        stats = {
            'algorithm': self.algorithm,
            'num_documents': len(self.searcher.documents)
        }
        
        if self.algorithm == 'bm25':
            if RANK_BM25_AVAILABLE and hasattr(self.searcher, 'bm25') and self.searcher.bm25:
                stats.update({
                    'bm25_variant': getattr(self.searcher, 'variant', 'okapi'),
                    'bm25_k1': self.searcher.k1,
                    'bm25_b': self.searcher.b,
                    'library': 'rank_bm25'
                })
            elif hasattr(self.searcher, 'term_matrix') and self.searcher.term_matrix is not None:
                stats.update({
                    'num_features': self.searcher.term_matrix.shape[1],
                    'avg_doc_length': getattr(self.searcher, 'avg_doc_length', 0),
                    'bm25_k1': self.searcher.k1,
                    'bm25_b': self.searcher.b,
                    'library': 'custom'
                })
        elif self.algorithm == 'tfidf' and hasattr(self.searcher, 'tfidf_matrix') and self.searcher.tfidf_matrix is not None:
            stats.update({
                'num_features': self.searcher.tfidf_matrix.shape[1]
            })
            
        return stats
