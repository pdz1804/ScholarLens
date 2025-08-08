"""
Vector store implementation for document embeddings.
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

# Small note: Embedded in FAISS Index
# The embeddings are stored inside the index.faiss file itself. FAISS stores the vectors internally.

class FaissStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """Initialize vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
        self.embeddings = None  # Store raw embeddings
        self.logger = logging.getLogger(__name__)
        
    def build_index(self, embeddings: np.ndarray, documents: List[Dict], metadata: List[Dict] = None):
        """Build FAISS index from embeddings.
        
        Args:
            embeddings: Document embeddings array
            documents: List of document dictionaries
            metadata: Optional metadata for each document
        """
        self.logger.info(f"Building FAISS index with {len(embeddings)} documents")
        
        # Store raw embeddings
        self.embeddings = embeddings.copy()
        
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, len(embeddings) // 10))
            self.index.train(embeddings.astype(np.float32))
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype(np.float32))
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        
        self.logger.info(f"Index built successfully with {self.index.ntotal} vectors")
        
    def search(self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching documents with scores
        """
        if self.index is None:
            self.logger.error("Index not built. Call build_index first")
            return []
            
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:
                result = {
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'index': int(idx)
                }
                results.append(result)
                
        self.logger.info(f"Found {len(results)} documents above threshold {threshold}")
        return results
        
    def save(self, save_path: str, save_raw_embeddings: bool = False):
        """Save index and metadata to disk.
        
        Args:
            save_path: Directory to save index files
            save_raw_embeddings: Whether to save raw embeddings as .npy file
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, str(save_path / "index.faiss"))
            
            # Save raw embeddings if requested
            if save_raw_embeddings and self.embeddings is not None:
                np.save(save_path / "embeddings.npy", self.embeddings)
                self.logger.info(f"Raw embeddings saved to {save_path / 'embeddings.npy'}")
            
            # Save documents and metadata
            with open(save_path / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
                
            with open(save_path / "metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata, f)
                
            # Save configuration
            config = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'num_documents': len(self.documents)
            }
            with open(save_path / "config.pkl", 'wb') as f:
                pickle.dump(config, f)
                
            self.logger.info(f"Vector store saved to {save_path}")
        else:
            self.logger.warning("No index to save")
            
    def load(self, load_path: str) -> bool:
        """Load index and metadata from disk.
        
        Args:
            load_path: Directory containing index files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = Path(load_path)
        
        try:
            # Load configuration
            with open(load_path / "config.pkl", 'rb') as f:
                config = pickle.load(f)
                
            self.dimension = config['dimension']
            self.index_type = config['index_type']
            
            # Load FAISS index
            self.index = faiss.read_index(str(load_path / "index.faiss"))
            
            # Load raw embeddings if available
            embeddings_path = load_path / "embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
                self.logger.info(f"Raw embeddings loaded from {embeddings_path}")
            
            # Load documents and metadata
            with open(load_path / "documents.pkl", 'rb') as f:
                self.documents = pickle.load(f)
                
            with open(load_path / "metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
                
            self.logger.info(f"Vector store loaded from {load_path}")
            self.logger.info(f"Loaded {len(self.documents)} documents with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if self.index is None:
            return {'status': 'not_built'}
            
        return {
            'status': 'ready',
            'num_documents': len(self.documents),
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
