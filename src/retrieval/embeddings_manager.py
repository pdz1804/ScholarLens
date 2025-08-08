"""
Embeddings manager for creating and managing document embeddings.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm


class EmbeddingsManager:
    """Manages document embeddings creation and storage."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embeddings manager.
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            self.logger.info(f"Loading embeddings model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embeddings model loaded successfully")
            
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        self.load_model()
        
        self.logger.info(f"Creating embeddings for {len(texts)} texts")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(embeddings)
        self.logger.info(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings
        
    def process_documents(self, documents: List[Dict], text_fields: List[str] = None) -> tuple:
        """Process documents to create embeddings.
        
        Args:
            documents: List of document dictionaries
            text_fields: Fields to use for embedding (default: ['title', 'abstract'])
            
        Returns:
            Tuple of (embeddings, processed_documents)
        """
        if text_fields is None:
            text_fields = ['title', 'abstract']
            
        self.logger.info(f"Processing {len(documents)} documents")
        
        # Extract text for embeddings
        texts = []
        processed_docs = []
        
        for doc in documents:
            # Combine specified text fields
            text_parts = []
            for field in text_fields:
                if field in doc and doc[field]:
                    text_parts.append(str(doc[field]))
                    
            if text_parts:
                combined_text = " ".join(text_parts)
                texts.append(combined_text)
                processed_docs.append(doc)
            else:
                self.logger.warning(f"Document missing text fields: {text_fields}")
                
        self.logger.info(f"Extracted text from {len(texts)} documents")
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        return embeddings, processed_docs
        
    def process_csv_dataset(self, csv_path: str, text_fields: List[str] = None) -> tuple:
        """Process CSV dataset to create embeddings.
        
        Args:
            csv_path: Path to CSV file
            text_fields: Fields to use for embedding
            
        Returns:
            Tuple of (embeddings, documents)
        """
        self.logger.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(df)} rows from dataset")
        
        # Convert to list of dictionaries
        documents = df.to_dict('records')
        
        # Process documents
        embeddings, processed_docs = self.process_documents(documents, text_fields)
        
        return embeddings, processed_docs
        
    def save_embeddings(self, embeddings: np.ndarray, documents: List[Dict], save_path: str):
        """Save embeddings and documents to disk.
        
        Args:
            embeddings: Embeddings array
            documents: List of documents
            save_path: Directory to save files
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(save_path / "embeddings.npy", embeddings)
        
        # Save documents
        with open(save_path / "documents.pkl", 'wb') as f:
            pickle.dump(documents, f)
            
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_documents': len(documents),
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
            
        self.logger.info(f"Embeddings saved to {save_path}")
        self.logger.info(f"Saved {len(documents)} documents with {embeddings.shape} embeddings")
        
    def load_embeddings(self, load_path: str) -> tuple:
        """Load embeddings and documents from disk.
        
        Args:
            load_path: Directory containing saved files
            
        Returns:
            Tuple of (embeddings, documents, metadata)
        """
        load_path = Path(load_path)
        
        try:
            # Load embeddings
            embeddings = np.load(load_path / "embeddings.npy")
            
            # Load documents
            with open(load_path / "documents.pkl", 'rb') as f:
                documents = pickle.load(f)
                
            # Load metadata
            with open(load_path / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                
            self.logger.info(f"Loaded embeddings from {load_path}")
            self.logger.info(f"Loaded {len(documents)} documents with {embeddings.shape} embeddings")
            
            return embeddings, documents, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {str(e)}")
            return None, None, None
            
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding
        """
        self.load_model()
        return self.model.encode([query])[0]



