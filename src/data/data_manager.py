"""
Data management for TechAuthor system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
import logging
import hashlib
from tqdm import tqdm

from ..core.models import Paper, DatasetInfo
from ..core.config import config_manager
from ..utils.logger import setup_logger
from ..retrieval.embeddings_manager import EmbeddingsManager
from ..retrieval.hybrid_search import HybridSearcher
from .processed_data_manager import ProcessedDataManager


class DataManager:
    """Manages data loading, processing, and storage for the TechAuthor system."""
    
    def __init__(self):
        """Initialize data manager."""
        self.config = config_manager.config
        self.logger = setup_logger("DataManager", self.config.system.log_level)
        
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.dataset_info: Dict[str, DatasetInfo] = {}
        self.papers: List[Paper] = []
        self.authors_index: Dict[str, List[int]] = {}  # author -> paper indices
        self.domain_index: Dict[str, List[int]] = {}   # domain -> paper indices
        self.subject_index: Dict[str, List[int]] = {}  # subject -> paper indices
        
        # Retrieval components - hybrid search only
        self.embeddings_manager = EmbeddingsManager()
        
        # Get sparse search configuration from config
        sparse_config = self.config.retrieval.sparse_search or {}
        
        # Hybrid searcher for comprehensive search capabilities
        self.hybrid_searcher = HybridSearcher(self.embeddings_manager, sparse_config)
        
        # Processed data manager for caching and storage
        self.processed_data_manager = ProcessedDataManager(
            processed_data_path=self.config.data.processed_data_path
        )
        
        self.is_initialized = False
    
    async def initialize(self, indexing_options: Optional[Dict[str, bool]] = None):
        """Initialize the data manager with optional indexing control.
        
        Args:
            indexing_options: Dictionary with indexing control options:
                - force_reindex: Force complete rebuild of all indices
                - update_index: Update indices incrementally
                - clear_cache: Clear cache before initialization
                - test_mode: Use only 1% of the dataset for testing
        """
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Initializing data manager")
            
            # Handle indexing options
            if indexing_options is None:
                indexing_options = {}
            
            force_reindex = indexing_options.get('force_reindex', False)
            update_index = indexing_options.get('update_index', False)
            clear_cache = indexing_options.get('clear_cache', False)
            test_mode = indexing_options.get('test_mode', False)
            
            if force_reindex:
                self.logger.info("Force reindex enabled - will rebuild all indices from scratch")
            elif update_index:
                self.logger.info("Update index enabled - will update indices incrementally")
            elif clear_cache:
                self.logger.info("Clear cache enabled - cleared cache before initialization")
            
            if test_mode:
                self.logger.info("TEST MODE enabled - using only 1% of dataset for testing")
            
            # Load datasets
            await self._load_datasets(test_mode=test_mode)
            
            # Process data
            await self._process_data()
            
            # Build indices
            await self._build_indices()
            
            # Build embeddings and semantic search with indexing options
            await self._build_embeddings(force_rebuild=force_reindex, update_only=update_index)
            
            self.is_initialized = True
            self.logger.info(f"Data manager initialized successfully with {len(self.papers)} papers")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data manager: {str(e)}")
            raise
    
    async def _load_datasets(self, test_mode: bool = False):
        """Load datasets from configuration.
        
        Args:
            test_mode: If True, sample only 1% of the data for testing
        """
        datasets_config = self.config.data.datasets
        
        for dataset_name, dataset_config in datasets_config.items():
            try:
                # Check if this is the new multi-file format
                if 'files' in dataset_config:
                    # New format with multiple files
                    self.logger.info(f"Loading multi-file dataset: {dataset_name}")
                    combined_df = pd.DataFrame()
                    
                    for file_config in dataset_config['files']:
                        dataset_path = Path(file_config['path'])
                        year = file_config.get('year', 'unknown')
                        
                        if not dataset_path.exists():
                            self.logger.warning(f"Dataset file not found: {dataset_path}")
                            continue
                        
                        self.logger.info(f"  Loading {year} data from: {dataset_path}")
                        
                        # Load CSV file
                        df = pd.read_csv(dataset_path)
                        
                        # Add year column for tracking
                        df['source_year'] = year
                        df['source_file'] = str(dataset_path)
                        
                        # Combine with existing data
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
                    
                    if not combined_df.empty:
                        # Remove duplicates based on Paper ID
                        initial_count = len(combined_df)
                        paper_id_col = dataset_config.get('columns', {}).get('id', 'Paper ID')
                        combined_df = combined_df.drop_duplicates(subset=[paper_id_col], keep='first')
                        final_count = len(combined_df)
                        
                        if initial_count != final_count:
                            duplicate_count = initial_count - final_count
                            self.logger.info(f"Removed {duplicate_count} duplicate papers based on Paper ID")
                        
                        # Apply test mode sampling if enabled
                        if test_mode:
                            original_count = len(combined_df)
                            sample_size = max(1, int(original_count * 0.01))  # 1% sample, minimum 1 record
                            combined_df = combined_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                            self.logger.info(f"TEST MODE: Sampled {len(combined_df)} papers (1%) from {original_count} total papers")
                        
                        self.datasets[dataset_name] = combined_df
                        
                        # Create dataset info
                        self.dataset_info[dataset_name] = DatasetInfo(
                            name=dataset_name,
                            path=f"Multi-file dataset: {len(dataset_config['files'])} files",
                            total_papers=len(combined_df),
                            date_range=self._get_date_range(combined_df, dataset_config),
                            domains=self._get_unique_values(combined_df, dataset_config.get('columns', {}).get('domain', 'Domain')),
                            subjects=self._get_unique_subjects(combined_df, dataset_config),
                            last_updated=datetime.now()
                        )
                        
                        self.logger.info(f"Loaded {len(combined_df)} papers from {dataset_name} ({len(dataset_config['files'])} files)")
                    else:
                        self.logger.warning(f"No valid files found for dataset: {dataset_name}")
                        
                elif 'path' in dataset_config:
                    # Legacy format with single file
                    dataset_path = Path(dataset_config['path'])
                    
                    if not dataset_path.exists():
                        self.logger.warning(f"Dataset not found: {dataset_path}")
                        continue
                    
                    self.logger.info(f"Loading single-file dataset: {dataset_name}")
                    
                    # Load CSV file
                    df = pd.read_csv(dataset_path)
                    
                    # Apply test mode sampling if enabled
                    if test_mode:
                        original_count = len(df)
                        sample_size = max(1, int(original_count * 0.01))  # 1% sample, minimum 1 record
                        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                        self.logger.info(f"TEST MODE: Sampled {len(df)} papers (1%) from {original_count} total papers")
                    
                    self.datasets[dataset_name] = df
                    
                    # Create dataset info
                    self.dataset_info[dataset_name] = DatasetInfo(
                        name=dataset_name,
                        path=str(dataset_path),
                        total_papers=len(df),
                        date_range=self._get_date_range(df, dataset_config),
                        domains=self._get_unique_values(df, dataset_config.get('columns', {}).get('domain', 'Domain')),
                        subjects=self._get_unique_subjects(df, dataset_config),
                        last_updated=datetime.now()
                    )
                    
                    self.logger.info(f"Loaded {len(df)} papers from {dataset_name}")
                else:
                    self.logger.error(f"Invalid dataset configuration for {dataset_name}: missing 'files' or 'path' key")
                    continue
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue
    
    async def _process_data(self):
        """Process loaded datasets into Paper objects."""
        self.logger.info("Processing data into Paper objects")
        
        # Calculate total number of papers to process
        total_papers = sum(len(df) for df in self.datasets.values())
        
        # Initialize progress bar
        pbar = tqdm(total=total_papers, desc="Processing papers", unit="papers")
        
        try:
            for dataset_name, df in self.datasets.items():
                dataset_config = self.config.data.datasets.get(dataset_name, {})
                columns = dataset_config.get('columns', {})
                
                # Map column names
                title_col = columns.get('title', 'Paper Title')
                id_col = columns.get('paper_id', 'Paper ID')
                authors_col = columns.get('authors', 'Authors')
                abstract_col = columns.get('abstract', 'Abstract')
                domain_col = columns.get('domain', 'Domain')
                primary_subject_col = columns.get('primary_subject', 'Primary Subject')
                subjects_col = columns.get('subjects', 'Subjects')
                date_col = columns.get('date', 'Date Submitted')
                abstract_url_col = columns.get('abstract_url', 'Abstract URL')
                pdf_url_col = columns.get('pdf_url', 'PDF URL')
                
                # Update progress bar description with current dataset
                pbar.set_description(f"Processing {dataset_name}")
                
                for _, row in df.iterrows():
                    try:
                        # Create Paper object
                        paper = Paper(
                            title=str(row[title_col]),
                            paper_id=str(row[id_col]),
                            authors=row[authors_col],
                            abstract=str(row[abstract_col]),
                            domain=str(row[domain_col]),
                            primary_subject=str(row[primary_subject_col]),
                            subjects=row[subjects_col],
                            date_submitted=row[date_col],
                            abstract_url=str(row[abstract_url_col]),
                            pdf_url=str(row[pdf_url_col])
                        )
                        
                        self.papers.append(paper)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process paper {row.get(id_col, 'unknown')}: {e}")
                        continue
                    finally:
                        # Update progress bar
                        pbar.update(1)
                        
        finally:
            # Close progress bar
            pbar.close()
        
        self.logger.info(f"Processed {len(self.papers)} papers")
    
    async def _build_indices(self):
        """Build search indices for efficient querying."""
        self.logger.info("Building search indices")
        
        # Build author index
        for i, paper in enumerate(self.papers):
            for author in paper.authors:
                author_clean = author.strip()
                if author_clean not in self.authors_index:
                    self.authors_index[author_clean] = []
                self.authors_index[author_clean].append(i)
        
        # Build domain index
        for i, paper in enumerate(self.papers):
            domain = paper.domain.strip()
            if domain not in self.domain_index:
                self.domain_index[domain] = []
            self.domain_index[domain].append(i)
        
        # Build subject index
        for i, paper in enumerate(self.papers):
            for subject in paper.subjects:
                subject_clean = subject.strip()
                if subject_clean not in self.subject_index:
                    self.subject_index[subject_clean] = []
                self.subject_index[subject_clean].append(i)
        
        self.logger.info(
            f"Built indices: {len(self.authors_index)} authors, "
            f"{len(self.domain_index)} domains, {len(self.subject_index)} subjects"
        )
    
    async def _build_embeddings(self, force_rebuild: bool = False, update_only: bool = False):
        """
        Build embeddings and vector database index with incremental support.
        
        Args:
            force_rebuild: If True, rebuild all indices from scratch
            update_only: If True, only update with new data (not implemented yet)
        """
        self.logger.info("Building embeddings and vector database index")
        
        # Check if we can load processed data from cache
        cache_key = self._generate_data_cache_key()
        
        if not force_rebuild:
            # Try to load from processed data cache
            cached_papers = self.processed_data_manager.load_papers()
            if cached_papers and len(cached_papers) == len(self.papers):
                self.logger.info(f"Found cached processed papers: {len(cached_papers)}")
                # Verify cache is still valid by comparing a few key attributes
                if self._validate_cached_papers(cached_papers):
                    self.logger.info("Using cached processed papers")
                    # Initialize hybrid searcher from cache
                    await self._initialize_hybrid_searcher_from_cache()
                    return
        
        # Convert papers to documents for hybrid search indexing
        self.logger.info(f"Processing {len(self.papers)} papers for hybrid search index")
        documents = []
        
        for paper in self.papers:
            # Create document for hybrid search
            doc = {
                'id': paper.paper_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'authors': ', '.join(paper.authors) if isinstance(paper.authors, list) else paper.authors,
                'domain': paper.domain,
                'subjects': ', '.join(paper.subjects) if isinstance(paper.subjects, list) else paper.subjects,
                'date': paper.date_submitted.isoformat() if paper.date_submitted else '',
                'primary_subject': paper.primary_subject,
                'abstract_url': paper.abstract_url,
                'pdf_url': paper.pdf_url
            }
            documents.append(doc)
        
        # Build hybrid search index
        self.logger.info(f"Building hybrid search index with {len(documents)} documents")
        try:
            # Build hybrid search index
            await self._build_hybrid_searcher_fallback(documents)
            
            self.logger.info("Successfully built hybrid search index")
            
            # Save processed papers to cache
            self.processed_data_manager.save_papers(self.papers)
            
            # Save indices for fast access
            indices = {
                'authors': self.authors_index,
                'domains': self.domain_index,
                'subjects': self.subject_index
            }
            self.processed_data_manager.save_indices(indices)
            
            # Save dataset info
            self.processed_data_manager.save_dataset_info(self.dataset_info)
            
            # Save detailed author statistics to populate /authors/ folder
            self._save_author_statistics()
            
            # Save subject mapping to populate /subjects/ folder
            self._save_subject_mapping()
            
            # Save domain statistics to populate /domains/ folder
            self._save_domain_statistics()
            
            self.logger.info("Saved processed data and indices to cache")
                
        except Exception as e:
            self.logger.error(f"Error building hybrid search index: {e}")
            raise
    
    async def _build_hybrid_searcher_fallback(self, documents: List[Dict[str, Any]]):
        """Build hybrid searcher as fallback or complement to vector database."""
        self.logger.info("Building hybrid searcher index")
        
        # Documents are already in the correct format with string fields
        # Build hybrid search index
        self.hybrid_searcher.build_index(documents, text_fields=['title', 'abstract', 'subjects'])
        
        # Save the indices for future use
        index_path = Path(self.config.data.embeddings_cache_path) / "index"
        self.hybrid_searcher.save_index(str(index_path))
        
        # Save metadata about current index
        self._save_index_metadata(index_path, len(documents))
        
        self.logger.info(f"Built and saved hybrid search index for {len(documents)} documents")
    
    async def _initialize_hybrid_searcher_from_cache(self):
        """Initialize hybrid searcher from cached index."""
        index_path = Path(self.config.data.embeddings_cache_path) / "index"
        
        self.logger.info(f"Checking for existing index at: {index_path}")
        if self._check_existing_index(index_path):
            try:
                self.logger.info("Attempting to load hybrid search index...")
                success = self.hybrid_searcher.load_index(str(index_path))
                if success:
                    self.logger.info("Successfully loaded existing hybrid search index")
                    return
                else:
                    self.logger.warning("Failed to load hybrid search index - load returned False")
            except Exception as e:
                self.logger.warning(f"Failed to load existing hybrid index: {e}")
        else:
            self.logger.info("Index validation failed")
        
        self.logger.info("No valid hybrid search index found, will rebuild when needed")
    
    def _generate_data_cache_key(self) -> str:
        """Generate a cache key based on current data state."""
        data_info = {
            'paper_count': len(self.papers),
            'datasets': list(self.datasets.keys()),
            'config_hash': hashlib.md5(str(self.config.data).encode()).hexdigest()[:8]
        }
        return hashlib.md5(str(data_info).encode()).hexdigest()
    
    def _validate_cached_papers(self, cached_papers: List[Paper]) -> bool:
        """Validate that cached papers match current data."""
        if len(cached_papers) != len(self.papers):
            return False
        
        # Sample a few papers to verify they match
        sample_size = min(10, len(self.papers))
        for i in range(0, len(self.papers), len(self.papers) // sample_size):
            if i < len(cached_papers):
                current = self.papers[i]
                cached = cached_papers[i]
                if (current.paper_id != cached.paper_id or 
                    current.title != cached.title):
                    return False
        
        return True
        
    def _check_existing_index(self, index_path: Path) -> bool:
        """
        Check if existing index is valid and current.
        
        Args:
            index_path: Path to index directory
            
        Returns:
            True if existing valid index found
        """
        try:
            # Check if all required index files exist
            semantic_path = index_path / "semantic"
            sparse_path = index_path / "sparse"
            
            required_files = [
                semantic_path / "index.faiss",
                semantic_path / "documents.pkl",
                sparse_path / "bm25_model.pkl",
                sparse_path / "sparse_documents.pkl",
                index_path / "index_metadata.json"
            ]
            
            if not all(f.exists() for f in required_files):
                self.logger.info("Some index files missing, will rebuild")
                self.logger.debug(f"Missing files: {[str(f) for f in required_files if not f.exists()]}")
                return False
                
            # Check metadata to see if rebuild needed
            import json
            with open(index_path / "index_metadata.json", 'r') as f:
                metadata = json.load(f)
                
            # Compare document count
            if metadata.get('document_count', 0) != len(self.papers):
                self.logger.info(f"Document count changed ({metadata.get('document_count', 0)} -> {len(self.papers)}), will rebuild")
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking existing index: {str(e)}")
            return False
            
    def _save_index_metadata(self, index_path: Path, document_count: int):
        """
        Save metadata about the current index for incremental updates.
        
        Args:
            index_path: Path to index directory
            document_count: Number of documents indexed
        """
        import json
        from datetime import datetime
        
        metadata = {
            'document_count': document_count,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(index_path / "index_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_date_range(self, df: pd.DataFrame, dataset_config: Dict) -> Dict[str, datetime]:
        """Get date range from dataset."""
        date_col = dataset_config.get('columns', {}).get('date', 'Date Submitted')
        
        if date_col not in df.columns:
            return {"start": datetime.now(), "end": datetime.now()}
        
        try:
            # Convert to datetime
            dates = pd.to_datetime(df[date_col], errors='coerce')
            dates = dates.dropna()
            
            return {
                "start": dates.min().to_pydatetime(),
                "end": dates.max().to_pydatetime()
            }
        except Exception:
            return {"start": datetime.now(), "end": datetime.now()}
    
    def _get_unique_values(self, df: pd.DataFrame, column: str) -> List[str]:
        """Get unique values from a column."""
        if column not in df.columns:
            return []
        return df[column].dropna().unique().tolist()
    
    def _get_unique_subjects(self, df: pd.DataFrame, dataset_config: Dict) -> List[str]:
        """Get unique subjects from dataset."""
        subjects_col = dataset_config.get('columns', {}).get('subjects', 'Subjects')
        
        if subjects_col not in df.columns:
            return []
        
        all_subjects = set()
        for subjects_str in df[subjects_col].dropna():
            try:
                # Parse subjects list
                if isinstance(subjects_str, str):
                    subjects_str = subjects_str.strip("[]'\"")
                    subjects = [s.strip().strip("'\"") for s in subjects_str.split(',')]
                    all_subjects.update(subjects)
            except Exception:
                continue
        
        return list(all_subjects)
    
    def get_papers_by_author(self, author_name: str, exact_match: bool = False) -> List[Paper]:
        """Get papers by author name.
        
        Args:
            author_name: Author name to search for
            exact_match: Whether to use exact matching
            
        Returns:
            List of papers by the author
        """
        if exact_match:
            indices = self.authors_index.get(author_name, [])
        else:
            # Fuzzy matching
            indices = []
            author_lower = author_name.lower()
            for author in self.authors_index:
                if author_lower in author.lower() or author.lower() in author_lower:
                    indices.extend(self.authors_index[author])
        
        return [self.papers[i] for i in indices]
    
    def get_papers_by_domain(self, domain: str) -> List[Paper]:
        """Get papers by domain.
        
        Args:
            domain: Domain to search for
            
        Returns:
            List of papers in the domain
        """
        indices = self.domain_index.get(domain, [])
        return [self.papers[i] for i in indices]
    
    def get_papers_by_subject(self, subject: str) -> List[Paper]:
        """Get papers by subject.
        
        Args:
            subject: Subject to search for
            
        Returns:
            List of papers with the subject
        """
        indices = self.subject_index.get(subject, [])
        return [self.papers[i] for i in indices]
    
    def search_papers(
        self,
        query: str,
        search_fields: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Paper]:
        """Search papers by text query using traditional keyword search.
        
        Args:
            query: Search query
            search_fields: Fields to search in (title, abstract, authors, etc.)
            filters: Additional filters (domain, date_range, etc.)
            limit: Maximum number of results
            
        Returns:
            List of matching papers
        """
        if search_fields is None:
            search_fields = ['title', 'abstract', 'authors']
        
        query_lower = query.lower()
        results = []
        
        for paper in self.papers:
            # Check if query matches any search field
            match = False
            
            if 'title' in search_fields and query_lower in paper.title.lower():
                match = True
            elif 'abstract' in search_fields and query_lower in paper.abstract.lower():
                match = True
            elif 'authors' in search_fields:
                for author in paper.authors:
                    if query_lower in author.lower():
                        match = True
                        break
            
            if not match:
                continue
            
            # Apply filters
            if filters:
                if 'domain' in filters and paper.domain not in filters['domain']:
                    continue
                if 'date_range' in filters:
                    date_range = filters['date_range']
                    if paper.date_submitted < date_range.get('start', datetime.min):
                        continue
                    if paper.date_submitted > date_range.get('end', datetime.max):
                        continue
            
            results.append(paper)
            
            if len(results) >= limit:
                break
        
        return results
    
    def search_papers(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        score_threshold: Optional[float] = None,
        alpha: float = 0.7
    ) -> List[Tuple[Paper, float]]:
        """Search papers using hybrid search (combines sparse and dense).
        
        Args:
            query: Search query
            filters: Additional filters (domain, date_range, etc.)
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            alpha: Weight for combining sparse and dense scores (0.0 = sparse only, 1.0 = dense only)
            
        Returns:
            List of (paper, score) tuples ordered by relevance
        """
        try:
            # Use hybrid search for comprehensive results
            results = self.hybrid_searcher.search(
                query=query,
                k=limit,
                alpha=alpha,
                score_threshold=score_threshold or 0.0
            )
            
            # Convert results to papers with scores
            paper_results = []
            for result in results:
                # Get paper by ID from result
                paper_id = result.get('id', '')
                score = result.get('score', 0.0)
                paper = self._get_paper_by_id(paper_id)
                if paper:
                    paper_results.append((paper, score))
            
            return paper_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID (handles both original and unique IDs)."""
        # First try direct match
        for paper in self.papers:
            if paper.paper_id == paper_id:
                return paper
        
        # If not found, try to extract original ID from unique ID format
        if '_dup_' in paper_id:
            original_id = paper_id.split('_dup_')[0]
            for paper in self.papers:
                if paper.paper_id == original_id:
                    return paper
        
        return None
    
    def _passes_filters(self, paper: Paper, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if paper passes the given filters."""
        if not filters:
            return True
        
        if 'domain' in filters:
            if isinstance(filters['domain'], list):
                if paper.domain not in filters['domain']:
                    return False
            else:
                if paper.domain != filters['domain']:
                    return False
        
        if 'date_range' in filters:
            date_range = filters['date_range']
            if paper.date_submitted < date_range.get('start', datetime.min):
                return False
            if paper.date_submitted > date_range.get('end', datetime.max):
                return False
        
        if 'authors' in filters:
            author_list = filters['authors'] if isinstance(filters['authors'], list) else [filters['authors']]
            if not any(author.lower() in [a.lower() for a in paper.authors] for author in author_list):
                return False
        
        if 'subjects' in filters:
            subject_list = filters['subjects'] if isinstance(filters['subjects'], list) else [filters['subjects']]
            if not any(subject.lower() in [s.lower() for s in paper.subjects] for subject in subject_list):
                return False
        
        return True
    
    def get_author_statistics(self, author_name: str) -> Dict[str, Any]:
        """Get statistics for an author.
        
        Args:
            author_name: Author name
            
        Returns:
            Author statistics
        """
        papers = self.get_papers_by_author(author_name, exact_match=False)
        
        if not papers:
            return {}
        
        # Calculate statistics
        domains = list(set(paper.domain for paper in papers))
        subjects = list(set(subject for paper in papers for subject in paper.subjects))
        years = [paper.date_submitted.year for paper in papers]
        
        # Collaborators
        collaborators = set()
        for paper in papers:
            for author in paper.authors:
                if author.lower() != author_name.lower():
                    collaborators.add(author)
        
        return {
            'total_papers': len(papers),
            'domains': domains,
            'subjects': subjects,
            'years_active': list(set(years)),
            'first_publication': min(paper.date_submitted for paper in papers),
            'last_publication': max(paper.date_submitted for paper in papers),
            'collaborators': list(collaborators),
            'papers_per_year': {year: years.count(year) for year in set(years)}
        }
    
    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Domain statistics
        """
        papers = self.get_papers_by_domain(domain)
        
        if not papers:
            return {}
        
        # Calculate statistics
        authors = set()
        subjects = set()
        years = []
        
        for paper in papers:
            authors.update(paper.authors)
            subjects.update(paper.subjects)
            years.append(paper.date_submitted.year)
        
        # Author publication counts
        author_counts = {}
        for paper in papers:
            for author in paper.authors:
                author_counts[author] = author_counts.get(author, 0) + 1
        
        return {
            'total_papers': len(papers),
            'unique_authors': len(authors),
            'subjects': list(subjects),
            'years_active': list(set(years)),
            'first_publication': min(paper.date_submitted for paper in papers),
            'last_publication': max(paper.date_submitted for paper in papers),
            'top_authors': sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:20],
            'papers_per_year': {year: years.count(year) for year in set(years)}
        }
    
    def get_collaboration_network(self, author_name: str, depth: int = 1) -> Dict[str, Any]:
        """Get collaboration network for an author.
        
        Args:
            author_name: Author name
            depth: Network depth (1 = direct collaborators, 2 = collaborators of collaborators)
            
        Returns:
            Collaboration network data
        """
        papers = self.get_papers_by_author(author_name, exact_match=False)
        
        if not papers:
            return {}
        
        # Direct collaborators
        direct_collaborators = {}
        for paper in papers:
            for author in paper.authors:
                if author.lower() != author_name.lower():
                    if author not in direct_collaborators:
                        direct_collaborators[author] = []
                    direct_collaborators[author].append(paper.paper_id)
        
        network = {
            'focal_author': author_name,
            'direct_collaborators': {
                author: {
                    'collaboration_count': len(paper_ids),
                    'shared_papers': paper_ids
                }
                for author, paper_ids in direct_collaborators.items()
            }
        }
        
        # Extended network if depth > 1
        if depth > 1:
            extended_collaborators = {}
            for collaborator in direct_collaborators:
                collab_papers = self.get_papers_by_author(collaborator, exact_match=False)
                for paper in collab_papers:
                    for author in paper.authors:
                        if (author.lower() != author_name.lower() and 
                            author not in direct_collaborators and
                            author not in extended_collaborators):
                            extended_collaborators[author] = extended_collaborators.get(author, 0) + 1
            
            network['extended_collaborators'] = extended_collaborators
        
        return network
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on data manager.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "datasets_loaded": len(self.datasets),
            "total_papers": len(self.papers),
            "authors_indexed": len(self.authors_index),
            "domains_indexed": len(self.domain_index),
            "subjects_indexed": len(self.subject_index),
            "dataset_info": {name: info.dict() for name, info in self.dataset_info.items()}
        }
    
    def _save_author_statistics(self):
        """Generate and save detailed author statistics."""
        self.logger.info("Generating comprehensive author statistics...")
        from collections import Counter
        
        author_stats = {}
        for author, paper_indices in self.authors_index.items():
            papers = [self.papers[i] for i in paper_indices]
            
            # Initialize comprehensive statistics
            stats = {
                # Keep original stats for compatibility
                "total_papers": len(papers),
                "years_active": [],
                "domains": set(),
                "subjects": set(),
                "collaborators": set(),
                "first_paper_date": None,
                "last_paper_date": None,
                "num_collaborators": 0,
                
                # Enhanced comprehensive stats
                "institutions": set(),
                "num_institutions": 0,
                "primary_subjects": Counter(),
                "domain_diversity": 0,
                "research_breadth": 0,
                "collaboration_intensity": "Low",
                "career_start": None,
                "career_latest": None,
                "years_span": 0,
                "avg_papers_per_year": 0.0,
                "most_active_year": None,
                "peak_year_papers": 0,
                "papers": [],  # Store paper IDs for reference
                "llm_parsed": True,  # Flag to indicate LLM-parsed data
                "generated_at": None
            }
            
            # Collect publication years, dates, and detailed info
            years = []
            dates = []
            yearly_paper_count = Counter()
            paper_ids = []
            
            for paper in papers:
                # Store paper ID
                paper_ids.append(paper.paper_id)
                
                # Process dates
                if paper.date_submitted:
                    try:
                        if isinstance(paper.date_submitted, str):
                            import pandas as pd
                            date_obj = pd.to_datetime(paper.date_submitted, errors='coerce')
                            if pd.notna(date_obj):
                                years.append(date_obj.year)
                                dates.append(date_obj)
                                yearly_paper_count[date_obj.year] += 1
                        else:
                            years.append(paper.date_submitted.year)
                            dates.append(paper.date_submitted)
                            yearly_paper_count[paper.date_submitted.year] += 1
                    except:
                        pass
                
                # Collect domains and subjects with enhanced tracking
                if paper.domain:
                    stats["domains"].add(paper.domain)
                
                if paper.subjects:
                    if isinstance(paper.subjects, str):
                        subject_list = [s.strip() for s in paper.subjects.split(',')]
                        stats["subjects"].update(subject_list)
                    elif isinstance(paper.subjects, list):
                        stats["subjects"].update(paper.subjects)
                
                if paper.primary_subject:
                    stats["primary_subjects"][paper.primary_subject] += 1
                
                # Collect collaborators
                if paper.authors:
                    for co_author in paper.authors:
                        if co_author != author:
                            stats["collaborators"].add(co_author)
                
                # Extract institutions from LLM-parsed data
                if hasattr(paper, 'author_institutions') and paper.author_institutions:
                    author_institutions = paper.author_institutions.get(author, [])
                    for institution in author_institutions:
                        if institution and institution.strip():
                            stats["institutions"].add(institution.strip())
            
            # Calculate comprehensive metrics
            years_list = sorted(list(set(years)))
            stats["years_active"] = years_list
            stats["domains"] = list(stats["domains"])
            stats["subjects"] = list(stats["subjects"])
            stats["collaborators"] = list(stats["collaborators"])[:50]  # Limit for performance
            stats["institutions"] = list(stats["institutions"])
            stats["papers"] = paper_ids[:100]  # Limit to first 100 papers
            
            # Calculate derived metrics
            stats["num_collaborators"] = len(stats["collaborators"])
            stats["num_institutions"] = len(stats["institutions"])
            stats["domain_diversity"] = len(stats["domains"])
            stats["research_breadth"] = len(stats["subjects"])
            
            # Career span and productivity metrics
            if years_list:
                stats["career_start"] = min(years_list)
                stats["career_latest"] = max(years_list)
                stats["years_span"] = len(years_list)
                stats["avg_papers_per_year"] = round(len(papers) / len(years_list), 2)
                
                # Find most productive year
                if yearly_paper_count:
                    most_productive_year, max_papers = yearly_paper_count.most_common(1)[0]
                    stats["most_active_year"] = most_productive_year
                    stats["peak_year_papers"] = max_papers
            
            # Date range
            if dates:
                stats["first_paper_date"] = min(dates).isoformat()
                stats["last_paper_date"] = max(dates).isoformat()
            
            # Collaboration intensity classification
            num_collaborators = stats["num_collaborators"]
            if num_collaborators > 50:
                stats["collaboration_intensity"] = "High"
            elif num_collaborators > 20:
                stats["collaboration_intensity"] = "Medium" 
            else:
                stats["collaboration_intensity"] = "Low"
            
            # Convert primary_subjects Counter to dict for JSON serialization
            stats["primary_subjects"] = dict(stats["primary_subjects"].most_common(10))
            
            # Add generation timestamp
            from datetime import datetime
            stats["generated_at"] = datetime.now().isoformat()
            
            author_stats[author] = stats
        
        # Save the comprehensive statistics
        try:
            self.processed_data_manager.save_author_statistics(author_stats)
            self.logger.info(f"Saved comprehensive statistics for {len(author_stats)} authors")
            self.logger.info("Enhanced author statistics now include:")
            self.logger.info("  - Institution affiliations from LLM parsing")
            self.logger.info("  - Career span and productivity metrics") 
            self.logger.info("  - Research breadth and domain diversity")
            self.logger.info("  - Collaboration intensity classification")
            self.logger.info("  - Primary subject rankings")
            self.logger.info("  - Peak productivity analysis")
        except Exception as e:
            self.logger.error(f"Failed to save author statistics: {e}")
            raise
    
    def _save_subject_mapping(self):
        """Generate and save subject mapping."""
        self.logger.info("Generating subject mapping...")
        
        subject_mapping = {}
        for subject, paper_indices in self.subject_index.items():
            # Get paper IDs for this subject
            paper_ids = [self.papers[i].paper_id for i in paper_indices]
            subject_mapping[subject] = paper_ids
        
        try:
            self.processed_data_manager.save_subject_mapping(subject_mapping)
            self.logger.info(f"Saved mapping for {len(subject_mapping)} subjects")
        except Exception as e:
            self.logger.error(f"Failed to save subject mapping: {e}")
    
    def _save_domain_statistics(self):
        """Generate and save domain statistics to domains folder."""
        self.logger.info("Generating domain statistics...")
        
        domain_stats = {}
        for domain, paper_indices in self.domain_index.items():
            papers = [self.papers[i] for i in paper_indices]
            
            # Calculate domain statistics
            stats = {
                "total_papers": len(papers),
                "unique_authors": set(),
                "subjects": set(),
                "years": set(),
                "paper_ids": []
            }
            
            for paper in papers:
                stats["paper_ids"].append(paper.paper_id)
                
                # Collect authors
                if paper.authors:
                    if isinstance(paper.authors, list):
                        stats["unique_authors"].update(paper.authors)
                    else:
                        stats["unique_authors"].update([a.strip() for a in paper.authors.split(',')])
                
                # Collect subjects  
                if paper.subjects:
                    if isinstance(paper.subjects, str):
                        stats["subjects"].update([s.strip() for s in paper.subjects.split(',')])
                    elif isinstance(paper.subjects, list):
                        stats["subjects"].update(paper.subjects)
                
                # Collect years
                if paper.date_submitted:
                    try:
                        if isinstance(paper.date_submitted, str):
                            import pandas as pd
                            date_obj = pd.to_datetime(paper.date_submitted, errors='coerce')
                            if pd.notna(date_obj):
                                stats["years"].add(date_obj.year)
                        else:
                            stats["years"].add(paper.date_submitted.year)
                    except:
                        pass
            
            # Convert sets to lists for JSON serialization
            stats["unique_authors"] = list(stats["unique_authors"])
            stats["subjects"] = list(stats["subjects"])
            stats["years"] = sorted(list(stats["years"]))
            stats["num_authors"] = len(stats["unique_authors"])
            stats["num_subjects"] = len(stats["subjects"])
            
            domain_stats[domain] = stats
        
        # Save domain statistics to domains folder
        try:
            domains_path = self.processed_data_manager.processed_data_path / "domains"
            filepath = domains_path / "domain_stats.json"
            
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(domain_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved statistics for {len(domain_stats)} domains to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save domain statistics: {e}")
