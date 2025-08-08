"""
Processed data manager for TechAuthor system.
Handles caching, preprocessing, and storage of processed research data.
"""

import pickle
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import pandas as pd

from ..core.models import Paper, DatasetInfo


class ProcessedDataManager:
    """Manages processed data storage and retrieval."""
    
    def __init__(self, processed_data_path: str = "./data/processed/"):
        """Initialize processed data manager.
        
        Args:
            processed_data_path: Path to store processed data
        """
        self.processed_data_path = Path(processed_data_path)
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for processed data."""
        directories = [
            self.processed_data_path,
            self.processed_data_path / "papers",
            self.processed_data_path / "authors", 
            self.processed_data_path / "subjects",
            self.processed_data_path / "domains",
            self.processed_data_path / "statistics",
            self.processed_data_path / "indices",
            self.processed_data_path / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Processed data directories created at: {self.processed_data_path}")
    
    def save_papers(self, papers: List[Paper], filename: str = "papers.pkl") -> str:
        """Save processed papers to disk.
        
        Args:
            papers: List of Paper objects
            filename: Name of the file to save
            
        Returns:
            Path to saved file
        """
        filepath = self.processed_data_path / "papers" / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(papers, f)
            
            self.logger.info(f"Saved {len(papers)} papers to {filepath}")
            
            # Also save metadata
            metadata = {
                'count': len(papers),
                'saved_at': datetime.now().isoformat(),
                'filename': filename,
                'file_size_bytes': filepath.stat().st_size
            }
            
            metadata_path = filepath.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save papers: {e}")
            raise
    
    def load_papers(self, filename: str = "papers.pkl") -> List[Paper]:
        """Load processed papers from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of Paper objects
        """
        filepath = self.processed_data_path / "papers" / filename
        
        if not filepath.exists():
            self.logger.warning(f"Papers file not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'rb') as f:
                papers = pickle.load(f)
            
            self.logger.info(f"Loaded {len(papers)} papers from {filepath}")
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to load papers: {e}")
            return []
    
    def save_author_statistics(self, author_stats: Dict[str, Any], filename: str = "author_stats.json"):
        """Save author statistics to disk.
        
        Args:
            author_stats: Dictionary with author statistics
            filename: Name of the file to save
        """
        filepath = self.processed_data_path / "authors" / filename
        
        try:
            # Convert any non-serializable objects to strings
            serializable_stats = {}
            for author, stats in author_stats.items():
                if isinstance(stats, dict):
                    clean_stats = {}
                    for key, value in stats.items():
                        if isinstance(value, datetime):
                            clean_stats[key] = value.isoformat()
                        elif isinstance(value, (list, dict, str, int, float, bool)):
                            clean_stats[key] = value
                        else:
                            clean_stats[key] = str(value)
                    serializable_stats[author] = clean_stats
                else:
                    serializable_stats[author] = str(stats)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved author statistics to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save author statistics: {e}")
            raise
    
    def load_author_statistics(self, filename: str = "author_stats.json") -> Dict[str, Any]:
        """Load author statistics from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary with author statistics
        """
        filepath = self.processed_data_path / "authors" / filename
        
        if not filepath.exists():
            self.logger.warning(f"Author statistics file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            self.logger.info(f"Loaded author statistics from {filepath}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to load author statistics: {e}")
            return {}
    
    def save_subject_mapping(self, subject_mapping: Dict[str, List[str]], filename: str = "subject_mapping.json"):
        """Save subject mapping to disk.
        
        Args:
            subject_mapping: Dictionary mapping subjects to paper IDs
            filename: Name of the file to save
        """
        filepath = self.processed_data_path / "subjects" / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(subject_mapping, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved subject mapping to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save subject mapping: {e}")
            raise
    
    def save_dataset_info(self, dataset_info: Dict[str, DatasetInfo], filename: str = "dataset_info.json"):
        """Save dataset information to disk.
        
        Args:
            dataset_info: Dictionary with dataset information
            filename: Name of the file to save
        """
        filepath = self.processed_data_path / "statistics" / filename
        
        try:
            # Convert DatasetInfo objects to dictionaries
            serializable_info = {}
            for name, info in dataset_info.items():
                if hasattr(info, 'dict'):
                    # Pydantic model
                    info_dict = info.dict()
                    # Convert datetime objects
                    for key, value in info_dict.items():
                        if isinstance(value, datetime):
                            info_dict[key] = value.isoformat()
                        elif isinstance(value, dict) and 'start' in value and 'end' in value:
                            # Date range handling
                            if isinstance(value['start'], datetime):
                                value['start'] = value['start'].isoformat()
                            if isinstance(value['end'], datetime):
                                value['end'] = value['end'].isoformat()
                    serializable_info[name] = info_dict
                else:
                    serializable_info[name] = str(info)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved dataset info to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset info: {e}")
            raise
    
    def save_indices(self, indices: Dict[str, Dict[str, List[int]]], filename: str = "search_indices.pkl"):
        """Save search indices to disk.
        
        Args:
            indices: Dictionary with search indices (authors, domains, subjects)
            filename: Name of the file to save
        """
        filepath = self.processed_data_path / "indices" / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(indices, f)
            
            # Save summary statistics
            summary = {}
            for index_name, index_data in indices.items():
                summary[index_name] = {
                    'unique_keys': len(index_data),
                    'total_mappings': sum(len(papers) for papers in index_data.values())
                }
            
            summary_path = filepath.with_suffix('.summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Saved indices to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save indices: {e}")
            raise
    
    def load_indices(self, filename: str = "search_indices.pkl") -> Dict[str, Dict[str, List[int]]]:
        """Load search indices from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary with search indices
        """
        filepath = self.processed_data_path / "indices" / filename
        
        if not filepath.exists():
            self.logger.warning(f"Indices file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'rb') as f:
                indices = pickle.load(f)
            
            self.logger.info(f"Loaded indices from {filepath}")
            return indices
            
        except Exception as e:
            self.logger.error(f"Failed to load indices: {e}")
            return {}
    
    def export_papers_csv(self, papers: List[Paper], filename: str = "papers_export.csv"):
        """Export papers to CSV format.
        
        Args:
            papers: List of Paper objects
            filename: Name of the CSV file
        """
        filepath = self.processed_data_path / "papers" / filename
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                if not papers:
                    return
                
                # Get field names from first paper
                fieldnames = [
                    'paper_id', 'title', 'authors', 'abstract', 'domain',
                    'primary_subject', 'subjects', 'date_submitted',
                    'abstract_url', 'pdf_url'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for paper in papers:
                    row = {
                        'paper_id': paper.paper_id,
                        'title': paper.title,
                        'authors': ', '.join(paper.authors),
                        'abstract': paper.abstract,
                        'domain': paper.domain,
                        'primary_subject': paper.primary_subject,
                        'subjects': ', '.join(paper.subjects),
                        'date_submitted': paper.date_submitted.isoformat() if paper.date_submitted else '',
                        'abstract_url': paper.abstract_url,
                        'pdf_url': paper.pdf_url
                    }
                    writer.writerow(row)
            
            self.logger.info(f"Exported {len(papers)} papers to CSV: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export papers to CSV: {e}")
            raise
    
    def cache_query_result(self, query_hash: str, result: Any, ttl_hours: int = 24):
        """Cache query result for faster retrieval.
        
        Args:
            query_hash: Hash of the query for unique identification
            result: Result to cache
            ttl_hours: Time to live in hours
        """
        cache_file = self.processed_data_path / "cache" / f"{query_hash}.cache"
        
        try:
            cache_data = {
                'result': result,
                'cached_at': datetime.now().isoformat(),
                'ttl_hours': ttl_hours,
                'expires_at': (datetime.now().timestamp() + (ttl_hours * 3600))
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.debug(f"Cached query result: {query_hash}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache query result: {e}")
    
    def get_cached_result(self, query_hash: str) -> Optional[Any]:
        """Retrieve cached query result.
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            Cached result if valid, None otherwise
        """
        cache_file = self.processed_data_path / "cache" / f"{query_hash}.cache"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is still valid
            if datetime.now().timestamp() > cache_data['expires_at']:
                # Cache expired, remove file
                cache_file.unlink()
                self.logger.debug(f"Cache expired for: {query_hash}")
                return None
            
            self.logger.debug(f"Cache hit for: {query_hash}")
            return cache_data['result']
            
        except Exception as e:
            self.logger.error(f"Failed to read cache: {e}")
            return None
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cache files.
        
        Args:
            older_than_hours: Only clear files older than this many hours. If None, clear all.
        """
        cache_dir = self.processed_data_path / "cache"
        
        if not cache_dir.exists():
            return
        
        cleared_count = 0
        current_time = datetime.now().timestamp()
        
        for cache_file in cache_dir.glob("*.cache"):
            try:
                if older_than_hours is not None:
                    file_time = cache_file.stat().st_mtime
                    if (current_time - file_time) < (older_than_hours * 3600):
                        continue
                
                cache_file.unlink()
                cleared_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        self.logger.info(f"Cleared {cleared_count} cache files")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored processed data.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'base_path': str(self.processed_data_path),
            'directories': {},
            'total_size_mb': 0
        }
        
        try:
            for subdir in ['papers', 'authors', 'subjects', 'domains', 'statistics', 'indices', 'cache']:
                dir_path = self.processed_data_path / subdir
                if dir_path.exists():
                    files = list(dir_path.glob("*"))
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    
                    stats['directories'][subdir] = {
                        'file_count': len([f for f in files if f.is_file()]),
                        'size_mb': total_size / (1024 * 1024),
                        'files': [f.name for f in files if f.is_file()]
                    }
                    
                    stats['total_size_mb'] += total_size / (1024 * 1024)
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            stats['error'] = str(e)
        
        return stats
