"""
TechAuthor - Research Paper Analysis System

A comprehensive system for analyzing relationships between authors and technology domains
using ArXiv research papers as bridges. Employs RAG + Multi-agent architecture for
sophisticated analysis and insights.
"""

try:
    from .core.system import TechAuthorSystem, create_system
    from .core.models import Query, QueryType
    from .core.config import config_manager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    from core.system import TechAuthorSystem, create_system
    from core.models import Query, QueryType
    from core.config import config_manager

__version__ = "1.0.0"
__author__ = "TechAuthor Team"
__email__ = "contact@techauthor.dev"

# Main exports
__all__ = [
    "TechAuthorSystem",
    "create_system",
    "Query",
    "QueryType",
    "config_manager"
]
