"""
Simple LLM utility for parsing tasks.
Lightweight wrapper around the existing LLM infrastructure for simple tasks.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from ..core.llm_manager import LLMManager


class LLMParsingUtility:
    """Simple LLM utility for parsing tasks like author extraction."""
    
    def __init__(self, config_path: str = None):
        """Initialize the LLM parsing utility."""
        self.logger = logging.getLogger(__name__)
        
        # If no config path provided, use default path relative to project root
        if config_path is None:
            import os
            current_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "config", "llm_config.yaml")
        
        self.llm_manager = LLMManager(config_path)
        
    async def parse_authors_and_institutions(self, author_string: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Parse author string using LLM to extract authors and institutions."""
        from ..prompts.author_parsing import AUTHOR_INSTITUTION_PARSING_SYSTEM_PROMPT, get_author_parsing_user_prompt
        
        if not author_string or author_string.strip() == '':
            return [], {}
        
        try:
            # Get prompts
            system_prompt = AUTHOR_INSTITUTION_PARSING_SYSTEM_PROMPT
            user_prompt = get_author_parsing_user_prompt(author_string)
            
            # Use LLM manager with classification agent (fast model)
            response = await self.llm_manager.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_name="classification",  # Use fast model for parsing
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=512
            )
            
            # Parse JSON response
            try:
                parsed_result = json.loads(response.strip())
                authors = parsed_result.get("authors", [])
                institutions = parsed_result.get("author_institutions", {})
                
                # Validate and clean results
                authors = [author.strip() for author in authors if author and author.strip()]
                cleaned_institutions = {}
                for author, author_insts in institutions.items():
                    if author.strip() and author_insts:
                        cleaned_institutions[author.strip()] = [
                            inst.strip() for inst in author_insts if inst and inst.strip()
                        ]
                
                self.logger.debug(f"LLM parsed {len(authors)} authors and {len(cleaned_institutions)} institutional mappings")
                return authors, cleaned_institutions
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM JSON response: {e}")
                self.logger.warning(f"LLM Response: {response}")
                # Fallback to empty results
                return [], {}
                
        except Exception as e:
            self.logger.error(f"LLM author parsing failed: {e}")
            # Return empty results on error
            return [], {}
    
    def parse_authors_and_institutions_sync(self, author_string: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Synchronous wrapper for parse_authors_and_institutions."""
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to use asyncio.create_task()
            # and then await it, but since this is a sync function, we'll create
            # a new thread to run the async code
            import concurrent.futures
            import threading
            
            # Create a new event loop in a separate thread
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.parse_authors_and_institutions(author_string))
                finally:
                    new_loop.close()
            
            # Run in a thread pool to avoid blocking the main event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
                
        except RuntimeError:
            # No running loop, we can use asyncio.run()
            return asyncio.run(self.parse_authors_and_institutions(author_string))


# Global instance for easy access
_llm_parser = None

def get_llm_parser() -> LLMParsingUtility:
    """Get global LLM parser instance."""
    global _llm_parser
    if _llm_parser is None:
        _llm_parser = LLMParsingUtility()
    return _llm_parser
