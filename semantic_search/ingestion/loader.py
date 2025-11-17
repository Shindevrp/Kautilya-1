"""
Documentation loader for Twitter API Postman documentation
Handles cloning and loading various document formats
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import requests


logger = logging.getLogger("semantic_search")


class DocumentationLoader:
    """Load documentation from GitHub repository or local directory"""
    
    def __init__(self, docs_path: Path, repo_url: str = None):
        """
        Initialize documentation loader.
        
        Args:
            docs_path: Path to documentation directory
            repo_url: GitHub repository URL (for cloning)
        """
        self.docs_path = docs_path
        self.repo_url = repo_url
        self.documents = []
    
    def clone_repo(self) -> bool:
        """
        Clone GitHub repository if not already present.
        
        Returns:
            True if successful, False otherwise
        """
        if self.docs_path.exists() and any(self.docs_path.iterdir()):
            logger.info(f"Documentation already exists at {self.docs_path}")
            return True
        
        if not self.repo_url:
            logger.error("Repository URL not provided for cloning")
            return False
        
        try:
            logger.info(f"Cloning repository from {self.repo_url}...")
            self.docs_path.parent.mkdir(parents=True, exist_ok=True)
            
            subprocess.run(
                ["git", "clone", self.repo_url, str(self.docs_path)],
                check=True,
                capture_output=True,
                timeout=300
            )
            
            logger.info(f"Successfully cloned repository to {self.docs_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during clone: {e}")
            return False
    
    def load_markdown_files(self) -> List[Dict[str, Any]]:
        """
        Load all Markdown files from documentation.
        
        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []
        
        for md_file in self.docs_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                relative_path = md_file.relative_to(self.docs_path)
                
                doc = {
                    "id": str(relative_path),
                    "path": str(md_file),
                    "relative_path": str(relative_path),
                    "content": content,
                    "file_type": "markdown",
                    "size": len(content),
                    "title": md_file.stem
                }
                
                documents.append(doc)
                logger.debug(f"Loaded markdown: {relative_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load {md_file}: {e}")
        
        return documents
    
    def load_json_files(self) -> List[Dict[str, Any]]:
        """
        Load all JSON files (Postman collections, etc).
        
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for json_file in self.docs_path.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                relative_path = json_file.relative_to(self.docs_path)
                
                # Convert JSON to readable text format
                content = self._json_to_text(data)
                
                doc = {
                    "id": str(relative_path),
                    "path": str(json_file),
                    "relative_path": str(relative_path),
                    "content": content,
                    "raw_json": data,
                    "file_type": "json",
                    "size": len(content),
                    "title": json_file.stem
                }
                
                documents.append(doc)
                logger.debug(f"Loaded JSON: {relative_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return documents
    
    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all documentation files.
        
        Returns:
            List of all documents
        """
        logger.info("Loading documentation files...")
        
        if not self.docs_path.exists():
            logger.error(f"Documentation path does not exist: {self.docs_path}")
            return []
        
        documents = []
        documents.extend(self.load_markdown_files())
        documents.extend(self.load_json_files())
        
        logger.info(f"Loaded {len(documents)} documents")
        self.documents = documents
        
        return documents
    
    @staticmethod
    def _json_to_text(data: Any, indent: int = 0) -> str:
        """
        Convert JSON data to readable text format.
        
        Args:
            data: JSON data to convert
            indent: Current indentation level
            
        Returns:
            Text representation of JSON
        """
        text_parts = []
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text_parts.append(f"{indent_str}{key}:")
                    text_parts.append(DocumentationLoader._json_to_text(value, indent + 1))
                else:
                    text_parts.append(f"{indent_str}{key}: {value}")
        
        elif isinstance(data, list):
            for item in data:
                text_parts.append(DocumentationLoader._json_to_text(item, indent))
        
        else:
            text_parts.append(f"{indent_str}{data}")
        
        return "\n".join(text_parts)
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get loaded documents."""
        return self.documents
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search documents by keyword.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of matching documents
        """
        results = []
        keyword_lower = keyword.lower()
        
        for doc in self.documents:
            if keyword_lower in doc["content"].lower():
                results.append(doc)
        
        return results
