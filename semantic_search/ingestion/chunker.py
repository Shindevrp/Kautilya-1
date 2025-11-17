"""
Intelligent document chunking with semantic awareness
Respects document structure: headers, code blocks, sections
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm


logger = logging.getLogger("semantic_search")


class SemanticChunker:
    """Intelligent chunking that respects document structure and context"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (approx)
            chunk_overlap: Overlap between consecutive chunks in tokens
            min_chunk_size: Minimum chunk size to create
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.chunks = []
    
    def chunk_document(
        self,
        doc: Dict[str, Any],
        preserve_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Intelligently chunk a single document respecting structure.
        
        Args:
            doc: Document dictionary with 'content' and metadata
            preserve_context: Include parent section headers in chunks
            
        Returns:
            List of chunks with metadata
        """
        content = doc.get("content", "")
        file_type = doc.get("file_type", "unknown")
        
        if file_type == "markdown":
            chunks = self._chunk_markdown(content, preserve_context)
        else:
            chunks = self._chunk_text(content, preserve_context)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk["source"] = doc.get("relative_path", "unknown")
            chunk["source_full"] = doc.get("path", "")
            chunk["doc_id"] = doc.get("id", "")
            chunk["file_type"] = file_type
        
        return chunks
    
    def _chunk_markdown(
        self,
        content: str,
        preserve_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk markdown respecting headers and structure.
        
        Args:
            content: Markdown content
            preserve_context: Include parent headers
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        lines = content.split('\n')
        
        # Extract sections based on headers
        sections = self._extract_sections(lines)
        
        for section in sections:
            section_chunks = self._process_section(section, preserve_context)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _extract_sections(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extract sections based on header hierarchy.
        
        Args:
            lines: Document lines
            
        Returns:
            List of section dictionaries
        """
        sections = []
        current_section = {
            "level": 0,
            "title": "Root",
            "content": [],
            "children": []
        }
        
        stack = [current_section]
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Create new section
                new_section = {
                    "level": level,
                    "title": title,
                    "content": [],
                    "children": [],
                    "line": line
                }
                
                # Find parent level
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                
                if stack:
                    stack[-1]["children"].append(new_section)
                else:
                    sections.append(new_section)
                
                stack.append(new_section)
            
            else:
                if stack:
                    stack[-1]["content"].append(line)
        
        return sections
    
    def _process_section(
        self,
        section: Dict[str, Any],
        preserve_context: bool = True,
        parent_headers: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a section and create chunks.
        
        Args:
            section: Section dictionary
            preserve_context: Include parent headers
            parent_headers: Headers from parent sections
            
        Returns:
            List of chunks
        """
        if parent_headers is None:
            parent_headers = []
        
        chunks = []
        
        # Build context headers
        context_headers = parent_headers + [section.get("title", "")]
        context_str = " > ".join([h for h in context_headers if h])
        
        # Get section content
        section_text = "\n".join(section.get("content", []))
        
        # Split section into chunks
        if section_text.strip():
            section_chunks = self._chunk_text(
                section_text,
                preserve_context=False,
                add_prefix=context_str if preserve_context else None
            )
            
            for chunk in section_chunks:
                chunk["section"] = section.get("title", "")
                chunk["section_hierarchy"] = context_str
                chunks.append(chunk)
        
        # Process children recursively
        for child in section.get("children", []):
            child_chunks = self._process_section(
                child,
                preserve_context=preserve_context,
                parent_headers=context_headers
            )
            chunks.extend(child_chunks)
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        preserve_context: bool = True,
        add_prefix: str = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks respecting boundaries.
        
        Args:
            text: Text to chunk
            preserve_context: Include context in chunks
            add_prefix: Prefix to add to each chunk
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split on paragraph boundaries first
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = self._estimate_tokens(para)
            
            # If adding this paragraph would exceed size, save current chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    if add_prefix:
                        chunk_text = f"{add_prefix}\n\n{chunk_text}"
                    
                    chunk_dict = {
                        "content": chunk_text,
                        "size": len(chunk_text),
                        "token_count": self._estimate_tokens(chunk_text)
                    }
                    chunks.append(chunk_dict)
                
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                if add_prefix:
                    chunk_text = f"{add_prefix}\n\n{chunk_text}"
                
                chunk_dict = {
                    "content": chunk_text,
                    "size": len(chunk_text),
                    "token_count": self._estimate_tokens(chunk_text)
                }
                chunks.append(chunk_dict)
        
        return chunks
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count (1 token ≈ 4 characters).
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        logger.info(f"Chunking {len(documents)} documents...")
        
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        self.chunks = all_chunks
        
        return all_chunks
    
    def get_chunks(self) -> List[Dict[str, Any]]:
        """Get all created chunks."""
        return self.chunks
    
    def save_chunks(self, output_path: Path) -> None:
        """
        Save chunks to file for inspection.
        
        Args:
            output_path: Path to save chunks JSON
        """
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.chunks)} chunks to {output_path}")
