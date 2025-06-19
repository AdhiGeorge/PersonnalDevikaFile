from typing import List, Dict, Any, Tuple
import re
from Agentres.llm.llm import LLM

class TextProcessor:
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.llm = LLM()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """Process text into chunks with metadata and embeddings."""
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Split into chunks
        chunks = self._create_chunks(cleaned_text)
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.llm.get_embedding(chunk)
            
            # Create metadata
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "is_code": self._is_code_chunk(chunk),
                "language": self._detect_language(chunk) if self._is_code_chunk(chunk) else None
            }
            
            processed_chunks.append({
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata
            })
        
        return processed_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep code blocks
        text = re.sub(r'[^\w\s\.,;:!?()\[\]{}<>/\-+=*&^%$#@~`]', '', text)
        
        return text.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks of text."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a natural break point
            break_point = self._find_break_point(text, end)
            
            # Create the chunk
            chunk = text[start:break_point]
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = break_point - self.overlap
        
        return chunks

    def _find_break_point(self, text: str, position: int) -> int:
        """Find a natural break point in the text."""
        # Look for paragraph breaks
        para_break = text.rfind('\n\n', 0, position)
        if para_break != -1:
            return para_break + 2
        
        # Look for sentence breaks
        sent_break = text.rfind('. ', 0, position)
        if sent_break != -1:
            return sent_break + 2
        
        # Look for word breaks
        word_break = text.rfind(' ', 0, position)
        if word_break != -1:
            return word_break + 1
        
        return position

    def _is_code_chunk(self, text: str) -> bool:
        """Detect if a chunk contains code."""
        # Check for common code indicators
        code_indicators = [
            r'```[\w]*\n',  # Code block markers
            r'^\s*(def|class|import|from|if|for|while|try|except)\s',  # Python keywords
            r'^\s*(function|const|let|var|if|for|while|try|catch)\s',  # JavaScript keywords
            r'^\s*(public|private|protected|class|interface|void|int|string)\s',  # Java/C# keywords
            r'^\s*(<[a-z]+>|</[a-z]+>)',  # HTML tags
            r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*',  # Variable assignment
        ]
        
        return any(re.search(pattern, text, re.MULTILINE) for pattern in code_indicators)

    def _detect_language(self, code: str) -> str:
        """Detect the programming language of a code chunk."""
        # Simple language detection based on keywords and patterns
        patterns = {
            'python': [
                r'^\s*(def|class|import|from)\s',
                r'^\s*(if|for|while|try|except)\s:',
                r'print\(',
            ],
            'javascript': [
                r'^\s*(function|const|let|var)\s',
                r'^\s*(if|for|while|try|catch)\s\(',
                r'console\.log\(',
            ],
            'java': [
                r'^\s*(public|private|protected)\s',
                r'^\s*(class|interface)\s',
                r'System\.out\.println\(',
            ],
            'html': [
                r'^\s*<[a-z]+>',
                r'^\s*</[a-z]+>',
                r'<[a-z]+\s+[a-z-]+=',
            ],
        }
        
        for lang, lang_patterns in patterns.items():
            if any(re.search(pattern, code, re.MULTILINE) for pattern in lang_patterns):
                return lang
        
        return 'unknown' 