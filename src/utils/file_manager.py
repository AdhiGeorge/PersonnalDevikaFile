import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class FileManager:
    """Handles all file operations for the agent system."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the file manager with a base directory.
        
        Args:
            base_dir: Base directory for all file operations. If None, uses 'output' in current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / 'output'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.code_dir = self.base_dir / 'code'
        self.code_dir.mkdir(exist_ok=True)
        
        self.responses_dir = self.base_dir / 'responses'
        self.responses_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.base_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"FileManager initialized with base directory: {self.base_dir}")
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for a programming language."""
        extensions = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'java': 'java',
            'c++': 'cpp',
            'c#': 'cs',
            'go': 'go',
            'rust': 'rs',
            'ruby': 'rb',
            'php': 'php',
            'html': 'html',
            'css': 'css',
            'sql': 'sql',
            'json': 'json',
            'yaml': 'yaml',
            'markdown': 'md',
            'text': 'txt',
        }
        return extensions.get(language.lower().strip(), 'txt')
    
    async def save_code(
        self, 
        code: str, 
        language: str, 
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Save generated code to a file.
        
        Args:
            code: The code content to save
            language: Programming language (used to determine file extension)
            filename: Optional custom filename (without extension)
            metadata: Optional metadata to include in response
            
        Returns:
            Dict containing file information
        """
        try:
            # Generate filename if not provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not filename:
                filename = f"generated_{timestamp}"
            
            # Ensure filename is safe
            safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)
            
            # Get appropriate extension
            extension = self._get_extension(language)
            
            # Create file path
            filepath = self.code_dir / f"{safe_name}.{extension}"
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write code to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Code saved to {filepath}")
            
            return {
                'status': 'success',
                'filepath': str(filepath),
                'filename': filepath.name,
                'language': language,
                'extension': extension,
                'size': len(code),
                'metadata': metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Error saving code: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'filename': filename,
                'language': language
            }
    
    async def save_response(
        self, 
        content: str, 
        response_type: str = 'text',
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Save a text response to a file.
        
        Args:
            content: The text content to save
            response_type: Type of response ('text', 'markdown', 'json', etc.)
            filename: Optional custom filename (without extension)
            metadata: Optional metadata to include in response
            
        Returns:
            Dict containing file information
        """
        try:
            # Generate filename if not provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not filename:
                filename = f"response_{timestamp}"
            
            # Ensure filename is safe
            safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)
            
            # Determine extension
            extension = self._get_extension(response_type)
            
            # Create file path
            filepath = self.responses_dir / f"{safe_name}.{extension}"
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Response saved to {filepath}")
            
            return {
                'status': 'success',
                'filepath': str(filepath),
                'filename': filepath.name,
                'type': response_type,
                'size': len(content),
                'metadata': metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Error saving response: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'filename': filename,
                'type': response_type
            }
    
    async def read_file(self, filepath: Union[str, Path]) -> str:
        """Read the contents of a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            raise
    
    async def list_files(self, directory: Optional[Union[str, Path]] = None) -> List[Dict]:
        """List all files in a directory with their metadata."""
        try:
            dir_path = Path(directory) if directory else self.base_dir
            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Directory not found: {dir_path}")
                
            files = []
            for item in dir_path.rglob('*'):
                if item.is_file():
                    stat = item.stat()
                    files.append({
                        'name': item.name,
                        'path': str(item),
                        'size': stat.st_size,
                        'created': stat.st_ctime,
                        'modified': stat.st_mtime,
                        'is_dir': False
                    })
                else:
                    files.append({
                        'name': item.name,
                        'path': str(item),
                        'is_dir': True
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {str(e)}")
            raise
    
    async def delete_file(self, filepath: Union[str, Path]) -> Dict[str, str]:
        """Delete a file."""
        try:
            path = Path(filepath)
            if not path.exists():
                return {'status': 'error', 'error': 'File not found'}
                
            path.unlink()
            return {'status': 'success', 'filepath': str(filepath)}
            
        except Exception as e:
            logger.error(f"Error deleting file {filepath}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
