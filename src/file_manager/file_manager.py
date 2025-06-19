import os
import logging
from datetime import datetime
from typing import Optional

from config.config import Config

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations for the agent."""
    
    def __init__(self, config: Config):
        """Initialize the file manager.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.output_dir = config.get('output', 'directory')
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the file manager.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Test file operations
            test_file = os.path.join(self.output_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('Test')
            os.remove(test_file)
            
            self._initialized = True
            logger.info(f"File manager initialized with output directory: {self.output_dir}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize file manager: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    async def save_text(self, content: str, filename: Optional[str] = None) -> str:
        """Save text content to a file.
        
        Args:
            content: Text content to save
            filename: Optional filename (default: timestamp.txt)
            
        Returns:
            str: Path to saved file
        """
        if not self._initialized:
            raise RuntimeError("File manager not initialized")
            
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'output_{timestamp}.txt'
                
            # Ensure filename has .txt extension
            if not filename.endswith('.txt'):
                filename += '.txt'
                
            # Save file
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved text to: {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Failed to save text file: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    async def save_code(self, content: str, language: str = 'python') -> str:
        """Save code content to a file.
        
        Args:
            content: Code content to save
            language: Programming language (default: python)
            
        Returns:
            str: Path to saved file
        """
        if not self._initialized:
            raise RuntimeError("File manager not initialized")
            
        try:
            # Map language to file extension
            extensions = {
                'python': '.py',
                'javascript': '.js',
                'typescript': '.ts',
                'java': '.java',
                'cpp': '.cpp',
                'c': '.c',
                'html': '.html',
                'css': '.css',
                'sql': '.sql'
            }
            
            ext = extensions.get(language.lower(), '.txt')
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'code_{timestamp}{ext}'
            
            # Save file
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved code to: {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Failed to save code file: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) 