import os
import logging
from typing import Any, Optional
from dotenv import load_dotenv
import re
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the application."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self._config = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
    def _load_env(self):
        """Load environment variables."""
        try:
            load_dotenv()
            logger.info("Environment variables loaded")
        except Exception as e:
            logger.error(f"Failed to load environment variables: {str(e)}")
            raise ValueError(f"Environment loading failed: {str(e)}")
        
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the workspace root.
        
        Args:
            path: Path to resolve
            
        Returns:
            Absolute path
        """
        try:
            if not path:
                return None
            if os.path.isabs(path):
                return path
            return os.path.abspath(os.path.join(os.getcwd(), path))
        except Exception as e:
            logger.error(f"Failed to resolve path: {str(e)}")
            return None
        
    def _validate_api_key(self, key: str, name: str) -> bool:
        """Validate an API key.
        
        Args:
            key: API key to validate
            name: Name of the API key for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not key or not isinstance(key, str):
            logger.error(f"{name} is missing or invalid")
            return False
        if key.strip() == '':
            logger.error(f"{name} is empty")
            return False
        return True
        
    def _validate_endpoint(self, endpoint: str, name: str) -> bool:
        """Validate an endpoint URL.
        
        Args:
            endpoint: Endpoint URL to validate
            name: Name of the endpoint for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not endpoint or not isinstance(endpoint, str):
            logger.error(f"{name} is missing or invalid")
            return False
        if endpoint.strip() == '':
            logger.error(f"{name} is empty")
            return False
        if not re.match(r'^https?://', endpoint):
            logger.error(f"{name} must be a valid URL")
            return False
        return True
        
    def _validate_model(self, model: str, name: str) -> bool:
        """Validate a model name.
        
        Args:
            model: Model name to validate
            name: Name of the model for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not model or not isinstance(model, str):
            logger.error(f"{name} is missing or invalid")
            return False
        if model.strip() == '':
            logger.error(f"{name} is empty")
            return False
        valid_models = {
            'gpt-4': ['gpt-4', 'gpt-4-32k', 'gpt-4o'],
            'gpt-3.5': ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k'],
            'embedding': ['text-embedding-3-small', 'text-embedding-ada-002']
        }
        for model_type, models in valid_models.items():
            if model in models:
                return True
        logger.error(f"{name} must be a valid model name")
        return False
        
    def _validate_path(self, path: str, name: str) -> bool:
        """Validate a file path.
        
        Args:
            path: Path to validate
            name: Name of the path for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not path or not isinstance(path, str):
            logger.error(f"{name} is missing or invalid")
            return False
        if path.strip() == '':
            logger.error(f"{name} is empty")
            return False
        try:
            if os.path.exists(path):
                # If it's a file, check if it's writable
                if os.path.isfile(path):
                    with open(path, 'a'):
                        pass
                else:
                    # If it's a directory, check if we can write to it
                    test_file = os.path.join(path, '.test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
            else:
                # If it doesn't exist, check if we can write to the parent directory
                parent_dir = os.path.dirname(path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                if parent_dir:
                    test_file = os.path.join(parent_dir, '.test')
                else:
                    test_file = '.test'
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            return True
        except Exception as e:
            logger.error(f"{name} is not writable: {str(e)}")
            return False
        
    async def initialize(self) -> bool:
        """Initialize the configuration.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        async with self._init_lock:
            if self._initialized:
                return True
                
            try:
                # Load environment variables
                self._load_env()
                
                # Load configuration from environment
                self._config = {
                    'google': {
                        'api_key': os.getenv('GOOGLE_API_KEY'),
                        'cse_id': os.getenv('GOOGLE_CSE_ID')
                    },
                    'tavily': {
                        'api_key': os.getenv('TAVILY_API_KEY')
                    },
                    'openai': {
                        'api_key': os.getenv('OPENAI_API_KEY'),
                        'azure_api_key': os.getenv('AZURE_OPENAI_API_KEY'),
                        'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
                        'azure_api_version_chat': os.getenv('AZURE_API_VERSION_CHAT'),
                        'azure_api_version_embeddings': os.getenv('AZURE_API_VERSION_EMBEDDINGS'),
                        'azure_deployment': os.getenv('AZURE_OPENAI_DEPLOYMENT'),
                        'azure_embedding_deployment': os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
                    },
                    'qdrant': {
                        'url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
                        'collection': os.getenv('QDRANT_COLLECTION', 'knowledge_base')
                    },
                    'llm': {
                        'model': os.getenv('LLM_MODEL', 'gpt-4o'),
                        'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
                    },
                    'database': {
                        'sqlite_path': self._resolve_path(os.getenv('SQLITE_DB', 'data/database.sqlite'))
                    },
                    'output': {
                        'directory': self._resolve_path(os.getenv('OUTPUT_DIR', 'data/output'))
                    },
                    'storage': {
                        'directory': self._resolve_path(os.getenv('STORAGE_DIR', 'data/storage'))
                    }
                }
                
                logger.info(f"[DIAGNOSTIC] QDRANT_URL loaded in config: {self._config['qdrant']['url']}")
                
                # Validate configuration
                if not self._validate_config():
                    raise ValueError("Configuration validation failed")
                    
                # Create necessary directories
                if not self._create_directories():
                    raise ValueError("Failed to create necessary directories")
                    
                self._initialized = True
                logger.info("Configuration initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize configuration: {str(e)}")
                raise ValueError(f"Configuration initialization failed: {str(e)}")
                
    def _validate_config(self) -> bool:
        """Validate the configuration.
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Validate API keys
            if not self._validate_api_key(self._config['google']['api_key'], 'Google API Key'):
                return False
            if not self._validate_api_key(self._config['tavily']['api_key'], 'Tavily API Key'):
                return False
                
            # Validate OpenAI configuration
            openai_config = self._config['openai']
            if not openai_config['api_key'] and not openai_config['azure_api_key']:
                logger.error("Either OpenAI API Key or Azure OpenAI API Key must be provided")
                return False
                
            if openai_config['azure_api_key']:
                if not self._validate_api_key(openai_config['azure_api_key'], 'Azure OpenAI API Key'):
                    return False
                if not self._validate_endpoint(openai_config['azure_endpoint'], 'Azure OpenAI Endpoint'):
                    return False
                if not openai_config['azure_api_version_chat']:
                    logger.error("Azure API Version for Chat is missing")
                    return False
                if not openai_config['azure_api_version_embeddings']:
                    logger.error("Azure API Version for Embeddings is missing")
                    return False
                if not openai_config['azure_deployment']:
                    logger.error("Azure OpenAI Deployment is missing")
                    return False
                if not openai_config['azure_embedding_deployment']:
                    logger.error("Azure OpenAI Embedding Deployment is missing")
                    return False
            elif not self._validate_api_key(openai_config['api_key'], 'OpenAI API Key'):
                return False
                
            # Validate endpoints
            if not self._validate_endpoint(self._config['qdrant']['url'], 'Qdrant URL'):
                return False
                
            # Validate models
            if not self._validate_model(self._config['llm']['model'], 'LLM Model'):
                return False
            if not self._validate_model(self._config['llm']['embedding_model'], 'Embedding Model'):
                return False
                
            # Validate paths
            if not self._validate_path(self._config['database']['sqlite_path'], 'SQLite Database Path'):
                return False
            if not self._validate_path(self._config['output']['directory'], 'Output Directory'):
                return False
            if not self._validate_path(self._config['storage']['directory'], 'Storage Directory'):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
            
    def _create_directories(self) -> bool:
        """Create necessary directories.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create database directory
            db_dir = os.path.dirname(self._config['database']['sqlite_path'])
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                
            # Create output directory
            if not os.path.exists(self._config['output']['directory']):
                os.makedirs(self._config['output']['directory'], exist_ok=True)
                
            # Create storage directory
            if not os.path.exists(self._config['storage']['directory']):
                os.makedirs(self._config['storage']['directory'], exist_ok=True)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            return False
            
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (optional)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if not self._initialized:
            raise RuntimeError("Configuration not initialized. Call initialize() first.")
            
        try:
            if key is None:
                return self._config.get(section, default)
            return self._config.get(section, {}).get(key, default)
        except Exception as e:
            logger.error(f"Failed to get configuration value: {str(e)}")
            return default
            
    def get_sqlite_db(self) -> str:
        """Get SQLite database path.
        
        Returns:
            str: SQLite database path
        """
        if not self._initialized:
            raise RuntimeError("Configuration not initialized. Call initialize() first.")
        return self._config['database']['sqlite_path']
        
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return
            
        try:
            # Reset state
            self._initialized = False
            self._config.clear()
            
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
            raise ValueError(f"Failed to cleanup: {str(e)}")

async def main():
    """Main function for testing."""
    try:
        # Create config
        config = Config()
        await config.initialize()
        
        # Test configuration
        print(f"Google API Key: {config.get('google', 'api_key')}")
        print(f"Qdrant URL: {config.get('qdrant', 'url')}")
        print(f"LLM Model: {config.get('llm', 'model')}")
        print(f"SQLite DB: {config.get_sqlite_db()}")
        
        # Cleanup
        await config.cleanup()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 