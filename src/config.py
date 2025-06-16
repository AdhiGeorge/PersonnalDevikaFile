import yaml
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def __init__(self):
        """Initialize configuration with environment variables."""
        # Load environment variables from .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        logger.info(f"Loading environment variables from: {env_path}")
        load_dotenv(env_path, override=True)
        
        # Azure OpenAI Configuration
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.model = os.getenv("MODEL_NAME", "gpt-4")
        self.model_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        # Model Configuration
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        
        # Search Engine Configuration
        self.search_engine = os.getenv("SEARCH_ENGINE", "google")
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
        
        # Google Search Configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        # Tavily Configuration
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Qdrant Configuration
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "devika_kb")
        
        # Embedding Model Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Log the loaded configuration (without sensitive values)
        logger.info("Configuration loaded with:")
        logger.info(f"Google API Key present: {bool(self.google_api_key)}")
        logger.info(f"Google CSE ID present: {bool(self.google_cse_id)}")
        logger.info(f"Tavily API Key present: {bool(self.tavily_api_key)}")
        
        # Debug log the actual values (first few characters only)
        if self.google_api_key:
            logger.info(f"Google API Key value: {self.google_api_key[:8]}...")
        if self.google_cse_id:
            logger.info(f"Google CSE ID value: {self.google_cse_id[:8]}...")
        if self.tavily_api_key:
            logger.info(f"Tavily API Key value: {self.tavily_api_key[:8]}...")
        
        # Validate required configuration
        self._validate_config()
        
        # Model configurations
        self.model_configs = {
            "gpt-4o": {
                "type": "chat",
                "max_tokens": 4000,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "gpt-35-turbo": {
                "type": "chat",
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "text-embedding-ada-002": {
                "type": "embedding",
                "max_tokens": 8191
            }
        }

        logger.info("Configuration initialized")

    def _load_config(self):
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}. "
                                "Please create a config.yaml file in the project root.")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) or {}
        
        # Set default values if not present in config
        defaults = {
            'LLM': {
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2048,
                'api_key': ''
            },
            'SERVER': {
                'host': '0.0.0.0',
                'port': 8080,
                'debug': True
            },
            'LOGGING': {
                'level': 'INFO',
                'file': 'logs/app.log',
                'LOG_REST_API': 'false',
                'LOG_PROMPTS': 'false'
            },
            'MEMORY': {
                'enabled': True,
                'type': 'local'
            },
            'API_KEYS': {
                'BING': '',
                'GOOGLE_SEARCH': '',
                'GOOGLE_SEARCH_ENGINE_ID': '',
                'CLAUDE': '',
                'OPENAI': '',
                'GEMINI': '',
                'MISTRAL': '',
                'GROQ': '',
                'TAVILY': ''
            },
            'API_ENDPOINTS': {
                'BING': 'https://api.bing.microsoft.com/v7.0/search',
                'GOOGLE': 'https://www.googleapis.com/customsearch/v1',
                'GOOGLE_SEARCH': 'https://www.googleapis.com/customsearch/v1',
                'LM_STUDIO': 'http://localhost:1234/v1',
                'OPENAI': 'https://api.openai.com/v1'
            },
            'STORAGE': {
                'LOGS_DIR': 'logs',
                'SCREENSHOTS_DIR': 'data/screenshots',
                'PDFS_DIR': 'data/pdfs',
                'PROJECTS_DIR': 'data/projects',
                'SQLITE_DB': 'data/database.sqlite'
            },
            'TIMEOUT': {
                'INFERENCE': 30
            }
        }
        
        # Update defaults with user config
        self._update_nested_dict(defaults, self.config)
        self.config = defaults
            
        # Apply environment-variable overrides
        self._apply_env_overrides()

    def _update_nested_dict(self, defaults: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update a nested dictionary with values from another."""
        for key, value in updates.items():
            if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
                self._update_nested_dict(defaults[key], value)
            else:
                defaults[key] = value

    def get_config(self):
        return self.config

    def get_bing_api_endpoint(self):
        return self.config["API_ENDPOINTS"]["BING"]

    def get_bing_api_key(self):
        return self.config["API_KEYS"]["BING"]

    def get_google_search_api_key(self):
        return self.config["API_KEYS"]["GOOGLE_SEARCH"]

    def get_google_search_engine_id(self):
        return self.config["API_KEYS"]["GOOGLE_SEARCH_ENGINE_ID"]

    def get_google_search_api_endpoint(self):
        return self.config["API_ENDPOINTS"]["GOOGLE_SEARCH"]

    def get_lmstudio_api_endpoint(self):
        return self.config["llm_providers"]["lmstudio"]["endpoint"]

    def get_claude_api_key(self):
        return self.config["llm_providers"]["anthropic"]["api_key"]

    def get_openai_api_key(self):
        return self.config["llm_providers"]["openai"]["api_key"]

    def get_openai_api_base_url(self):
        return self.config["llm_providers"]["openai"]["endpoint"]

    def get_gemini_api_key(self):
        return self.config["llm_providers"]["google"]["api_key"]

    def get_mistral_api_key(self):
        return self.config["llm_providers"]["mistral"]["api_key"]

    def get_groq_api_key(self):
        return self.config["llm_providers"]["groq"]["api_key"]

    def get_tavily_api_key(self):
        return self.config["API_KEYS"]["TAVILY"]

    def get_sqlite_db(self):
        return self.config["storage"]["sqlite_db"]

    def get_screenshots_dir(self):
        return self.config["storage"]["screenshots_dir"]

    def get_pdfs_dir(self):
        return self.config["storage"]["pdfs_dir"]

    def get_projects_dir(self):
        return self.config["storage"]["projects_dir"]

    def get_logs_dir(self):
        return self.config["storage"]["logs_dir"]

    def get_logging_rest_api(self):
        return self.config["logging"]["log_rest_api"]

    def get_logging_prompts(self):
        return self.config["logging"]["log_prompts"]
    
    def get_timeout_inference(self):
        return self.config["azure_openai"]["timeout"]

    def get_azure_openai_endpoint(self):
        return self.config["azure_openai"]["endpoint"]

    def get_azure_openai_api_key(self):
        return self.config["azure_openai"]["api_key"]

    def set_bing_api_key(self, key):
        self.config["search_engines"]["bing"]["api_key"] = key
        self.save_config()

    def set_bing_api_endpoint(self, endpoint):
        self.config["search_engines"]["bing"]["endpoint"] = endpoint
        self.save_config()

    def set_google_search_api_key(self, key):
        self.config["search_engines"]["google"]["api_key"] = key
        self.save_config()

    def set_google_search_engine_id(self, key):
        self.config["search_engines"]["google"]["search_engine_id"] = key
        self.save_config()

    def set_google_search_api_endpoint(self, endpoint):
        self.config["search_engines"]["google"]["endpoint"] = endpoint
        self.save_config()

    def set_lmstudio_api_endpoint(self, endpoint):
        self.config["llm_providers"]["lmstudio"]["endpoint"] = endpoint
        self.save_config()

    def set_claude_api_key(self, key):
        self.config["llm_providers"]["anthropic"]["api_key"] = key
        self.save_config()

    def set_openai_api_key(self, key):
        self.config["llm_providers"]["openai"]["api_key"] = key
        self.save_config()

    def set_openai_api_endpoint(self, endpoint):
        self.config["llm_providers"]["openai"]["endpoint"] = endpoint
        self.save_config()

    def set_gemini_api_key(self, key):
        self.config["llm_providers"]["google"]["api_key"] = key
        self.save_config()

    def set_mistral_api_key(self, key):
        self.config["llm_providers"]["mistral"]["api_key"] = key
        self.save_config()

    def set_groq_api_key(self, key):
        self.config["llm_providers"]["groq"]["api_key"] = key
        self.save_config()

    def set_tavily_api_key(self, key):
        self.config["API_KEYS"]["TAVILY"] = key
        self.save_config()

    def set_logging_rest_api(self, value):
        self.config["logging"]["log_rest_api"] = value
        self.save_config()

    def set_logging_prompts(self, value):
        self.config["logging"]["log_prompts"] = value
        self.save_config()

    def set_timeout_inference(self, value):
        self.config["azure_openai"]["timeout"] = value
        self.save_config()

    def save_config(self):
        with open("config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)

    def update_config(self, data):
        config_path = "config.yaml"
        # Load current config
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        # Apply updates
        self._update_nested_dict(config_data, data)
        
        # Save updated config
        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f)
        
        # Reload config
        self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        # Convert key to lowercase and replace underscores with dots
        attr_name = key.lower().replace('.', '_')
        return getattr(self, attr_name, default)

    def __getattr__(self, item):
        """Allow accessing config values as attributes."""
        if item in self.config:
            return self.config[item]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'OPENAI_API_KEY': ('llm_providers.openai.api_key', str),
            'CLAUDE_API_KEY': ('llm_providers.anthropic.api_key', str),
            'GEMINI_API_KEY': ('llm_providers.google.api_key', str),
            'MISTRAL_API_KEY': ('llm_providers.mistral.api_key', str),
            'GROQ_API_KEY': ('llm_providers.groq.api_key', str),
            'BING_API_KEY': ('search_engines.bing.api_key', str),
            'GOOGLE_SEARCH_API_KEY': ('search_engines.google.api_key', str),
            'GOOGLE_SEARCH_ENGINE_ID': ('search_engines.google.search_engine_id', str),
            'TAVILY_API_KEY': ('API_KEYS.TAVILY', str),
            'LOG_LEVEL': ('logging.level', str),
            'LOG_REST_API': ('logging.log_rest_api', str),
            'LOG_PROMPTS': ('logging.log_prompts', str),
            'INFERENCE_TIMEOUT': ('timeout.inference', int)
        }

        # Add Azure OpenAI specific environment variable mappings
        if "AZURE_OPENAI_API_KEY" in os.environ:
            self._set_nested_value(self.config, ["azure_openai", "api_key"], os.environ["AZURE_OPENAI_API_KEY"])
        if "AZURE_OPENAI_ENDPOINT" in os.environ:
            self._set_nested_value(self.config, ["azure_openai", "endpoint"], os.environ["AZURE_OPENAI_ENDPOINT"])

        for env_var, (config_path, type_cast) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if type_cast:
                    value = type_cast(value)
                self._set_nested_value(self.config, config_path.split('.'), value)

    def _set_nested_value(self, config_dict, path, value):
        """Set a value in a nested dictionary using a path."""
        if len(path) == 1:
            config_dict[path[0]] = value
        else:
            if path[0] not in config_dict:
                config_dict[path[0]] = {}
            self._set_nested_value(config_dict[path[0]], path[1:], value)

    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        return self.model_configs.get(model_id)

    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.azure_api_key:
            raise ValueError("Azure OpenAI API key not configured")
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not configured")
        return True

    def _validate_config(self):
        """Validate required configuration."""
        if not self.google_api_key:
            logger.warning("GOOGLE_API_KEY not set in environment variables")
        if not self.google_cse_id:
            logger.warning("GOOGLE_CSE_ID not set in environment variables")
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set in environment variables")

    def get_model_config(self) -> dict:
        """Get model configuration for Azure OpenAI."""
        return {
            "api_key": self.azure_api_key,
            "api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
            "deployment_name": self.deployment_name
        }
