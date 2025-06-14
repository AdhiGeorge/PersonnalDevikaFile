import yaml
import os
from typing import Dict, Any


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
<<<<<<< HEAD
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
                'NETLIFY': '',
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
                'REPOS_DIR': 'data/repos',
                'SQLITE_DB': 'data/database.sqlite'
            },
            'TIMEOUT': {
                'INFERENCE': 30
            }
        }
        
        # Update defaults with user config
        self._update_nested_dict(defaults, self.config)
        self.config = defaults
=======
        # Load the existing config.yaml file
        if not os.path.exists("config.yaml"):
            raise FileNotFoundError("config.yaml file not found. Please ensure it exists in the project root.")
        
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
>>>>>>> 925f80e (fifth commit)
            
        # ------------------------------------------------------------------
        # Apply environment-variable overrides (e.g. .env file)
        # ------------------------------------------------------------------
        self._apply_env_overrides()

    def get_config(self):
        return self.config

    def get_bing_api_endpoint(self):
        return self.config["search_engines"]["bing"]["endpoint"]

    def get_bing_api_key(self):
<<<<<<< HEAD
        return self.config["API_KEYS"]["BING"]
        
    def _update_nested_dict(self, defaults: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update a nested dictionary with values from another."""
        for key, value in updates.items():
            if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
                self._update_nested_dict(defaults[key], value)
            else:
                defaults[key] = value
=======
        return self.config["search_engines"]["bing"]["api_key"]
>>>>>>> 925f80e (fifth commit)

    def get_google_search_api_key(self):
        return self.config["search_engines"]["google"]["api_key"]

    def get_google_search_engine_id(self):
        return self.config["search_engines"]["google"]["search_engine_id"]

    def get_google_search_api_endpoint(self):
<<<<<<< HEAD
        return self.config["API_ENDPOINTS"]["GOOGLE_SEARCH"]
=======
        return self.config["search_engines"]["google"]["endpoint"]
>>>>>>> 925f80e (fifth commit)

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

    def get_netlify_api_key(self):
        return self.config["deployment"]["netlify"]["api_key"]

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

    def get_repos_dir(self):
        return self.config["storage"]["repos_dir"]

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

    def set_netlify_api_key(self, key):
        self.config["deployment"]["netlify"]["api_key"] = key
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
<<<<<<< HEAD
            yaml.safe_dump(self.config, f)

    def update_config(self, data):
        config_path = "config.yaml"
        # Load current config
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        # Apply updates
        self._update_nested_dict(config_data, data)
        self.config = config_data

        # Save back to file
        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f)

    # Dictionary-style retrieval with dotted path support
    def get(self, path: str, default=None):
        """Retrieve a value using dotted path notation, e.g. "server.port".

        Searches case-sensitively first, then tries upper/lower case variants at
        each level. Returns *default* if any part of the path is missing.
        """
        parts = path.split(".") if isinstance(path, str) else [path]
        cur = self.config
        for part in parts:
            if not isinstance(cur, dict):
                return default
            if part in cur:
                cur = cur[part]
            elif part.upper() in cur:
                cur = cur[part.upper()]
            elif part.lower() in cur:
                cur = cur[part.lower()]
            else:
                return default
        return cur

    # ------------------------------------------------------------------
    # Dynamic attribute access helpers
    # ------------------------------------------------------------------
    def __getattr__(self, item):
        """Enable attribute-style access to config values.

        Allows chained access like ``config.monitoring.metrics.prometheus.port``.
        Keys are looked up case-sensitively first, then as upper- or lower-case
        variants to provide some flexibility between defaults (often UPPERCASE)
        and user YAML keys (often lowercase).
        """
        if item in self.config:
            value = self.config[item]
        elif item.upper() in self.config:
            value = self.config[item.upper()]
        elif item.lower() in self.config:
            value = self.config[item.lower()]
        else:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

        # Wrap nested dictionaries in a namespace for further dot access
        if isinstance(value, dict):
            return _ConfigNamespace(value)
        return value

    def _apply_env_overrides(self):
        """Update loaded config with matching environment variables."""
        import os

        env_map = {
            # Azure / OpenAI keys & endpoint
            "AZURE_OPENAI_API_KEY": ("API_KEYS", "OPENAI"),
            "OPENAI_API_KEY": ("API_KEYS", "OPENAI"),
            "AZURE_OPENAI_ENDPOINT": ("API_ENDPOINTS", "OPENAI"),

            # Model selection
            "AZURE_OPENAI_DEPLOYMENT": ("LLM", "model"),

            # Google search
            "GOOGLE_API_KEY": ("API_KEYS", "GOOGLE_SEARCH"),
            "GOOGLE_CSE_ID": ("API_KEYS", "GOOGLE_SEARCH_ENGINE_ID"),

            # Tavily
            "TAVILY_API_KEY": ("API_KEYS", "TAVILY"),
        }

        for env, path in env_map.items():
            val = os.getenv(env)
            if not val:
                continue
            section, key = path
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = val


# ----------------------------------------------------------------------
# Helper namespace class for nested configuration dictionaries
# ----------------------------------------------------------------------
class _ConfigNamespace:
    """Simple wrapper that provides recursive attribute access to dicts."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, item):
        if item in self._data:
            value = self._data[item]
        elif isinstance(item, str) and item.upper() in self._data:
            value = self._data[item.upper()]
        elif isinstance(item, str) and item.lower() in self._data:
            value = self._data[item.lower()]
        else:
            raise AttributeError(f"No attribute '{item}' in config section")

        if isinstance(value, dict):
            return _ConfigNamespace(value)
        return value

    # Allow dict-style access as well
    def __getitem__(self, key):
        return self._data[key]

    # Dictionary-style retrieval with dotted path support
    def get(self, key, default=None):
        """Dict-like .get() for namespaces."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __repr__(self):
        return repr(self._data)
=======
            yaml.dump(self.config, f)

    def update_config(self, data):
        for key, value in data.items():
            if key in self.config:
                with open("config.yaml", "r+") as f:
                    config = yaml.safe_load(f)
                    for sub_key, sub_value in value.items():
                        self.config[key][sub_key] = sub_value
                        config[key][sub_key] = sub_value
                    f.seek(0)
                    yaml.dump(config, f)
>>>>>>> 925f80e (fifth commit)
