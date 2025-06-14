import asyncio
<<<<<<< HEAD
from typing import Tuple
=======
import tiktoken
from functools import lru_cache
from typing import List, Tuple
import os
>>>>>>> 925f80e (fifth commit)

from src.socket_instance import emit_agent
from .azure_openai_client import AzureOpenAI
from src.state import AgentState
from src.config import Config
from src.utils.token_tracker import TokenTracker
import logging
from datetime import datetime, timezone
import tiktoken

TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

logger = logging.getLogger(__name__)
agentState = AgentState()
config = Config()

class LLM:
    _cache = {}
    _rate_limit = {}
    _lock = asyncio.Lock()
    _config = Config()
    _token_tracker = TokenTracker()

    def __init__(self, model_id: str = None):
        self.model_id = model_id
        self.log_prompts = config.get_logging_prompts()
        self.timeout_inference = config.get_timeout_inference()
        self.config_dict = config.get_config()
        self.rate_limit_config = self.config_dict.get("server", {}).get("rate_limit", {})
        self.requests_per_minute = self.rate_limit_config.get("requests_per_minute", 60)
        self.burst_size = self.rate_limit_config.get("burst_size", 10)
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        self.models = {
            "AZURE_OPENAI": [
                (deployment_name, deployment_name),
            ]
        }

    def list_models(self) -> dict:
        return self.models

    def model_enum(self, model_name: str) -> Tuple[str, str]:
        """Return (provider_enum, internal_model_id) for *model_name*.

        Accepts either the display name (e.g. "GPT-4o") or the internal ID
        (e.g. "gpt-4o"), case-insensitive.
        """
        if not model_name:
            return (None, None)

        query = model_name.strip().lower()

        for provider_enum, models in self.models.items():
            for display_name, internal_id in models:
                if query in {display_name.lower(), internal_id.lower()}:
                    return provider_enum, internal_id

        # Fallback: assume Azure OpenAI with the provided model_name
        return ("AZURE_OPENAI", model_name)

    @staticmethod
    def update_global_token_usage(string: str, project_name: str):
        token_usage = len(TIKTOKEN_ENC.encode(string))
        agentState.update_token_usage(project_name, token_usage)

        total = agentState.get_latest_token_usage(project_name) + token_usage
        emit_agent("tokens", {"token_usage": total})

    # ------------------------------------------------------------------
    # Public synchronous entrypoint – safe for normal (blocking) calls.
    # Internally delegates to the async implementation.
    # ------------------------------------------------------------------
    async def ainference(self, prompt: str, project_name: str) -> str:
        cache_key = (self.model_id, prompt)
        if cache_key in self._cache:
            logger.info(f"LLM cache hit for {cache_key}")
            return self._cache[cache_key]

        logger.info(f"Attempting inference with model_id: {self.model_id}")

        # Rate limiting
        now = datetime.now(timezone.utc)
        window = now.replace(second=0, microsecond=0)
        if self.model_id not in self._rate_limit:
            self._rate_limit[self.model_id] = {}
        if window not in self._rate_limit[self.model_id]:
            self._rate_limit[self.model_id][window] = 0
        if self._rate_limit[self.model_id][window] >= self.requests_per_minute:
            logger.warning(f"LLM rate limit exceeded for {self.model_id}")
            await asyncio.sleep(60)
        self._rate_limit[self.model_id][window] += 1

        # Error handling and retries
        error_handling = self.config_dict.get("error_handling", {})
        max_retries = error_handling.get("max_retries", 3)
        retry_delay = error_handling.get("retry_delay", 2)
        backoff_factor = error_handling.get("backoff_factor", 2)
        attempt = 0
        while attempt < max_retries:
            try:
                model_enum, model_name = self.model_enum(self.model_id)
                if model_enum is None:
                    raise ValueError(f"Model {self.model_id} not supported")
                if model_enum == "AZURE_OPENAI":
                    client = AzureOpenAI()
                    response = await client.inference(model_name, prompt)
                else:
                    raise ValueError(f"Unsupported model enum: {model_enum}")
                # Token/cost tracking
                self._token_tracker.track_usage(self.model_id, prompt, response, {"project_name": project_name})
                self._cache[cache_key] = response
                return response
            except Exception as e:
                logger.error(f"LLM inference error: {str(e)} (attempt {attempt+1})")
                await asyncio.sleep(retry_delay * (backoff_factor ** attempt))
                attempt += 1
        raise RuntimeError(f"LLM inference failed after {max_retries} attempts")

    def inference(self, prompt: str, project_name: str) -> str:
        """Blocking helper for synchronous callers (runs its own event loop)."""
        return asyncio.run(self.ainference(prompt, project_name))
