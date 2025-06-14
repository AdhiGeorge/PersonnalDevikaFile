from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import asyncio
from functools import lru_cache
import logging
from src.config import Config

logger = logging.getLogger(__name__)

class SentenceBert:
    _instance = None
    _model = None
    _keybert = None
    _cache = {}
    _config = Config()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentenceBert, cls).__new__(cls)
            cls._initialize_models()
        return cls._instance

    @classmethod
    def _initialize_models(cls):
        """Initialize BERT models with caching."""
        try:
            model_name = cls._config.get("embedding.model", "all-MiniLM-L6-v2")
            cls._model = SentenceTransformer(model_name)
            cls._keybert = KeyBERT(model=cls._model)
            logger.info(f"Initialized BERT models with {model_name}")
        except Exception as e:
            logger.error(f"Error initializing BERT models: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text."""
        return self._model.encode(text)

    async def extract_keywords_async(self, text: str, top_n: int = 5) -> List[str]:
        """Asynchronously extract keywords from text."""
        try:
            # Check cache first
            cache_key = f"{text}_{top_n}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            keywords = await loop.run_in_executor(
                None, 
                lambda: self._keybert.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    top_n=top_n,
                    diversity=0.5
                )
            )
            
            # Cache results
            self._cache[cache_key] = [k[0] for k in keywords]
            return self._cache[cache_key]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Synchronously extract keywords from text."""
        try:
            # Check cache first
            cache_key = f"{text}_{top_n}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            keywords = self._keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_n,
                diversity=0.5
            )
            
            # Cache results
            self._cache[cache_key] = [k[0] for k in keywords]
            return self._cache[cache_key]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        try:
            return self._get_cached_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return np.array([])

    def clear_cache(self):
        """Clear the keyword cache."""
        self._cache.clear()
        self._get_cached_embedding.cache_clear() 