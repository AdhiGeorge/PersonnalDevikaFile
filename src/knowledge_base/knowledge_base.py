import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from src.config import Config

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, config: Config):
        """Initialize the knowledge base with configuration."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        self.config = config
        self._client = None
        self._model = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        try:
            self._initialize()
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
            raise

    def _initialize(self):
        """Initialize the knowledge base with retries."""
        for attempt in range(self.max_retries):
            try:
                # Initialize Qdrant client
                self._client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key
                )
                
                # Initialize sentence transformer
                self._model = SentenceTransformer(self.config.embedding_model)
                
                # Create collection if it doesn't exist
                self._ensure_collection()
                
                logger.info(f"Knowledge base initialized with Qdrant at {self.config.qdrant_url}")
                logger.info(f"Knowledge base initialized with model: {self.config.embedding_model}")
                return
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to initialize knowledge base after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(self.retry_delay * (attempt + 1))

    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration."""
        try:
            collections = self._client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.config.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=models.VectorParams(
                        size=self._model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {str(e)}")
            raise

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the knowledge base with retries."""
        try:
            # Generate embedding
            embedding = self._model.encode(text)
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Prepare point
            point = models.PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload={
                    "text": text,
                    **(metadata or {})
                }
            )
            
            # Add to collection with retries
            for attempt in range(self.max_retries):
                try:
                    self._client.upsert(
                        collection_name=self.config.collection_name,
                        points=[point]
                    )
                    logger.info(f"Added document {doc_id} to knowledge base")
                    return doc_id
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to add document after {self.max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base with retries."""
        try:
            # Generate query embedding
            query_embedding = self._model.encode(query)
            
            # Search with retries
            for attempt in range(self.max_retries):
                try:
                    results = self._client.search(
                        collection_name=self.config.collection_name,
                        query_vector=query_embedding.tolist(),
                        limit=limit
                    )
                    
                    # Format results
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            "id": result.id,
                            "score": result.score,
                            "text": result.payload.get("text", ""),
                            "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                        })
                    
                    return formatted_results
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to search after {self.max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the knowledge base with retries."""
        for attempt in range(self.max_retries):
            try:
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.PointIdsList(
                        points=[doc_id]
                    )
                )
                logger.info(f"Deleted document {doc_id} from knowledge base")
                return True
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to delete document after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(self.retry_delay * (attempt + 1))

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document from the knowledge base with retries."""
        for attempt in range(self.max_retries):
            try:
                result = self._client.retrieve(
                    collection_name=self.config.collection_name,
                    ids=[doc_id]
                )
                
                if not result:
                    return None
                    
                point = result[0]
                return {
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                }
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to get document after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(self.retry_delay * (attempt + 1))

    def clear_collection(self) -> bool:
        """Clear all documents from the collection with retries."""
        for attempt in range(self.max_retries):
            try:
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[]
                        )
                    )
                )
                logger.info(f"Cleared collection {self.config.collection_name}")
                return True
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to clear collection after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(self.retry_delay * (attempt + 1)) 