from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class KnowledgeBase:
    def __init__(self):
        """Initialize the knowledge base with Qdrant and embedding model."""
        # Get configuration from environment
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "agent_knowledge")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in your .env file.")
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=self.qdrant_url)
        
        # Initialize embedding model based on configuration
        if self.embedding_model_name == "text-embedding-3-small":
            self.use_openai = True
            self.vector_size = 1536  # text-embedding-3-small
        else:
            self.use_openai = False
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"Knowledge base initialized with model: {self.embedding_model_name}")

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector size from the embedding model
                vector_size = self.vector_size if self.use_openai else self.embedding_model.get_sentence_embedding_dimension()
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using the configured model."""
        if not text or not isinstance(text, str):
            raise ValueError("Text for embedding must be a non-empty string.")
        try:
            if self.use_openai:
                response = openai.Embedding.create(
                    input=text,
                    model=self.embedding_model_name,
                    api_key=self.openai_api_key
                )
                return response['data'][0]['embedding']
            else:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def add_document(self, 
                    text: str, 
                    metadata: Dict[str, Any], 
                    document_id: Optional[str] = None) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            text: The text content to store
            metadata: Additional metadata about the document
            document_id: Optional custom ID for the document
            
        Returns:
            The ID of the added document
        """
        if not text or not isinstance(metadata, dict):
            raise ValueError("Text and metadata are required.")
        try:
            # Generate embedding
            vector = self._get_embedding(text)
            
            # Prepare metadata
            full_metadata = {
                **metadata,
                "text": text,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Generate document ID if not provided
            if not document_id:
                document_id = f"doc_{datetime.utcnow().timestamp()}"
            
            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=document_id,
                        vector=vector,
                        payload=full_metadata
                    )
                ]
            )
            
            logger.info(f"Added document to knowledge base: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {str(e)}")
            raise

    def search(self, 
               query: str, 
               limit: int = 5, 
               score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for similar documents.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of matching documents with their metadata and scores
        """
        if not query:
            raise ValueError("Query must be a non-empty string.")
        try:
            # Generate query embedding
            query_vector = self._get_embedding(query)
            
            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                results.append({
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "metadata": scored_point.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            raise

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document metadata if found, None otherwise
        """
        if not document_id:
            raise ValueError("Document ID is required.")
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id]
            )
            
            if result and len(result) > 0:
                return result[0].payload
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not document_id:
            raise ValueError("Document ID is required.")
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[document_id]
                )
            )
            logger.info(f"Deleted document from knowledge base: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def update_document(self, 
                       document_id: str, 
                       text: Optional[str] = None, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing document in the knowledge base.
        
        Args:
            document_id: The ID of the document to update
            text: New text content (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not document_id:
            raise ValueError("Document ID is required.")
        try:
            # Get existing document
            existing = self.get_document(document_id)
            if not existing:
                return False
            
            # Prepare new metadata
            new_metadata = existing.copy()
            if text:
                new_metadata["text"] = text
                vector = self._get_embedding(text)
            else:
                vector = self._get_embedding(existing["text"])
            
            if metadata:
                new_metadata.update(metadata)
            
            new_metadata["updated_at"] = datetime.utcnow().isoformat()
            
            # Update in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=document_id,
                        vector=vector,
                        payload=new_metadata
                    )
                ]
            )
            
            logger.info(f"Updated document in knowledge base: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                match=models.MatchAny(any=["*"])
                            )
                        ]
                    )
                )
            )
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

if __name__ == "__main__":
    # Real, practical example usage
    try:
        # Initialize knowledge base
        kb = KnowledgeBase()
        
        # Example document about Python async programming
        doc_text = """
        Asynchronous programming in Python allows you to write concurrent code that can handle multiple tasks efficiently.
        The asyncio library provides the infrastructure for writing single-threaded concurrent code using coroutines,
        multiplexing I/O access over sockets and other resources, running network clients and servers, and other related
        primitives.
        """
        
        # Add document with metadata
        doc_id = kb.add_document(
            text=doc_text,
            metadata={
                "title": "Python Async Programming",
                "category": "Programming",
                "tags": ["python", "async", "asyncio", "concurrency"]
            }
        )
        print(f"Added document with ID: {doc_id}")
        
        # Search for similar documents
        query = "How to write concurrent code in Python?"
        results = kb.search(query, limit=3)
        
        print("\nSearch Results:")
        for result in results:
            print(f"\nScore: {result['score']}")
            print(f"Title: {result['metadata'].get('title')}")
            print(f"Text: {result['metadata'].get('text', '')[:200]}...")
        
        # Update document
        updated_text = doc_text + "\n\nKey features include async/await syntax, event loops, and coroutines."
        kb.update_document(
            document_id=doc_id,
            text=updated_text,
            metadata={"updated": True}
        )
        print("\nDocument updated successfully")
        
        # Retrieve updated document
        updated_doc = kb.get_document(doc_id)
        print("\nUpdated Document:")
        print(f"Title: {updated_doc.get('title')}")
        print(f"Text: {updated_doc.get('text', '')[:200]}...")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
