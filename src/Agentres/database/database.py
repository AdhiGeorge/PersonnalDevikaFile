import sqlite3
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
from Agentres.llm.llm import LLM
from Agentres.config.config import Config
import os
import asyncio
import logging
import aiosqlite

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, config: Config):
        """Initialize database with configuration."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        self.config = config
        self.sqlite_path = config.get_sqlite_db()
        self.sqlite_conn = None
        self.qdrant_client = None
        self._llm = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def llm(self) -> LLM:
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = LLM(self.config)
        return self._llm

    async def initialize(self):
        """Initialize async components."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
                
                # Initialize SQLite connection
                self.sqlite_conn = await aiosqlite.connect(self.sqlite_path)
                await self.setup_sqlite()
                
                # Initialize Qdrant client
                self.qdrant_client = qdrant_client.QdrantClient(
                    url=self.config.get('qdrant_url', 'http://localhost:6333')
                )
                await self.setup_qdrant()
                
                # Test SQLite connection
                try:
                    async with self.sqlite_conn.cursor() as cursor:
                        await cursor.execute("SELECT 1")
                        result = await cursor.fetchone()
                        if result[0] != 1:
                            raise ValueError("Failed to query SQLite database")
                        logger.info("SQLite connection test successful")
                except Exception as e:
                    logger.error(f"SQLite connection test failed: {str(e)}")
                    raise ValueError(f"Failed to connect to SQLite: {str(e)}")
                
                # Test Qdrant connection
                try:
                    collections = self.qdrant_client.get_collections()
                    if not collections:
                        raise ValueError("Failed to get Qdrant collections")
                    logger.info("Qdrant connection test successful")
                except Exception as e:
                    logger.error(f"Qdrant connection test failed: {str(e)}")
                    raise ValueError(f"Failed to connect to Qdrant: {str(e)}")
                
                self._initialized = True
                logger.info("Database async components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize async components: {str(e)}")
                raise ValueError(f"Async initialization failed: {str(e)}")

    def _ensure_initialized(self):
        """Ensure database is initialized before use."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        if not self.sqlite_conn or not self.qdrant_client:
            raise RuntimeError("Database components not properly initialized")

    async def setup_sqlite(self):
        """Setup SQLite database with necessary tables."""
        async with self.sqlite_conn.cursor() as cursor:
            # Create conversations table
            await cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create messages table
            await cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
            ''')
            
            # Create query_cache table
            await cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_embedding BLOB,
                conversation_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
            ''')
            
            # Create state table
            await cursor.execute('''
            CREATE TABLE IF NOT EXISTS state (
                project_id TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            await self.sqlite_conn.commit()

    async def setup_qdrant(self):
        """Setup Qdrant collections."""
        # Create collection for knowledge base
        self.qdrant_client.recreate_collection(
            collection_name="knowledge_base",
            vectors_config=VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=Distance.COSINE
            )
        )

    async def generate_conversation_name(self, query: str) -> str:
        """Generate a brief name for the conversation based on the first query."""
        self._ensure_initialized()
        prompt = f"Generate a brief, descriptive name (max 5 words) for a conversation that starts with this query: {query}"
        response = await self.llm.generate(prompt, max_tokens=20)
        return response.strip()

    async def create_conversation(self, query: str) -> int:
        """Create a new conversation and return its ID."""
        self._ensure_initialized()
        name = await self.generate_conversation_name(query)
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO conversations (name) VALUES (?)",
                (name,)
            )
            conversation_id = cursor.lastrowid
            await self.sqlite_conn.commit()
            return conversation_id

    async def add_message(self, conversation_id: int, role: str, content: str):
        """Add a message to a conversation."""
        self._ensure_initialized()
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content)
            )
            await self.sqlite_conn.commit()

    async def get_conversation_history(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get the message history for a conversation."""
        self._ensure_initialized()
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            rows = await cursor.fetchall()
            return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows]

    def hash_query(self, query: str) -> str:
        """Generate a hash for the query."""
        return hashlib.sha256(query.encode()).hexdigest()

    async def check_query_cache(self, query: str) -> Optional[int]:
        """Check if a query exists in cache and return conversation_id if found."""
        self._ensure_initialized()
        query_hash = self.hash_query(query)
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "SELECT conversation_id FROM query_cache WHERE query_hash = ?",
                (query_hash,)
            )
            result = await cursor.fetchone()
            return result[0] if result else None

    async def cache_query(self, query: str, conversation_id: int, embedding: List[float]):
        """Cache a query with its embedding."""
        self._ensure_initialized()
        query_hash = self.hash_query(query)
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO query_cache (query_hash, query_text, query_embedding, conversation_id) VALUES (?, ?, ?, ?)",
                (query_hash, query, json.dumps(embedding), conversation_id)
            )
            await self.sqlite_conn.commit()

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of specified size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    async def store_knowledge(self, text: str, metadata: Dict[str, Any]):
        """Store text in the knowledge base."""
        self._ensure_initialized()
        chunks = self.chunk_text(text)
        embeddings = await self.llm.get_embeddings(chunks)
        
        for chunk, embedding in zip(chunks, embeddings):
            self.qdrant_client.upsert(
                collection_name="knowledge_base",
                points=[
                    models.PointStruct(
                        id=hashlib.md5(chunk.encode()).hexdigest(),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "metadata": metadata
                        }
                    )
                ]
            )

    async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information."""
        self._ensure_initialized()
        query_embedding = await self.llm.get_embeddings([query])[0]
        
        results = self.qdrant_client.search(
            collection_name="knowledge_base",
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            {
                "text": hit.payload["text"],
                "metadata": hit.payload["metadata"],
                "score": hit.score
            }
            for hit in results
        ]

    async def create_state(self, project: str, state: Dict[str, Any]) -> None:
        """Create a new state for a project."""
        self._ensure_initialized()
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO state (project_id, state_data) VALUES (?, ?)",
                (project, json.dumps(state))
            )
            await self.sqlite_conn.commit()

    async def update_state(self, project: str, state: Dict[str, Any]) -> None:
        """Update the state for a project."""
        self._ensure_initialized()
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "UPDATE state SET state_data = ?, updated_at = CURRENT_TIMESTAMP WHERE project_id = ?",
                (json.dumps(state), project)
            )
            await self.sqlite_conn.commit()

    async def delete_state(self, project: str) -> None:
        """Delete the state for a project."""
        self._ensure_initialized()
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM state WHERE project_id = ?",
                (project,)
            )
            await self.sqlite_conn.commit()

    async def get_latest_state(self, project: str) -> Optional[Dict[str, Any]]:
        """Get the latest state for a project."""
        self._ensure_initialized()
        async with self.sqlite_conn.cursor() as cursor:
            await cursor.execute(
                "SELECT state_data FROM state WHERE project_id = ? ORDER BY updated_at DESC LIMIT 1",
                (project,)
            )
            result = await cursor.fetchone()
            return json.loads(result[0]) if result else None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return
            
        try:
            # Close SQLite connection
            if self.sqlite_conn:
                await self.sqlite_conn.close()
                
            # Close Qdrant client
            if self.qdrant_client:
                self.qdrant_client.close()
                
            # Reset state
            self._initialized = False
            self.sqlite_conn = None
            self.qdrant_client = None
            
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
            raise ValueError(f"Failed to cleanup: {str(e)}")

async def main():
    """Main function for testing."""
    try:
        # Create config
        config = Config()
        await config.initialize()
        
        # Create database
        db = Database(config)
        await db.initialize()
        
        # Test database operations
        project = "test_project"
        test_state = {"test_key": "test_value"}
        
        # Test state creation
        await db.create_state(project, test_state)
        
        # Test state update
        test_state["new_key"] = "new_value"
        await db.update_state(project, test_state)
        
        # Test state retrieval
        latest_state = await db.get_latest_state(project)
        print(f"Latest state: {latest_state}")
        
        # Test state deletion
        await db.delete_state(project)
        
        # Cleanup
        await db.cleanup()
        await config.cleanup()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 