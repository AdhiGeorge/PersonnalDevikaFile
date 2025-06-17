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
from Agentres.config import Config

class Database:
    def __init__(self):
        self.sqlite_conn = sqlite3.connect('agent_state.db')
        self.setup_sqlite()
        
        # Initialize Qdrant client
        self.qdrant_client = qdrant_client.QdrantClient(
            path="./qdrant_data"  # Local storage
        )
        self.setup_qdrant()
        
        self.llm = LLM()
        self.config = Config()

    def setup_sqlite(self):
        """Setup SQLite database with necessary tables."""
        cursor = self.sqlite_conn.cursor()
        
        # Create conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create messages table
        cursor.execute('''
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
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            query_embedding BLOB,
            conversation_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        self.sqlite_conn.commit()

    def setup_qdrant(self):
        """Setup Qdrant collections."""
        # Create collection for knowledge base
        self.qdrant_client.recreate_collection(
            collection_name="knowledge_base",
            vectors_config=VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=Distance.COSINE
            )
        )

    def generate_conversation_name(self, query: str) -> str:
        """Generate a brief name for the conversation based on the first query."""
        prompt = f"Generate a brief, descriptive name (max 5 words) for a conversation that starts with this query: {query}"
        response = self.llm.generate(prompt, max_tokens=20)
        return response.strip()

    def create_conversation(self, query: str) -> int:
        """Create a new conversation and return its ID."""
        name = self.generate_conversation_name(query)
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (name) VALUES (?)",
            (name,)
        )
        conversation_id = cursor.lastrowid
        self.sqlite_conn.commit()
        return conversation_id

    def add_message(self, conversation_id: int, role: str, content: str):
        """Add a message to a conversation."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content)
        )
        self.sqlite_conn.commit()

    def get_conversation_history(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get the message history for a conversation."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in cursor.fetchall()]

    def hash_query(self, query: str) -> str:
        """Generate a hash for the query."""
        return hashlib.sha256(query.encode()).hexdigest()

    def check_query_cache(self, query: str) -> Optional[int]:
        """Check if a query exists in cache and return conversation_id if found."""
        query_hash = self.hash_query(query)
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "SELECT conversation_id FROM query_cache WHERE query_hash = ?",
            (query_hash,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def cache_query(self, query: str, conversation_id: int, embedding: List[float]):
        """Cache a query with its embedding."""
        query_hash = self.hash_query(query)
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO query_cache (query_hash, query_text, query_embedding, conversation_id) VALUES (?, ?, ?, ?)",
            (query_hash, query, json.dumps(embedding), conversation_id)
        )
        self.sqlite_conn.commit()

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

    def store_knowledge(self, text: str, metadata: Dict[str, Any]):
        """Store knowledge in Qdrant with proper chunking."""
        chunks = self.chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = self.llm.get_embedding(chunk)
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name="knowledge_base",
                points=[
                    models.PointStruct(
                        id=f"{metadata['id']}_{i}",
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "metadata": metadata,
                            "chunk_index": i
                        }
                    )
                ]
            )

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base using semantic search."""
        query_embedding = self.llm.get_embedding(query)
        
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

    def close(self):
        """Close database connections."""
        self.sqlite_conn.close()
        self.qdrant_client.close() 