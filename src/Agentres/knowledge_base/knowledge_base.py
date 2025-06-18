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
from openai import OpenAI, AzureOpenAI
from Agentres.config.config import Config
import uuid
from sqlmodel import Field, Session, SQLModel, create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Project(SQLModel, table=True):
    """Project metadata and state."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "active"  # active, completed, archived
    project_metadata: str  # JSON string of additional metadata

class AgentState(SQLModel, table=True):
    """Agent state and metadata."""
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    agent_type: str  # researcher, planner, etc.
    state: str  # JSON string of agent state
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    agent_metadata: str  # JSON string of additional metadata

class Knowledge(SQLModel, table=True):
    """Knowledge base entry."""
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    agent_state_id: Optional[int] = Field(default=None, foreign_key="agentstate.id")
    content: str
    meta: Optional[str] = Field(default=None, sa_column_kwargs={"nullable": True})  # Store as JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class KnowledgeBase:
    def __init__(self, config: Config):
        """Initialize the knowledge base with Qdrant and SQLite."""
        if not isinstance(config, Config):
            raise ValueError("config must be an instance of Config")
            
        # Get configuration from config object
        self.qdrant_url = config.get('qdrant', 'url') or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.embedding_model_name = config.get('llm', 'embedding_model') or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        # Force collection name to be 'knowledge_base' to match existing collection
        self.collection_name = "knowledge_base"
        
        # Initialize SQLite
        sqlite_path = config.get('database', 'sqlite_path')
        self.engine = create_engine(f"sqlite:///{sqlite_path}")
        SQLModel.metadata.create_all(self.engine)
        logger.info(f"SQLite knowledge base initialized at {sqlite_path}")
        
        # Check for OpenAI or Azure OpenAI configuration
        self.openai_api_key = config.get('openai', 'api_key')
        self.azure_openai_api_key = config.get('openai', 'azure_api_key')
        self.azure_openai_endpoint = config.get('openai', 'azure_endpoint')
        
        if not (self.openai_api_key or (self.azure_openai_api_key and self.azure_openai_endpoint)):
            raise ValueError("Either OPENAI_API_KEY or (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT) must be set in your .env file.")
        
        # Initialize OpenAI client based on configuration
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            self.client = AzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=os.getenv("AZURE_API_VERSION_EMBEDDINGS", "2023-05-15"),
                azure_endpoint=self.azure_openai_endpoint
            )
            self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        else:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.deployment = self.embedding_model_name
        
        # Initialize embedding model based on configuration
        if self.embedding_model_name == "text-embedding-3-small":
            self.use_openai = True
            self.vector_size = 1536  # text-embedding-3-small
        else:
            self.use_openai = False
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize Qdrant client if available
        self.use_qdrant = False
        self._initialized = False

    async def initialize(self):
        """Initialize async components."""
        try:
            # Initialize Qdrant client
            try:
                self.qdrant_client = QdrantClient(url=self.qdrant_url)
                self._ensure_collection()
                self.use_qdrant = True
                logger.info(f"Knowledge base initialized with Qdrant at {self.qdrant_url}")
            except Exception as e:
                logger.error(f"Qdrant initialization failed: {str(e)}", exc_info=True)
                return False
            
            # Test embedding generation
            test_text = "Test embedding generation"
            try:
                self._get_embedding(test_text)
                logger.info("Embedding generation tested successfully")
            except Exception as e:
                logger.error(f"Error testing embedding generation: {str(e)}", exc_info=True)
                return False
            
            self._initialized = True
            logger.info(f"Knowledge base initialized with model: {self.embedding_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize async components: {str(e)}", exc_info=True)
            return False

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        if not self.use_qdrant:
            return
            
        try:
            # Get vector size based on model
            vector_size = self.vector_size if self.use_openai else self.embedding_model.get_sentence_embedding_dimension()
            
            # Create collection if it doesn't exist
            try:
                collections = self.qdrant_client.get_collections().collections
                collection_names = [collection.name for collection in collections]
            except Exception as e:
                logger.error(f"Error getting collections: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to connect to Qdrant at {self.qdrant_url}. Please ensure Qdrant is running.")
            
            if self.collection_name not in collection_names:
                try:
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created new collection: {self.collection_name}")
                except Exception as e:
                    logger.error(f"Error creating collection: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Failed to create collection {self.collection_name}. Error: {str(e)}")
            else:
                # Update collection if it exists but has wrong vector size
                try:
                    collection_info = self.qdrant_client.get_collection(self.collection_name)
                    if collection_info.config.params.vectors.size != vector_size:
                        # Delete and recreate collection
                        self.qdrant_client.delete_collection(self.collection_name)
                        self.qdrant_client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=vector_size,
                                distance=Distance.COSINE
                            )
                        )
                        logger.info(f"Recreated collection {self.collection_name} with correct vector size")
                except Exception as e:
                    logger.error(f"Error updating collection: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Failed to update collection {self.collection_name}. Error: {str(e)}")
                    
            # Verify collection was created successfully
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                if collection_info.config.params.vectors.size != vector_size:
                    raise RuntimeError(f"Collection {self.collection_name} was created but has incorrect vector size")
                logger.info(f"Collection {self.collection_name} verified successfully")
            except Exception as e:
                logger.error(f"Error verifying collection: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to verify collection {self.collection_name}. Error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}", exc_info=True)
            self.use_qdrant = False
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using the configured model."""
        if not text or not isinstance(text, str):
            raise ValueError("Text for embedding must be a non-empty string.")
        
        try:
            if self.use_openai:
                response = None
                try:
                    # For Azure OpenAI or standard OpenAI
                    if hasattr(self, 'azure_openai_api_key') and self.azure_openai_api_key:
                        # Azure OpenAI
                        response = self.client.embeddings.create(
                            input=text,
                            model=self.deployment
                        )
                    else:
                        # Standard OpenAI
                        response = self.client.embeddings.create(
                            input=text,
                            model=self.embedding_model_name
                        )
                    
                    # Handle the response
                    if response and hasattr(response, 'data') and response.data:
                        if isinstance(response.data, list) and len(response.data) > 0:
                            if hasattr(response.data[0], 'embedding'):
                                return response.data[0].embedding
                            elif isinstance(response.data[0], dict) and 'embedding' in response.data[0]:
                                return response.data[0]['embedding']
                    
                    # If we get here, the response format was unexpected
                    error_msg = f"Unexpected response format from embedding API: {response}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                except Exception as api_error:
                    error_msg = f"Error calling embedding API: {str(api_error)}"
                    logger.error(error_msg, exc_info=True)
                    raise ValueError(error_msg) from api_error
            
            else:
                # For local models
                try:
                    return self.embedding_model.encode(text).tolist()
                except Exception as local_error:
                    error_msg = f"Error with local embedding model: {str(local_error)}"
                    logger.error(error_msg, exc_info=True)
                    raise ValueError(error_msg) from local_error
        
        except Exception as e:
            error_msg = f"Unexpected error in _get_embedding: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    async def search(self, query: str, limit: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search the knowledge base for similar documents."""
        try:
            # Ensure async components are initialized
            if not self._initialized:
                await self.initialize()
                
            if not query:
                raise ValueError("Query must be a non-empty string.")
                
            if not self.use_qdrant:
                return []
                
            # Generate query embedding
            query_vector = self._get_embedding(query)
            
            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []

    def add_project(self, name: str, description: str, metadata: Dict[str, Any] = None) -> int:
        """Add a new project."""
        try:
            project = Project(
                name=name,
                description=description,
                project_metadata=json.dumps(metadata or {})
            )
            with Session(self.engine) as session:
                session.add(project)
                session.commit()
                session.refresh(project)
                return project.id
        except Exception as e:
            logger.error(f"Error adding project: {str(e)}")
            raise

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        try:
            with Session(self.engine) as session:
                project = session.get(Project, project_id)
                if project:
                    return {
                        "id": project.id,
                        "name": project.name,
                        "description": project.description,
                        "created_at": project.created_at,
                        "updated_at": project.updated_at,
                        "status": project.status,
                        "metadata": json.loads(project.project_metadata)
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting project: {str(e)}")
            return None

    def update_project(self, project_id: int, **kwargs) -> bool:
        """Update project metadata."""
        try:
            with Session(self.engine) as session:
                project = session.get(Project, project_id)
                if project:
                    for key, value in kwargs.items():
                        if key == "metadata" and isinstance(value, dict):
                            value = json.dumps(value)
                            key = "project_metadata"
                        setattr(project, key, value)
                    project.updated_at = datetime.utcnow()
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating project: {str(e)}")
            return False

    def add_agent_state(self, project_id: int, agent_type: str, state: Dict[str, Any], metadata: Dict[str, Any] = None) -> int:
        """Add agent state."""
        try:
            agent_state = AgentState(
                project_id=project_id,
                agent_type=agent_type,
                state=json.dumps(state),
                agent_metadata=json.dumps(metadata or {})
            )
            with Session(self.engine) as session:
                session.add(agent_state)
                session.commit()
                session.refresh(agent_state)
                return agent_state.id
        except Exception as e:
            logger.error(f"Error adding agent state: {str(e)}")
            raise

    def get_agent_state(self, agent_state_id: int) -> Optional[Dict[str, Any]]:
        """Get agent state by ID."""
        try:
            with Session(self.engine) as session:
                agent_state = session.get(AgentState, agent_state_id)
                if agent_state:
                    return {
                        "id": agent_state.id,
                        "project_id": agent_state.project_id,
                        "agent_type": agent_state.agent_type,
                        "state": json.loads(agent_state.state),
                        "created_at": agent_state.created_at,
                        "updated_at": agent_state.updated_at,
                        "metadata": json.loads(agent_state.agent_metadata)
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting agent state: {str(e)}")
            return None

    def update_agent_state(self, agent_state_id: int, state: Dict[str, Any], metadata: Dict[str, Any] = None) -> bool:
        """Update agent state."""
        try:
            with Session(self.engine) as session:
                agent_state = session.get(AgentState, agent_state_id)
                if agent_state:
                    agent_state.state = json.dumps(state)
                    if metadata:
                        agent_state.agent_metadata = json.dumps(metadata)
                    agent_state.updated_at = datetime.utcnow()
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating agent state: {str(e)}")
            return False

    def get_project_agent_states(self, project_id: int, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all agent states for a project."""
        try:
            with Session(self.engine) as session:
                query = session.query(AgentState).filter(AgentState.project_id == project_id)
                if agent_type:
                    query = query.filter(AgentState.agent_type == agent_type)
                agent_states = query.all()
                return [{
                    "id": state.id,
                    "project_id": state.project_id,
                    "agent_type": state.agent_type,
                    "state": json.loads(state.state),
                    "created_at": state.created_at,
                    "updated_at": state.updated_at,
                    "metadata": json.loads(state.agent_metadata)
                } for state in agent_states]
        except Exception as e:
            logger.error(f"Error getting project agent states: {str(e)}")
            return []

    def add_knowledge(self, tag: str, contents: str, metadata: Dict[str, Any] = None) -> int:
        """Add knowledge to the base."""
        try:
            # Generate embedding
            embedding = self._get_embedding(contents)
            
            # Store in SQLite
            knowledge = Knowledge(
                tag=tag,
                content=contents,
                meta=json.dumps(metadata) if metadata else None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            with Session(self.engine) as session:
                session.add(knowledge)
                session.commit()
                session.refresh(knowledge)
                
            # Store in Qdrant if available
            if self.use_qdrant:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=str(knowledge.id),
                            vector=embedding,
                            payload={
                                "tag": tag,
                                "content": contents,
                                "meta": knowledge.meta
                            }
                        )
                    ]
                )
            
            return knowledge.id
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            raise

    def get_knowledge(self, tag: str) -> Optional[Dict[str, Any]]:
        """Get knowledge by tag."""
        try:
            with Session(self.engine) as session:
                knowledge = session.query(Knowledge).filter(Knowledge.tag == tag).first()
                if knowledge:
                    return {
                        "id": knowledge.id,
                        "tag": knowledge.tag,
                        "content": knowledge.content,
                        "meta": json.loads(knowledge.meta) if knowledge.meta else None,
                        "created_at": knowledge.created_at,
                        "updated_at": knowledge.updated_at
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting knowledge: {str(e)}")
            return None

    def update_knowledge(self, tag: str, contents: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update knowledge."""
        try:
            with Session(self.engine) as session:
                knowledge = session.query(Knowledge).filter(Knowledge.tag == tag).first()
                if knowledge:
                    if contents:
                        knowledge.content = contents
                        # Update embedding
                        embedding = self._get_embedding(contents)
                        knowledge.meta = json.dumps(metadata) if metadata else None
                    if metadata:
                        knowledge.meta = json.dumps(metadata)
                    knowledge.updated_at = datetime.utcnow()
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating knowledge: {str(e)}")
            return False

    def delete_knowledge(self, tag: str) -> bool:
        """Delete knowledge."""
        try:
            with Session(self.engine) as session:
                knowledge = session.query(Knowledge).filter(Knowledge.tag == tag).first()
                if knowledge:
                    # Delete from Qdrant if available
                    if self.use_qdrant:
                        self.qdrant_client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(
                                points=[str(knowledge.id)]
                            )
                        )
                    session.delete(knowledge)
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting knowledge: {str(e)}")
            return False

    def clear_knowledge(self) -> bool:
        """Clear all knowledge."""
        try:
            with Session(self.engine) as session:
                session.query(Knowledge).delete()
                session.commit()
                
            # Clear Qdrant if available
            if self.use_qdrant:
                self.qdrant_client.delete_collection(self.collection_name)
                self._ensure_collection()
                
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge: {str(e)}")
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
        doc_id = kb.add_knowledge(
            tag="Python Async Programming",
            contents=doc_text,
            metadata={
                "category": "Programming",
                "tags": ["python", "async", "asyncio", "concurrency"]
            }
        )
        print(f"Added knowledge with ID: {doc_id}")
        
        # Search for similar documents
        query = "How to write concurrent code in Python?"
        results = kb.search(query, limit=3)
        
        print("\nSearch Results:")
        for result in results:
            print(f"\nScore: {result['score']}")
            print(f"Tag: {result['payload'].get('tag')}")
            print(f"Text: {result['payload'].get('content', '')[:200]}...")
        
        # Update knowledge
        updated_text = doc_text + "\n\nKey features include async/await syntax, event loops, and coroutines."
        kb.update_knowledge(
            tag="Python Async Programming",
            contents=updated_text,
            metadata={"updated": True}
        )
        print("\nKnowledge updated successfully")
        
        # Retrieve updated knowledge
        updated_knowledge = kb.get_knowledge("Python Async Programming")
        print("\nUpdated Knowledge:")
        print(f"Tag: {updated_knowledge.get('tag')}")
        print(f"Text: {updated_knowledge.get('content', '')[:200]}...")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
