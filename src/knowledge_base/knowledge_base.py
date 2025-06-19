from typing import List, Dict, Any, Optional, Tuple, Iterator
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
from config.config import Config
import uuid
from sqlmodel import Field, Session, SQLModel, create_engine
import tiktoken
import re

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
        self.config = config
        self.qdrant_url = "http://localhost:6333"
        logger.info(f"[FORCED] Qdrant URL being used: {self.qdrant_url}")
        # Get configuration
        # Initialize Qdrant client with explicit URL handling
        env_qdrant_url = os.getenv("QDRANT_URL")
        config_qdrant_url = config.get('qdrant', 'url')
        
        # Debug logging
        logger.info(f"[DEBUG] Environment QDRANT_URL: {env_qdrant_url}")
        logger.info(f"[DEBUG] Config qdrant.url: {config_qdrant_url}")
        
        # Force localhost if the URL contains 'qdrant:'
        if env_qdrant_url and 'qdrant:' in env_qdrant_url:
            logger.warning(f"Replacing QDRANT_URL from {env_qdrant_url} to http://localhost:6333")
            env_qdrant_url = "http://localhost:6333"
            
        if config_qdrant_url and 'qdrant:' in config_qdrant_url:
            logger.warning(f"Replacing config qdrant.url from {config_qdrant_url} to http://localhost:6333")
            config_qdrant_url = "http://localhost:6333"
        
        # Use environment variable if set, otherwise use config, otherwise default to localhost
        self.qdrant_url = env_qdrant_url or config_qdrant_url or "http://localhost:6333"
        logger.info(f"[DEBUG] Final Qdrant URL: {self.qdrant_url}")
        
        # Force collection name to be consistent
        self.collection_name = "knowledge_base"
        self.embedding_model_name = config.get('llm', 'embedding_model') or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
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

    def _get_qdrant_url(self):
        url = self.config.get('qdrant', 'url')
        if not url:
            url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        if not url or url == 'None':
            url = 'http://localhost:6333'
        return url

    def _truncate_to_token_limit(self, text: str, max_tokens: int = 8192, model: str = None) -> str:
        """Truncate text to a maximum number of tokens using tiktoken for accuracy."""
        model_name = model or self.embedding_model_name
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            logger.warning(f"Truncating text from {len(tokens)} to {max_tokens} tokens for model {model_name}.")
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        return text

    async def initialize(self):
        """Initialize async components."""
        try:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(url="http://localhost:6333")
            self._ensure_collection()
            self.use_qdrant = True
            logger.info(f"[FORCED] Knowledge base initialized with Qdrant at {self.qdrant_url}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise

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

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex while preserving newlines as sentence boundaries."""
        # Split on period followed by space/newline, question mark, or exclamation mark
        # Also split on newlines to preserve document structure
        sentences = []
        # First split by newlines to preserve document structure
        paragraphs = text.split('\n')
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            # Split paragraph into sentences
            paragraph_sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
            sentences.extend(paragraph_sentences)
        return [s.strip() for s in sentences if s.strip()]

    def _get_chunk_size_tokens(self, model: str = "text-embedding-3-small") -> int:
        """Get the optimal chunk size in tokens for the given model."""
        if model == "text-embedding-3-small":
            # Use a slightly smaller chunk size to allow for overlap
            return 6000  # Allows for overlap while staying under 8192 limit
        else:
            # For other models, use a conservative default
            return 1500

    def _estimate_tokens(self, text: str, model: str = "text-embedding-3-small") -> int:
        """Estimate the number of tokens in a text string."""
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

    def _create_chunks_with_overlap(self, sentences: List[str], 
                                  model: str = "text-embedding-3-small",
                                  overlap_sentences: int = 2) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Create overlapping chunks from sentences that fit within token limits. All data is chunked and retained."""
        enc = tiktoken.encoding_for_model(model)
        chunk_size_tokens = self._get_chunk_size_tokens(model)
        current_chunk = []
        current_chunk_tokens = 0
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(enc.encode(sentence))
            # If a single sentence exceeds chunk size, split it further by tokens
            if sentence_tokens > chunk_size_tokens:
                logger.warning(f"Single sentence exceeds chunk size, splitting: {sentence[:100]}...")
                words = sentence.split()
                word_chunk = []
                word_chunk_tokens = 0
                for word in words:
                    word_tokens = len(enc.encode(word + ' '))
                    if word_chunk_tokens + word_tokens > chunk_size_tokens:
                        chunk_text = ' '.join(word_chunk)
                        logger.info(f"Yielding word-level chunk (partial): {chunk_text[:60]}... [{word_chunk_tokens} tokens]")
                        yield chunk_text, {
                            'start_idx': i,
                            'end_idx': i,
                            'is_partial': True
                        }
                        word_chunk = [word]
                        word_chunk_tokens = word_tokens
                    else:
                        word_chunk.append(word)
                        word_chunk_tokens += word_tokens
                if word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    logger.info(f"Yielding word-level chunk (partial): {chunk_text[:60]}... [{word_chunk_tokens} tokens]")
                    yield chunk_text, {
                        'start_idx': i,
                        'end_idx': i,
                        'is_partial': True
                    }
                continue
            # If adding this sentence would exceed chunk size, yield current chunk
            if current_chunk_tokens + sentence_tokens > chunk_size_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                metadata = {
                    'start_idx': max(0, i - len(current_chunk)),
                    'end_idx': i - 1,
                    'is_partial': False
                }
                logger.info(f"Yielding chunk: {chunk_text[:60]}... [{current_chunk_tokens} tokens]")
                yield chunk_text, metadata
                # Start new chunk with overlap
                if overlap_sentences > 0:
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_chunk_tokens = sum(len(enc.encode(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_chunk_tokens = 0
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
        # Yield any remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = {
                'start_idx': max(0, len(sentences) - len(current_chunk)),
                'end_idx': len(sentences) - 1,
                'is_partial': False
            }
            logger.info(f"Yielding final chunk: {chunk_text[:60]}... [{current_chunk_tokens} tokens]")
            yield chunk_text, metadata

    def _get_embeddings_for_text(self, text: str) -> List[Tuple[List[float], Dict[str, Any]]]:
        """Get embeddings for text by chunking and embedding each chunk. All data is retained."""
        sentences = self._split_into_sentences(text)
        chunks_with_metadata = list(self._create_chunks_with_overlap(
            sentences,
            model=self.embedding_model_name
        ))
        logger.info(f"Total chunks to embed: {len(chunks_with_metadata)}")
        embeddings_with_metadata = []
        for idx, (chunk_text, metadata) in enumerate(chunks_with_metadata):
            try:
                logger.info(f"Embedding chunk {idx+1}/{len(chunks_with_metadata)} [{len(chunk_text)} chars]")
                embedding = self._get_embedding(chunk_text)
                embeddings_with_metadata.append((embedding, metadata))
            except Exception as e:
                logger.error(f"Error embedding chunk: {str(e)}", exc_info=True)
                continue
        return embeddings_with_metadata

    def _get_embedding(self, text: str) -> list:
        try:
            truncated_text = self._truncate_to_token_limit(text, 8192, self.embedding_model_name)
            response = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=truncated_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in _get_embedding: {str(e)}")
            raise

    async def search(self, query: str, limit: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity."""
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
            
        try:
            # Truncate query to token limit before embedding
            truncated_query = self._truncate_to_token_limit(query, max_tokens=8192, model=self.embedding_model_name)
            query_embedding = self._get_embedding(truncated_query)
            
            if self.use_qdrant:
                search_result = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit * 2,  # Get more results to account for chunks
                    score_threshold=score_threshold
                )
                
                # Group results by original content and combine scores
                grouped_results = {}
                for hit in search_result:
                    content_id = hit.payload.get('content_id')
                    if content_id not in grouped_results:
                        grouped_results[content_id] = {
                            'content': hit.payload.get('content', ''),
                            'metadata': hit.payload.get('metadata', {}),
                            'score': hit.score,
                            'chunk_scores': [hit.score],
                            'chunk_metadata': [hit.payload.get('chunk_metadata', {})]
                        }
                    else:
                        # Update score with max of chunks
                        grouped_results[content_id]['score'] = max(
                            grouped_results[content_id]['score'],
                            hit.score
                        )
                        grouped_results[content_id]['chunk_scores'].append(hit.score)
                        grouped_results[content_id]['chunk_metadata'].append(
                            hit.payload.get('chunk_metadata', {})
                        )
                
                # Sort by max score and take top results
                results = sorted(
                    grouped_results.values(),
                    key=lambda x: x['score'],
                    reverse=True
                )[:limit]
                
                return results
            else:
                logger.warning("Qdrant not available, returning empty results")
                return []
                
        except Exception as e:
            logger.error(f"Error in search: {str(e)}", exc_info=True)
            return []

    async def add(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add content to the knowledge base with chunking support."""
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        try:
            # Generate unique ID for this content
            content_id = str(uuid.uuid4())
            # Get embeddings for all chunks
            embeddings_with_metadata = self._get_embeddings_for_text(content)
            if self.use_qdrant:
                # Prepare points for Qdrant
                points = []
                for i, (embedding, chunk_metadata) in enumerate(embeddings_with_metadata):
                    point_id = str(uuid.uuid4())  # Use a new UUID for each point
                    payload = {
                        'content': content,
                        'content_id': content_id,
                        'chunk_index': i,
                        'chunk_metadata': chunk_metadata,
                        'metadata': metadata or {}
                    }
                    points.append(models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    ))
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    try:
                        self.qdrant_client.upsert(
                            collection_name=self.collection_name,
                            points=batch
                        )
                    except Exception as e:
                        logger.error(f"Error uploading batch to Qdrant: {str(e)}", exc_info=True)
                        raise
                logger.info(f"Added content with {len(points)} chunks to knowledge base")
                return content_id
            else:
                logger.warning("Qdrant not available, content not stored")
                return content_id
        except Exception as e:
            logger.error(f"Error adding content to knowledge base: {str(e)}", exc_info=True)
            raise

    async def delete(self, content_id: str) -> bool:
        """Delete content and all its chunks from the knowledge base."""
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
            
        if not self.use_qdrant:
            logger.warning("Qdrant not available, nothing to delete")
            return False
            
        try:
            # Delete all points with matching content_id
            filter_query = models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_id",
                        match=models.MatchValue(value=content_id)
                    )
                ]
            )
            
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=filter_query
            )
            
            logger.info(f"Deleted content {content_id} from knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting content from knowledge base: {str(e)}", exc_info=True)
            return False

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
            embedding = self._get_embedding(contents)
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
            if self.use_qdrant:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
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
