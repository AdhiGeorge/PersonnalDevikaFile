import pytest
import os
from datetime import datetime
from agent.core.knowledge_base import KnowledgeBase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def kb():
    """Fixture to create a KnowledgeBase instance for testing."""
    return KnowledgeBase()

@pytest.fixture
def sample_document():
    """Fixture to create a sample document for testing."""
    return {
        "text": """
        Asynchronous programming in Python allows you to write concurrent code that can handle multiple tasks efficiently.
        The asyncio library provides the infrastructure for writing single-threaded concurrent code using coroutines,
        multiplexing I/O access over sockets and other resources, running network clients and servers, and other related
        primitives.
        """,
        "metadata": {
            "title": "Python Async Programming",
            "category": "Programming",
            "tags": ["python", "async", "asyncio", "concurrency"]
        }
    }

def test_initialization(kb):
    """Test that the knowledge base initializes correctly."""
    assert kb is not None
    assert kb.collection_name == os.getenv("QDRANT_COLLECTION", "agent_knowledge")
    assert kb.embedding_model_name == os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def test_add_document(kb):
    doc_text = "This is a test document for the knowledge base."
    doc_id = kb.add_document(
        text=doc_text,
        metadata={"title": "Test Document", "category": "Test"}
    )
    assert doc_id is not None
    doc = kb.get_document(doc_id)
    assert doc is not None
    assert doc["text"] == doc_text
    assert doc["title"] == "Test Document"

def test_get_document(kb, sample_document):
    """Test retrieving a document from the knowledge base."""
    doc_id = kb.add_document(
        text=sample_document["text"],
        metadata=sample_document["metadata"]
    )
    doc = kb.get_document(doc_id)
    assert doc is not None
    assert doc["title"] == sample_document["metadata"]["title"]
    assert doc["text"] == sample_document["text"]

def test_search(kb):
    doc_text = "Python is a high-level programming language."
    kb.add_document(
        text=doc_text,
        metadata={"title": "Python Doc", "category": "Programming"}
    )
    results = kb.search("high-level programming language", limit=1)
    assert len(results) > 0
    assert results[0]["metadata"]["title"] == "Python Doc"

def test_update_document(kb):
    doc_text = "Original text."
    doc_id = kb.add_document(
        text=doc_text,
        metadata={"title": "Original", "category": "Test"}
    )
    updated_text = "Updated text."
    kb.update_document(
        document_id=doc_id,
        text=updated_text,
        metadata={"title": "Updated", "category": "Test"}
    )
    updated_doc = kb.get_document(doc_id)
    assert updated_doc["text"] == updated_text
    assert updated_doc["title"] == "Updated"

def test_delete_document(kb):
    doc_text = "Document to delete."
    doc_id = kb.add_document(
        text=doc_text,
        metadata={"title": "Delete Me", "category": "Test"}
    )
    assert kb.delete_document(doc_id) is True
    assert kb.get_document(doc_id) is None

def test_clear_collection(kb):
    doc_text = "Document to clear."
    kb.add_document(
        text=doc_text,
        metadata={"title": "Clear Me", "category": "Test"}
    )
    assert kb.clear_collection() is True
    results = kb.search("Document to clear", limit=1)
    assert len(results) == 0

def test_embedding_generation(kb):
    """Test that embeddings are generated correctly."""
    text = "Test embedding generation"
    embedding = kb._get_embedding(text)
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_invalid_document_id(kb):
    """Test handling of invalid document ID."""
    doc = kb.get_document("invalid_id")
    assert doc is None

def test_search_with_no_results(kb):
    """Test search when no documents match the query."""
    results = kb.search("This query should not match any documents", limit=1)
    assert len(results) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
