import pytest
from unittest.mock import patch, MagicMock
from src.rag.retriever import Retriever


@pytest.fixture
def mock_embedder():
    with patch("src.rag.retriever.Embedder") as MockEmbedder:
        instance = MagicMock()
        MockEmbedder.return_value = instance
        mock_collection = MagicMock()
        instance.get_collection.return_value = mock_collection
        instance._openai = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
        instance._openai.embeddings.create.return_value = mock_embed_response
        yield instance, mock_collection


def test_retrieve_for_report_uses_metadata_filter(mock_embedder):
    embedder_instance, mock_collection = mock_embedder
    mock_collection.get.return_value = {
        "documents": ["Trial NCT123 Phase 3", "FDA Approval NDA456"],
        "metadatas": [
            {"source": "clinicaltrials", "company": "TestPharma"},
            {"source": "fda_approval", "company": "TestPharma"}
        ],
        "ids": ["id1", "id2"]
    }
    retriever = Retriever(embedder=embedder_instance)
    results = retriever.retrieve_for_report("test_collection", "TestPharma")
    mock_collection.get.assert_called_once()
    assert len(results) == 2
    assert results[0]["text"] == "Trial NCT123 Phase 3"


def test_retrieve_for_chat_uses_similarity_search(mock_embedder):
    embedder_instance, mock_collection = mock_embedder
    mock_collection.query.return_value = {
        "documents": [["Trial NCT123 Phase 3"]],
        "metadatas": [[{"source": "clinicaltrials", "company": "TestPharma"}]],
        "distances": [[0.15]],
        "ids": [["id1"]]
    }
    retriever = Retriever(embedder=embedder_instance)
    results = retriever.retrieve_for_chat("test_collection", "What phase is Drug X in?")
    mock_collection.query.assert_called_once()
    assert len(results) == 1
    assert results[0]["text"] == "Trial NCT123 Phase 3"
