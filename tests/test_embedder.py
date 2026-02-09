import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.embedder import Embedder


@pytest.fixture
def mock_openai():
    with patch("src.ingestion.embedder.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_chroma():
    with patch("src.ingestion.embedder.chromadb") as mock_mod:
        mock_client = MagicMock()
        mock_mod.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        yield mock_collection


def test_embed_and_store(mock_openai, mock_chroma):
    embedder = Embedder(openai_api_key="test-key", chroma_path="/tmp/test_chroma")
    chunks = [
        {"text": "Clinical trial NCT123", "metadata": {"source": "clinicaltrials", "nct_id": "NCT123"}},
        {"text": "FDA approval NDA456", "metadata": {"source": "fda_approval", "application_number": "NDA456"}},
    ]
    embedder.embed_and_store(chunks, collection_name="test_company")
    mock_openai.embeddings.create.assert_called()
    mock_chroma.add.assert_called_once()
    call_args = mock_chroma.add.call_args
    assert len(call_args.kwargs["documents"]) == 2
    assert len(call_args.kwargs["ids"]) == 2


def test_embed_and_store_batches_large_inputs(mock_openai, mock_chroma):
    embedder = Embedder(openai_api_key="test-key", chroma_path="/tmp/test_chroma")
    chunks = [{"text": f"Chunk {i}", "metadata": {"source": "test"}} for i in range(150)]
    def side_effect(**kwargs):
        n = len(kwargs["input"])
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
        return mock_resp
    mock_openai.embeddings.create.side_effect = side_effect
    embedder.embed_and_store(chunks, collection_name="test_company")
    assert mock_openai.embeddings.create.call_count >= 2
