import pytest
from unittest.mock import patch, MagicMock
from src.rag.generator import Generator


@pytest.fixture
def mock_anthropic():
    with patch("src.rag.generator.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="This is a generated report about TestPharma.")]
        mock_client.messages.create.return_value = mock_message
        yield mock_client


def test_generate_report(mock_anthropic):
    generator = Generator(api_key="test-key")
    chunks = [
        {"text": "Clinical Trial NCT123 Phase 3 RECRUITING", "metadata": {"source": "clinicaltrials"}},
        {"text": "FDA Approval NDA456 TYGACIL", "metadata": {"source": "fda_approval"}},
    ]
    result = generator.generate_report("TestPharma", chunks)
    assert isinstance(result, str)
    assert len(result) > 0
    mock_anthropic.messages.create.assert_called_once()
    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert any("TestPharma" in msg.get("content", "") for msg in call_kwargs["messages"])


def test_generate_chat_response(mock_anthropic):
    generator = Generator(api_key="test-key")
    chunks = [
        {"text": "Clinical Trial NCT123 Phase 3 RECRUITING", "metadata": {"source": "clinicaltrials"}},
    ]
    history = [
        {"role": "user", "content": "Tell me about TestPharma's pipeline"},
        {"role": "assistant", "content": "TestPharma has several trials..."},
    ]
    result = generator.generate_chat_response("What phase is Drug X?", chunks, history)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_report_with_no_data(mock_anthropic):
    generator = Generator(api_key="test-key")
    result = generator.generate_report("UnknownCo", [])
    assert isinstance(result, str)


def test_generate_report_empty_response():
    """API returns empty content array — should return fallback message."""
    with patch("src.rag.generator.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = []
        mock_client.messages.create.return_value = mock_message

        generator = Generator(api_key="test-key")
        result = generator.generate_report("TestPharma", [{"text": "data", "metadata": {}}])
        assert "empty response" in result.lower()


def test_generate_chat_response_empty_response():
    """Chat API returns empty content — should return fallback message."""
    with patch("src.rag.generator.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = []
        mock_client.messages.create.return_value = mock_message

        generator = Generator(api_key="test-key")
        result = generator.generate_chat_response("test?", [], [])
        assert "try again" in result.lower()
