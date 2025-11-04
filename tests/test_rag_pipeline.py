import pytest
from unittest.mock import MagicMock, patch
import rag_pipeline


@pytest.fixture(scope="function", autouse=True)
def mock_env(monkeypatch):
    """Mock environment variables for API keys."""
    monkeypatch.setenv("COHERE_API_KEY", "fake-cohere-key")
    monkeypatch.setenv("PINECONE_API_KEY", "fake-pinecone-key")


def test_construct_prompt_cohere():
    """Test that construct_prompt_cohere builds a valid formatted prompt."""
    user_question = "What is inflation?"
    retrieved_docs = [
        MagicMock(page_content="Inflation is an increase in prices."),
        MagicMock(page_content="It decreases the purchasing power of money."),
    ]

    prompt = rag_pipeline.construct_prompt_cohere(user_question, retrieved_docs)

    # Validate structure and formatting
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert "<context>" in prompt[0]["content"]
    assert "</context>" in prompt[0]["content"]
    assert "Inflation" in prompt[0]["content"]
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"] == user_question


@patch("rag_pipeline.requests.post")
def test_generate_with_deepseek(mock_post):
    """Test that generate_with_deepseek handles response correctly."""
    # Mock DeepSeek-style API response
    mock_response = {
        "choices": [
            {"message": {"content": "<think>Reasoning...</think>Final answer text."}}
        ]
    }
    mock_post.return_value.json.return_value = mock_response

    result = rag_pipeline.generate_with_deepseek(
        [{"role": "user", "content": "Explain GDP"}]
    )

    mock_post.assert_called_once()

    # Your current implementation does NOT split off <think> text, so we assert full content
    assert "<think>" in result
    assert "Final answer text." in result


def test_retrieve_top_k_docs():
    """Test retrieve_top_k_docs calls similarity_search correctly."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = ["doc1", "doc2"]

    mock_retriever = MagicMock()
    mock_retriever.vectorstore = mock_vectorstore

    result = rag_pipeline.retrieve_top_k_docs(mock_retriever, "test query", 2)

    mock_vectorstore.similarity_search.assert_called_once_with("test query", 2)
    assert result == ["doc1", "doc2"]