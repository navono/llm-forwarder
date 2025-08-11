"""
Integration tests for the chat handler.
These tests connect to the actual running service on port 8008.
"""

import json

import pytest
import requests

from src.llm_forwarder.http.message_type import AVAILABLE_MODELS
from tests.integration.base import HTTP_NOT_FOUND, HTTP_OK, SERVICE_URL, TEST_CHAT_MODEL, requires_service


@requires_service
def test_chat_completion():
    """Test the chat completion endpoint of the running service"""
    request_data = {"model": TEST_CHAT_MODEL, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, how are you?"}], "temperature": 0.7, "max_tokens": 100}

    response = requests.post(f"{SERVICE_URL}/v1/chat/completions", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert data["model"] == AVAILABLE_MODELS[TEST_CHAT_MODEL]["model_name_in_request"]
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    assert data["choices"][0]["message"]["role"] == "assistant"


@requires_service
def test_chat_completion_streaming():
    """Test the streaming chat completion endpoint of the running service"""
    request_data = {"model": TEST_CHAT_MODEL, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, how are you?"}], "temperature": 0.7, "max_tokens": 100, "stream": True}

    response = requests.post(f"{SERVICE_URL}/v1/chat/completions", json=request_data, stream=True)

    assert response.status_code == HTTP_OK
    assert response.headers["content-type"].startswith("text/event-stream")

    # Read and validate streaming response
    content = ""
    for chunk in response.iter_lines():
        if not chunk:
            continue

        decoded_chunk = chunk.decode("utf-8")
        if not (decoded_chunk.startswith("data: ") and not decoded_chunk.endswith("[DONE]")):
            continue

        try:
            data = json.loads(decoded_chunk[6:])  # Skip "data: " prefix
            if "choices" in data and len(data["choices"]) > 0 and "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                content += data["choices"][0]["delta"]["content"]
        except json.JSONDecodeError:
            pass

    # Verify we received some content
    assert len(content) > 0


@requires_service
def test_invalid_model():
    """Test chat completion with an invalid model"""
    request_data = {"model": "non-existent-model", "messages": [{"role": "user", "content": "Hello"}]}

    response = requests.post(f"{SERVICE_URL}/v1/chat/completions", json=request_data)

    assert response.status_code == HTTP_NOT_FOUND
    data = response.json()
    assert "detail" in data
    assert "Model non-existent-model not found" in data["detail"]


# 搜索接口测试
@requires_service
def test_search_with_json_body():
    """Test the search endpoint with JSON body"""
    # 查找一个 search 类型的模型
    search_model = next((model for model, info in AVAILABLE_MODELS.items() if info["type"] == "search"), None)
    if not search_model:
        pytest.skip("No search model found in AVAILABLE_MODELS")

    request_data = {"model": search_model, "query": "artificial intelligence", "top_k": 5}

    response = requests.get(f"{SERVICE_URL}/v1/search", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert "model" in data
    assert data["model"] == search_model

    # 验证搜索结果格式
    if data["data"]:
        result = data["data"][0]
        assert "title" in result
        assert "url" in result
        assert "description" in result


@requires_service
def test_search_with_query_params():
    """Test the search endpoint with query parameters"""
    # 查找一个 search 类型的模型
    search_model = next((model for model, info in AVAILABLE_MODELS.items() if info["type"] == "search"), None)
    if not search_model:
        pytest.skip("No search model found in AVAILABLE_MODELS")

    params = {"model": search_model, "query": "machine learning", "top_k": 3}

    response = requests.get(f"{SERVICE_URL}/v1/search", params=params)

    assert response.status_code == HTTP_OK
    data = response.json()
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert "model" in data
    assert data["model"] == search_model


@requires_service
def test_search_invalid_model():
    """Test search with an invalid model"""
    request_data = {"model": "non-existent-model", "query": "test query"}

    response = requests.get(f"{SERVICE_URL}/v1/search", json=request_data)

    assert response.status_code == HTTP_NOT_FOUND
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower() or "not a search model" in data["detail"].lower()
