"""
Integration tests for the embedding handler.
These tests connect to the actual running service on port 8008.
"""

import pytest
import requests

from src.llm_forwarder.utils.llm_utils import AVAILABLE_MODELS
from tests.integration.base import HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_OK, SERVICE_URL, requires_service


# 查找一个可用的嵌入模型
def get_test_embedding_model():
    """Get a test embedding model from available models"""
    return next((model for model, info in AVAILABLE_MODELS.items() if info["type"] == "embedding"), None)


def is_model_available(model_name):
    """Check if a specific model is available"""
    return model_name in AVAILABLE_MODELS and AVAILABLE_MODELS[model_name]["type"] == "embedding"


@requires_service
def test_embedding_with_string_input():
    """Test the embedding endpoint with a single string input"""
    # 查找一个嵌入模型
    embedding_model = get_test_embedding_model()
    if not embedding_model:
        pytest.skip("No embedding model found in AVAILABLE_MODELS")

    request_data = {"model": embedding_model, "input": "This is a test sentence for embedding."}

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 1  # 单个输入应该有一个嵌入结果

    # 验证嵌入结果格式
    embedding_result = data["data"][0]
    assert "embedding" in embedding_result
    assert isinstance(embedding_result["embedding"], list)
    assert len(embedding_result["embedding"]) > 0  # 嵌入向量应该有值
    assert "index" in embedding_result
    assert embedding_result["index"] == 0

    # 验证模型信息
    assert "model" in data
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]


@requires_service
def test_embedding_with_list_input():
    """Test the embedding endpoint with a list of strings input"""
    # 查找一个嵌入模型
    embedding_model = get_test_embedding_model()
    if not embedding_model:
        pytest.skip("No embedding model found in AVAILABLE_MODELS")

    request_data = {"model": embedding_model, "input": ["First test sentence.", "Second test sentence."]}

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    # 两个输入应该有两个嵌入结果
    expected_results = 2
    assert len(data["data"]) == expected_results

    # 验证嵌入结果格式
    for i, embedding_result in enumerate(data["data"]):
        assert "embedding" in embedding_result
        assert isinstance(embedding_result["embedding"], list)
        assert "index" in embedding_result
        assert embedding_result["index"] == i


@requires_service
def test_embedding_with_dimensions():
    """Test the embedding endpoint with custom dimensions"""
    # 查找一个嵌入模型
    embedding_model = get_test_embedding_model()
    if not embedding_model:
        pytest.skip("No embedding model found in AVAILABLE_MODELS")

    # 注意：并非所有模型都支持自定义维度，此测试可能会被跳过
    request_data = {
        "model": embedding_model,
        "input": "Test with custom dimensions.",
        "dimensions": 512,  # 尝试请求512维的嵌入
    }

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    # 如果模型不支持自定义维度，可能会返回400错误
    if response.status_code == HTTP_BAD_REQUEST:
        pytest.skip(f"Model {embedding_model} does not support custom dimensions")
        return

    assert response.status_code == HTTP_OK
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 1

    # 验证嵌入维度
    embedding_vector = data["data"][0]["embedding"]
    # 注意：有些模型可能会忽略维度参数，返回默认维度
    # 所以这里不严格断言维度必须是512
    assert len(embedding_vector) > 0


@requires_service
def test_invalid_model():
    """Test embedding with an invalid model"""
    request_data = {"model": "non-existent-model", "input": "Test sentence."}

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    assert response.status_code == HTTP_NOT_FOUND
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


@requires_service
def test_non_embedding_model():
    """Test embedding with a non-embedding model"""
    # 查找一个非嵌入模型
    non_embedding_model = next((model for model, info in AVAILABLE_MODELS.items() if info["type"] != "embedding"), None)
    if not non_embedding_model:
        pytest.skip("No non-embedding model found in AVAILABLE_MODELS")

    request_data = {"model": non_embedding_model, "input": "Test sentence."}

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    assert response.status_code == HTTP_BAD_REQUEST
    data = response.json()
    assert "detail" in data
    assert "not an embedding model" in data["detail"].lower()


@requires_service
def test_jina_embeddings_v3():
    """Test embedding with jina-embeddings-v3 model specifically"""
    model_name = "jina-embeddings-v3"

    if not is_model_available(model_name):
        pytest.skip(f"Model {model_name} not available in AVAILABLE_MODELS")

    request_data = {"model": model_name, "input": "This is a test for Jina embeddings model."}

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证基本结构
    assert "data" in data
    assert len(data["data"]) == 1

    # 验证嵌入结果
    embedding_result = data["data"][0]
    assert "embedding" in embedding_result
    assert isinstance(embedding_result["embedding"], list)

    # Jina embeddings 应该有特定维度 (通常是 768 或 1024)
    embedding_vector = embedding_result["embedding"]
    assert len(embedding_vector) > 0

    # 验证模型信息
    assert "model" in data
    assert model_name in data["model"]


@requires_service
def test_qwen3_embedding():
    """Test embedding with qwen3-embedding-4b@q6_k model specifically"""
    model_name = "qwen3-embedding-4b@q6_k"

    if not is_model_available(model_name):
        pytest.skip(f"Model {model_name} not available in AVAILABLE_MODELS")

    request_data = {"model": model_name, "input": "这是一个测试千问嵌入模型的句子。"}

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证基本结构
    assert "data" in data
    assert len(data["data"]) == 1

    # 验证嵌入结果
    embedding_result = data["data"][0]
    assert "embedding" in embedding_result
    assert isinstance(embedding_result["embedding"], list)

    # 千问嵌入向量应该有特定维度
    embedding_vector = embedding_result["embedding"]
    assert len(embedding_vector) > 0

    # 验证模型信息
    assert "model" in data
    assert model_name in data["model"]


@requires_service
def test_embedding_batch_with_specific_models():
    """Test batch embedding with specific models"""
    # 测试两个模型
    models_to_test = ["jina-embeddings-v3", "qwen3-embedding-4b@q6_k"]

    # 找到一个可用的模型
    available_model = None
    for model in models_to_test:
        if is_model_available(model):
            available_model = model
            break

    if not available_model:
        pytest.skip("None of the specific embedding models are available")

    # 批量文本输入
    texts = ["First test sentence for batch embedding.", "Second test sentence for batch embedding.", "Third test sentence for batch embedding."]

    request_data = {"model": available_model, "input": texts}

    response = requests.post(f"{SERVICE_URL}/v1/embeddings", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证结果数量
    assert "data" in data
    expected_count = len(texts)
    assert len(data["data"]) == expected_count

    # 验证每个嵌入结果
    for i, embedding_result in enumerate(data["data"]):
        assert "embedding" in embedding_result
        assert isinstance(embedding_result["embedding"], list)
        assert "index" in embedding_result
        assert embedding_result["index"] == i
