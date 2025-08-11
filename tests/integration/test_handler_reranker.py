"""
Integration tests for the reranker handler.
These tests connect to the actual running service on port 8008.
"""

import pytest
import requests

from src.llm_forwarder.utils.llm_utils import AVAILABLE_MODELS
from tests.integration.base import HTTP_NOT_FOUND, HTTP_OK, SERVICE_URL, requires_service


def get_test_rerank_model() -> str | None:
    """Get a test rerank model from available models"""
    return next((model for model, info in AVAILABLE_MODELS.items() if info["type"] == "rerank"), None)


def is_model_available(model_name: str) -> bool:
    """Check if a specific model is available"""
    return model_name in AVAILABLE_MODELS and AVAILABLE_MODELS[model_name]["type"] == "rerank"


@requires_service
def test_rerank_basic():
    """Test the rerank endpoint with basic input"""
    # 查找一个重排序模型
    rerank_model = get_test_rerank_model()
    if not rerank_model:
        pytest.skip("No rerank model found in AVAILABLE_MODELS")

    request_data = {
        "model": rerank_model,
        "query": "What is machine learning?",
        "documents": ["Machine learning is a branch of artificial intelligence.", "Python is a programming language.", "Machine learning algorithms can learn from data.", "The weather is sunny today."],
    }

    response = requests.post(f"{SERVICE_URL}/v1/rerank", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证基本结构
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)

    # 验证重排序结果
    assert len(data["data"]) > 0
    for result in data["data"]:
        assert "index" in result
        assert "relevance_score" in result
        assert "document" in result
        assert isinstance(result["relevance_score"], float)

    # 验证模型信息
    assert "model" in data
    assert "usage" in data


@requires_service
def test_rerank_with_top_k():
    """Test the rerank endpoint with top_k parameter"""
    rerank_model = get_test_rerank_model()
    if not rerank_model:
        pytest.skip("No rerank model found in AVAILABLE_MODELS")

    top_k = 2
    request_data = {
        "model": rerank_model,
        "query": "What is machine learning?",
        "documents": ["Machine learning is a branch of artificial intelligence.", "Python is a programming language.", "Machine learning algorithms can learn from data.", "The weather is sunny today."],
        "top_k": top_k,
    }

    response = requests.post(f"{SERVICE_URL}/v1/rerank", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证结果数量不超过 top_k
    assert len(data["data"]) <= top_k


@requires_service
def test_jina_reranker_v2():
    """Test reranking with jina-reranker-v2-base-multilingual model specifically"""
    model_name = "jina-reranker-v2-base-multilingual"

    if not is_model_available(model_name):
        pytest.skip(f"Model {model_name} not available in AVAILABLE_MODELS")

    request_data = {
        "model": model_name,
        "query": "机器学习是什么？",  # 多语言查询
        "documents": ["机器学习是人工智能的一个分支。", "Python是一种编程语言。", "机器学习算法可以从数据中学习。", "今天天气晴朗。"],
    }

    response = requests.post(f"{SERVICE_URL}/v1/rerank", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证基本结构
    assert "data" in data
    assert len(data["data"]) > 0

    # 验证重排序结果 - 相关文档应该排在前面
    results = data["data"]
    # 找出包含"机器学习"的文档的索引
    ml_indices = [i for i, doc in enumerate(request_data["documents"]) if "机器学习" in doc]

    # 验证相关性分数
    min_relevance_score = 0.5  # 相关文档的最小相关性分数
    for result in results:
        if result["index"] in ml_indices:
            # 相关文档的分数应该较高
            assert result["relevance_score"] > min_relevance_score

    # 验证模型信息
    assert "model" in data
    assert model_name in data["model"]


@requires_service
def test_invalid_model():
    """Test reranking with an invalid model"""
    request_data = {"model": "non-existent-model", "query": "What is machine learning?", "documents": ["Machine learning is a branch of artificial intelligence."]}

    response = requests.post(f"{SERVICE_URL}/v1/rerank", json=request_data)

    assert response.status_code == HTTP_NOT_FOUND
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


@requires_service
def test_non_rerank_model():
    """Test reranking with a non-rerank model"""
    # 查找一个非重排序模型
    non_rerank_model = next((model for model, info in AVAILABLE_MODELS.items() if info["type"] != "rerank"), None)
    if not non_rerank_model:
        pytest.skip("No non-rerank model found in AVAILABLE_MODELS")

    request_data = {"model": non_rerank_model, "query": "What is machine learning?", "documents": ["Machine learning is a branch of artificial intelligence."]}

    response = requests.post(f"{SERVICE_URL}/v1/rerank", json=request_data)

    # 处理程序应该返回 404 错误
    assert response.status_code == HTTP_NOT_FOUND
    data = response.json()
    assert "detail" in data
    assert "not a rerank model" in data["detail"].lower()


@requires_service
def test_rerank_empty_documents():
    """Test reranking with empty documents list"""
    rerank_model = get_test_rerank_model()
    if not rerank_model:
        pytest.skip("No rerank model found in AVAILABLE_MODELS")

    request_data = {"model": rerank_model, "query": "What is machine learning?", "documents": []}

    response = requests.post(f"{SERVICE_URL}/v1/rerank", json=request_data)

    # 空文档列表应该返回空结果，而不是错误
    assert response.status_code == HTTP_OK
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 0
