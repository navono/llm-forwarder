"""
Integration tests for the classification handler.
These tests connect to the actual running service on port 8008.
"""

import os
import pytest
import requests

from src.llm_forwarder.utils.llm_utils import AVAILABLE_MODELS
from tests.integration.base import HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_OK, SERVICE_URL, requires_service

def get_test_classify_model():
    """Get a test classification model from available models"""
    return next((model for model, info in AVAILABLE_MODELS.items() if info["type"] == "classify"), None)



def has_jina_api_key():
    """Check if JINA_SECRET_KEY environment variable is set"""
    return bool(os.getenv("JINA_SECRET_KEY"))


def is_model_available(model_name):
    """Check if a specific model is available"""
    return model_name in AVAILABLE_MODELS and AVAILABLE_MODELS[model_name]["type"] == "classify"


@requires_service
def test_classify_single_text():
    """Test classification with a single text input"""
    classify_model = get_test_classify_model()
    if not classify_model:
        pytest.skip("No classification model found in AVAILABLE_MODELS")
    
    # 如果没有设置Jina API密钥，跳过测试
    if "jina" in AVAILABLE_MODELS.get(classify_model, {}).get("owned_by", "") and not has_jina_api_key():
        pytest.skip("JINA_SECRET_KEY not set, skipping Jina API test")

    request_data = {"model": classify_model, "input": "This movie was fantastic and I enjoyed every minute of it.", "labels": ["positive", "negative", "neutral"]}

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证基本结构
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 1  # 单个输入文本应该有一个结果列表

    # 验证分类结果
    results = data["data"][0]
    assert isinstance(results, list)
    assert len(results) == len(request_data["labels"])  # 结果数量应该与标签数量相同

    for result in results:
        assert "label" in result
        assert "score" in result
        assert "index" in result
        assert isinstance(result["score"], float)
        assert result["label"] in request_data["labels"]

    # 验证模型信息
    assert "model" in data
    assert "usage" in data


@requires_service
def test_classify_multiple_texts():
    """Test classification with multiple text inputs"""
    classify_model = get_test_classify_model()
    if not classify_model:
        pytest.skip("No classification model found in AVAILABLE_MODELS")
    
    # 如果没有设置Jina API密钥，跳过测试
    if "jina" in AVAILABLE_MODELS.get(classify_model, {}).get("owned_by", "") and not has_jina_api_key():
        pytest.skip("JINA_SECRET_KEY not set, skipping Jina API test")

    request_data = {
        "model": classify_model,
        "input": ["This movie was fantastic and I enjoyed every minute of it.", "The service was terrible and the food was cold.", "The weather is neither good nor bad today."],
        "labels": ["positive", "negative", "neutral"],
    }

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证基本结构
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == len(request_data["input"])  # 结果数量应该与输入文本数量相同

    # 验证每个文本的分类结果
    for text_results in data["data"]:
        assert isinstance(text_results, list)
        assert len(text_results) == len(request_data["labels"])  # 每个文本的结果数量应该与标签数量相同

        for result in text_results:
            assert "label" in result
            assert "score" in result
            assert "index" in result
            assert isinstance(result["score"], float)
            assert result["label"] in request_data["labels"]

    # 验证模型信息
    assert "model" in data
    assert "usage" in data


@requires_service
def test_classify_custom_labels():
    """Test classification with custom labels"""
    classify_model = get_test_classify_model()
    if not classify_model:
        pytest.skip("No classification model found in AVAILABLE_MODELS")
    
    # 如果没有设置Jina API密钥，跳过测试
    if "jina" in AVAILABLE_MODELS.get(classify_model, {}).get("owned_by", "") and not has_jina_api_key():
        pytest.skip("JINA_SECRET_KEY not set, skipping Jina API test")

    request_data = {"model": classify_model, "input": "The sky is blue and the sun is shining.", "labels": ["weather", "food", "sports", "technology", "politics"]}

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证结果
    results = data["data"][0]
    assert len(results) == len(request_data["labels"])  # 结果数量应该与标签数量相同

    # 验证所有自定义标签都在结果中
    result_labels = [result["label"] for result in results]
    for label in request_data["labels"]:
        assert label in result_labels


@requires_service
def test_classify_multilingual():
    """Test classification with multilingual text"""
    classify_model = get_test_classify_model()
    if not classify_model:
        pytest.skip("No classification model found in AVAILABLE_MODELS")
    
    # 如果没有设置Jina API密钥，跳过测试
    if "jina" in AVAILABLE_MODELS.get(classify_model, {}).get("owned_by", "") and not has_jina_api_key():
        pytest.skip("JINA_SECRET_KEY not set, skipping Jina API test")

    request_data = {
        "model": classify_model,
        "input": [
            "这部电影非常精彩，我很喜欢。",  # 中文：这部电影非常精彩，我很喜欢。
            "Ce film était terrible, je ne l'ai pas aimé.",  # 法语：这部电影很糟糕，我不喜欢。
        ],
        "labels": ["positive", "negative", "neutral"],
    }

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    assert response.status_code == HTTP_OK
    data = response.json()

    # 验证结果数量
    assert len(data["data"]) == len(request_data["input"])

    # 每个文本都应该有分类结果
    for text_results in data["data"]:
        assert len(text_results) == len(request_data["labels"])


@requires_service
def test_invalid_model():
    """Test classification with an invalid model"""
    request_data = {"model": "non-existent-model", "input": "This is a test.", "labels": ["positive", "negative"]}

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    assert response.status_code == HTTP_NOT_FOUND
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


@requires_service
def test_non_classify_model():
    """Test classification with a non-classification model"""
    # 查找一个非分类模型
    non_classify_model = next((model for model, info in AVAILABLE_MODELS.items() if info["type"] != "classify"), None)
    if not non_classify_model:
        pytest.skip("No non-classification model found in AVAILABLE_MODELS")
        
    # 如果是Jina模型但没有API密钥，跳过测试
    if "jina" in AVAILABLE_MODELS.get(non_classify_model, {}).get("owned_by", "") and not has_jina_api_key():
        pytest.skip("JINA_SECRET_KEY not set, skipping Jina API test")

    request_data = {"model": non_classify_model, "input": "This is a test.", "labels": ["positive", "negative"]}

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    # 处理程序应该返回 400 错误
    assert response.status_code == HTTP_BAD_REQUEST
    data = response.json()
    assert "detail" in data
    assert "not a classify model" in data["detail"].lower()


@requires_service
def test_classify_empty_labels():
    """Test classification with empty labels list"""
    classify_model = get_test_classify_model()
    if not classify_model:
        pytest.skip("No classification model found in AVAILABLE_MODELS")

    request_data = {"model": classify_model, "input": "This is a test.", "labels": []}

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    # 空标签列表应该返回错误
    assert response.status_code != HTTP_OK


@requires_service
def test_classify_multi_label():
    """Test multi-label classification"""
    classify_model = get_test_classify_model()
    if not classify_model:
        pytest.skip("No classification model found in AVAILABLE_MODELS")
    
    # 如果没有设置Jina API密钥，跳过测试
    if "jina" in AVAILABLE_MODELS.get(classify_model, {}).get("owned_by", "") and not has_jina_api_key():
        pytest.skip("JINA_SECRET_KEY not set, skipping Jina API test")

    request_data = {
        "model": classify_model,
        "input": "The movie had great acting but a poor storyline and terrible special effects.",
        "labels": ["good_acting", "good_plot", "good_effects", "bad_acting", "bad_plot", "bad_effects"],
        "multi_label": True,
    }

    response = requests.post(f"{SERVICE_URL}/v1/classify", json=request_data)

    # 多标签分类可能不被所有模型支持，所以我们只检查请求是否成功
    if response.status_code == HTTP_OK:
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 1  # 单个输入

        # 在多标签分类中，多个标签可能有较高的分数
        results = data["data"][0]
        assert len(results) == len(request_data["labels"])
