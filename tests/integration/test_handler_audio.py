"""
Integration tests for the index-tts server.
These tests connect to the actual running index-tts service.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest
import requests

from tests.integration.base import HTTP_OK

# Constants for testing
INDEX_TTS_SERVICE_URL = "http://localhost:8008"  # Default port for index-tts server

SPEECH_ENDPOINT = "/v1/audio/speech"


# Skip marker for tests that require a running index-tts service
def is_index_tts_service_running():
    """Check if the index-tts service is running on port 8008"""
    try:
        response = requests.get(f"{INDEX_TTS_SERVICE_URL}/health", timeout=2)
        return response.status_code == HTTP_OK
    except requests.RequestException:
        return False


requires_index_tts_service = pytest.mark.skipif(not is_index_tts_service_running(), reason="llm forwarder service is not running on port 8008")

# Test data
TEST_TEXT = "这是一个测试文本，用于测试 IndexTTS 服务。"
TEST_AUDIO_PATH = Path(__file__).parent.parent.parent / "packages" / "index-tts" / "audio_prompt" / "Female-成熟_01.wav"


@requires_index_tts_service
def test_root_endpoint():
    """Test the root endpoint of the running index-tts service"""
    response = requests.get(f"{INDEX_TTS_SERVICE_URL}/")
    assert response.status_code == HTTP_OK
    data = response.json()
    assert "message" in data


@requires_index_tts_service
def test_health_endpoint():
    """Test the health endpoint of the running index-tts service"""
    response = requests.get(f"{INDEX_TTS_SERVICE_URL}/health")
    assert response.status_code == HTTP_OK
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@requires_index_tts_service
def test_tts_endpoint_with_text():
    """Test the TTS endpoint with text input"""
    request_data = {
        "input": TEST_TEXT,
        "model": "index-tts",
        "response_format": "mp3",
        "max_text_tokens_per_sentence": 100,
        "sentences_bucket_max_size": 4,
        "verbose": True,
        "repetition_penalty": 10.0,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 1.0,
        "length_penalty": 0.0,
        "num_beams": 3,
        "max_mel_tokens": 600,
        "do_sample": True,
    }
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("outputs", f"tts_output_{current_time}_{hash(TEST_TEXT)}.wav")

    # 使用stream=True参数来处理流式响应
    response = requests.post(f"{INDEX_TTS_SERVICE_URL}{SPEECH_ENDPOINT}", json=request_data, stream=True)

    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] in ["audio/mpeg", "audio/wav"]

    # 创建outputs目录（如果不存在）
    os.makedirs("outputs", exist_ok=True)

    # 确保文件扩展名与内容类型匹配
    content_type = response.headers.get("Content-Type")
    if "mp3" in output_path and "wav" in content_type:
        output_path = output_path.replace(".mp3", ".wav")
    elif "wav" in output_path and "mpeg" in content_type:
        output_path = output_path.replace(".wav", ".mp3")

    # 使用块读取流式响应并写入文件
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # 过滤掉保持连接活跃的空块
                f.write(chunk)

    # 检查文件是否存在且有内容
    file_size = os.path.getsize(output_path)
    assert os.path.exists(output_path)
    assert file_size > 0

    print(f"音频文件已保存至: {output_path} (大小: {file_size} 字节)")


# @requires_index_tts_service
# def test_tts_endpoint_with_audio_prompt():
#     """Test the TTS endpoint with text input and audio prompt"""
#     # Skip if the test audio prompt doesn't exist
#     if not TEST_AUDIO_PATH.exists():
#         pytest.skip(f"Test audio prompt not found at {TEST_AUDIO_PATH}")

#     # Read the audio file and encode it as base64
#     with open(TEST_AUDIO_PATH, "rb") as audio_file:
#         audio_data = audio_file.read()
#         audio_base64 = base64.b64encode(audio_data).decode("utf-8")

#     request_data = {"text": TEST_TEXT, "audio_prompt": audio_base64, "output_format": "mp3"}

#     response = requests.post(f"{INDEX_TTS_SERVICE_URL}{SPEECH_ENDPOINT}", json=request_data)

#     assert response.status_code == HTTP_OK
#     assert response.headers["Content-Type"] == "audio/mpeg"

#     # Check that we got audio data
#     assert len(response.content) > 0


# @requires_index_tts_service
# def test_speech_endpoint_openai_compatible():
#     """Test the speech endpoint that's compatible with OpenAI API"""
#     request_data = {"model": "index-tts-1.5", "input": TEST_TEXT, "voice": None, "response_format": {"type": "mp3"}, "speed": 1.0}

#     response = requests.post(f"{INDEX_TTS_SERVICE_URL}/v1/audio/speech", json=request_data)

#     assert response.status_code == HTTP_OK
#     assert response.headers["Content-Type"] == "audio/mpeg"

#     # Check that we got audio data
#     assert len(response.content) > 0


# @requires_index_tts_service
# def test_invalid_request():
#     """Test the TTS endpoint with invalid parameters"""
#     # Missing required text parameter
#     request_data = {"output_format": "mp3"}

#     response = requests.post(f"{INDEX_TTS_SERVICE_URL}{SPEECH_ENDPOINT}", json=request_data)

#     # Should return a 422 Unprocessable Entity or 400 Bad Request
#     assert response.status_code in [422, HTTP_BAD_REQUEST]


# @requires_index_tts_service
# def test_wav_output_format():
#     """Test the TTS endpoint with WAV output format"""
#     request_data = {"text": TEST_TEXT, "output_format": "wav"}

#     response = requests.post(f"{INDEX_TTS_SERVICE_URL}{SPEECH_ENDPOINT}", json=request_data)

#     assert response.status_code == HTTP_OK
#     assert response.headers["Content-Type"] == "audio/wav"

#     # Check that we got audio data
#     assert len(response.content) > 0
