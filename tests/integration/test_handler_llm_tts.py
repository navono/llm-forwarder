"""
Integration tests for the llm-forwarder TTS handler.
These tests connect to the actual running service on port 8008.
"""

import base64
from pathlib import Path

import pytest
import requests

from tests.integration.base import HTTP_BAD_REQUEST, HTTP_OK, SERVICE_URL, requires_service

# Test data
TEST_TEXT = "这是一个测试文本，用于测试 llm-forwarder 的 TTS 功能。"
TEST_AUDIO_PROMPT_PATH = Path(__file__).parent.parent.parent / "packages" / "index-tts" / "audio_prompt" / "Female-成熟_01.wav"


def is_tts_available():
    """Check if TTS functionality is available in the service"""
    try:
        response = requests.get(f"{SERVICE_URL}/v1/models")
        if response.status_code != HTTP_OK:
            return False
        
        models = response.json()
        # 检查是否有任何模型具有 TTS 能力
        # 首先检查是否有 index-tts 模型
        for model in models.get("data", []):
            if model.get("id") == "index-tts":
                # 如果找到 index-tts 模型，则认为 TTS 功能可用
                # 即使没有明确的 capabilities 字段
                return True
        
        # 如果没有找到 index-tts 模型，则检查 capabilities 字段
        return any(model.get("capabilities", {}).get("tts", False) for model in models.get("data", []))
    except requests.RequestException:
        return False


requires_tts = pytest.mark.skipif(not is_tts_available(), reason="No TTS-capable models available in the service")


@requires_service
@requires_tts
def test_tts_speech_endpoint_basic():
    """Test the TTS speech endpoint with basic parameters"""
    # Prepare request data
    request_data = {
        "model": "index-tts",  # Use the model key from models.json
        "input": TEST_TEXT,
        "response_format": "mp3",
    }

    # Add voice_file_path if test audio prompt exists
    if TEST_AUDIO_PROMPT_PATH.exists():
        request_data["voice_file_path"] = str(TEST_AUDIO_PROMPT_PATH)

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_tts_speech_endpoint_with_base64_voice():
    """Test the TTS speech endpoint with base64-encoded voice prompt"""
    # Skip if test audio prompt doesn't exist
    if not TEST_AUDIO_PROMPT_PATH.exists():
        pytest.skip(f"Test audio prompt not found at {TEST_AUDIO_PROMPT_PATH}")

    # Read and encode the audio file
    with open(TEST_AUDIO_PROMPT_PATH, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    # Prepare request data
    request_data = {
        "model": "index-tts",  # Use the model key from models.json
        "input": TEST_TEXT,
        "voice": audio_base64,
        "response_format": "mp3",
    }

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_tts_speech_endpoint_with_wav_format():
    """Test the TTS speech endpoint with WAV output format"""
    # Prepare request data
    request_data = {
        "model": "index-tts",  # Use the model key from models.json
        "input": TEST_TEXT,
        "response_format": "wav",
    }

    # Add voice_file_path if test audio prompt exists
    if TEST_AUDIO_PROMPT_PATH.exists():
        request_data["voice_file_path"] = str(TEST_AUDIO_PROMPT_PATH)

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/wav"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_tts_speech_endpoint_with_advanced_params():
    """Test the TTS speech endpoint with advanced generation parameters"""
    # Prepare request data with advanced parameters
    request_data = {
        "model": "index-tts",  # Use the model key from models.json
        "input": TEST_TEXT,
        "response_format": "mp3",
        "speed": 1.2,  # Slightly faster speech
        "max_text_tokens_per_sentence": 50,  # Smaller chunks
        "sentences_bucket_max_size": 2,  # Smaller bucket size
        "verbose": True,  # Enable verbose output
    }

    # Add voice_file_path if test audio prompt exists
    if TEST_AUDIO_PROMPT_PATH.exists():
        request_data["voice_file_path"] = str(TEST_AUDIO_PROMPT_PATH)

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/audio/speech", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_tts_speech_endpoint_missing_input():
    """Test the TTS speech endpoint with missing input parameter"""
    # Prepare request data with missing input
    request_data = {
        "model": "index-tts",  # Use the model key from models.json
        "response_format": "mp3",
    }

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/audio/speech", json=request_data)

    # Verify response
    assert response.status_code == HTTP_BAD_REQUEST
    data = response.json()
    assert "detail" in data


@requires_service
@requires_tts
def test_tts_speech_endpoint_invalid_model():
    """Test the TTS speech endpoint with an invalid model name"""
    # Prepare request data with invalid model
    request_data = {"model": "non-existent-tts-model", "input": TEST_TEXT, "response_format": "mp3"}

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/audio/speech", json=request_data)

    # Verify response
    assert response.status_code == HTTP_BAD_REQUEST
    data = response.json()
    assert "detail" in data


@requires_service
@requires_tts
def test_tts_speech_endpoint_long_text():
    """Test the TTS speech endpoint with a longer text input"""
    # Prepare request data with longer text
    long_text = TEST_TEXT * 5  # Repeat the test text 5 times
    request_data = {
        "model": "index-tts",  # Use the model key from models.json
        "input": long_text,
        "response_format": "mp3",
    }

    # Add voice_file_path if test audio prompt exists
    if TEST_AUDIO_PROMPT_PATH.exists():
        request_data["voice_file_path"] = str(TEST_AUDIO_PROMPT_PATH)

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/audio/speech", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content
