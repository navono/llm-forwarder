"""
Integration tests for the llm-forwarder TTS handler /v1/tts endpoint.
These tests connect to the actual running service on port 8008.
"""

import base64

import pytest
import requests

from tests.integration.base import HTTP_BAD_REQUEST, HTTP_OK, SERVICE_URL, requires_service
from tests.integration.test_handler_llm_tts import TEST_AUDIO_PROMPT_PATH, TEST_TEXT, is_tts_available

requires_tts = pytest.mark.skipif(not is_tts_available(), reason="No TTS-capable models available in the service")


@requires_service
@requires_tts
def test_v1_tts_endpoint_basic():
    """Test the /v1/tts endpoint with basic parameters"""
    # Prepare request data - using OpenAI format
    request_data = {"text": TEST_TEXT, "response_format": {"type": "mp3"}}

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_v1_tts_endpoint_with_base64_voice():
    """Test the /v1/tts endpoint with base64-encoded voice prompt"""
    # Skip if test audio prompt doesn't exist
    if not TEST_AUDIO_PROMPT_PATH.exists():
        pytest.skip(f"Test audio prompt not found at {TEST_AUDIO_PROMPT_PATH}")

    # Read and encode the audio file
    with open(TEST_AUDIO_PROMPT_PATH, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    # Prepare request data - using OpenAI format
    request_data = {"text": TEST_TEXT, "voice": audio_base64, "response_format": {"type": "mp3"}}

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_v1_tts_endpoint_with_wav_format():
    """Test the /v1/tts endpoint with WAV output format"""
    # Prepare request data - using OpenAI format
    request_data = {"text": TEST_TEXT, "response_format": {"type": "wav"}}

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/wav"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_v1_tts_endpoint_with_advanced_params():
    """Test the /v1/tts endpoint with advanced generation parameters"""
    # Prepare request data with advanced parameters - using OpenAI format
    request_data = {
        "text": TEST_TEXT,
        "response_format": {"type": "mp3"},
        "speed": 1.2,  # Slightly faster speech
        "max_text_tokens_per_sentence": 50,  # Smaller chunks
        "sentences_bucket_max_size": 2,  # Smaller bucket size
        "verbose": True,  # Enable verbose output
    }

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content


@requires_service
@requires_tts
def test_v1_tts_endpoint_missing_text():
    """Test the /v1/tts endpoint with missing text parameter"""
    # Prepare request data with missing input - using OpenAI format
    request_data = {"response_format": {"type": "mp3"}}

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_BAD_REQUEST
    data = response.json()
    assert "detail" in data


@requires_service
@requires_tts
def test_v1_tts_endpoint_long_text():
    """Test the /v1/tts endpoint with a longer text input"""
    # Prepare request data with longer text - using OpenAI format
    long_text = TEST_TEXT * 5  # Repeat the test text 5 times
    request_data = {"text": long_text, "response_format": {"type": "mp3"}}

    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)

    # Verify response
    assert response.status_code == HTTP_OK
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert len(response.content) > 0  # Should have audio content
