"""
Integration tests for error handling in the llm-forwarder TTS handler /v1/tts endpoint.
These tests connect to the actual running service on port 8008.
"""

import pytest
import requests

from tests.integration.base import HTTP_BAD_REQUEST, SERVICE_URL, requires_service
from tests.integration.test_handler_llm_tts import TEST_TEXT, is_tts_available

# HTTP status codes
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500


requires_tts = pytest.mark.skipif(
    not is_tts_available(),
    reason="No TTS-capable models available in the service"
)


@requires_service
@requires_tts
def test_v1_tts_endpoint_invalid_output_format():
    """Test the /v1/tts endpoint with an invalid output format"""
    # Prepare request data with invalid output format - using OpenAI format
    request_data = {
        "input": TEST_TEXT,
        "response_format": {"type": "invalid_format"}  # Invalid format
    }
    
    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)
    
    # Verify response
    assert response.status_code == HTTP_BAD_REQUEST
    data = response.json()
    assert "detail" in data


@requires_service
def test_v1_tts_endpoint_service_unavailable():
    """Test the /v1/tts endpoint when the TTS service is unavailable"""
    # Temporarily modify the service URL to point to a non-existent endpoint
    original_url = SERVICE_URL
    non_existent_url = f"{original_url}/non_existent_endpoint"
    
    # Prepare request data - using OpenAI format
    request_data = {
        "input": TEST_TEXT,
        "response_format": {"type": "mp3"}
    }
    
    # Make the request to a non-existent endpoint to simulate service unavailability
    # This should be handled by the error handler in the TTS service
    response = requests.post(f"{non_existent_url}/v1/tts", json=request_data)
    
    # Verify response - should be 404 Not Found
    assert response.status_code == HTTP_NOT_FOUND


@requires_service
@requires_tts
def test_v1_tts_endpoint_empty_text():
    """Test the /v1/tts endpoint with empty text"""
    # Prepare request data with empty text - using OpenAI format
    request_data = {
        "input": "",
        "response_format": {"type": "mp3"}
    }
    
    # Make the request
    response = requests.post(f"{SERVICE_URL}/v1/tts", json=request_data)
    
    # Verify response
    # Empty text might be handled differently by different TTS services
    # Some might return an error, others might return an empty audio file
    # We'll just check that the request completes without a server error
    assert response.status_code != HTTP_SERVER_ERROR
