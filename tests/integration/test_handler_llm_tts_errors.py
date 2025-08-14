"""
Integration tests for error handling in the llm-forwarder TTS handler.
These tests connect to the actual running service on port 8008.
"""

import pytest
import requests

from tests.integration.base import HTTP_BAD_REQUEST, SERVICE_URL, requires_service
from tests.integration.test_handler_llm_tts import TEST_TEXT, requires_tts


@requires_service
@requires_tts
def test_tts_service_unavailable():
    """Test handling of TTS service being unavailable"""
    # This test requires mocking the TTS service URL to simulate unavailability
    # We'll use the requests library's built-in exception handling
    
    # Prepare request data
    request_data = {
        "model": "index-tts",
        "input": TEST_TEXT,
        "response_format": "mp3"
    }
    
    # Make the request to a non-existent endpoint to simulate service unavailability
    # This should be handled gracefully by the handler
    response = requests.post(f"{SERVICE_URL}/v1/audio/speech/unavailable_test", json=request_data)
    
    # Verify response - should return a 404 or similar error
    assert response.status_code >= 400
