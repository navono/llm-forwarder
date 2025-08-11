"""
Base module for integration tests.
Contains shared functionality for all integration tests.
"""

import pytest
import requests

# Constants for testing
SERVICE_URL = "http://localhost:8008"
TEST_CHAT_MODEL = "qwen3_235b_a22b_instruct_2507_fp8"

# TEST_CHAT_MODEL = next((model for model, info in AVAILABLE_MODELS.items() if info["type"] == "chat"), None)
# if not TEST_CHAT_MODEL:
#     raise ValueError("No chat model found in AVAILABLE_MODELS")

# HTTP status codes
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_BAD_REQUEST = 400


def is_service_running():
    """Check if the service is running on port 8008"""
    try:
        response = requests.get(f"{SERVICE_URL}/", timeout=2)
        return response.status_code == HTTP_OK
    except requests.RequestException:
        return False


# Skip marker for tests that require a running service
requires_service = pytest.mark.skipif(not is_service_running(), reason="Service is not running on port 8008")
