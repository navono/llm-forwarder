"""
Integration tests for the root handler.
These tests connect to the actual running service on port 8008.
"""

import requests

from tests.integration.base import HTTP_OK, SERVICE_URL, requires_service


@requires_service
def test_root_endpoint():
    """Test the root endpoint of the running service"""
    response = requests.get(f"{SERVICE_URL}/")
    assert response.status_code == HTTP_OK
    data = response.json()
    assert "message" in data
    assert data["message"] == "Hello, FastAPI!"
    assert "dask result" in data


@requires_service
def test_models_endpoint():
    """Test the models endpoint of the running service"""
    response = requests.get(f"{SERVICE_URL}/v1/models")
    assert response.status_code == HTTP_OK
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)

    # Verify each model in the response
    for model_info in data["data"]:
        assert "id" in model_info
        assert "object" in model_info
        assert model_info["object"] == "model"
        assert "created" in model_info
        assert "owned_by" in model_info
