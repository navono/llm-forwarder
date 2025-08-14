import base64
import os

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from loguru import logger

from packages.llm_forwarder.utils.llm_message_type import AudioSpeechRequest, AudioTranscriptionRequest
from packages.llm_forwarder.utils.llm_utils import AVAILABLE_MODELS, HTTP_OK

# Create router
handler_router = APIRouter()

# Get index-tts model configuration
INDEX_TTS_MODEL_KEY = "index-tts"


def get_tts_service_url():
    """
    Get the URL for the TTS service from model configuration.
    """
    if INDEX_TTS_MODEL_KEY not in AVAILABLE_MODELS:
        raise HTTPException(status_code=500, detail=f"TTS model {INDEX_TTS_MODEL_KEY} not found in configuration")

    model_config = AVAILABLE_MODELS[INDEX_TTS_MODEL_KEY]
    url = model_config.get("url")

    if not url:
        raise HTTPException(status_code=500, detail=f"No URL configured for TTS model {INDEX_TTS_MODEL_KEY}")

    return url


async def check_tts_service():
    """
    Check if the TTS service is available.
    """
    try:
        base_url = get_tts_service_url()
        # 使用正确的健康检查端点
        health_url = f"{base_url}/health"
        logger.trace(f"Checking TTS service health at: {health_url}")
        async with aiohttp.ClientSession() as session, session.get(health_url) as response:
            if response.status == HTTP_OK:
                return True
            else:
                logger.error(f"TTS service health check failed: {response.status}")
                return False
    except Exception as e:
        logger.error(f"Failed to connect to TTS service: {str(e)}")
        return False


@handler_router.post("/v1/audio/speech", response_class=Response)
async def create_speech(request: Request):
    """
    Endpoint for text-to-speech synthesis using docker-index-tts service.
    """
    try:
        # Parse request body
        body = await request.json()
        tts_request = AudioSpeechRequest(**body)

        # Check if TTS service is available
        if not await check_tts_service():
            raise HTTPException(status_code=503, detail="TTS service is not available")

        # Process audio prompt
        audio_prompt_base64 = await process_audio_prompt(tts_request)

        # Forward request to docker-index-tts service
        return await perform_tts_inference(tts_request, audio_prompt_base64)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing speech request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing speech request: {str(e)}") from e


def _parse_voice_prompt(body):
    """Parse voice prompt from various request structures."""
    if "voice" in body:
        return body["voice"]
    elif "audio_prompt" in body:
        return body["audio_prompt"]
    return None


async def process_audio_prompt(tts_request: AudioSpeechRequest) -> str:
    """
    Process the audio prompt from the request and return as base64 string.
    """
    if tts_request.voice:
        # Already base64-encoded, return as is
        return tts_request.voice
    elif tts_request.voice_file_path:
        # Read file and encode as base64
        try:
            if not os.path.exists(tts_request.voice_file_path):
                raise HTTPException(status_code=400, detail=f"Audio prompt file not found: {tts_request.voice_file_path}")

            with open(tts_request.voice_file_path, "rb") as f:
                audio_data = f.read()
                return base64.b64encode(audio_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read audio file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read audio file: {str(e)}") from e
    else:
        # No voice prompt provided, return None
        return None


async def perform_tts_inference(tts_request: AudioSpeechRequest, audio_prompt_base64: str = None) -> Response:
    """
    Forward TTS request to docker-index-tts service and return the audio response.
    """
    logger.info(f"Forwarding TTS request to docker-index-tts service for text: {tts_request.input[:50]}...")

    try:
        # Prepare request payload for docker-index-tts service based on the provided test script format
        payload = {
            "text": tts_request.input,
            "response_format": {"type": tts_request.response_format.lower()},
            "max_text_tokens_per_sentence": tts_request.max_text_tokens_per_sentence,
            "sentences_bucket_max_size": tts_request.sentences_bucket_max_size,
            "verbose": tts_request.verbose,
        }

        # Add audio prompt if provided
        if audio_prompt_base64:
            payload["voice"] = audio_prompt_base64

        # Add speed parameter if provided
        if tts_request.speed is not None:
            payload["speed"] = tts_request.speed

        # Send request to docker-index-tts service
        base_url = get_tts_service_url()
        logger.trace(f"TTS url: {base_url}")
        async with aiohttp.ClientSession() as session, session.post(f"{base_url}/v1/tts", json=payload) as response:
            if response.status != HTTP_OK:
                error_detail = await response.text()
                logger.error(f"TTS service returned error: {response.status}, {error_detail}")
                raise HTTPException(status_code=response.status, detail=f"TTS service error: {error_detail}")

            # Get content type from response headers
            content_type = response.headers.get("Content-Type", "audio/mpeg")

            # Read response content
            audio_data = await response.read()

            # Return audio response
            return Response(content=audio_data, media_type=content_type)

    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to TTS service: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to TTS service: {str(e)}") from e
    except Exception as e:
        logger.error(f"TTS inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}") from e


@handler_router.post("/v1/audio/transcriptions", response_class=Response)
async def create_transcriptions(request: AudioTranscriptionRequest):
    """
    Alternative endpoint for speech-to-text synthesis using docker-index-tts service.
    This endpoint follows OpenAI's TTS API format.
    """
    try:
        pass

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing TTS request at /v1/tts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing TTS request: {str(e)}") from e
