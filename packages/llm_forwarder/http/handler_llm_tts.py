import base64
import io
import os

import torch
import torchaudio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

# Import IndexTTS from the index-tts package
try:
    from indextts.infer import IndexTTS
except ImportError:
    logger.error("Failed to import IndexTTS. Make sure the index-tts package is installed.")
    IndexTTS = None

# Create router
handler_router = APIRouter()

# Global TTS model instance
_tts_model = None


def get_tts_model(
    cfg_path: str = "checkpoints/config.yaml",
    model_dir: str = "checkpoints",
    is_fp16: bool = True,
    device: str | None = None,
    use_cuda_kernel: bool | None = None,
) -> IndexTTS:
    """
    Get or initialize the TTS model instance.
    """
    # Using a nonlocal approach with a function attribute instead of global
    if not hasattr(get_tts_model, "_model_instance") or get_tts_model._model_instance is None:
        if IndexTTS is None:
            raise HTTPException(status_code=500, detail="IndexTTS module not available")
        
        try:
            logger.info(f"Initializing IndexTTS model from {model_dir}")
            get_tts_model._model_instance = IndexTTS(
                cfg_path=cfg_path,
                model_dir=model_dir,
                is_fp16=is_fp16,
                device=device,
                use_cuda_kernel=use_cuda_kernel,
            )
            logger.info("IndexTTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IndexTTS model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize TTS model: {str(e)}") from e
    
    return get_tts_model._model_instance


class TTSRequest(BaseModel):
    """
    Request model for text-to-speech synthesis.
    """
    model: str = Field(..., description="Model name to use for TTS")
    input: str = Field(..., description="Text to synthesize")
    voice: str | None = Field(None, description="Base64-encoded audio prompt or path to audio file")
    voice_file_path: str | None = Field(None, description="Path to audio prompt file (alternative to voice)")
    response_format: str = Field("mp3", description="Audio format to return (mp3 or wav)")
    speed: float | None = Field(1.0, description="Speech speed factor")
    
    # Advanced generation parameters
    max_text_tokens_per_sentence: int | None = Field(100, description="Maximum text tokens per sentence")
    sentences_bucket_max_size: int | None = Field(4, description="Maximum bucket size for sentences")
    verbose: bool | None = Field(False, description="Enable verbose output")


@handler_router.post("/v1/audio/speech", response_class=Response)
async def create_speech(request: Request):
    """
    Endpoint for text-to-speech synthesis using IndexTTS.
    """
    try:
        # Parse request body
        body = await request.json()
        tts_request = TTSRequest(**body)
        
        # Get or initialize TTS model
        tts_model = get_tts_model()
        
        # Process audio prompt
        audio_prompt_path = await process_audio_prompt(tts_request)
        
        # Prepare generation parameters
        generation_kwargs = {}
        if tts_request.speed is not None:
            generation_kwargs["speed"] = tts_request.speed
        
        # Perform TTS inference and get response
        return await perform_tts_inference(
            tts_model=tts_model,
            tts_request=tts_request,
            audio_prompt_path=audio_prompt_path,
            generation_kwargs=generation_kwargs
        )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing TTS request: {str(e)}") from e


async def process_audio_prompt(tts_request: TTSRequest) -> str:
    """
    Process the audio prompt from the request.
    """
    audio_prompt_path = None
    
    if tts_request.voice:
        try:
            # Handle base64-encoded audio
            audio_data = base64.b64decode(tts_request.voice)
            # Save to temporary file
            audio_prompt_path = os.path.join(os.getcwd(), "temp_prompt.wav")
            with open(audio_prompt_path, "wb") as f:
                f.write(audio_data)
            logger.debug(f"Saved audio prompt to {audio_prompt_path}")
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}") from e
    elif tts_request.voice_file_path:
        # Use provided file path
        audio_prompt_path = tts_request.voice_file_path
        if not os.path.exists(audio_prompt_path):
            raise HTTPException(status_code=400, detail=f"Audio prompt file not found: {audio_prompt_path}")
    else:
        raise HTTPException(status_code=400, detail="Either 'voice' or 'voice_file_path' must be provided")
    
    return audio_prompt_path


async def perform_tts_inference(tts_model, tts_request: TTSRequest, audio_prompt_path: str, generation_kwargs: dict) -> Response:
    """
    Perform TTS inference and return the audio response.
    """
    # Create output buffer for audio
    output_buffer = io.BytesIO()
    
    # Perform TTS inference
    logger.info(f"Performing TTS inference for text: {tts_request.input[:50]}...")
    try:
        # Use infer_fast method for better performance
        result = tts_model.infer_fast(
            audio_prompt=audio_prompt_path,
            text=tts_request.input,
            output_path=None,  # We'll handle the output ourselves
            verbose=tts_request.verbose,
            max_text_tokens_per_sentence=tts_request.max_text_tokens_per_sentence,
            sentences_bucket_max_size=tts_request.sentences_bucket_max_size,
            **generation_kwargs
        )
        
        # Process the result
        content_type = "audio/mpeg" if tts_request.response_format.lower() == "mp3" else "audio/wav"
        
        if isinstance(result, tuple):
            # Result is (sample_rate, audio_data)
            sample_rate, audio_data = result
            # Convert to tensor for torchaudio
            audio_tensor = torch.tensor(audio_data).T  # Transpose to get [channels, samples]
            
            # Save to buffer in requested format
            format_type = "mp3" if tts_request.response_format.lower() == "mp3" else "wav"
            torchaudio.save(output_buffer, audio_tensor, sample_rate, format=format_type)
        else:
            # Result is a file path
            with open(result, "rb") as f:
                output_buffer.write(f.read())
        
        # Clean up temporary file if created
        if tts_request.voice and audio_prompt_path and os.path.exists(audio_prompt_path):
            os.remove(audio_prompt_path)
        
        # Reset buffer position for reading
        output_buffer.seek(0)
        
        # Return audio response
        return Response(
            content=output_buffer.read(),
            media_type=content_type
        )
        
    except Exception as e:
        logger.error(f"TTS inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}") from e
