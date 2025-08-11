#!/usr/bin/env python3

import base64
import os
import sys
import tempfile
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Add the index-tts package to the path if needed
index_tts_path = Path(__file__).parent / "index-tts"
if index_tts_path.exists():
    sys.path.append(str(index_tts_path))

# Import IndexTTS
try:
    from indextts.infer import IndexTTS
except ImportError as e:
    logger.error(f"Failed to import IndexTTS: {e}")
    logger.error("Make sure index-tts is installed or in the Python path")
    sys.exit(1)

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("index-tts-server.log", rotation="10 MB", level="INFO")

# Create FastAPI app
app = FastAPI(
    title="IndexTTS API",
    description="API for text-to-speech synthesis using IndexTTS",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global IndexTTS instance
tts_instance = None

# Model configuration
MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints")
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(MODEL_DIR, "config.yaml"))
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"
USE_CUDA_KERNEL = os.environ.get("USE_CUDA_KERNEL", "1") == "1"


# Request models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    audio_prompt: str | None = Field(None, description="Base64-encoded audio prompt or path to audio file")
    output_format: str = Field("mp3", description="Output audio format (mp3 or wav)")
    max_text_tokens_per_sentence: int = Field(100, description="Maximum text tokens per sentence")
    sentences_bucket_max_size: int = Field(4, description="Maximum sentences per bucket")
    verbose: bool = Field(False, description="Enable verbose output")


# Helper functions
def get_tts_instance():
    """Get or initialize the IndexTTS instance"""
    # Using a global instance is necessary for FastAPI to reuse the model across requests
    # pylint: disable=global-statement
    global tts_instance
    if tts_instance is None:
        logger.info(f"Initializing IndexTTS model with config {CONFIG_PATH} and model dir {MODEL_DIR}")
        try:
            tts_instance = IndexTTS(cfg_path=CONFIG_PATH, model_dir=MODEL_DIR, is_fp16=USE_FP16, use_cuda_kernel=USE_CUDA_KERNEL)
            logger.info("IndexTTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IndexTTS: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize TTS model: {str(e)}") from e
    return tts_instance


async def process_audio_prompt(audio_prompt):
    """Process audio prompt from base64 or file path"""
    if not audio_prompt:
        return None

    # Check if it's a base64 string
    if audio_prompt.startswith("data:") or ";base64," in audio_prompt:
        try:
            # Extract the base64 part if it's a data URL
            if ";base64," in audio_prompt:
                audio_prompt = audio_prompt.split(";base64,")[1]

            # Decode base64
            audio_data = base64.b64decode(audio_prompt)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                return temp_file.name
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}") from e

    # Assume it's a file path
    if os.path.exists(audio_prompt):
        return audio_prompt
    else:
        raise HTTPException(status_code=404, detail=f"Audio prompt file not found: {audio_prompt}")


# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "IndexTTS API is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/v1/tts")
async def tts_endpoint(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate speech from text"""
    try:
        # Get TTS instance
        tts = get_tts_instance()

        # Process audio prompt if provided
        audio_prompt_path = None
        if request.audio_prompt:
            audio_prompt_path = await process_audio_prompt(request.audio_prompt)
            background_tasks.add_task(lambda: os.unlink(audio_prompt_path) if audio_prompt_path and os.path.exists(audio_prompt_path) else None)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=f".{request.output_format}", delete=False) as temp_file:
            output_path = temp_file.name

        # Add cleanup task
        background_tasks.add_task(lambda: os.unlink(output_path) if os.path.exists(output_path) else None)

        # Generate speech
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        # Call infer_fast and ignore the result as we're using the output file directly
        tts.infer_fast(
            audio_prompt=audio_prompt_path, text=request.text, output_path=output_path, verbose=request.verbose, max_text_tokens_per_sentence=request.max_text_tokens_per_sentence, sentences_bucket_max_size=request.sentences_bucket_max_size
        )

        # Stream the audio file
        def iterfile():
            with open(output_path, "rb") as f:
                yield from f

        content_type = "audio/mpeg" if request.output_format == "mp3" else "audio/wav"
        return StreamingResponse(iterfile(), media_type=content_type)

    except Exception as e:
        logger.error(f"Error in TTS generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Alternative endpoint matching the llm-forwarder API
@app.post("/v1/audio/speech")
async def speech_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Generate speech from text (compatible with llm-forwarder API)"""
    try:
        # Parse request body
        data = await request.json()

        # Convert to TTSRequest format
        tts_request = TTSRequest(
            text=data.get("input", ""),
            audio_prompt=data.get("voice", None),
            output_format=data.get("response_format", {}).get("type", "mp3"),
            max_text_tokens_per_sentence=data.get("max_text_tokens_per_sentence", 100),
            sentences_bucket_max_size=data.get("sentences_bucket_max_size", 4),
            verbose=data.get("verbose", False),
        )

        # Call the main TTS endpoint
        return await tts_endpoint(tts_request, background_tasks)

    except Exception as e:
        logger.error(f"Error in speech endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", "12234"))

    # Start server
    logger.info(f"Starting IndexTTS server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
