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
logger.add(sys.stderr, level="TRACE")
logger.add("index-tts-server.log", rotation="10 MB", level="TRACE")

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
MODEL_ROOT_DIR = os.environ.get("MODEL_ROOT_DIR", "/app/index-tts/")  # Default to current directory if not specified
MODEL_CHECKPOINTS_DIR = os.environ.get("MODEL_CHECKPOINTS_DIR", os.path.join(MODEL_ROOT_DIR, "checkpoints"))  # The directory containing bpe.model, config.yaml etc.
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(MODEL_CHECKPOINTS_DIR, "config.yaml"))
MODEL_DIR_FOR_INIT = os.environ.get("MODEL_DIR_FOR_INIT", MODEL_CHECKPOINTS_DIR)  # This is the directory passed to IndexTTS constructor
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"
USE_CUDA_KERNEL = os.environ.get("USE_CUDA_KERNEL", "1") == "1"

print(f"MODEL_ROOT_DIR: {MODEL_ROOT_DIR}")
print(f"MODEL_CHECKPOINTS_DIR: {MODEL_CHECKPOINTS_DIR}")
print(f"CONFIG_PATH: {CONFIG_PATH}")
print(f"MODEL_DIR_FOR_INIT: {MODEL_DIR_FOR_INIT}")
print(f"USE_FP16: {USE_FP16}")
print(f"USE_CUDA_KERNEL: {USE_CUDA_KERNEL}")


# Request models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    audio_prompt: str | None = Field(None, description="Base64-encoded audio prompt or path to audio file")
    output_format: str = Field("mp3", description="Output audio format (mp3 or wav)")
    max_text_tokens_per_sentence: int = Field(100, description="Maximum text tokens per sentence")
    sentences_bucket_max_size: int = Field(4, description="Maximum sentences per bucket")
    verbose: bool = Field(False, description="Enable verbose output")
    repetition_penalty: int = Field(10, description="Repetition penalty")
    top_p: float = Field(0.8, description="Top-p sampling parameter")
    top_k: int = Field(30, description="Top-k sampling parameter")
    temperature: float = Field(1.0, description="Sampling temperature")
    length_penalty: float = Field(0.0, description="Length penalty")
    num_beams: int = Field(3, description="Number of beams")
    max_mel_tokens: int = Field(600, description="Maximum mel tokens")
    do_sample: bool = Field(True, description="Do sample")


# Helper functions
def get_tts_instance():
    """Get or initialize the IndexTTS instance"""
    # Using a global instance is necessary for FastAPI to reuse the model across requests
    # pylint: disable=global-statement
    global tts_instance
    if tts_instance is None:
        logger.info(f"Initializing IndexTTS model with config {CONFIG_PATH} and model dir {MODEL_DIR_FOR_INIT}")
        try:
            tts_instance = IndexTTS(cfg_path=CONFIG_PATH, model_dir=MODEL_DIR_FOR_INIT, is_fp16=USE_FP16, use_cuda_kernel=USE_CUDA_KERNEL)
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
        else:
            audio_prompt_path = "./audio_prompt/Female-成熟_01.wav"
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=f".{request.output_format}", delete=False) as temp_file:
            output_path = temp_file.name

        # Add cleanup task
        background_tasks.add_task(lambda: os.unlink(output_path) if os.path.exists(output_path) else None)

        # Generate speech
        logger.info(f"Generating speech for text: {request.text[:50]}...")

        # 设置与成功案例相同的参数
        kwargs = {
            "do_sample": request.do_sample,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "temperature": request.temperature,
            "length_penalty": request.length_penalty,
            "num_beams": request.num_beams,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
        }

        # 使用infer方法替代infer_fast
        logger.trace(f"audio_prompt_path: {audio_prompt_path}")
        logger.trace(f"request.text: {request.text}")
        logger.trace(f"output_path: {output_path}")
        logger.trace(f"verbose: {request.verbose}")
        logger.trace(f"max_text_tokens_per_sentence: {request.max_text_tokens_per_sentence}")
        logger.trace(f"kwargs: {kwargs}")
        tts.infer(audio_prompt_path, request.text, output_path, verbose=request.verbose, max_text_tokens_per_sentence=request.max_text_tokens_per_sentence, **kwargs)

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
        # 处理response_format，确保它是一个字符串
        response_format = data.get("response_format", {})
        output_format = response_format.get("type", "mp3") if isinstance(response_format, dict) else "mp3"

        # 确保有文本输入
        text = data.get("input", "")
        if not text:
            text = "测试文本"  # 默认测试文本

        tts_request = TTSRequest(
            text=text,
            audio_prompt=data.get("voice", None),
            output_format=output_format,
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
