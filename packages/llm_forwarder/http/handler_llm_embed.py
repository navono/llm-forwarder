from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from loguru import logger

from ..utils.llm_message_type import EmbeddingRequest, EmbeddingResponse
from ..utils.llm_utils import AVAILABLE_MODELS, call_backend

load_dotenv()
# HTTP status codes
HTTP_OK = 200

handler_router = APIRouter()


@handler_router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """创建文本嵌入"""
    if request.model not in AVAILABLE_MODELS:
        logger.error(f"Model {request.model} not found")
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "embedding":
        logger.error(f"Model {request.model} is not an embedding model")
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not an embedding model")

    # 处理输入
    texts = [request.input] if isinstance(request.input, str) else request.input

    # 判断是否使用真实后端
    # 调用真实后端 API
    payload = {"model": request.model, "input": texts, "task": request.task, "dimensions": request.dimensions}

    # 移除 None 值
    payload = {k: v for k, v in payload.items() if v is not None}

    backend_response = await call_backend(request.model, "embedding", payload)

    # 返回后端响应（假设格式兼容）
    return EmbeddingResponse(**backend_response)
