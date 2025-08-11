from fastapi import APIRouter, HTTPException
from loguru import logger

from ..utils.llm_message_type import RerankRequest, RerankResponse, RerankResult
from ..utils.llm_utils import AVAILABLE_MODELS, call_backend

handler_router = APIRouter()


@handler_router.post("/v1/rerank")
async def create_rerank(request: RerankRequest) -> RerankResponse:
    """文档重排序"""
    if request.model not in AVAILABLE_MODELS:
        logger.error(f"Model {request.model} not found")
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "rerank":
        logger.error(f"Model {request.model} is not a rerank model")
        raise HTTPException(status_code=404, detail=f"Model {request.model} is not a rerank model")

    # 判断是否使用真实后端
    # 构建符合 Jina API 格式的请求载荷
    payload = {"model": request.model, "query": request.query, "documents": request.documents, "top_n": request.top_k}

    try:
        backend_response = await call_backend(request.model, "rerank", payload)

        # 检查响应格式并转换为我们的格式
        if "results" in backend_response:
            # Jina API 返回的格式
            jina_results = backend_response["results"]
            data = []
            for result in jina_results:
                data.append(
                    RerankResult(
                        index=result.get("index", 0),
                        relevance_score=result.get("relevance_score", 0.0),
                        document=result.get("document", {}).get("text", "") if isinstance(result.get("document"), dict) else str(result.get("document", "")),
                    )
                )

            usage = backend_response.get("usage", {"total_tokens": 0})

            return RerankResponse(data=data, model=request.model, usage=usage)
        else:
            # 如果响应格式已经兼容，直接返回
            return RerankResponse(**backend_response)

    except Exception as e:
        logger.error(f"Backend API call failed for model {request.model}: {str(e)}")
        # API 调用失败，返回错误
        raise HTTPException(status_code=502, detail=f"Backend API call failed for model {request.model}: {str(e)}") from e
