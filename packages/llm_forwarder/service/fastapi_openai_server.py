"""
OpenAI API 兼容的 FastAPI 服务器
支持文本生成、音频处理、嵌入、重排序等功能
"""

import os
import time
import uuid
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

# 创建 FastAPI 应用
app = FastAPI(title="OpenAI Compatible API Server", description="OpenAI API 兼容服务器，支持文本生成、音频处理、嵌入等功能", version="1.0.0")

# ==================== 模拟的后端服务配置 ====================

AVAILABLE_MODELS = {
    # 文本生成模型
    "qwen3_coder_30b_a3b": {"type": "chat", "url": "http://localhost:11234/v1/chat/completions", "model_name_in_request": "qwen3-coder-30b-a3b-instruct-1m", "owned_by": "custom"},
    "qwen3_0.6b": {"type": "chat", "url": "http://localhost:11234/v1/chat/completions", "model_name_in_request": "qwen3-0.6b", "owned_by": "custom"},
    "qwen3_235b_a22b_instruct_2507_fp8": {"type": "chat", "url": "http://localhost:18001/v1/chat/completions", "model_name_in_request": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "owned_by": "custom"},
    # TTS 模型
    "tts-1": {"type": "tts", "owned_by": "openai"},
    "tts-1-hd": {"type": "tts", "owned_by": "openai"},
    # ASR 模型
    "whisper-1": {"type": "asr", "owned_by": "openai"},
    # 嵌入模型
    "qwen3_embedding_4b": {"type": "embedding", "url": "http://localhost:11234/v1/embeddings", "model_name_in_request": "qwen3-embedding-4b@q6_k", "owned_by": "custom"},
    "jina-embeddings-v3": {"type": "embedding", "url": "https://api.jina.ai/v1/embeddings", "model_name_in_request": "jina-embeddings-v3", "owned_by": "jina"},
    # v4 是多模态
    "jina-embeddings-v4": {"type": "embedding", "url": "https://api.jina.ai/v1/embeddings", "model_name_in_request": "jina-embeddings-v4", "owned_by": "jina"},
    # 重排序模型
    "jina-reranker": {"type": "rerank", "url": "https://api.jina.ai/v1/rerank", "model_name_in_request": "jina-reranker-v2-base-multilingual", "owned_by": "jina"},
    # 深度搜索
    "jina-deepsearch": {"type": "deep-search", "url": "https://deepsearch.jina.ai/v1/chat/completions", "model_name_in_request": "jina-deepsearch-v1", "owned_by": "jina"},
    # 分类
    "jina-classify": {"type": "classify", "url": "https://api.jina.ai/v1/classify", "model_name_in_request": "jina-embeddings-v3", "owned_by": "jina"},
    # # Reader
    # "jina-reader": {
    #     "type": "reader",
    #     "url": "r.jina.ai",
    #     "owned_by": "jina"
    # }
    # Search
    "jina-search": {"type": "search", "url": "https://s.jina.ai", "model_name_in_request": "", "owned_by": "jina"},
}

# ==================== 数据模型定义 ====================


# ==================== 工具函数 ====================


def generate_id(prefix: str = "chatcmpl") -> str:
    """生成唯一ID"""
    return f"{prefix}-{uuid.uuid4().hex[:29]}"


def get_current_timestamp() -> int:
    """获取当前时间戳"""
    return int(time.time())


# ==================== 真实 HTTP 调用函数 ====================


async def call_real_backend(model_name: str, endpoint_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    调用真实的后端 API
    """
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model_config = AVAILABLE_MODELS[model_name]
    backend_url = model_config.get("url")

    if not backend_url:
        raise HTTPException(status_code=500, detail=f"No URL configured for model {model_name}")

    # 获取 API Key（如果需要）
    headers = {"Content-Type": "application/json"}

    # 为 Jina API 添加认证头
    if "jina.ai" in backend_url or "deepsearch.jina.ai" in backend_url:
        jina_secret_key = os.getenv("JINA_SECRET_KEY")
        if jina_secret_key:
            headers["Authorization"] = f"Bearer {jina_secret_key}"
        else:
            print("no secret key")

    # 调整请求体中的模型名称
    if "model_name_in_request" in model_config:
        payload["model"] = model_config["model_name_in_request"]

    try:
        # 为深度搜索等耗时操作增加更长的超时时间
        timeout_duration = 120.0 if endpoint_type == "deep-search" else 60.0
        async with httpx.AsyncClient(timeout=timeout_duration) as client:
            print(f"Calling real backend: {backend_url} with model {payload['model']}")
            response = await client.post(backend_url, json=payload, headers=headers)
            if response.status_code == 200:
                # print(response.json())
                return response.json()
            else:
                print(f"Request failed with status code: {response.status_code}")

    except httpx.HTTPStatusError as e:
        error_detail = f"Backend API error: {e.response.status_code} - {e.response.text}"
        print(f"HTTP Error: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

    except httpx.TimeoutException:
        error_detail = f"Backend API timeout for model {model_name}"
        print(f"Timeout Error: {error_detail}")
        raise HTTPException(status_code=504, detail=error_detail)

    except Exception as e:
        error_detail = f"Unexpected error calling backend {model_name}: {str(e)}"
        print(f"Unexpected Error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


async def should_use_real_backend(model_name: str) -> bool:
    """
    判断是否应该使用真实的后端调用
    """
    # 带 jina- 前缀的模型或配置了 URL 的模型使用真实后端
    if model_name.startswith("jina-"):
        return True

    model_config = AVAILABLE_MODELS.get(model_name, {})
    return "url" in model_config


# ==================== API 路由 ====================


@app.get("/")
async def root():
    """根路径"""
    return {"message": "OpenAI Compatible API Server", "version": "1.0.0"}


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """列出所有可用模型"""
    models = []
    for model_id, info in AVAILABLE_MODELS.items():
        models.append(ModelInfo(id=model_id, created=get_current_timestamp(), owned_by=info["owned_by"]))

    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """创建聊天完成"""
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "chat":
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not a chat model")

    # 判断是否使用真实后端
    if await should_use_real_backend(request.model):
        # 调用真实后端 API
        payload = {"model": request.model, "messages": [msg.dict() for msg in request.messages], "max_tokens": request.max_tokens, "temperature": request.temperature, "top_p": request.top_p, "stream": request.stream, "stop": request.stop}

        # 移除 None 值
        payload = {k: v for k, v in payload.items() if v is not None}

        backend_response = await call_real_backend(request.model, "chat", payload)

        # 返回后端响应（假设格式兼容）
        return ChatCompletionResponse(**backend_response)

    else:
        # 没有配置 URL 的模型不支持
        raise HTTPException(status_code=501, detail=f"Model {request.model} does not have a configured backend URL")


@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """文本转语音"""
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "tts":
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not a TTS model")

    # TTS 模型不支持真实后端调用，返回错误
    raise HTTPException(status_code=501, detail=f"TTS model {request.model} is not implemented")


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...), model: str = Form(...), language: str | None = Form(None), prompt: str | None = Form(None), response_format: str | None = Form("json"), temperature: float | None = Form(0)
) -> AudioTranscriptionResponse:
    """语音转文本"""
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    if AVAILABLE_MODELS[model]["type"] != "asr":
        raise HTTPException(status_code=400, detail=f"Model {model} is not an ASR model")

    # ASR 模型不支持真实后端调用，返回错误
    raise HTTPException(status_code=501, detail=f"ASR model {model} is not implemented")


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """创建文本嵌入"""
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "embedding":
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not an embedding model")

    # 处理输入
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # 判断是否使用真实后端
    if await should_use_real_backend(request.model):
        # 调用真实后端 API
        payload = {"model": request.model, "input": texts, "task": request.task, "dimensions": request.dimensions}

        # 移除 None 值
        payload = {k: v for k, v in payload.items() if v is not None}

        backend_response = await call_real_backend(request.model, "embedding", payload)

        # 返回后端响应（假设格式兼容）
        return EmbeddingResponse(**backend_response)

    else:
        # 没有配置 URL 的模型不支持
        raise HTTPException(status_code=501, detail=f"Model {request.model} does not have a configured backend URL")


@app.post("/v1/rerank")
async def create_rerank(request: RerankRequest) -> RerankResponse:
    """文档重排序"""
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "rerank":
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not a rerank model")

    # 判断是否使用真实后端
    if await should_use_real_backend(request.model):
        # 构建符合 Jina API 格式的请求载荷
        payload = {"model": request.model, "query": request.query, "documents": request.documents, "top_n": request.top_k}

        print(f"Sending rerank request to Jina API: {payload}")

        try:
            backend_response = await call_real_backend(request.model, "rerank", payload)

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
            print(f"Error calling Jina rerank API: {e}")
            # API 调用失败，返回错误
            raise HTTPException(status_code=502, detail=f"Backend API call failed for model {request.model}: {str(e)}")


@app.get("/v1/search", response_model=SearchResponse)
async def create_search(request: SearchRequest):
    """搜索接口"""
    if request.model not in AVAILABLE_MODELS or AVAILABLE_MODELS[request.model]["type"] != "search":
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found or not a search model")

    try:
        # 构建搜索URL
        base_url = AVAILABLE_MODELS[request.model]["url"]

        # 发起HTTP请求
        headers = {
            "Authorization": f"Bearer {os.getenv('JINA_SECRET_KEY')}",
            "Accept": "application/json",
        }
        if request.return_json:
            headers["Accept"] = "application/json"
        headers["X-Respond-With"] = "no-content"
        params = {"q": request.query}

        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"url: {base_url}")
            print(f"headers: {headers}")
            response = await client.get(base_url, headers=headers, params=params)

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Search API returned error: {response.text}")

            # 解析响应
            search_results = response.json()
            # print(f"search results: {search_results}")
            # 提取搜索结果
            results = []

            # 处理返回的数据结构
            if isinstance(search_results, dict) and "data" in search_results:
                # 如果是字典并包含 data 字段
                items = search_results["data"]
            elif isinstance(search_results, list):
                # 如果直接是列表
                items = search_results
            else:
                # 其他情况
                print(f"Unexpected search results format: {search_results}")
                items = []

            print(f"Processing {len(items)} items")

            # 限制结果数量
            for item in items[: request.top_k]:
                # 尝试不同的字段名称格式
                result = SearchResult(title=item.get("title", ""), url=item.get("url", ""), description=item.get("description", ""), date=item.get("date", None))
                results.append(result)

            # print(f"Final results: {results}")
            response_obj = SearchResponse(data=results, model=request.model)
            print(f"Response object: {response_obj}")
            return response_obj

    except Exception as e:
        print(f"Error in search API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/v1/classify", response_model=ClassifyResponse)
async def create_classify(request: ClassifyRequest) -> ClassifyResponse:
    """文本分类"""
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "classify":
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not a classify model")

    # 处理输入
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # 判断是否使用真实后端
    if await should_use_real_backend(request.model):
        # 调用真实后端 API
        payload = {
            "model": request.model,
            "input": texts,
            "labels": request.labels,
        }

        # 移除 None 值
        payload = {k: v for k, v in payload.items() if v is not None}

        backend_response = await call_real_backend(request.model, "classify", payload)

        print(f"Jina classify API response: {backend_response}")

        # 处理 Jina API 响应格式
        try:
            # Jina 分类 API 实际返回格式:
            # {"usage": {...}, "data": [{"object": "classification", "index": 0, "prediction": "label", "score": 0.x, "predictions": [{"label": "x", "score": 0.x}, ...]}, ...]}

            data = []
            usage = backend_response.get("usage", {"total_tokens": 0})

            # 获取分类结果
            if "data" in backend_response:
                classify_results = backend_response["data"]
            else:
                # 如果直接是结果数组
                classify_results = backend_response if isinstance(backend_response, list) else []

            # 为每个输入文本构建分类结果
            for i, text in enumerate(texts):
                text_results = []

                if i < len(classify_results) and isinstance(classify_results[i], dict):
                    # 获取当前文本的分类结果
                    text_classification = classify_results[i]

                    # 从 predictions 数组中提取每个标签的分数
                    if "predictions" in text_classification:
                        predictions = text_classification["predictions"]

                        # 为每个标签创建结果
                        for j, prediction in enumerate(predictions):
                            if isinstance(prediction, dict) and "label" in prediction and "score" in prediction:
                                text_results.append(ClassifyResult(label=prediction["label"], score=float(prediction["score"]), index=j))

                    # 如果没有找到 predictions，尝试从请求的标签列表创建默认结果
                    if not text_results:
                        for j, label in enumerate(request.labels):
                            text_results.append(ClassifyResult(label=label, score=0.0, index=j))
                else:
                    # 如果没有对应的结果，创建默认的零分结果
                    for j, label in enumerate(request.labels):
                        text_results.append(ClassifyResult(label=label, score=0.0, index=j))

                data.append(text_results)

            return ClassifyResponse(data=data, model=request.model, usage=usage)

        except Exception as e:
            print(f"Error processing Jina classify response: {e}")
            print(f"Raw response: {backend_response}")
            # 如果处理失败，返回错误
            raise HTTPException(status_code=502, detail=f"Failed to process backend response: {str(e)}")


# ==================== 错误处理 ====================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "invalid_request_error"}})


# ==================== 启动服务器 ====================

if __name__ == "__main__":
    uvicorn.run("fastapi_openai_server:app", host="0.0.0.0", port=8008, reload=True, log_level="info")
