import json
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from ..utils.llm_message_type import HTTP_OK, ClassifyRequest, ClassifyResponse, ClassifyResult, SearchRequest, SearchResponse, SearchResult
from ..utils.llm_utils import AVAILABLE_MODELS, call_backend

handler_router = APIRouter()


async def _decode_request_body(raw_body: bytes) -> str:
    """
    使用多种编码方式尝试解码请求体
    """
    try:
        # 先尝试UTF-8解码
        return raw_body.decode("utf-8")
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        logger.warning("UTF-8 decoding failed, attempting to detect encoding")
        try:
            # 尝试UTF-16
            return raw_body.decode("utf-16")
        except UnicodeDecodeError:
            try:
                # 尝试GB18030
                return raw_body.decode("gb18030")
            except UnicodeDecodeError:
                try:
                    # 尝试Big5
                    return raw_body.decode("big5")
                except UnicodeDecodeError:
                    # 最后尝试使用替换错误的UTF-8
                    logger.warning("All encoding attempts failed, using utf-8 with errors='replace'")
                    return raw_body.decode("utf-8", errors="replace")


async def _parse_json_body(raw_body: bytes) -> dict:
    """
    解码请求体并解析JSON
    """
    body_str = await _decode_request_body(raw_body)
    return json.loads(body_str)


async def stream_search_response(search_results: list[SearchResult], model: str):
    """
    流式返回搜索结果
    """
    # 生成唯一ID
    response_id = str(uuid.uuid4())
    created_time = int(time.time())

    # 逐个返回搜索结果
    for i, result in enumerate(search_results):
        # 将 result 按照标点符号进行拆分
        sentences = re.split(r"[\.\?\!\，\。\？\！]", result.description)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        for j, sentence in enumerate(sentences):
            response_chunk = {
                "id": response_id,
                "object": "search.chunk",
                "created": created_time,
                "model": model,
                "data": sentence,
                "index": j,
            }

            # 转换为JSON并返回
            yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"

    # 发送结束信号
    done_chunk = {"id": response_id, "object": "search.done", "created": created_time, "model": model}
    yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
    logger.info(f"Search streaming completed for {len(search_results)} results")


async def _summarize_search_results(query: str, search_results: list[SearchResult], stream: bool = False) -> str | AsyncGenerator[str, None]:
    """
    使用LLM总结搜索结果

    Args:
        query: 用户查询
        search_results: 搜索结果列表
        stream: 是否使用流式模式返回

    Returns:
        如果stream=False，返回总结文本
        如果stream=True，返回异步生成器，生成流式总结文本
    """
    # 从环境变量中获取默认模型和基础URL
    default_model = os.getenv("DEFAULT_LLM_MODEL", "qwen3_235b_a22b_instruct_2507_fp8")
    # 不使用环境变量中的DEFAULT_BASE_URL，而是使用外部LLM服务地址
    # 避免递归调用自身服务
    base_url = os.getenv("DEFAULT_BASE_URL", "http://localhost:8000")

    # 构建提示词
    prompt = f"""根据下面的搜索结果，回答用户的问题。请综合分析搜索结果，给出简洁明确的回答。
用户问题: {query}

搜索结果:
"""

    # 添加搜索结果到提示词中
    for i, result in enumerate(search_results):
        prompt += f"\n{i + 1}. {result.title}\n"
        prompt += f"   链接: {result.url}\n"
        prompt += f"   描述: {result.description}\n"

    # 构建请求体
    messages = [{"role": "system", "content": "你是一个智能搜索助手，可以根据搜索结果回答用户问题。请用中文回答。"}, {"role": "user", "content": prompt}]

    payload = {
        "model": default_model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": stream,  # 设置流式模式
    }

    # 调用本地的聊天补全API
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    if stream:
        # 流式模式
        async def stream_summary():
            response_id = str(uuid.uuid4())
            created_time = int(time.time())

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    logger.debug(f"Calling LLM for streaming summary with URL: {url}")
                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        response.raise_for_status()

                        # 处理流式响应
                        async for chunk in response.aiter_text():
                            yield chunk

                # 发送结束信号
                done_chunk = {"id": response_id, "object": "search.done", "created": created_time, "model": default_model}
                yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
                logger.info("Summary streaming completed")

            except Exception as e:
                logger.error(f"Error in streaming summary: {e}")
                error_chunk = {"id": response_id, "object": "search.error", "created": created_time, "model": default_model, "error": str(e)}
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

        return stream_summary()
    else:
        # 非流式模式
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug(f"Calling LLM for summary with URL: {url}")
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()
                logger.debug(f"LLM summary response: {result}")

                # 提取生成的文本
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return "无法生成总结。"
        except Exception as e:
            logger.error(f"Error calling LLM for summary: {e}")
            return f"生成总结时出错: {str(e)}"


async def _prepare_streaming_request(model_name: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, str], float]:
    """
    准备流式请求的参数
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
            logger.error("no secret key")

    # 确保流式参数设置为 True
    payload["stream"] = True

    # 调整请求体中的模型名称
    if "model_name_in_request" in model_config:
        payload["model"] = model_config["model_name_in_request"]

    # 为深度搜索等耗时操作增加更长的超时时间
    timeout_duration = 120.0 if model_config.get("type") == "deep-search" else 60.0

    return backend_url, payload, headers, timeout_duration


async def _process_streaming_line(line: str) -> tuple[str, bool]:
    """
    处理流式响应的单行数据
    返回：(处理后的数据, 是否结束)
    """
    chunk = line
    if chunk.startswith("data: "):
        # 移除 "data: " 前缀
        chunk = chunk[6:]

    if chunk.strip() == "":
        return "", False

    if chunk == "[DONE]":
        return "data: [DONE]\n\n", True

    try:
        # 尝试解析JSON以提取实际内容
        data = json.loads(chunk)
        # 检查是否有choices和delta
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                # 只返回实际内容部分
                content = choice["delta"]["content"]
                return f"data: {json.dumps({'choices': [{'delta': {'content': content}}]}, ensure_ascii=False)}\n\n", False
    except Exception as e:
        logger.warning(f"Error parsing streaming JSON: {e}, using raw chunk")

    # 如果无法解析或没有找到内容，返回原始数据
    return f"data: {chunk}\n\n", False


async def stream_chat_response(model_name: str, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    流式返回聊天响应
    """
    try:
        backend_url, payload, headers, timeout_duration = await _prepare_streaming_request(model_name, payload)

        async with httpx.AsyncClient(timeout=timeout_duration) as client:
            logger.debug(f"Streaming from backend: {backend_url} with model {payload['model']}")

            async with client.stream("POST", backend_url, json=payload, headers=headers) as response:
                if response.status_code != HTTP_OK:
                    error_text = await response.aread()
                    logger.error(f"Stream request failed with status code: {response.status_code}, {error_text}")
                    error_json = json.dumps({"error": {"message": f"Backend error: {response.status_code}", "type": "backend_error"}})
                    yield f"data: {error_json}\n\n"
                    return

                # 处理流式响应
                async for line in response.aiter_lines():
                    try:
                        result, done = await _process_streaming_line(line)
                        # logger.trace(f"Processing line: {line}")
                        if result:
                            yield result
                        if done:
                            break
                    except Exception as exc:
                        logger.error(f"Error processing line: {exc}, line: {line}")
                        continue
            logger.debug("Stream request completed")

    except httpx.HTTPStatusError as exc:
        error_detail = f"Backend API error: {exc.response.status_code} - {exc.response.text}"
        logger.error(f"HTTP Error: {error_detail}")
        error_json = json.dumps({"error": {"message": error_detail, "type": "http_error"}})
        yield f"data: {error_json}\n\n"

    except httpx.TimeoutException:
        error_detail = f"Backend API timeout for model {model_name}"
        logger.error(f"Timeout Error: {error_detail}")
        error_json = json.dumps({"error": {"message": error_detail, "type": "timeout"}})
        yield f"data: {error_json}\n\n"

    except Exception as exc:
        error_detail = f"Unexpected error streaming from backend {model_name}: {str(exc)}"
        logger.error(f"Unexpected Error: {error_detail}")
        error_json = json.dumps({"error": {"message": error_detail, "type": "server_error"}})
        yield f"data: {error_json}\n\n"


@handler_router.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    """创建聊天完成，使用 LiteLLM 处理请求"""
    try:
        # 解析请求体
        raw_body = await request.body()
        body = await _parse_json_body(raw_body)

        # 记录解析后的请求体以便调试
        logger.debug(f"Decoded request body: {str(body)[:100]}...")
        logger.debug(f"Received chat completion request: {body}")

        # 提取所有可能的参数
        model = body.get("model")
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        top_k = body.get("top_k")
        stream = body.get("stream", False)
        stop = body.get("stop")
        frequency_penalty = body.get("frequency_penalty")
        presence_penalty = body.get("presence_penalty")
        repetition_penalty = body.get("repetition_penalty")

        # 验证必要字段
        if not model:
            logger.error("Missing required field: model")
            raise HTTPException(status_code=400, detail="Missing required field: model")

        if not messages:
            logger.error("Missing required field: messages")
            raise HTTPException(status_code=400, detail="Missing required field: messages")

        # 验证模型是否存在
        if model not in AVAILABLE_MODELS:
            logger.error(f"Model {model} not found")
            raise HTTPException(status_code=404, detail=f"Model {model} not found")

        if AVAILABLE_MODELS[model]["type"] != "chat":
            logger.error(f"Model {model} is not a chat model")
            raise HTTPException(status_code=400, detail=f"Model {model} is not a chat model")

        target_llm = AVAILABLE_MODELS[model]
        # 准备 LiteLLM 参数
        litellm_params = {"model": target_llm.get("model_name_in_request", model), "api_base": target_llm.get("url"), "messages": messages, "stream": stream}

        # 添加可选参数（如果存在）
        if max_tokens is not None:
            litellm_params["max_tokens"] = max_tokens
        if temperature is not None:
            litellm_params["temperature"] = temperature
        if top_p is not None:
            litellm_params["top_p"] = top_p
        if top_k is not None:
            litellm_params["top_k"] = top_k
        if stop is not None:
            litellm_params["stop"] = stop
        if frequency_penalty is not None:
            litellm_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            litellm_params["presence_penalty"] = presence_penalty
        if repetition_penalty is not None:
            litellm_params["repetition_penalty"] = repetition_penalty

        # 使用原来的方式处理请求，避免使用 LiteLLM 的企业版功能
        logger.debug(f"Using backend API with params: {litellm_params}")

        # 构建请求载荷
        payload = {"model": litellm_params["model"], "messages": litellm_params["messages"], "stream": litellm_params["stream"]}

        # 添加可选参数
        for key in ["max_tokens", "temperature", "top_p", "top_k", "stop", "frequency_penalty", "presence_penalty", "repetition_penalty"]:
            if key in litellm_params and litellm_params[key] is not None:
                payload[key] = litellm_params[key]

        # 处理流式响应
        if stream:
            logger.debug(f"Streaming response for model {model}")
            return StreamingResponse(stream_chat_response(model, payload), media_type="text/event-stream")

        # 非流式响应
        backend_response = await call_backend(model, "chat", payload)
        return backend_response

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # 检查是否为模型不存在错误
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Model not found or provider not configured: {str(e)}") from e
        else:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e


@handler_router.get("/v1/search", response_model=SearchResponse)
async def create_search(request: Request):
    """搜索接口 - 支持查询参数或JSON请求体"""
    # 尝试从请求体获取JSON数据
    try:
        # 先尝试读取请求体
        raw_body = await request.body()
        if raw_body:
            try:
                # 使用辅助函数解析JSON请求体
                body = await _parse_json_body(raw_body)
                logger.debug(f"Received search request body: {body}")

                # 创建SearchRequest对象
                request_obj = SearchRequest(model=body.get("model"), query=body.get("query"), top_k=body.get("top_k", 10), return_json=body.get("return_json", False), summary=body.get("summary", False), stream=body.get("stream", False))
            except Exception as e:
                logger.warning(f"Failed to parse request body as JSON: {e}")
                # 如果无法解析JSON，尝试从查询参数获取
                request_obj = SearchRequest(
                    model=request.query_params.get("model"),
                    query=request.query_params.get("query"),
                    top_k=int(request.query_params.get("top_k", 10)),
                    return_json=request.query_params.get("return_json", "false").lower() == "true",
                    summary=request.query_params.get("summary", "false").lower() == "true",
                    stream=request.query_params.get("stream", "false").lower() == "true",
                )
        else:
            # 如果请求体为空，尝试从查询参数获取
            request_obj = SearchRequest(
                model=request.query_params.get("model"),
                query=request.query_params.get("query"),
                top_k=int(request.query_params.get("top_k", 10)),
                return_json=request.query_params.get("return_json", "false").lower() == "true",
                summary=request.query_params.get("summary", "false").lower() == "true",
                stream=request.query_params.get("stream", "false").lower() == "true",
            )
    except Exception as e:
        logger.error(f"Error parsing search request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}") from e

    # 验证模型
    if request_obj.model not in AVAILABLE_MODELS or AVAILABLE_MODELS[request_obj.model]["type"] != "search":
        raise HTTPException(status_code=404, detail=f"Model {request_obj.model} not found or not a search model")

    try:
        # 构建搜索URL
        base_url = AVAILABLE_MODELS[request_obj.model]["url"]

        # 发起HTTP请求
        headers = {
            "Authorization": f"Bearer {os.getenv('JINA_SECRET_KEY')}",
            "Accept": "application/json",
        }
        if request_obj.return_json:
            headers["Accept"] = "application/json"
        headers["X-Respond-With"] = "no-content"
        params = {"q": request_obj.query}

        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.debug(f"url: {base_url}")
            # logger.debug(f"headers: {headers}")
            response = await client.get(base_url, headers=headers, params=params)

            if response.status_code != HTTP_OK:
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
                logger.debug(f"Unexpected search results format: {search_results}")
                items = []

            # logger.debug(f"Processing {len(items)} items")

            # 限制结果数量
            for item in items[: request_obj.top_k]:
                # 尝试不同的字段名称格式
                result = SearchResult(title=item.get("title", ""), url=item.get("url", ""), description=item.get("description", ""), date=item.get("date", None))
                results.append(result)

            # 如果请求中指定了stream=true，使用流式响应
            if request_obj.stream:
                logger.info(f"Streaming search results for query: {request_obj.query}")

                # 如果同时请求了summary和stream，使用流式总结
                if request_obj.summary:
                    logger.debug(f"Generating streaming summary for search query: {request_obj.query}")
                    # 注意：需要使用await来获取协程结果，并传递给StreamingResponse
                    summary_generator = await _summarize_search_results(request_obj.query, results, stream=True)
                    return StreamingResponse(summary_generator, media_type="text/event-stream")
                else:
                    # 普通流式搜索结果
                    return StreamingResponse(stream_search_response(results, request_obj.model), media_type="text/event-stream")

            # 非流式模式下的总结
            if request_obj.summary:
                logger.info(f"Generating summary for search query: {request_obj.query}")
                summary = await _summarize_search_results(request_obj.query, results)

                # 创建一个包含总结的结果对象
                summary_result = SearchResult(title="AI 搜索总结", url="", description=summary)

                # 将总结结果放在最前面
                results = [summary_result]

            # 普通响应
            response_obj = SearchResponse(data=results, model=request_obj.model)
            logger.debug(f"Response object: {response_obj}")
            return response_obj

    except Exception as e:
        logger.error(f"Error in search API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@handler_router.post("/v1/classify", response_model=ClassifyResponse)
async def create_classify(request: ClassifyRequest) -> ClassifyResponse:
    """文本分类"""
    if request.model not in AVAILABLE_MODELS:
        logger.error(f"Model {request.model} not found")
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    if AVAILABLE_MODELS[request.model]["type"] != "classify":
        logger.error(f"Model {request.model} is not a classify model")
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not a classify model")

    # 处理输入
    texts = [request.input] if isinstance(request.input, str) else request.input

    payload = {
        "model": request.model,
        "input": texts,
        "labels": request.labels,
    }

    # 移除 None 值
    payload = {k: v for k, v in payload.items() if v is not None}

    backend_response = await call_backend(request.model, "classify", payload)

    # 处理 Jina API 响应格式
    try:
        data = []
        usage = backend_response.get("usage", {"total_tokens": 0})

        # 获取分类结果
        classify_results = backend_response["data"] if "data" in backend_response else backend_response if isinstance(backend_response, list) else []

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
        logger.error(f"Error processing Jina classify response: {e}")
        # 如果处理失败，返回错误
        raise HTTPException(status_code=502, detail=f"Failed to process backend response: {str(e)}") from e
