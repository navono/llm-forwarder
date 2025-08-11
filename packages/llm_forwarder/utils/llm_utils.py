import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from loguru import logger

from .llm_message_type import HTTP_OK

load_dotenv()


# 尝试从配置文件加载模型配置
def load_models_from_config() -> dict[str, Any]:
    """
    从配置文件加载模型配置
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "config" / "models.json"

    try:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                models = json.load(f)
                logger.info(f"从配置文件加载了 {len(models)} 个模型配置")

                # 创建模型名称到配置的映射
                model_map = {}
                for model_name, config in models.items():
                    # 使用 api_key 作为内部引用键
                    if "api_key" in config:
                        api_key = config["api_key"]
                        model_map[api_key] = config
                        model_map[api_key]["model_name_in_request"] = model_name
                    else:
                        # 向后兼容，如果没有 api_key，则使用模型名称作为键
                        model_map[model_name] = config

                logger.info(f"模型配置映射创建完成，共 {len(model_map)} 个模型")
                return model_map
        else:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {}
    except Exception as e:
        logger.error(f"加载模型配置失败: {str(e)}")
        return {}


# 加载模型配置
AVAILABLE_MODELS = load_models_from_config()


def generate_id(prefix: str = "chatcmpl") -> str:
    """生成唯一ID"""
    return f"{prefix}-{uuid.uuid4().hex[:29]}"


async def call_backend(model_name: str, endpoint_type: str, payload: dict[str, Any]) -> dict[str, Any]:
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
            logger.error("no secret key")

    # 调整请求体中的模型名称
    if "model_name" in model_config and model_config["model_name"]:
        # 如果model_name不为空，使用它
        payload["model"] = model_config["model_name"]

    try:
        # 为深度搜索等耗时操作增加更长的超时时间
        timeout_duration = 120.0 if endpoint_type == "deep-search" else 60.0
        async with httpx.AsyncClient(timeout=timeout_duration) as client:
            logger.debug(f"Calling real backend with post: {backend_url} with model {payload['model']}")
            response = await client.post(backend_url, json=payload, headers=headers)
            if response.status_code == HTTP_OK:
                logger.debug(f"Request success with status code: {response.status_code}")
                return response.json()
            else:
                logger.error(f"Request failed with status code: {response.status_code}, reason: {response.text}")
                error_detail = f"Backend API error: {response.status_code}"
                raise HTTPException(status_code=response.status_code, detail=error_detail)

    except httpx.HTTPStatusError as e:
        error_detail = f"Backend API error: {e.response.status_code} - {e.response.text}"
        logger.error(f"HTTP Error: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail) from e

    except httpx.TimeoutException as e:
        error_detail = f"Backend API timeout for model {model_name}"
        logger.error(f"Timeout Error: {error_detail}")
        raise HTTPException(status_code=504, detail=error_detail) from e

    except Exception as e:
        error_detail = f"Unexpected error calling backend {model_name}: {str(e)}"
        logger.error(f"Unexpected Error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail) from e
