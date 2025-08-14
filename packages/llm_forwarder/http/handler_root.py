import time

from dask.distributed import get_worker
from fastapi import APIRouter
from loguru import logger

from ..utils import get_dask_client
from ..utils.llm_message_type import ModelInfo, ModelsResponse
from ..utils.llm_utils import AVAILABLE_MODELS

handler_router = APIRouter()


def inc(x):
    logger.trace("process request")
    worker = get_worker()
    logger.trace(f"worker: {worker}")
    counter = worker.data.get("counter")
    return counter


def get_current_timestamp() -> int:
    """获取当前时间戳"""
    return int(time.time())


@handler_router.get("/")
async def home():
    try:
        logger.trace("receive request")
        client = get_dask_client()
        worker_addresses = list(client.scheduler_info()["workers"].keys())
        worker1_address = worker_addresses[0]
        result = client.submit(inc, 1, workers=[worker1_address]).result()
        return {"message": "Hello, FastAPI!", "dask result": result}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"message": "Error processing request", "error": str(e)}


@handler_router.get("/health")
async def health():
    return {"status": "ok"}


@handler_router.get("/v1/models")
async def list_models() -> ModelsResponse:
    """列出所有可用模型"""
    models = []
    for model_id, info in AVAILABLE_MODELS.items():
        # 根据模型类型设置能力
        capabilities = {}
        model_type = info.get("type", "")

        # 设置模型能力
        if model_type == "chat":
            capabilities["chat"] = True
        elif model_type == "embedding":
            capabilities["embedding"] = True
        elif model_type == "tts":
            capabilities["tts"] = True
        elif model_type == "asr":
            capabilities["asr"] = True
        elif model_type == "rerank":
            capabilities["rerank"] = True
        elif model_type == "classify":
            capabilities["classify"] = True
        elif model_type == "search":
            capabilities["search"] = True

        models.append(ModelInfo(id=model_id, created=get_current_timestamp(), owned_by=info["owned_by"], capabilities=capabilities))

    return ModelsResponse(data=models)
