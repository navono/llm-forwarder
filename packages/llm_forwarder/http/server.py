import uvicorn
from fastapi import FastAPI
from loguru import logger

from .handler_llm_embed import handler_router as embed_handler_router
from .handler_llm_reranker import handler_router as reranker_handler_router
from .handler_llm_texts import handler_router as texts_handler_router
from .handler_llm_tts import handler_router as tts_handler_router
from .handler_root import handler_router as root_handler_router

app = FastAPI()
app.include_router(root_handler_router)
app.include_router(texts_handler_router)
app.include_router(embed_handler_router)
app.include_router(reranker_handler_router)
app.include_router(tts_handler_router)


class HTTPServer:
    def __init__(self, config=None):
        self.host = config["host"] if config and "host" in config else "127.0.0.1"
        self.port = config["port"] if config and "port" in config else 8000
        self.server = None  # Initialize server instance

    async def start(self):
        logger.trace(f"http server listening at: {self.host}:{self.port}")
        config = uvicorn.Config(app, host=self.host, port=self.port, lifespan="on")  # Ensure lifespan is handled
        self.server = uvicorn.Server(config)  # Store server instance
        await self.server.serve()

    async def stop(self):
        if self.server:
            logger.info("Signalling Uvicorn server to stop...")
            self.server.should_exit = True
