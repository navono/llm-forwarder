import asyncio
import json
import socket

import websockets
from loguru import logger


class WSClient:
    def __init__(self, url, **kwargs):
        self.url = url
        # set some default values
        self.reply_timeout = kwargs.get("reply_timeout") or 15
        self.ping_timeout = kwargs.get("ping_timeout") or 15
        self.sleep_time = kwargs.get("sleep_time") or 5
        self.ws_connection = None
        self.message_handlers = {}

    async def connect(self):
        asyncio.create_task(self._listen_forever())

    async def disconnect(self):
        if self.ws_connection is not None:
            logger.debug("Closing WebSocket connection...")
            try:
                await self.ws_connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            finally:
                self.ws_connection = None

    def register_handler(self, message_type: str, handler: object):
        if message_type == "":
            logger.warning("Handler type cannot be empty")
            return None

        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    async def send_message(self, message_data: dict):
        if self.ws_connection is not None:
            await self.ws_connection.send(json.dumps(message_data))

    async def _listen_forever(self):
        while True:
            logger.trace(f"Creating new ws connection with url: {self.url}")
            try:
                async with websockets.connect(self.url) as ws_connection:
                    self.ws_connection = ws_connection
                    logger.trace(f"Connected to ws server: {self.url}")
                    while True:
                        try:
                            reply = await asyncio.wait_for(ws_connection.recv(), timeout=self.reply_timeout)
                        except (TimeoutError, websockets.exceptions.ConnectionClosed):
                            try:
                                pong = await ws_connection.ping()
                                await asyncio.wait_for(pong, timeout=self.ping_timeout)
                                # logger.trace("Ping OK, keeping connection alive...")
                                continue
                            except Exception as e:
                                logger.error(f"Ping error - retrying connection in {self.sleep_time} sec, error: {e}")
                                await asyncio.sleep(self.sleep_time)
                                break

                        data = json.loads(reply)
                        message_type = data.get("type")
                        # logger.trace(f"Get message type from data: {message_type}")
                        # if message_type in {"control", "full-text"}:
                        #     logger.trace(f"data :{data}")

                        if message_type in self.message_handlers:
                            for handler in self.message_handlers[message_type]:
                                await handler(data)
                        elif message_type is None:
                            # 遍历 message_handlers
                            for register_type, handlers in self.message_handlers.items():
                                for handler in handlers:
                                    data_with_type = {
                                        "type": register_type,
                                        "data": data,
                                    }
                                    await handler(data_with_type)

            except socket.gaierror:
                logger.error(f"Socket error - retrying connection in {self.sleep_time} sec")
                await asyncio.sleep(self.sleep_time)
                continue
            except ConnectionRefusedError:
                logger.error("Nobody seems to listen to this endpoint. Please check the URL.")
                logger.error(f"Retrying connection in {self.sleep_time} sec")
                await asyncio.sleep(self.sleep_time)
                continue
