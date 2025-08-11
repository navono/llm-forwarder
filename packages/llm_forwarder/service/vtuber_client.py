from loguru import logger

from .websocket import WSClient


class VtuberClient:
    def __init__(self, config: dict):
        self.config = config

        # self.asr_client = WebSocketClient(config["asr"]["url"], config["asr"]["retry_interval"], config["asr"]["max_retries"])
        self.tts_client = WSClient(config["tts"]["url"])

    async def start(self):
        logger.debug("Starting VtuberClient...")
        # await self.asr_client.connect()
        await self.tts_client.connect()
        logger.debug("VtuberClient started successfully")

        # await self.tts_client.send_message({"text": "三局两胜，我赢啦"})
        # await self.tts_client.send_message({"text": "三局两胜，我输啦"})
        # await self.tts_client.send_message({"text": "三局两胜，平局"})
        # await self.tts_client.send_message({"text": "好的，玩个剪刀石头布的游戏吧，三局两胜喔"})
        # await self.tts_client.send_message({"text": "准备好了吗，第一局开始咯，一、二、三"})
        # await self.tts_client.send_message({"text": "准备好了吗，第二局开始咯，一、二、三"})
        # await self.tts_client.send_message({"text": "准备好了吗，第三局开始咯，一、二、三"})

        # await self.tts_client.send_message({"text": "我没听清你说什么，请你再说一遍"})
        # await self.tts_client.send_message({"text": "我现在只会玩剪刀石头布游戏，其他的暂时都还不会"})
        # await self.tts_client.send_message({"text": "大家好，我是笨笨。我的激活词是“你好，笨笨”，激活后，我有三秒的时间来听取您的语音指令。"})
        # await self.tts_client.send_message({"text": "我在"})

        # await self.tts_client.send_message({"text": "好的主人，我来啦"})
        # await self.tts_client.send_message({"text": "好的，玩个剪刀石头布的游戏吧"})
        # await self.tts_client.send_message({"text": "准备好了吗，开始咯，一、二、三"})
        # await self.tts_client.send_message({"text": "出剪刀"})
        # await self.tts_client.send_message({"text": "出布"})
        # await self.tts_client.send_message({"text": "出石头"})
        # await self.tts_client.send_message({"text": "哈哈，我赢啦"})
        # await self.tts_client.send_message({"text": "我输啦"})
        # await self.tts_client.send_message({"text": "巧啦，平局"})

        # await self.tts_client.send_message({"text": "对不起主人，目前我只会玩剪刀石头布，成语接龙两个游戏。"})
        # await self.tts_client.send_message({"text": "好的我们开始游戏吧。"})
        # await self.tts_client.send_message({"text": "哈哈，我赢啦"})
        # await self.tts_client.send_message({"text": "1 2 3 开始"})

    async def stop(self):
        logger.debug("Stopping VtuberClient...")
        # await self.asr_client.disconnect()
        await self.tts_client.disconnect()
