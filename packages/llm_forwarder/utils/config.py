import os

import yaml
from loguru import logger


class Config:
    _instance = None
    _config = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not Config._initialized:
            try:
                current_dir = os.getcwd()
                logger.debug(f"Current dir: {current_dir}, Loading config file...")

                config_file = "../config/config.yaml" if current_dir.endswith("src") else "./config/config.yaml"
                logger.debug(f"Using config file path: {config_file}")

                with open(config_file, encoding="utf-8") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                    self.__config = config
                    f.close()

                logger.debug(f"Config loaded: {self.__config}")
                Config._initialized = True
            except Exception as e:
                logger.error(f"Error loading config file '{config_file}': {e}", exc_info=True)
                raise

    def get_config(self):
        return self.__config
