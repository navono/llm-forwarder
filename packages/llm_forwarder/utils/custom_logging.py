import logging
import os.path
import sys

from loguru import logger

loglevel_mapping = {
    50: "CRITICAL",
    40: "ERROR",
    30: "WARNING",
    20: "INFO",
    10: "DEBUG",
    0: "NOTSET",
}


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = loglevel_mapping[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log = logger.bind(request_id="app")
        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class MyInterceptor(logging.Handler):
    def handle(self, record):
        # 修改日志级别
        record.level = logging.WARNING

        # 添加自定义元数据
        # record.extra['my_custom_metadata'] = 'hello, world!'

        # 继续记录日志
        super().handle(record)

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = loglevel_mapping[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log = logger.bind(request_id="app")
        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class ModuleFilter:
    """Filter to exclude logs from specific modules"""

    def __init__(self, excluded_modules: list):
        self.excluded_modules = excluded_modules

    def __call__(self, record):
        # Check if the record's module is in the excluded list
        module_path = record["name"]

        # Check if the module or any parent module should be excluded
        return all(not (module_path == excluded or module_path.startswith(excluded + ".")) for excluded in self.excluded_modules)


class MessageFilter:
    """Filter to exclude logs based on message content"""

    def __init__(self, excluded_patterns: list):
        self.excluded_patterns = excluded_patterns

    def __call__(self, record):
        # Get the message content
        message = record["message"]

        # Check if the message contains any excluded pattern
        return all(pattern not in message for pattern in self.excluded_patterns)


class LoggerNameFilter:
    """Filter to exclude logs based on logger name"""

    def __init__(self, excluded_loggers: list):
        self.excluded_loggers = excluded_loggers

    def __call__(self, record):
        # Get the logger name
        logger_name = record.get("name", "")

        # Check if the logger name is in the excluded list
        return all(excluded not in logger_name for excluded in self.excluded_loggers)


class SourceFilter:
    """直接过滤特定源的日志记录"""

    def __init__(self, excluded_sources):
        # 将排除的源分解为模块和函数对
        self.excluded_pairs = []
        for source in excluded_sources:
            if ":" in source:
                module, function = source.split(":", 1)
                self.excluded_pairs.append((module, function))

    def __call__(self, record):
        try:
            # 获取记录中的模块名和函数名
            module = record.get("name", "")
            function = record.get("function", "")

            # 检查是否匹配任何排除的模块-函数对
            for excluded_module, excluded_function in self.excluded_pairs:
                if (excluded_module == module or (excluded_module == "src.g1_task.utils.custom_logging" and "custom_logging" in module)) and excluded_function == function:
                    return False
            return True
        except Exception:
            # 如果出现异常，则不过滤
            return True


class CustomizeLogger:
    __mqtt_client = None

    # 默认要排除的日志源 - 这是过滤第三方日志的主要方式
    __excluded_sources = [
        "logging:callHandlers",  # 排除 logging:callHandlers 函数的日志
        "src.g1_task.utils.custom_logging:handle",  # 排除 custom_logging:handle 函数的日志
    ]

    # 以下过滤器作为备用，可以在需要时启用
    # 默认要排除的模块
    __excluded_modules = [
        # "websockets",  # 排除所有 websockets 日志
    ]

    # 默认要排除的消息模式
    __excluded_patterns = [
        # 如果源过滤器不足以过滤掉所有不需要的日志，可以启用这些模式
        # "ping",
        # "keepalive",
    ]

    # 默认要排除的日志记录器名称
    __excluded_loggers = [
        # "loggingCallHandlers",
    ]

    @classmethod
    def make_logger(cls, config):
        # 获取当前脚本的绝对路径
        abs_filepath = config["path"]
        # 确保日志目录存在
        os.makedirs(abs_filepath, exist_ok=True)

        # 使用固定的文件名，让 loguru 处理轮换
        fixed_filename = f"{config['filename']}.log"

        logger = cls.customize_logging(
            # config['path'),
            filepath=os.path.join(abs_filepath, fixed_filename),
            level=config["level"],
            retention=config["retention"],
            rotation=config["rotation"],
            format=config["format"],
        )
        return logger

    @classmethod
    def set_excluded_modules(cls, modules: list):
        """Set modules to exclude from logging"""
        cls.__excluded_modules = modules

    @classmethod
    def get_excluded_modules(cls) -> list:
        """Get the list of excluded modules"""
        return cls.__excluded_modules

    @classmethod
    def set_excluded_patterns(cls, patterns: list):
        """Set message patterns to exclude from logging"""
        cls.__excluded_patterns = patterns

    @classmethod
    def get_excluded_patterns(cls) -> list:
        """Get the list of excluded message patterns"""
        return cls.__excluded_patterns

    @classmethod
    def set_excluded_loggers(cls, loggers: list):
        """Set logger names to exclude from logging"""
        cls.__excluded_loggers = loggers

    @classmethod
    def get_excluded_loggers(cls) -> list:
        """Get the list of excluded logger names"""
        return cls.__excluded_loggers

    @classmethod
    def set_excluded_functions(cls, functions: list):
        """Set function names to exclude from logging"""
        cls.__excluded_functions = functions

    @classmethod
    def get_excluded_functions(cls) -> list:
        """Get the list of excluded function names"""
        return cls.__excluded_functions

    @classmethod
    def customize_logging(cls, filepath: str, level: str, rotation: str, retention: str, format: str):
        logger.remove()

        # Create filters
        filters = []

        # 优先添加源过滤器，这是过滤第三方日志的主要方式
        if cls.__excluded_sources:
            filters.append(SourceFilter(cls.__excluded_sources))

        # 如果需要其他过滤器，可以启用以下代码
        # Add module filter if excluded modules are specified
        if cls.__excluded_modules:
            filters.append(ModuleFilter(cls.__excluded_modules))

        # Add message filter if excluded patterns are specified
        if cls.__excluded_patterns:
            filters.append(MessageFilter(cls.__excluded_patterns))

        # Add logger name filter if excluded loggers are specified
        if cls.__excluded_loggers:
            filters.append(LoggerNameFilter(cls.__excluded_loggers))

        # Combine all filters
        def combined_filter(record):
            return all(f(record) for f in filters) if filters else True

        # Add handlers with the combined filter
        logger.add(sys.stdout, enqueue=True, backtrace=True, level=level.upper(), format=format, filter=combined_filter if filters else None)
        logger.add(filepath, rotation=rotation, retention=retention, enqueue=True, backtrace=True, level=level.upper(), format=format, filter=combined_filter if filters else None)
        # logger.add(
        #     cls.__mqtt_sink,
        #     enqueue=True,
        #     backtrace=True,
        #     level=level.upper(),
        #     format=format
        # )

        logging.basicConfig(handlers=[InterceptHandler(), MyInterceptor()], level=0)
        logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
        for _log in ["uvicorn", "uvicorn.error", "fastapi"]:
            _logger = logging.getLogger(_log)
            _logger.handlers = [InterceptHandler()]

        return logger.bind(request_id=None, method=None)

    # @classmethod
    # def set_mqtt_client(cls, mqtt_client):
    #     cls.__mqtt_client = mqtt_client
    #
    # @classmethod
    # def __mqtt_sink(cls, message):
    #     if cls.__mqtt_client is None:
    #         time.sleep(1)
    #         return
    #     cls.__mqtt_client.publish(Topic.FLOWCHART_STATUS.value, message)
