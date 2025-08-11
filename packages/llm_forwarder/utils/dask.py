import asyncio
import atexit
import signal

from dask.distributed import LocalCluster, WorkerPlugin
from loguru import logger

from .config import Config


class DaskClientSingleton:
    _instance = None
    _client = None
    _cluster = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # 先创建实例，再初始化
        return cls._instance

    def __init__(self):
        # 只初始化一次
        if not DaskClientSingleton._initialized:
            try:
                logger.trace("Creating Dask client and local cluster")
                # 在实例创建时获取配置，避免模块级别的配置初始化
                config_data = Config().get_config()
                # Store the cluster in a class variable to be able to close it later
                self.__class__._cluster = LocalCluster(
                    scheduler_port=config_data["cluster"]["scheduler_port"],
                    dashboard_address=config_data["cluster"]["dashboard_address"],
                    n_workers=config_data["cluster"].get("n_workers", 4),  # 进程数，默认为4
                    threads_per_worker=config_data["cluster"].get("threads_per_worker", 2),  # 每个进程的线程数，默认为2
                    silence_logs=True,  # 减少日志输出
                )
                self.__class__._client = self.__class__._cluster.get_client()
                self.__class__._initialized = True

                # 注册退出处理函数，确保程序退出时关闭资源
                atexit.register(self._cleanup)

                # 注册信号处理函数，处理 Ctrl+C 中断
                self._setup_signal_handlers()

                dashboard_url = f"http://localhost{config_data['cluster']['dashboard_address']}"
                logger.trace(f"Dask client and cluster initialized successfully, open {dashboard_url} to view dashboard")
            except Exception as e:
                logger.error(f"Error initializing Dask client: {e}", exc_info=True)
                self._cleanup()
                raise

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        # 在Windows上只支持SIGINT和SIGTERM
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                signal.signal(sig, self._signal_handler)
                logger.trace(f"Registered signal handler for {sig}")
            except (ValueError, OSError) as e:
                # 可能在非主线程中调用，忽略错误
                logger.debug(f"Could not register signal handler for {sig}: {e}")

    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down Dask resources...")
        self._cleanup()
        # 不立即退出，允许正常的程序终止流程继续

    def _cleanup(self):
        """Clean up resources"""
        logger.trace("Cleaning up Dask resources")
        try:
            if self.__class__._client is not None:
                logger.trace("Closing Dask client...")
                try:
                    self.__class__._client.close(timeout=2)
                except Exception as e:
                    logger.debug(f"Error closing Dask client: {e}")
                finally:
                    self.__class__._client = None

            if self.__class__._cluster is not None:
                logger.trace("Closing Dask cluster...")
                try:
                    self.__class__._cluster.close(timeout=2)
                except Exception as e:
                    logger.debug(f"Error closing Dask cluster: {e}")
                finally:
                    self.__class__._cluster = None

            self.__class__._initialized = False
            logger.trace("Dask resources cleanup completed")
        except Exception as e:
            logger.error(f"Error during Dask cleanup: {e}", exc_info=True)

    def get_client(self):
        """Returns the Dask client instance."""
        return self._client

    @classmethod
    async def close(cls):
        """Closes the Dask client and the local cluster asynchronously."""
        logger.trace("DaskClientSingleton.close() called.")
        try:
            if cls._client is not None:
                logger.trace("Attempting to close Dask client...")
                try:
                    # Client.close() is synchronous for a local cluster and returns None.
                    # Running in a thread to avoid blocking the event loop if it hangs.
                    await asyncio.to_thread(cls._client.close, timeout=2)
                    logger.trace("Dask client close() method called successfully.")
                except Exception as e:
                    logger.debug(f"Exception during Dask client.close(): {e}")
                finally:
                    cls._client = None
                    logger.trace("Dask client set to None.")
            else:
                logger.trace("Dask client was already None.")

            if cls._cluster is not None:
                logger.trace("Attempting to close Dask local cluster...")
                try:
                    # LocalCluster.close() is synchronous. Using a timeout.
                    # Running in a thread to avoid blocking the event loop.
                    await asyncio.to_thread(cls._cluster.close, timeout=2)  # 缩短超时时间
                    logger.trace("Dask local cluster close() method called successfully.")
                except Exception as e:
                    logger.debug(f"Exception during Dask local cluster.close(): {e}")
                finally:
                    cls._cluster = None
                    logger.trace("Dask local cluster set to None.")
            else:
                logger.trace("Dask local cluster was already None.")

            cls._instance = None  # Reset instance so it can be recreated if needed
            cls._initialized = False  # 重置初始化状态
            logger.trace("DaskClientSingleton instance reset.")
        except Exception as e:
            logger.error(f"Error during Dask close: {e}", exc_info=True)


class Counter(WorkerPlugin):
    def __init__(self, worker1_address):
        self.worker1_address = worker1_address

    def setup(self, worker):
        if worker.address == self.worker1_address:
            self.counter = 10
            worker.data["counter"] = self.counter

    def teardown(self, worker):
        if worker.address == self.worker1_address:
            self.counter = 0

    def transition(self, key, start, finish, *args, **kwargs):
        if start == "executing":
            self.counter += 1


class MyWebSocketPlugin(WorkerPlugin):
    def __init__(self, base_ws_url="ws://example.com/"):
        self.base_ws_url = base_ws_url
        self.ws_connection = None
        self.worker_name = None
        print("MyWebSocketPlugin: Plugin instance created.")

    async def setup(self, worker):
        """在 Worker 启动时调用"""
        self.worker_name = worker.name
        ws_url = f"{self.base_ws_url}worker_{self.worker_name}"
        # self.ws_connection = WebSocketConnection(ws_url)
        await self.ws_connection.connect()

        # 将插件实例存储在 Worker 的属性中，以便任务函数访问
        # 推荐将资源直接挂载到 worker.data 字典中
        worker.data["ws_connection"] = self.ws_connection
        worker.data["plugin_initialized"] = True
        print(f"MyWebSocketPlugin on {self.worker_name}: Setup complete.")

    async def teardown(self, worker):
        """在 Worker 关闭时调用"""
        if self.ws_connection:
            await self.ws_connection.close()
        print(f"MyWebSocketPlugin on {self.worker_name}: Teardown complete.")

    # 可以在这里添加其他生命周期方法，例如：
    # async def transition(self, key, start, finish, **kwargs):
    #     """在任务状态转换时调用"""
    #     if finish == 'executing':
    #         print(f"Worker {self.worker_name}: Starting task {key}")
    #     elif finish == 'finished':
    #         print(f"Worker {self.worker_name}: Finished task {key}")


def init_dask():
    singleton = DaskClientSingleton()
    client = singleton.get_client()
    worker_addresses = list(client.scheduler_info()["workers"].keys())
    worker1_address = worker_addresses[0]
    print(f"Available Worker Addresses: {worker_addresses}")

    client.register_worker_plugin(Counter(worker1_address), "counter")


def get_dask_client():
    """Gets the Dask client from the singleton."""
    try:
        singleton = DaskClientSingleton()
        return singleton.get_client()
    except Exception as e:
        logger.error(f"Error getting Dask client: {e}", exc_info=True)
        # 在出错时返回 None，调用者需要处理这种情况
        return None
