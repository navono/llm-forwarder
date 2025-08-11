import asyncio
import signal
import sys

from .http import HTTPServer
from .utils import Config, CustomizeLogger, get_dask_client, init_dask

CONTEXT_SETTINGS = {"auto_envvar_prefix": "ASSISTANTS"}

gen_config = Config().get_config()
logger = CustomizeLogger.make_logger(gen_config["log"])

_SHUTDOWN_CALLED = False


class ApplicationManager:
    """Class to manage application components and state"""

    def __init__(self):
        self.shutdown_called = False
        # self.asr_client = None
        self.dask_client = None


# Create a single instance of the application manager
app_manager = ApplicationManager()


async def start():
    # app_manager.asr_client = VtuberClient(gen_config)
    # asr_task = asyncio.create_task(app_manager.asr_client.start())

    # Initialize Dask and register plugins
    init_dask()
    app_manager.dask_client = get_dask_client()

    # Start HTTP server
    http_server = HTTPServer(gen_config["http"])  # http_server instance
    http_task = asyncio.create_task(http_server.start(), name="HTTPServerTask")

    # Setup signal handlers for graceful shutdown
    current_loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Pass http_server and http_task to the shutdown coroutine
            current_loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(shutdown(http_server=http_server, http_task=http_task, loop=current_loop, signal_received=s), name=f"ShutdownTaskSignal-{s.name if hasattr(s, 'name') else str(s)}")
            )
    else:
        # On Windows, signal.signal runs in a separate thread, so we schedule shutdown in the loop
        def handle_signal_win(signum, frame):
            if not current_loop.is_closed():
                logger.info(f"Windows signal {signum} received, scheduling shutdown.")
                asyncio.run_coroutine_threadsafe(shutdown(http_server=http_server, http_task=http_task, loop=current_loop, signal_received=signum), current_loop)

        signal.signal(signal.SIGINT, handle_signal_win)
        signal.signal(signal.SIGTERM, handle_signal_win)

    tasks = [http_task]

    try:
        # Wait for all primary tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            task_name = tasks[i].get_name() if hasattr(tasks[i], "get_name") else f"Task-{i}"
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.error(f"{task_name} failed with exception: {result}", exc_info=result)
            elif isinstance(result, asyncio.CancelledError):
                logger.info(f"{task_name} was cancelled.")

    except asyncio.CancelledError:
        logger.info("Main application gather was cancelled. Initiating shutdown if not already started.")
    finally:
        await shutdown(http_server=http_server, http_task=http_task, loop=current_loop)


async def shutdown(http_server=None, http_task=None, loop=None, signal_received=None):
    """Gracefully shutdown the application"""
    global _SHUTDOWN_CALLED
    if _SHUTDOWN_CALLED:
        # If shutdown is called again while already in progress, log and return.
        # This can happen if multiple signals are received or if the finally block in start() also calls it.
        logger.info("Shutdown already in progress, ignoring duplicate call or ensuring http_task is awaited.")
        if http_task and not http_task.done():
            try:
                # Ensure we wait for the task if it's still running from a prior shutdown call
                await asyncio.wait_for(http_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("Duplicate shutdown call: http_task did not complete quickly or was cancelled.")
            except Exception as e:
                logger.error(f"Duplicate shutdown call: Error awaiting http_task: {e}")
        return

    _SHUTDOWN_CALLED = True

    if signal_received:
        signal_name = signal.strsignal(signal_received) if isinstance(signal_received, signal.Signals) else str(signal_received)
        logger.info(f"Starting application shutdown sequence due to signal: {signal_name}...")
    else:
        logger.info("Starting application shutdown sequence...")

    # Request HTTP server to stop
    if http_server is not None and http_task is not None:
        if not http_task.done():
            logger.info(f"Requesting HTTP server to stop (task state: {http_task.done()}, cancelled: {http_task.cancelled()})...")
            if hasattr(http_server, "stop") and asyncio.iscoroutinefunction(http_server.stop):
                await http_server.stop()  # Signal the server to stop
                logger.info("HTTP server stop() signalled.")
            else:
                logger.error("HTTPServer instance does not have a valid stop() coroutine method. Cannot signal graceful shutdown.")
                # Fallback or error handling if stop() is not available
                http_task.cancel()  # Fallback to direct cancellation
                logger.warning("Falling back to http_task.cancel()")

            try:
                logger.info("Awaiting HTTP server task to complete (timeout: 10s)...")
                await asyncio.wait_for(http_task, timeout=10.0)
                logger.info(f"HTTP server task completed (state after await: {http_task.done()}, cancelled: {http_task.cancelled()}).")
            except asyncio.CancelledError:
                logger.warning("HTTP server task was cancelled while awaiting its completion during shutdown.")
            except TimeoutError:
                logger.warning("HTTP server task did not complete within 10s timeout. Forcing cancellation.")
                http_task.cancel()
                try:
                    await http_task  # Await the cancellation
                except asyncio.CancelledError:
                    logger.info("HTTP server task successfully force-cancelled.")
                except Exception as e_cancel:
                    logger.error(f"Error during force-cancellation of HTTP server task: {e_cancel}", exc_info=e_cancel)
            except Exception as e:
                logger.error(f"Exception while awaiting HTTP server task during shutdown: {e}", exc_info=e)

            if http_task.done() and http_task.cancelled():
                logger.info("HTTP server task processing complete (was cancelled).")
            elif http_task.done():
                logger.info("HTTP server task processing complete (exited normally).")
            else:
                logger.warning("HTTP server task processing incomplete after shutdown attempt.")
        else:
            logger.info(f"HTTP server task already done before shutdown logic (state: {http_task.done()}, cancelled: {http_task.cancelled()}).")
    elif http_server is None:
        logger.warning("HTTPServer instance not provided for shutdown.")
    elif http_task is None:
        logger.warning("HTTP task not provided for shutdown.")

    if loop:
        tasks_to_cancel = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task(loop=loop) and t is not http_task]
        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} other tasks...")
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()

            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            for i, res in enumerate(results):
                task_name_str = tasks_to_cancel[i].get_name() if hasattr(tasks_to_cancel[i], "get_name") else f"OtherTask-{i}"
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    logger.error(f"Error cancelling task {task_name_str}: {res}", exc_info=res)
                elif isinstance(res, asyncio.CancelledError):
                    logger.info(f"Task {task_name_str} successfully cancelled.")
            logger.info("Other tasks cancellation processed.")

    logger.info("Application shutdown sequence complete.")
