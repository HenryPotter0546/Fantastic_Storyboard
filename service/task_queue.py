import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self, max_concurrent_tasks=1):
        self.queue = asyncio.Queue()
        self.running_tasks = 0
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_status = {}  # task_id -> status: queued|running|completed|failed
        self.task_futures = {} # task_id -> asyncio.Future

    async def worker(self):
        while True:
            if self.running_tasks >= self.max_concurrent_tasks:
                # 限制最大并发，等待释放资源
                await asyncio.sleep(0.1)
                continue

            try:
                task_id, coro_func, future = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            self.running_tasks += 1
            self.task_status[task_id] = "running"
            logger.info(f"Start executing task {task_id}")

            try:
                result = await coro_func()
                future.set_result(result)
                self.task_status[task_id] = "completed"
                logger.info(f"Task {task_id} completed")
            except Exception as e:
                future.set_exception(e)
                self.task_status[task_id] = f"failed: {e}"
                logger.error(f"Task {task_id} failed: {e}")
            finally:
                self.running_tasks -= 1
                self.queue.task_done()

    def start_worker(self):
        # 启动后台worker任务
        asyncio.create_task(self.worker())

    def submit_task(self, coro_func):
        """提交异步生成任务，返回task_id和future"""
        task_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self.task_status[task_id] = "queued"
        self.task_futures[task_id] = future
        self.queue.put_nowait((task_id, coro_func, future))
        logger.info(f"Task {task_id} queued")
        return task_id, future

    def get_status(self, task_id):
        """获取任务状态"""
        return self.task_status.get(task_id, "unknown")

    def get_queue_position(self, task_id):
        """获取任务在队列中的位置（0-based），如果已开始返回-1"""
        if self.get_status(task_id) != "queued":
            return -1
        try:
            items = list(self.queue._queue)  # 访问内部队列，非公开API，风险可控
            for idx, (tid, _, _) in enumerate(items):
                if tid == task_id:
                    return idx
            return -1
        except Exception:
            return -1