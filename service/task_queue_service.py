import asyncio
import uuid
import logging
from collections import deque
from typing import Dict, Optional, Callable, Awaitable, Any
import time

logger = logging.getLogger(__name__)

class TaskQueueService:
    def __init__(self, max_concurrent_tasks: int = 2):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.pending_queue = deque()
        self.task_status: Dict[str, dict] = {}
        self.lock = asyncio.Lock()
        self.task_callbacks: Dict[str, Callable] = {}
        
    async def submit_task(
        self,
        task_func: Callable[..., Awaitable],
        *args,
        **kwargs
    ) -> str:
        """提交新任务到队列"""
        task_id = str(uuid.uuid4())
        
        async with self.lock:
            self.task_status[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0,
                "message": "Waiting in queue",
                "position": len(self.pending_queue) + 1,
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "user_id": kwargs.get("user_id", "unknown")
            }
            
            # 存储回调函数以便任务完成时调用
            self.task_callbacks[task_id] = kwargs.pop("callback", None)
            
            if len(self.active_tasks) < self.max_concurrent_tasks:
                # 有空闲资源，立即执行
                task = asyncio.create_task(self._execute_task(task_id, task_func, *args, **kwargs))
                self.active_tasks[task_id] = task
                self.task_status[task_id]["status"] = "running"
                self.task_status[task_id]["message"] = "Starting generation"
                self.task_status[task_id]["started_at"] = time.time()
                self.task_status[task_id]["position"] = 0  # 正在运行的任务位置为0
            else:
                # 加入等待队列
                self.pending_queue.append((task_id, task_func, args, kwargs))
                self.task_status[task_id]["message"] = f"Queued at position {len(self.pending_queue)}"
                
        return task_id
    
    async def _execute_task(
        self, 
        task_id: str,
        task_func: Callable[..., Awaitable],
        *args,
        **kwargs
    ):
        """执行任务"""
        try:
            # 执行实际任务
            result = await task_func(*args, **kwargs)
            
            async with self.lock:
                self.task_status[task_id]["status"] = "completed"
                self.task_status[task_id]["result"] = result
                self.task_status[task_id]["completed_at"] = time.time()
                
                # 如果有回调函数，调用它
                if self.task_callbacks.get(task_id):
                    await self.task_callbacks[task_id](task_id, result)
                    
                del self.task_callbacks[task_id]
                
        except Exception as e:
            async with self.lock:
                self.task_status[task_id]["status"] = "failed"
                self.task_status[task_id]["message"] = f"Error: {str(e)}"
                logger.error(f"Task {task_id} failed: {str(e)}")
                
        finally:
            async with self.lock:
                # 移除完成的任务
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                
                # 从队列中取出新任务
                if self.pending_queue:
                    next_task_id, next_func, next_args, next_kwargs = self.pending_queue.popleft()
                    task = asyncio.create_task(
                        self._execute_task(next_task_id, next_func, *next_args, **next_kwargs)
                    )
                    self.active_tasks[next_task_id] = task
                    self.task_status[next_task_id]["status"] = "running"
                    self.task_status[next_task_id]["message"] = "Starting generation"
                    self.task_status[next_task_id]["started_at"] = time.time()
                    self.task_status[next_task_id]["position"] = 0  # 正在运行的任务位置为0
                    
                    # 更新队列位置
                    for i, (queued_id, _, _, _) in enumerate(self.pending_queue):
                        if queued_id in self.task_status:
                            self.task_status[queued_id]["position"] = i + 1
    
    def update_progress(self, task_id: str, progress: int, message: str):
        """更新任务进度"""
        if task_id in self.task_status:
            self.task_status[task_id]["progress"] = progress
            self.task_status[task_id]["message"] = message
    
    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        async with self.lock:
            return self.task_status.get(task_id)
    
    async def get_user_tasks(self, user_id: str) -> list:
        """获取用户的所有任务"""
        async with self.lock:
            return [task for task in self.task_status.values() if task["user_id"] == user_id]
    
    async def cancel_task(self, task_id: str):
        """取消任务"""
        async with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
                self.task_status[task_id]["status"] = "cancelled"
                self.task_status[task_id]["message"] = "Task cancelled by user"
                
            # 如果任务在队列中，从队列中移除
            for i, (queued_id, _, _, _) in enumerate(self.pending_queue):
                if queued_id == task_id:
                    self.pending_queue.remove((queued_id, _, _, _))
                    self.task_status[task_id]["status"] = "cancelled"
                    self.task_status[task_id]["message"] = "Task cancelled by user"
                    
                    # 更新队列位置
                    for j, (q_id, _, _, _) in enumerate(self.pending_queue):
                        if q_id in self.task_status:
                            self.task_status[q_id]["position"] = j + 1
                    break

# 全局任务队列实例
task_queue = TaskQueueService(max_concurrent_tasks=2)  # 根据GPU资源调整