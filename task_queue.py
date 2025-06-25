import threading
import time
import uuid
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self):
        self.queue = deque()
        self.lock = threading.Lock()
        self.current_task = None
        self.processing = False
        self.task_history = {}
        
    def add_task(self, task_data):
        """添加新任务到队列"""
        task_id = str(uuid.uuid4())
        with self.lock:
            task = {
                "id": task_id,
                "data": task_data,
                "status": "queued",
                "position": len(self.queue) + 1,
                "created_at": datetime.utcnow().isoformat(),
                "started_at": None,
                "completed_at": None
            }
            self.queue.append(task)
            self.task_history[task_id] = task
            return task_id, len(self.queue)
    
    def get_task(self, task_id):
        """获取任务信息"""
        with self.lock:
            return self.task_history.get(task_id)
    
    def get_queue_state(self):
        """获取当前队列状态"""
        with self.lock:
            return {
                "pending": len(self.queue),
                "running": 1 if self.current_task else 0,
                "tasks": list(self.queue),
                "current": self.current_task
            }
    
    def start_next_task(self):
        """从队列中取出下一个任务执行"""
        with self.lock:
            if self.queue and not self.processing:
                self.current_task = self.queue.popleft()
                self.current_task["status"] = "processing"
                self.current_task["started_at"] = datetime.utcnow().isoformat()
                self.processing = True
                return self.current_task
            return None
    
    def complete_task(self, task_id, result):
        """标记任务完成"""
        with self.lock:
            if task_id in self.task_history:
                self.task_history[task_id]["status"] = "completed"
                self.task_history[task_id]["result"] = result
                self.task_history[task_id]["completed_at"] = datetime.utcnow().isoformat()
            self.processing = False
            self.current_task = None
    
    def fail_task(self, task_id, error):
        """标记任务失败"""
        with self.lock:
            if task_id in self.task_history:
                self.task_history[task_id]["status"] = "failed"
                self.task_history[task_id]["error"] = str(error)
                self.task_history[task_id]["completed_at"] = datetime.utcnow().isoformat()
            self.processing = False
            self.current_task = None
    
    def estimate_wait_time(self, position):
        """估计等待时间"""
        if position <= 1:
            return 0
            
        # 简单算法：每任务平均2分钟
        avg_time_per_task = 120
        return max(10, (position - 1) * avg_time_per_task)

# 全局任务队列实例
task_queue = TaskQueue()