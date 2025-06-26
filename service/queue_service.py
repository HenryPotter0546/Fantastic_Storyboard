import os
import uuid
import psutil
import time
import logging
from typing import Optional, Dict
from redis import Redis
from rq import Queue, Worker
from rq.job import Job
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class TaskQueueService:
    def __init__(self):
        self.redis_conn = Redis(host='localhost', port=6379, db=0)
        self.task_queue = Queue(connection=self.redis_conn)
        self.resource_check_interval = 5  # 资源检查间隔(秒)
        self.max_concurrent_tasks = 1    # 最大并发任务数
        self.min_gpu_memory = 2 * 1024  # 2GB显存需求(MB)
        self.min_free_memory = 500       # 500MB内存需求(MB)
        self.max_cpu_usage = 80          # CPU最大使用率(%)
        
    def check_system_resources(self) -> bool:
        """检查系统资源是否满足新任务需求"""
        try:
            # 检查CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > self.max_cpu_usage:
                return False
                
            # 检查内存
            mem = psutil.virtual_memory()
            if mem.available / (1024 * 1024) < self.min_free_memory:
                return False
                
            # 如果有GPU则检查显存
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_gpu_mem = mem_info.free / (1024 * 1024)  # MB
                if free_gpu_mem < self.min_gpu_memory:
                    return False
            except ImportError:
                pass
                
            return True
        except Exception as e:
            logger.error(f"资源检查失败: {str(e)}")
            return False
            
    def submit_task(self, task_data: Dict) -> Dict:
        """提交新任务到队列"""
        if len(self.task_queue) + len(Worker.all(connection=self.redis_conn)) > 20:
            raise HTTPException(status_code=429, detail="队列已满，请稍后再试")
            
        job_id = str(uuid.uuid4())
        job = self.task_queue.enqueue(
            'main.run_comic_task',
            task_data,
            job_id=job_id,
            result_ttl=86400  # 结果保留24小时
        )
        
        # 获取队列位置
        jobs = self.task_queue.get_jobs()
        position = next((i for i, j in enumerate(jobs) if j.id == job_id), 0)
        
        return {
            "job_id": job_id,
            "position": position + 1,
            "status": "queued" if position > 0 else "running"
        }
        
    def get_task_status(self, job_id: str) -> Optional[Dict]:
        """获取任务状态"""
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            
            if job.is_finished:
                return {"status": "completed", "result": job.result}
            elif job.is_failed:
                return {"status": "failed", "error": str(job.exc_info)}
            elif job.is_started:
                return {"status": "running", "progress": job.meta.get("progress", 0)}
            else:
                # 获取队列位置
                jobs = self.task_queue.get_jobs()
                position = next((i for i, j in enumerate(jobs) if j.id == job_id), 0)
                return {
                    "status": "queued",
                    "position": position + 1,
                    "total_in_queue": len(jobs)
                }
        except Exception:
            return None