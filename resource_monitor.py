import psutil
import time
import threading
import logging
import os

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self):
        self.cpu_threshold = 85  # CPU使用率阈值(%)
        self.mem_threshold = 90  # 内存使用率阈值(%)
        self.gpu_threshold = 90  # GPU显存使用率阈值(%)
        self.update_interval = 5  # 监控更新间隔(秒)
        self.lock = threading.Lock()
        self.status = "idle"
        self._start_monitor()
        
    def _get_gpu_usage(self):
        """获取GPU使用率"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return 0
                
            total_used = 0
            total_total = 0
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_used += mem_info.used
                total_total += mem_info.total
            return (total_used / total_total) * 100 if total_total > 0 else 0
        except ImportError:
            logger.warning("pynvml not installed, GPU monitoring disabled")
            return 0
        except Exception as e:
            logger.error(f"获取GPU使用率失败: {str(e)}")
            return 0
    
    def _monitor_loop(self):
        """监控循环"""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                mem_percent = psutil.virtual_memory().percent
                gpu_percent = self._get_gpu_usage()
                
                with self.lock:
                    if (cpu_percent > self.cpu_threshold or 
                        mem_percent > self.mem_threshold or 
                        gpu_percent > self.gpu_threshold):
                        self.status = "busy"
                    else:
                        self.status = "idle"
            except Exception as e:
                logger.error(f"资源监控错误: {str(e)}")
            
            time.sleep(self.update_interval)
    
    def _start_monitor(self):
        """启动监控线程"""
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
    
    def get_status(self):
        """获取当前资源状态"""
        with self.lock:
            return self.status

# 全局资源监控实例
resource_monitor = ResourceMonitor()