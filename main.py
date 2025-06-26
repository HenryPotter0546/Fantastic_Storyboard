from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from service.deepseek_service import DeepSeekService
from service.diffusion_service import LocalDiffusionService
import asyncio
from typing import List, Dict, Optional, Callable
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import traceback
import logging
import json
import uvicorn
import os
import uuid
import base64
import shutil
import time
import threading
import psutil
from redis import Redis
from rq import Queue, Worker
from rq.job import Job
import datetime
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="小说转漫画API")

app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建全局的diffusion服务实例
diffusion_service = LocalDiffusionService()

# 初始化Redis和任务队列
redis_conn = Redis(host='localhost', port=6379, db=0)
task_queue = Queue(connection=redis_conn)

class NovelRequest(BaseModel):
    text: str
    num_scenes: int = 10

class SceneResponse(BaseModel):
    description: str
    image_url: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("templates/index.html")

@app.get("/available-models")
async def get_available_models():
    """获取可用的模型列表"""
    try:
        models = diffusion_service.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"获取可用模型失败: {str(e)}")
        return {"models": [], "error": str(e)}

@app.post("/set-model")
async def set_model(request: dict):
    """设置要使用的模型（不立即加载）"""
    try:
        model_name = request.get("model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="Missing model_name parameter")

        diffusion_service.set_model(model_name)
        return {"success": True, "message": f"模型已设置为: {model_name}"}

    except Exception as e:
        logger.error(f"设置模型失败: {str(e)}")
        return {"success": False, "message": str(e)}

@app.post("/load-model")
async def load_model():
    """预加载当前设置的模型"""
    try:
        if diffusion_service.model_name is None:
            raise HTTPException(status_code=400, detail="请先选择模型")

        logger.info(f"开始预加载模型: {diffusion_service.model_name}")

        # 在后台线程中加载模型，避免阻塞
        def load_model_sync():
            try:
                diffusion_service.preload_model()
                return True
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                return False

        # 使用线程池执行模型加载
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(load_model_sync)
            success = future.result(timeout=300)  # 5分钟超时

        if success:
            return {"success": True, "message": f"模型 {diffusion_service.model_name} 加载完成"}
        else:
            return {"success": False, "message": "模型加载失败，请查看日志"}

    except Exception as e:
        logger.error(f"预加载模型失败: {str(e)}")
        return {"success": False, "message": str(e)}

@app.get("/model-status")
async def get_model_status():
    """获取模型加载状态"""
    try:
        is_loaded = diffusion_service.is_model_loaded()
        current_model = diffusion_service.model_name

        return {
            "is_loaded": is_loaded,
            "current_model": current_model,
            "message": f"模型 {current_model} 已加载" if is_loaded and current_model else "未加载模型"
        }
    except Exception as e:
        logger.error(f"获取模型状态失败: {str(e)}")
        return {"is_loaded": False, "current_model": None, "message": str(e)}

# 任务队列相关端点
@app.post("/submit-task")
async def submit_task(request: NovelRequest):
    """提交新任务到队列"""
    try:
        # 检查模型是否已加载
        if not diffusion_service.is_model_loaded():
            raise HTTPException(status_code=400, detail="请先加载模型")
            
        # 生成唯一的 job_id
        job_id = str(uuid.uuid4())
        
        task_data = {
            "text": request.text,
            "num_scenes": request.num_scenes,
            "steps": 30,
            "guidance": 7.5,
            "width": 512,
            "height": 512,
            "model_name": diffusion_service.model_name,
            "job_id": job_id  # 确保包含 job_id
        }
        
        # 检查队列长度
        if len(task_queue) + len(Worker.all(connection=redis_conn)) > 20:
            raise HTTPException(status_code=429, detail="队列已满，请稍后再试")
            
        # 提交任务到队列
        job = task_queue.enqueue(
            run_comic_task,
            task_data,
            job_id=job_id,  # 使用相同的 job_id
            result_ttl=86400  # 结果保留24小时
        )
        
        # 获取队列位置
        jobs = task_queue.get_jobs()
        position = next((i for i, j in enumerate(jobs) if j.id == job_id), 0)
        
        return {
            "job_id": job_id,
            "position": position + 1,
            "status": "queued" if position > 0 else "running"
        }
        
    except Exception as e:
        logger.error(f"提交任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/task-status/{job_id}")
async def get_task_status(job_id: str):
    """获取任务状态"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            return {"status": "completed", "result": job.result}
        elif job.is_failed:
            return {"status": "failed", "error": str(job.exc_info)}
        elif job.is_started:
            return {"status": "running", "progress": job.meta.get("progress", 0)}
        else:
            # 获取队列位置
            jobs = task_queue.get_jobs()
            position = next((i for i, j in enumerate(jobs) if j.id == job_id), 0)
            return {
                "status": "queued",
                "position": position + 1,
                "total_in_queue": len(jobs)
            }
    except Exception:
        raise HTTPException(status_code=404, detail="任务不存在")

@app.websocket("/ws/task-updates/{job_id}")
async def websocket_task_updates(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            status = await get_task_status(job_id)
            await websocket.send_json(status)
            
            if status["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
        await websocket.send_json({"error": str(e)})

def run_comic_task(task_data):
    """
    执行漫画生成任务的核心函数
    参数:
        task_data (dict): 包含任务参数的字典，必须包含以下键:
            - job_id: 任务ID
            - text: 小说文本
            - num_scenes: 场景数量
            - steps: 生成步骤数 (可选，默认30)
            - guidance: 引导系数 (可选，默认7.5)
            - width: 图片宽度 (可选，默认512)
            - height: 图片高度 (可选，默认512)
            - model_name: 模型名称
    """
    try:
        # ========== 参数验证 ==========
        required_keys = ['job_id', 'text', 'num_scenes', 'model_name']
        for key in required_keys:
            if key not in task_data:
                raise ValueError(f"任务数据缺少必要参数: {key}")

        job_id = task_data['job_id']
        job = Job.fetch(job_id, connection=redis_conn)
        
        # 设置默认参数
        steps = task_data.get('steps', 30)
        guidance = task_data.get('guidance', 7.5)
        width = task_data.get('width', 512)
        height = task_data.get('height', 512)
        text = task_data['text']
        num_scenes = task_data['num_scenes']
        model_name = task_data['model_name']

        # ========== 初始化服务 ==========
        deepseek = DeepSeekService()
        diffusion_service.set_model(model_name)
        
        # ========== 场景分割 ==========
        job.meta['progress'] = 5
        job.meta['stage'] = '正在分割场景'
        job.save_meta()
        
        scenes_cn = deepseek.split_into_scenes_cn(text, num_scenes)
        total_scenes = len(scenes_cn)
        
        # ========== 任务进度跟踪 ==========
        results = []
        for i, scene_cn in enumerate(scenes_cn):
            # 更新进度
            progress = int((i + 1) / total_scenes * 90) + 5  # 5-95%
            job.meta.update({
                'progress': progress,
                'stage': f'正在生成场景 {i+1}/{total_scenes}',
                'current_scene': scene_cn
            })
            job.save_meta()
            
            # ========== 生成英文提示词 ==========
            english_prompt = deepseek.translate_to_english(scene_cn)
            prompt = f"comic style, {english_prompt}, detailed, high quality"
            
            # ========== 图片生成 ==========
            try:
                image_url = diffusion_service.generate_image_with_style(
                    prompt=prompt,
                    style="comic",
                    steps=steps,
                    guidance=guidance,
                    width=width,
                    height=height
                )
            except Exception as e:
                logger.error(f"场景 {i} 生成失败: {str(e)}")
                image_url = None
                
            # ========== 保存结果 ==========
            results.append({
                'scene_id': i,
                'scene_text': scene_cn,
                'prompt': prompt,
                'image_url': image_url,
                'timestamp': datetime.now().isoformat()
            })
            
            job.meta['last_generated'] = image_url
            job.save_meta()

        # ========== 任务完成 ==========
        job.meta.update({
            'progress': 100,
            'stage': '已完成',
            'completed_at': datetime.now().isoformat()
        })
        job.save_meta()
        
        return {
            'status': 'success',
            'scenes': results,
            'parameters': {
                'model_used': model_name,
                'steps': steps,
                'guidance': guidance,
                'resolution': f"{width}x{height}"
            }
        }
        
    except Exception as e:
        # ========== 错误处理 ==========
        error_msg = f"任务执行失败: {str(e)}"
        logger.error(error_msg)
        
        if 'job' in locals():
            job.meta.update({
                'progress': 0,
                'stage': '失败',
                'error': error_msg,
                'traceback': traceback.format_exc()
            })
            job.save_meta()
        
        raise  # 重新抛出异常以便RQ记录失败状态
# 资源监控线程
def resource_monitor():
    """后台资源监控线程"""
    while True:
        try:
            # 检查系统资源
            cpu_usage = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            
            # 如果有GPU则检查显存
            gpu_ok = True
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_gpu_mem = mem_info.free / (1024 * 1024)  # MB
                gpu_ok = free_gpu_mem > 2000  # 2GB显存需求
            except ImportError:
                pass
                
            # 如果资源充足且队列中有任务，启动worker
            if (cpu_usage < 80 and mem.available / (1024 * 1024) > 500 and gpu_ok 
                and len(task_queue) > 0 and len(Worker.all(connection=redis_conn)) == 0):
                worker = Worker([task_queue], connection=redis_conn)
                worker.work(burst=True)  # 只处理一个任务
                
            time.sleep(5)
        except Exception as e:
            logger.error(f"资源监控错误: {str(e)}")
            time.sleep(10)

# 启动资源监控线程
monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
monitor_thread.start()

if __name__ == "__main__":
    logger.info("启动服务器...")
    uvicorn.run(app, host="localhost", port=8000)