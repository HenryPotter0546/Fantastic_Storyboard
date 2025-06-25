import time
import psutil
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service.deepseek_service import DeepSeekService
from service.diffusion_service import LocalDiffusionService
from typing import List
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import traceback
import logging
import json
import uvicorn
import os
import uuid
import base64
import shutil
import threading
import concurrent.futures
from resource_monitor import resource_monitor
from task_queue import task_queue

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="小说转漫画API")

app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建全局的diffusion服务实例
diffusion_service = LocalDiffusionService()

class NovelRequest(BaseModel):
    text: str
    num_scenes: int = 10
    steps: int = 30
    guidance: float = 7.5
    width: int = 512
    height: int = 512

class SceneResponse(BaseModel):
    description: str
    image_url: str

# 定义异步的漫画生成任务
async def generate_comic_task(
    text: str,
    num_scenes: int,
    steps: int,
    guidance: float,
    width: int,
    height: int
) -> dict:
    """异步执行漫画生成任务"""
    try:
        logger.info(f"开始生成漫画任务：场景数={num_scenes}, 步数={steps}, 分辨率={width}x{height}")
        
        # 使用相同的逻辑进行场景分割和图像生成
        session_id = str(uuid.uuid4())
        temp_dir = os.path.join("temp", session_id)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        image_paths = []
        deepseek = DeepSeekService()
        
        # 检查模型是否已加载
        if diffusion_service.model_name is None:
            raise Exception("请先选择并加载模型")
            
        if not diffusion_service.is_model_loaded():
            logger.info("模型未加载，开始自动加载...")
            try:
                diffusion_service.preload_model()
            except Exception as e:
                raise Exception(f"模型加载失败: {str(e)}")
        
        # 分割场景
        scenes_cn = deepseek.split_into_scenes_cn(text, num_scenes)
        logger.info(f"场景分割完成，共 {len(scenes_cn)} 个场景")
        
        results = []
        # 为每个场景生成图片
        for i, scene_cn in enumerate(scenes_cn):
            logger.info(f"处理第 {i+1}/{len(scenes_cn)} 个场景")
            
            try:
                # 翻译场景描述
                english_prompt = deepseek.translate_to_english(scene_cn)
                logger.info(f"场景翻译完成: {english_prompt}")
                
                # 构建提示词
                prompt = f"comic style, {english_prompt}, detailed, high quality"
                
                # 生成图像
                image_url = await asyncio.get_event_loop().run_in_executor(
                    None,
                    diffusion_service.generate_image_with_style,
                    prompt, "comic", steps, guidance, width, height, None
                )
                
                # 保存图片
                image_data = base64.b64decode(image_url.split(',')[1])
                image_path = os.path.join(temp_dir, f"scene_{i+1}.png")
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                # 添加到结果
                results.append({
                    "index": i,
                    "description": scene_cn,
                    "image_url": image_url,
                    "image_path": image_path
                })
                
            except Exception as e:
                logger.error(f"场景 {i+1} 生成失败: {str(e)}")
                results.append({
                    "index": i,
                    "description": scene_cn,
                    "error": str(e)
                })
                
        # 返回结果
        return {
            "session_id": session_id,
            "scenes": results
        }
        
    except Exception as e:
        logger.error(f"漫画生成任务失败: {str(e)}")
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# 后台工作线程
def worker_thread():
    """后台任务处理线程"""
    while True:
        # 检查资源状态
        if resource_monitor.get_status() == "busy":
            time.sleep(5)
            continue
            
        # 获取下一个任务
        task = task_queue.start_next_task()
        if not task:
            time.sleep(2)
            continue
            
        task_id = task["id"]
        task_data = task["data"]
        
        try:
            logger.info(f"开始处理任务: {task_id}")
            
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 执行任务
            result = loop.run_until_complete(generate_comic_task(
                task_data["text"],
                task_data["num_scenes"],
                task_data["steps"],
                task_data["guidance"],
                task_data["width"],
                task_data["height"]
            ))
            
            # 标记任务完成
            task_queue.complete_task(task_id, result)
            logger.info(f"任务完成: {task_id}")
            
        except Exception as e:
            logger.error(f"任务处理失败: {task_id}, {str(e)}")
            task_queue.fail_task(task_id, e)
        
        time.sleep(1)

# 启动工作线程
threading.Thread(target=worker_thread, daemon=True).start()

# 任务队列API
@app.post("/submit-task")
async def submit_task(request: NovelRequest):
    """提交漫画生成任务"""
    try:
        # 检查模型是否已加载
        if not diffusion_service.is_model_loaded():
            return {"success": False, "message": "请先加载模型"}
        
        # 创建任务数据
        task_data = {
            "text": request.text,
            "num_scenes": request.num_scenes,
            "steps": request.steps,
            "guidance": request.guidance,
            "width": request.width,
            "height": request.height
        }
        
        # 添加到任务队列
        task_id, position = task_queue.add_task(task_data)
        
        # 获取等待时间估计
        wait_time = task_queue.estimate_wait_time(position)
        
        return {
            "success": True,
            "task_id": task_id,
            "position": position,
            "wait_time": wait_time
        }
        
    except Exception as e:
        logger.error(f"提交任务失败: {str(e)}")
        return {"success": False, "message": str(e)}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 添加资源状态信息
    resource_status = resource_monitor.get_status()
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "position": task.get("position", 0),
        "resource_status": resource_status,
        "created_at": task["created_at"],
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "result": task.get("result"),
        "error": task.get("error")
    }

@app.get("/system-status")
async def get_system_status():
    """获取系统资源状态"""
    try:
        cpu_percent = psutil.cpu_percent()
        mem_percent = psutil.virtual_memory().percent
        gpu_percent = resource_monitor._get_gpu_usage()
        
        return {
            "cpu": cpu_percent,
            "memory": mem_percent,
            "gpu": gpu_percent,
            "resource_status": resource_monitor.get_status(),
            "queue_length": len(task_queue.queue),
            "processing": task_queue.processing
        }
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        return {"error": str(e)}

# 模型管理API
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

# 流式生成API（保留原功能）
@app.get("/novel-to-comic-stream")
async def novel_to_comic_stream(
    text: str,
    num_scenes: int = 10,
    steps: int = 30,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
):
    return StreamingResponse(
        generate_scenes(text, num_scenes, steps, guidance, width, height), 
        media_type="text/event-stream"
    )

async def generate_scenes(
    text: str,
    num_scenes: int,
    steps: int,
    guidance: float,
    width: int,
    height: int,
):
    session_id = str(uuid.uuid4())
    try:
        logger.info(f"开始处理请求, 参数: num_scenes={num_scenes}, steps={steps}, guidance={guidance}")
        logger.info(f"开始处理请求，会话ID: {session_id}")
        
        # 创建一个临时目录来存放生成的图片
        temp_dir = os.path.join("temp", session_id)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        image_paths = []
        # 初始化服务
        logger.info("初始化服务...")
        deepseek = DeepSeekService()
        
        # 检查是否已设置模型
        if diffusion_service.model_name is None:
            error_msg = "请先选择并加载模型"
            logger.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            return

        # 检查模型是否已加载，如果没有则自动加载
        if not diffusion_service.is_model_loaded():
            logger.info("模型未加载，开始自动加载...")
            yield f"data: {json.dumps({'type': 'info', 'message': f'模型 {diffusion_service.model_name} 未加载，正在自动加载中...'})}\n\n"
            try:
                diffusion_service.preload_model()
                yield f"data: {json.dumps({'type': 'info', 'message': '模型加载完成，开始生成...'})}\n\n"
            except Exception as e:
                error_msg = f"模型加载失败: {str(e)}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

        # 分割场景（得到中文描述）
        logger.info("向前端发送\"开始分割场景\"...")
        yield f"data: {json.dumps({'type': 'info', 'message': '正在分割场景，这可能需要一些时间...'})}\n\n"

        logger.info("开始分割场景")
        scenes_cn = deepseek.split_into_scenes_cn(text, num_scenes)
        logger.info(f"场景分割完成，共 {len(scenes_cn)} 个场景")

        # 发送场景总数，以便前端初始化
        yield f"data: {json.dumps({'type': 'start', 'total_scenes': len(scenes_cn)})}\n\n"
        
        # 为每个场景生成图片
        logger.info("开始生成图片...")
        for i, scene_cn in enumerate(scenes_cn):
            logger.info(f"处理第 {i+1}/{len(scenes_cn)} 个场景")
            try:
                # 定义一个回调函数，用于在生成过程中发送进度
                queue = asyncio.Queue()

                def progress_callback(step, timestep, latents):
                    try:
                        queue.put_nowait({
                            "type": "progress", "index": i, "step": step + 1, "total_steps": steps
                        })
                    except asyncio.QueueFull:
                        # 如果队列满了，可以忽略这次更新，避免阻塞
                        pass

                # 将中文场景描述翻译成英文prompt
                english_prompt = deepseek.translate_to_english(scene_cn)
                logger.info(f"场景翻译完成: {english_prompt}")

                # 构建适合本地diffusion的英文提示词
                prompt = f"comic style, {english_prompt}, detailed, high quality"
                logger.info(f"生成提示词: {prompt}")

                # 启动图像生成任务
                loop = asyncio.get_event_loop()
                executor = loop.run_in_executor
                gen_task = executor(
                    None,
                    diffusion_service.generate_image_with_style,
                    prompt, "comic", steps, guidance, width, height, progress_callback
                )

                # 同时监听进度队列
                done = False
                while not done:
                    try:
                        # 等待队列消息或生成任务完成
                        progress_task = asyncio.create_task(queue.get())
                        finished, pending = await asyncio.wait(
                            [gen_task, progress_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        if progress_task in finished:
                            progress_data = progress_task.result()
                            yield f"data: {json.dumps(progress_data)}\n\n"

                        if gen_task in finished:
                            image_url = gen_task.result()
                            # 确保所有剩余的进度消息都被处理
                            while not queue.empty():
                                progress_data = queue.get_nowait()
                                yield f"data: {json.dumps(progress_data)}\n\n"
                            
                            logger.info(f"图片生成完成")
                
                            # 保存图片到临时目录
                            image_data = base64.b64decode(image_url.split(',')[1])
                            image_path = os.path.join(temp_dir, f"scene_{i+1}.png")
                            with open(image_path, "wb") as f:
                                f.write(image_data)
                            image_paths.append(image_path)

                            # 只发送中文描述
                            yield f"data: {json.dumps({'type': 'scene', 'index': i, 'description': scene_cn, 'image_url': image_url})}\n\n"
                            done = True
                            # 取消可能仍在等待的progress_task
                            if not progress_task.done():
                                progress_task.cancel()
                        
                    except Exception as task_e:
                        logger.error(f"任务执行中发生错误: {task_e}")
                        done = True
                        if 'progress_task' in locals() and not progress_task.done():
                            progress_task.cancel()
                        raise task_e

            except Exception as e:
                error_msg = f"场景 {i+1} 生成失败: {str(e)}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'scene_error', 'index': i, 'message': error_msg})}\n\n"
        
        # 发送图片生成完成的消息，并包含会话ID
        yield f"data: {json.dumps({'type': 'all_images_generated', 'session_id': session_id})}\n\n"
        
        # 发送完成消息，以便前端正常关闭连接
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
    except Exception as e:
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

@app.post("/regenerate-image")
async def regenerate_image(request: dict):
    try:
        scene_index = request.get("sceneIndex")

        logger.info(f"收到重新生成图片请求: scene_index={scene_index}")

        if scene_index is None:
            raise HTTPException(status_code=400, detail="Missing scene_index parameter")

        # 这里可以实现重新生成逻辑
        # 暂时返回错误信息
        return {
            "success": False,
            "message": "重新生成功能暂未实现"
        }

    except Exception as e:
        logger.error(f"重新生成图片失败: {str(e)}")
        return {
            "success": False,
            "message": str(e)
        }

@app.get("/video", response_class=HTMLResponse)
async def get_video_page():
    """获取视频播放页面"""
    return FileResponse("templates/video.html")

@app.post("/create-video")
async def create_video(request: dict):
    """根据会话ID创建视频"""
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing session_id parameter")
        
        logger.info(f"开始创建视频，会话ID: {session_id}")
        
        temp_dir = os.path.join("temp", session_id)
        if not os.path.exists(temp_dir):
            raise HTTPException(status_code=404, detail="Session not found")
        
        image_paths = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".png")])
        
        if not image_paths:
            raise HTTPException(status_code=400, detail="No images found for this session")
        
        # 生成视频
        video_output_dir = os.path.join("static", "videos")
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        
        video_path = os.path.join(video_output_dir, f"{session_id}.mp4")
        diffusion_service.create_video_from_images(image_paths, video_path)
        logger.info(f"视频生成成功: {video_path}")
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        logger.info(f"已清理临时目录: {temp_dir}")
        
        return {"success": True, "video_url": f"/static/videos/{session_id}.mp4"}
        
    except Exception as e:
        logger.error(f"创建视频失败: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    logger.info("启动服务器...")
    uvicorn.run(app, host="localhost", port=8000)