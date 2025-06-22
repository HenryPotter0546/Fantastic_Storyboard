from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service.deepseek_service import DeepSeekService
from service.diffusion_service import LocalDiffusionService
import asyncio
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


@app.get("/novel-to-comic-stream")
async def novel_to_comic_stream(text: str, num_scenes: int = 10):
    return StreamingResponse(
        generate_scenes(text, num_scenes),
        media_type="text/event-stream"
    )


async def generate_scenes(text: str, num_scenes: int):
    try:
        logger.info("开始处理请求")
        
        # 创建一个临时目录来存放生成的图片
        session_id = str(uuid.uuid4())
        temp_dir = os.path.join("temp", session_id)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        image_paths = []
        
        # 初始化服务
        logger.info("初始化服务...")
        deepseek = DeepSeekService()
        
        # 检查是否已设置模型
        if diffusion_service.model_name is None:
            error_msg = "请先选择模型"
            logger.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            return
        
        # 分割场景（得到中文描述）
        logger.info("开始分割场景...")
        scenes_cn = deepseek.split_into_scenes_cn(text, num_scenes)
        logger.info(f"场景分割完成，共 {len(scenes_cn)} 个场景")
        
        # 为每个场景生成图片
        logger.info("开始生成图片...")
        for i, scene_cn in enumerate(scenes_cn):
            logger.info(f"处理第 {i+1}/{len(scenes_cn)} 个场景")
            try:
                # 将中文场景描述翻译成英文prompt
                english_prompt = deepseek.translate_to_english(scene_cn)
                logger.info(f"场景翻译完成: {english_prompt}")
                # 构建适合本地diffusion的英文提示词
                prompt = f"comic style, {english_prompt}, detailed, high quality"
                logger.info(f"生成提示词: {prompt}")
                # 生成图片
                image_url = diffusion_service.generate_image_with_style(prompt, "comic")
                logger.info(f"图片生成完成")
                
                # 保存图片到临时目录
                image_data = base64.b64decode(image_url.split(',')[1])
                image_path = os.path.join(temp_dir, f"scene_{i+1}.png")
                with open(image_path, "wb") as f:
                    f.write(image_data)
                image_paths.append(image_path)
                
                # 只发送中文描述
                yield f"data: {json.dumps({'type': 'scene', 'index': i, 'description': scene_cn, 'image_url': image_url})}\n\n"
            except Exception as e:
                error_msg = f"场景 {i+1} 生成失败: {str(e)}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'scene_error', 'index': i, 'message': error_msg})}\n\n"
        
        # 生成视频
        video_path = None
        if image_paths:
            try:
                logger.info("开始生成视频...")
                video_output_dir = os.path.join("static", "videos")
                if not os.path.exists(video_output_dir):
                    os.makedirs(video_output_dir)
                
                video_path = os.path.join(video_output_dir, f"{session_id}.mp4")
                diffusion_service.create_video_from_images(image_paths, video_path)
                logger.info(f"视频生成成功: {video_path}")
                
                yield f"data: {json.dumps({'type': 'video', 'video_url': f'/static/videos/{session_id}.mp4'})}\n\n"
                
            except Exception as e:
                error_msg = f"视频生成失败: {str(e)}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时目录: {temp_dir}")
            
        # 发送完成消息
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


if __name__ == "__main__":
    logger.info("启动服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8080)