from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.deepseek import DeepSeekService
from services.diffusion_service import LocalDiffusionService
import asyncio
from typing import List
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import traceback
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="小说转漫画API")

app.mount("/static", StaticFiles(directory="static"), name="static")

class NovelRequest(BaseModel):
    text: str
    num_scenes: int = 10

class SceneResponse(BaseModel):
    description: str
    image_url: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("templates/index.html")

async def generate_scenes(text: str, num_scenes: int):
    try:
        logger.info("开始处理请求")
        
        # 初始化服务
        logger.info("初始化服务...")
        deepseek = DeepSeekService()
        diffusion = LocalDiffusionService()
        
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
                image_url = diffusion.generate_image_with_style(prompt, "comic")
                logger.info(f"图片生成完成")
                # 只发送中文描述
                yield f"data: {json.dumps({'type': 'scene', 'index': i, 'description': scene_cn, 'image_url': image_url})}\n\n"
            except Exception as e:
                error_msg = f"场景 {i+1} 生成失败: {str(e)}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'scene_error', 'index': i, 'message': error_msg})}\n\n"
            
        # 发送完成消息
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
    except Exception as e:
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

@app.get("/novel-to-comic-stream")
async def novel_to_comic_stream(text: str, num_scenes: int = 10):
    return StreamingResponse(
        generate_scenes(text, num_scenes),
        media_type="text/event-stream"
    )

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

if __name__ == "__main__":
    import uvicorn
    logger.info("启动服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 