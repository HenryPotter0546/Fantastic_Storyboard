from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.deepseek import DeepSeekService
from services.diffusion_service import LocalDiffusionService
import asyncio
from typing import List
from fastapi.responses import HTMLResponse, StreamingResponse
import traceback
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="小说转漫画API")

class NovelRequest(BaseModel):
    text: str
    num_scenes: int = 10

class SceneResponse(BaseModel):
    description: str
    image_url: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>小说转漫画服务</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                textarea { width: 100%; height: 200px; margin: 10px 0; }
                button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                #result { margin-top: 20px; }
                .scene { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9; }
                .scene h3 { color: #333; margin-bottom: 15px; }
                .scene p { color: #666; line-height: 1.6; margin-bottom: 15px; font-size: 16px; }
                .scene img { max-width: 100%; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .error { color: red; padding: 10px; border: 1px solid red; margin: 10px 0; border-radius: 4px; }
                .loading { color: #666; font-style: italic; }
                .image-actions { margin-top: 10px; text-align: center; }
                .image-actions button { 
                    margin: 0 5px;
                    padding: 5px 15px;
                    font-size: 14px;
                }
                .regenerate-btn { background-color: #4CAF50; }
                .image-container { position: relative; }
                .image-status { 
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    padding: 5px 10px;
                    border-radius: 4px;
                    background-color: rgba(0,0,0,0.7);
                    color: white;
                    display: none;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>小说转漫画服务 (本地Diffusion)</h1>
                <div>
                    <textarea id="novelText" placeholder="请输入小说文本..."></textarea>
                    <div>
                        <label for="numScenes">场景数量：</label>
                        <input type="number" id="numScenes" value="10" min="1" max="20">
                    </div>
                    <button onclick="generateComic()">生成漫画</button>
                </div>
                <div id="result"></div>
            </div>

            <script>
                async function regenerateImage(sceneIndex) {
                    const sceneDiv = document.querySelector(`#scene-${sceneIndex}`);
                    const statusDiv = sceneDiv.querySelector('.image-status');
                    statusDiv.style.display = 'block';
                    statusDiv.textContent = '重新生成中...';
                    
                    try {
                        const response = await fetch('/regenerate-image', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                sceneIndex: sceneIndex
                            })
                        });
                        
                        const result = await response.json();
                        if (result.success) {
                            statusDiv.textContent = '生成成功！';
                            // 更新图片URL
                            const img = sceneDiv.querySelector('img');
                            img.src = result.imageUrl;
                            setTimeout(() => {
                                statusDiv.style.display = 'none';
                            }, 2000);
                        } else {
                            statusDiv.textContent = '生成失败：' + result.message;
                        }
                    } catch (error) {
                        statusDiv.textContent = '请求失败：' + error.message;
                    }
                }

                async function generateComic() {
                    const text = document.getElementById('novelText').value;
                    const numScenes = document.getElementById('numScenes').value;
                    const resultDiv = document.getElementById('result');
                    
                    resultDiv.innerHTML = '<div class="loading">正在生成中...</div>';
                    
                    try {
                        const eventSource = new EventSource(`/novel-to-comic-stream?text=${encodeURIComponent(text)}&num_scenes=${numScenes}`);
                        
                        eventSource.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            
                            if (data.type === 'error') {
                                resultDiv.innerHTML = `<div class="error">生成失败：${data.message}</div>`;
                                eventSource.close();
                                return;
                            }
                            
                            if (data.type === 'scene') {
                                const sceneDiv = document.createElement('div');
                                sceneDiv.className = 'scene';
                                sceneDiv.id = `scene-${data.index}`;
                                sceneDiv.innerHTML = `
                                    <h3>场景 ${data.index + 1}</h3>
                                    <p>${data.description}</p>
                                    <div class="image-container">
                                        <img src="${data.image_url}" alt="场景 ${data.index + 1}">
                                        <div class="image-status"></div>
                                    </div>
                                    <div class="image-actions">
                                        <button class="regenerate-btn" onclick="regenerateImage(${data.index})">重新生成</button>
                                    </div>
                                `;
                                
                                if (data.index === 0) {
                                    resultDiv.innerHTML = '';
                                }
                                
                                resultDiv.appendChild(sceneDiv);
                            }
                            
                            if (data.type === 'scene_error') {
                                const errorDiv = document.createElement('div');
                                errorDiv.className = 'scene error';
                                errorDiv.innerHTML = `<h3>场景 ${data.index + 1}</h3><div>${data.message}</div>`;
                                if (data.index === 0) {
                                    resultDiv.innerHTML = '';
                                }
                                resultDiv.appendChild(errorDiv);
                            }
                            
                            if (data.type === 'complete') {
                                eventSource.close();
                            }
                        };
                        
                        eventSource.onerror = function(error) {
                            resultDiv.innerHTML = '<div class="error">连接错误，请重试</div>';
                            eventSource.close();
                        };
                        
                    } catch (error) {
                        resultDiv.innerHTML = `<div class="error">生成失败：${error.message}</div>`;
                    }
                }
            </script>
        </body>
    </html>
    """

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