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
from service import datamodels, auth, schemas
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, status  
from service.database import get_db

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

# --- 登录端点 ---
@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(db: AsyncSession = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = await auth.get_user(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- 注册端点 ---
@app.post("/users/register", response_model=schemas.User)
async def register_user(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    """
    处理用户注册请求。
    FastAPI 会自动处理 Depends(get_db)，将一个可用的数据库会话 (AsyncSession) 赋值给 db 参数。
    """
    
    # 1. 检查用户名是否已存在。将【正确的 db 对象】传递给 auth.get_user
    db_user = await auth.get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # 2. 创建新用户
    hashed_password = auth.get_password_hash(user.password)
    # 注意：这里应该是 datamodels.User，因为你的模型文件是 datamodels.py
    new_user = datamodels.User(username=user.username, hashed_password=hashed_password)
    
    # 3. 使用【正确的 db 对象】进行数据库操作
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    return new_user

# --- 获取当前用户信息端点 (受保护) ---
@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: datamodels.User = Depends(auth.get_current_user)):
    return current_user

# 强制登录页面
@app.get("/", response_class=HTMLResponse)
async def read_login_page():
    return FileResponse("templates/login.html")

# 登录后的主应用页面
@app.get("/index", response_class=HTMLResponse)
async def read_main_app():
    return FileResponse("templates/index.html")

# 为了登录页面的 js 也能访问，修改 login 路由
@app.get("/login", response_class=HTMLResponse)
async def read_login_page_again():
    return FileResponse("templates/login.html")


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
async def novel_to_comic_stream(
    text: str,
    num_scenes: int = 10,
    steps: int = 30,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
    db: AsyncSession = Depends(get_db),
    current_user: datamodels.User = Depends(auth.get_current_user) # 保护该端点
):
    """
    处理小说到漫画的流式生成请求。
    1. 验证用户身份。
    2. 检查并扣除积分。
    3. 调用核心生成器函数。
    """
    # 计算所需积分
    required_credits = num_scenes * 100
    if current_user.credits < required_credits:
        # 这里不能直接 raise HTTPException，因为前端期望的是流式响应。
        # 我们需要通过流发送一个错误消息。
        async def insufficient_credits_stream():
            error_data = {
                "type": "error",
                "message": f"积分不足。需要 {required_credits} 积分，但您只有 {current_user.credits}。"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(insufficient_credits_stream(), media_type="text/event-stream")

    # 在生成开始前扣除积分
    current_user.credits -= required_credits
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user) # 刷新以获取数据库中的最新状态

    # 调用独立的生成器函数，并将所有必要的参数传递给它
    return StreamingResponse(
        generate_scenes(
            text=text,
            num_scenes=num_scenes,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            db=db, # 传递 db 会话
            current_user=current_user # 传递当前用户对象
        ),
        media_type="text/event-stream"
    )


async def generate_scenes(
    text: str,
    num_scenes: int,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    db: AsyncSession,          
    current_user: datamodels.User,
):
    logger.info(f"User {current_user.username} starting generation. Credits left: {current_user.credits}")

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
        logger.info("向前端发送“开始分割场景”...")
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

        # 最新积分信息
        yield f"data: {json.dumps({'type': 'complete', 'new_credit_balance': current_user.credits})}\n\n"
            
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
