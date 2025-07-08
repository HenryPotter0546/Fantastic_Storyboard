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

import functools
from fastapi import Query

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="小说转漫画API")

app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建全局的diffusion服务实例
diffusion_service = LocalDiffusionService()

# 全局变量用于控制重新生成
current_generation_task = None
pending_regeneration_request = None  # 新增：存储待处理的重新生成请求


class NovelRequest(BaseModel):
    text: str
    num_scenes: int = 10


class SceneResponse(BaseModel):
    description: str
    image_url: str
class LoadLoraRequest(BaseModel):
    lora_name: str

class SetLoraScaleRequest(BaseModel):
    scale: float

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


@app.get("/pic", response_class=HTMLResponse)
async def pic_page():
    return FileResponse("templates/pic.html")

from fastapi import Request

# ... 你已有代码 ...
@app.get("/available-loras")
async def available_loras():
    try:
        loras = diffusion_service.get_available_loras()
        return {"loras": loras}
    except Exception as e:
        logger.error(f"获取LoRA列表失败: {str(e)}")
        return {"loras": [], "error": str(e)}
    
class SetLoraRequest(BaseModel):
    lora_name: str
    scale: float = 1.0

@app.post("/set-lora")
async def set_lora(request: SetLoraRequest):
    try:
        diffusion_service.load_lora_weights(request.lora_name, request.scale)
        lora_info = next((l for l in diffusion_service.get_available_loras() if l['name'] == request.lora_name), None)
        prompt = lora_info.get('prompt', '') if lora_info else ''
        return {
            "success": True,
            "message": f"LoRA {request.lora_name} 加载成功，权重scale={request.scale}",
            "prompt": prompt
        }
    except Exception as e:
        logger.error(f"设置LoRA失败: {str(e)}")
        return {"success": False, "message": str(e)}
@app.post("/unload-lora")
async def unload_lora():
    try:
        diffusion_service.unload_lora_weights()
        return {"success": True, "message": "LoRA权重已卸载"}
    except Exception as e:
        logger.error(f"卸载LoRA失败: {str(e)}")
        return {"success": False, "message": str(e)}

@app.get("/lora-status")
async def lora_status():
    return {
        "loaded": diffusion_service.lora_weights_loaded,
        "name": None,
        "path": diffusion_service.lora_weights_path,
        "scale": getattr(diffusion_service, "lora_scale", 1.0)
    }

@app.post("/load-lora")
async def load_lora(request: LoadLoraRequest):
    """
    只加载LoRA权重，不调整scale，默认1.0
    """
    try:
        diffusion_service.load_lora_weights(request.lora_name)
        return {"success": True, "message": f"LoRA {request.lora_name} 权重加载成功，scale为默认1.0"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/set-lora-scale")
async def set_lora_scale(request: SetLoraScaleRequest):
    """
    仅调整已加载LoRA的权重scale，无需重新加载权重
    """
    try:
        diffusion_service.set_lora_scale(request.scale)
        return {"success": True, "message": f"LoRA权重scale调整为 {request.scale}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/unload-lora")
async def unload_lora():
    try:
        diffusion_service.unload_lora_weights()
        return {"success": True, "message": "LoRA权重已卸载"}
    except Exception as e:
        return {"success": False, "message": f"卸载LoRA失败: {str(e)}"}

@app.get("/lora-status")
async def lora_status():
    return {
        "loaded": diffusion_service.lora_weights_loaded,
        "path": diffusion_service.lora_weights_path
    }
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
    global current_generation_task, pending_regeneration_request

    logger.info(f"User {current_user.username} starting generation. Credits left: {current_user.credits}")
    session_id = str(uuid.uuid4())
    try:
        logger.info(f"开始处理请求, 参数: num_scenes={num_scenes}, steps={steps}, guidance={guidance}")
        logger.info(f"开始处理请求，会话ID: {session_id}")

        temp_dir = os.path.join("temp", session_id)
        os.makedirs(temp_dir, exist_ok=True)
        image_paths = []
        deepseek = DeepSeekService()


        # 确保模型已经加载

        if diffusion_service.model_name is None:
            error_msg = "请先选择并加载模型"
            logger.error(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            return

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
        yield f"data: {json.dumps({'type': 'info', 'message': '正在构思故事情节和分镜...'})}\n\n"



        # 【核心修改】明确使用 process_novel_to_scenes
        scenes_data = await deepseek.process_novel_to_scenes(text, num_scenes)


        # --- 3. 【核心修改】适配前端协议 ---
        # 提取纯中文字符串列表，以匹配前端在 'start' 事件中对 data.scenes 的期望
        chinese_descriptions_list = [scene.get("chinese_description", "") for scene in scenes_data]
        logger.info(f"场景分割完成，共 {len(scenes_data)} 个场景")

        # 发送 'start' 事件，其数据结构与你同事版本的前端期望完全一致

        yield f"data: {json.dumps({'type': 'start', 'total_scenes': len(scenes_data), 'scenes': chinese_descriptions_list, 'session_id': session_id})}\n\n"

        # --- 【新增部分】获取当前加载的LoRA prompt ---
        current_lora_prompt = ""
        if diffusion_service.lora_weights_loaded and diffusion_service.lora_weights_path:
            loras = diffusion_service.get_available_loras()
            # 兼容路径格式差异，使用 normpath
            current_path_norm = os.path.normpath(diffusion_service.lora_weights_path)
            for lora in loras:
                if os.path.normpath(lora['path']) == current_path_norm:
                    current_lora_prompt = lora.get('prompt', '')
                    logger.info(f"检测到已加载LoRA: {lora['name']}，提示词: {current_lora_prompt}")
                    break
        else:
            logger.info("未检测到已加载LoRA")

        # --- 生成图片 ---
        logger.info("开始生成图片...")
        for i, scene_info in enumerate(scenes_data):
            logger.info(f"处理第 {i + 1}/{len(scenes_data)} 个场景")
            try:
                chinese_description = scene_info.get("chinese_description", "（无描述）")
                english_prompt = scene_info.get("english_prompt", "")

                if not english_prompt:
                    logger.warning(f"场景 {i+1} 没有有效的英文提示词，已跳过。")
                    yield f"data: {json.dumps({'type': 'scene_error', 'index': i, 'message': 'AI未能生成有效的绘图指令。'})}\n\n"
                    continue

                # 【核心修改】拼接LoRA提示词和原英文提示词
                prompt_for_drawing = f" {current_lora_prompt}, {english_prompt}, detailed, high quality"
                logger.info(f"生成提示词: {prompt_for_drawing}")

                queue = asyncio.Queue()

                def progress_callback(step, timestep, latents):
                    try:
                        queue.put_nowait({"type": "progress", "index": i, "step": step + 1, "total_steps": steps})
                    except asyncio.QueueFull:
                        pass



                loop = asyncio.get_event_loop()

                import functools

                target_for_executor = functools.partial(
                    diffusion_service.generate_image_with_style,
                    prompt=prompt_for_drawing, style="comic",
                    steps=steps, guidance_scale=guidance,
                    width=width, height=height, callback=progress_callback
                )
                gen_task = loop.run_in_executor(None, target_for_executor)
                current_generation_task = gen_task  # 【修改1】记录当前任务

                done = False
                while not done:
                    progress_task = asyncio.create_task(queue.get())
                    finished, pending = await asyncio.wait([gen_task, progress_task],
                                                           return_when=asyncio.FIRST_COMPLETED)

                    if progress_task in finished:
                        yield f"data: {json.dumps(progress_task.result())}\n\n"

                    if gen_task in finished:
                        image_url = gen_task.result()
                        while not queue.empty():
                            yield f"data: {json.dumps(queue.get_nowait())}\n\n"

                        logger.info("图片生成完成")

                        image_data = base64.b64decode(image_url.split(',')[1])
                        image_path = os.path.join(temp_dir, f"scene_{i + 1}.png")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        image_paths.append(image_path)

                        


                        # 发送 'scene' 事件，数据结构与你同事版本前端期望一致

                        yield f"data: {json.dumps({'type': 'scene', 'index': i, 'description': chinese_description, 'image_url': image_url})}\n\n"
                        done = True
                        if not progress_task.done(): progress_task.cancel()

                        current_generation_task = None  # 【修改2】清空当前任务记录

                        # 【修改3】检查是否有待处理的重新生成请求
                        if pending_regeneration_request:
                            logger.info("检测到待处理的重新生成请求，当前图片已完成，即将处理重新生成...")
                            break  # 跳出循环，停止后续图片生成

            except Exception as e:
                current_generation_task = None  # 【修改4】异常时也要清空任务记录
                error_msg = f"场景 {i + 1} 生成失败: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'scene_error', 'index': i, 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'all_images_generated', 'session_id': session_id})}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'new_credit_balance': current_user.credits})}\n\n"

    except Exception as e:
        current_generation_task = None  # 【修改5】异常时清空任务记录
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"


@app.post("/save-scene-descriptions")
async def save_scene_descriptions(request: dict):
    """保存场景描述到txt文件"""
    try:
        session_id = request.get("session_id")
        descriptions = request.get("descriptions")

        if not session_id or not descriptions:
            raise HTTPException(status_code=400, detail="Missing session_id or descriptions")

        # 确保temptxt目录存在
        temp_txt_dir = "temptxt"
        if not os.path.exists(temp_txt_dir):
            os.makedirs(temp_txt_dir)

        # 生成txt文件内容
        txt_content = ""
        for i, description in enumerate(descriptions):
            if description:  # 只保存非空描述
                txt_content += f"场景{i + 1}：{description}\n"

        # 保存到文件
        txt_file_path = os.path.join(temp_txt_dir, f"{session_id}.txt")
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)

        logger.info(f"场景描述已保存到: {txt_file_path}")
        return {"success": True, "message": "场景描述保存成功"}

    except Exception as e:
        logger.error(f"保存场景描述失败: {str(e)}")
        return {"success": False, "message": str(e)}

@app.post("/regenerate-image")
async def regenerate_image(request: dict):
    global current_generation_task, pending_regeneration_request

    try:
        description = request.get("description")
        scene_index = request.get("sceneIndex")
        steps = request.get("steps", 30)
        session_id = request.get("session_id")  # 新增：获取session_id

        logger.info(f"收到重新生成图片请求: scene_index={scene_index}, description={description}, steps={steps}, session_id={session_id}")

        if scene_index is None or description is None:
            raise HTTPException(status_code=400, detail="Missing description or scene_index parameter")

        # 检查是否已设置模型
        if diffusion_service.model_name is None:
            return {
                "success": False,
                "message": "请先选择并加载模型"
            }

        # 检查模型是否已加载
        if not diffusion_service.is_model_loaded():
            return {
                "success": False,
                "message": "模型未加载，请先加载模型"
            }

        # 【修改6】检查是否有正在进行的生成任务
        if current_generation_task and not current_generation_task.done():
            logger.info("检测到正在进行的图片生成任务，将在当前图片完成后进行重新生成...")
            # 存储重新生成请求，等待当前任务完成
            pending_regeneration_request = {
                "description": description,
                "scene_index": scene_index,
                "steps": steps,
                "session_id": session_id  # 新增：存储session_id
            }

            # 等待当前任务完成
            logger.info("等待当前生成任务完成...")
            await current_generation_task
            logger.info("当前生成任务已完成，开始处理重新生成请求")

            # 清空待处理请求标记
            pending_regeneration_request = None

        try:
            # 初始化服务
            deepseek = DeepSeekService()

            # 将中文描述翻译成英文
            english_prompt = deepseek.translate_to_english(description)
            logger.info(f"翻译完成: {english_prompt}")

            # 转换为适合图像生成的提示词
            image_prompt = deepseek.convert_to_image_prompt(english_prompt)
            logger.info(f"图像提示词生成完成: {image_prompt}")

            # 构建最终提示词
            final_prompt = f" {image_prompt}, detailed, high quality"
            logger.info(f"最终提示词: {final_prompt}")

            # 生成图片，使用传入的steps参数
            image_url = diffusion_service.generate_image_with_style(
                final_prompt,
               
                steps,  # 使用传入的steps参数
                7.5,  # guidance_scale
                512,  # width
                512  # height
            )

            # 新增：如果提供了session_id，则保存图片到对应目录
            if session_id:
                try:
                    temp_dir = os.path.join("temp", session_id)
                    if os.path.exists(temp_dir):
                        # 解码base64图片数据
                        image_data = base64.b64decode(image_url.split(',')[1])
                        # 覆盖原有图片文件
                        image_path = os.path.join(temp_dir, f"scene_{scene_index + 1}.png")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        logger.info(f"重新生成的图片已保存并覆盖: {image_path}")
                    else:
                        logger.warning(f"临时目录不存在: {temp_dir}")
                except Exception as save_error:
                    logger.error(f"保存重新生成的图片失败: {str(save_error)}")
                    # 不影响主要流程，继续返回结果

            logger.info("重新生成图片完成")

            return {
                "success": True,
                "image_url": image_url
            }

        except Exception as e:
            logger.error(f"重新生成图片失败: {str(e)}")
            return {
                "success": False,
                "message": f"重新生成失败: {str(e)}"
            }

    except Exception as e:
        logger.error(f"重新生成图片请求处理失败: {str(e)}")
        return {
            "success": False,
            "message": str(e)
        }

@app.get("/video", response_class=HTMLResponse)
async def get_video_page():
    """获取视频播放页面"""
    return FileResponse("templates/video.html")


from service.video_service import VideoService

# 在你的main.py文件中添加VideoService实例
video_service = VideoService()


@app.post("/create-video")
async def create_video(request: dict):
    """根据会话ID创建带语音的视频"""
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing session_id parameter")

        logger.info(f"开始创建视频，会话ID: {session_id}")

        temp_dir = os.path.join("temp", session_id)
        if not os.path.exists(temp_dir):
            raise HTTPException(status_code=404, detail="Session not found")

        # 获取图片路径
        image_paths = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".png")])

        if not image_paths:
            raise HTTPException(status_code=400, detail="No images found for this session")

        # 读取场景描述
        txt_file_path = os.path.join("temptxt", f"{session_id}.txt")
        scene_descriptions = []

        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                scene_descriptions = [line.strip() for line in f.readlines() if line.strip()]

        if not scene_descriptions:
            raise HTTPException(status_code=400, detail="No scene descriptions found")

        if len(image_paths) != len(scene_descriptions):
            logger.warning(f"图片数量({len(image_paths)})与描述数量({len(scene_descriptions)})不匹配")
            # 调整到较小的数量
            min_count = min(len(image_paths), len(scene_descriptions))
            image_paths = image_paths[:min_count]
            scene_descriptions = scene_descriptions[:min_count]

        # 生成视频
        video_output_dir = os.path.join("static", "videos")
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        video_path = os.path.join(video_output_dir, f"{session_id}.mp4")

        # 使用新的视频服务创建带语音的视频
        video_service.create_video_from_images_with_audio(
            image_paths, scene_descriptions, video_path, session_id
        )

        logger.info(f"视频生成成功: {video_path}")

        # 清理临时图片文件
        shutil.rmtree(temp_dir)
        logger.info(f"已清理临时目录: {temp_dir}")

        # 保留场景描述文件
        if os.path.exists(txt_file_path):
            logger.info(f"场景描述文件保留在: {txt_file_path}")

        return {"success": True, "video_url": f"/static/videos/{session_id}.mp4"}

    except Exception as e:
        logger.error(f"创建视频失败: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    logger.info("启动服务器...")
    uvicorn.run(app, host="localhost", port=8080)

