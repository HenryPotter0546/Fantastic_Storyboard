from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import base64
import io
import os
import logging
import uvicorn
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
import requests
from typing import Optional
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="图生图API")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局变量存储pipeline（从文生图服务获取）
img2img_pipeline = None
current_model_name = None


def get_model_status_from_main():
    """从主服务获取模型状态"""
    try:
        response = requests.get("http://127.0.0.1:8080/model-status")
        if response.status_code == 200:
            return response.json()
        return {"is_loaded": False, "current_model": None}
    except:
        return {"is_loaded": False, "current_model": None}


def create_img2img_pipeline_from_loaded_model():
    """从已加载的文生图模型创建图生图pipeline"""
    global img2img_pipeline, current_model_name

    try:
        # 获取主服务的模型状态
        model_status = get_model_status_from_main()

        if not model_status["is_loaded"]:
            logger.error("主服务中没有加载的模型")
            return False

        model_name = model_status["current_model"]

        # 如果已经为同一模型创建了pipeline，直接返回
        if img2img_pipeline is not None and current_model_name == model_name:
            logger.info(f"图生图pipeline已存在，模型: {model_name}")
            return True

        logger.info(f"为模型 {model_name} 创建图生图pipeline")

        # 读取模型配置
        import yaml
        config_path = "config/model.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if model_name not in config:
            logger.error(f"找不到模型配置: {model_name}")
            return False

        model_config = config[model_name]
        model_path = model_config['path']
        single_files = model_config.get('single_files', False)
        use_safetensors = model_config.get('use_safetensors', True)
        model_type = model_config.get('type', 'sd')

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 根据模型类型创建对应的图生图pipeline
        if model_type.lower() == 'sdxl':
            if single_files:
                img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=use_safetensors,
                    local_files_only=False
                )
            else:
                img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=use_safetensors,
                    local_files_only=False
                )
        else:
            if single_files:
                img2img_pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=use_safetensors,
                    local_files_only=False
                )
            else:
                img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=use_safetensors,
                    local_files_only=False
                )

        # 移动到设备并优化
        img2img_pipeline = img2img_pipeline.to(device)

        if device == "cuda":
            img2img_pipeline.enable_attention_slicing()
            img2img_pipeline.enable_vae_slicing()

        current_model_name = model_name
        logger.info(f"图生图pipeline创建完成，模型: {model_name}")
        return True

    except Exception as e:
        logger.error(f"创建图生图pipeline失败: {str(e)}")
        return False


@app.get("/", response_class=HTMLResponse)
async def get_img2img_page():
    """返回图生图页面"""
    return FileResponse("templates/img2img.html")


@app.get("/model-status")
async def get_img2img_model_status():
    """获取图生图模型状态"""
    main_status = get_model_status_from_main()

    return {
        "main_service_loaded": main_status["is_loaded"],
        "main_service_model": main_status["current_model"],
        "img2img_ready": img2img_pipeline is not None,
        "current_model": current_model_name
    }


@app.post("/prepare-img2img")
async def prepare_img2img():
    """手动准备图生图pipeline"""
    success = create_img2img_pipeline_from_loaded_model()

    if success:
        return {
            "success": True,
            "message": f"图生图pipeline创建成功，使用模型: {current_model_name}"
        }
    else:
        return {
            "success": False,
            "message": "创建图生图pipeline失败，请确保主服务已加载模型"
        }


@app.post("/generate")
async def generate_img2img(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        strength: float = Form(0.8),
        guidance_scale: float = Form(7.5),
        steps: int = Form(20),
        width: int = Form(512),
        height: int = Form(512)
):
    """生成图生图"""
    try:
        # 确保pipeline准备就绪
        if img2img_pipeline is None:
            raise HTTPException(status_code=400, detail="图生图服务未准备就绪，请先点击'创建图生图pipeline'按钮")

        # 读取上传的图片
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 调整图片大小
        input_image = input_image.resize((width, height))

        logger.info(f"开始生成图生图，prompt: {prompt}")

        # 生成图片
        result = img2img_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )

        generated_image = result.images[0]

        # 将图片转换为base64
        def image_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        input_base64 = image_to_base64(input_image)
        output_base64 = image_to_base64(generated_image)

        # 保存生成的图片到静态目录
        if not os.path.exists("static/generated"):
            os.makedirs("static/generated")

        import uuid
        filename = f"img2img_{uuid.uuid4().hex}.png"
        filepath = os.path.join("static/generated", filename)
        generated_image.save(filepath)

        logger.info("图生图生成完成")

        return {
            "success": True,
            "input_image": input_base64,
            "output_image": output_base64,
            "download_url": f"/static/generated/{filename}"
        }

    except Exception as e:
        logger.error(f"图生图生成失败: {str(e)}")
        return {
            "success": False,
            "message": str(e)
        }


if __name__ == "__main__":
    logger.info("启动图生图API服务器...")
    # 启动时不再自动创建pipeline
    uvicorn.run(app, host="localhost", port=8081)