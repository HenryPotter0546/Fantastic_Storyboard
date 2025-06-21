import os
import time
import logging
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import base64
import io
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class LocalDiffusionService:
    def __init__(self):
        self.model_id = os.getenv('DIFFUSION_MODEL_ID', 'runwayml/stable-diffusion-v1-5')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self._load_model()
        
    def _load_model(self):
        """加载diffusion模型"""
        try:
            logger.info(f"正在加载模型: {self.model_id}")
            logger.info(f"使用设备: {self.device}")
            
            # 创建pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # 使用更快的scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # 移动到指定设备
            self.pipeline = self.pipeline.to(self.device)
            
            # 启用内存优化
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def generate_image(self, prompt: str, negative_prompt: str = None) -> str:
        """生成图片并返回base64编码的图片数据"""
        try:
            logger.info(f"开始生成图片，提示词: {prompt}")
            
            if not self.pipeline:
                raise Exception("模型未加载")
            
            # 设置默认的负面提示词
            if negative_prompt is None:
                negative_prompt = "low quality, bad quality, sketches, blurry, blur, out of focus, grainy, text, watermark, logo, banner, extra digits, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed limbs, missing arms, missing legs, floating limbs, disconnected limbs, long neck, long body, mutated hands and fingers, out of frame, double, two heads, blurred, ugly, disfigured, too many limbs, deformed, repetitive, black and white, grainy, extra limbs, bad anatomy, high pass filter, airbrush, portrait, zoomed, soft light, smooth skin, closeup, deformed, extra limbs, extra faces, mutated hands, bad anatomy, bad proportions, blind, bad eyes, ugly eyes, dead eyes, blur, vignette, out of shot, out of focus, gaussian, closeup, monochrome, grainy, noisy, text, writing, watermark, logo, overexposed, underexposed, over-saturated, under-saturated"
            
            # 生成图片
            with torch.no_grad():
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=512,
                    height=512
                ).images[0]
            
            # 转换为base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 返回data URL格式
            image_url = f"data:image/png;base64,{img_str}"
            
            logger.info("图片生成完成")
            return image_url
            
        except Exception as e:
            logger.error(f"生成图片时发生错误: {str(e)}")
            raise
    
    def generate_image_with_style(self, prompt: str, style: str = "comic") -> str:
        """根据风格生成图片"""
        try:
            # 根据风格调整提示词
            style_prompts = {
                "comic": "comic style, manga style, anime style, vibrant colors, bold lines, dynamic composition",
                "realistic": "photorealistic, highly detailed, professional photography, sharp focus",
                "artistic": "artistic style, painterly, oil painting, masterpiece, beautiful composition",
                "cartoon": "cartoon style, cute, colorful, simple lines, friendly",
                "sketch": "sketch style, pencil drawing, black and white, artistic sketch"
            }
            
            style_prompt = style_prompts.get(style, style_prompts["comic"])
            full_prompt = f"{style_prompt}, {prompt}, high quality, detailed"
            
            return self.generate_image(full_prompt)
            
        except Exception as e:
            logger.error(f"生成风格化图片时发生错误: {str(e)}")
            raise
    
    async def handle_image_action(self, task_id: str, action: str, index: int) -> dict:
        """
        处理图片操作（重新生成变体）
        
        Args:
            task_id: 任务ID（本地diffusion中不使用）
            action: 操作类型（本地diffusion中只支持重新生成）
            index: 图片索引（本地diffusion中不使用）
            
        Returns:
            dict: 包含操作结果的字典
        """
        try:
            logger.info(f"收到图片操作请求: action={action}")
            
            # 本地diffusion不支持upscale和variation，只能重新生成
            # 这里可以扩展为保存原始prompt并重新生成
            return {
                "success": False,
                "message": "本地diffusion暂不支持图片操作，请重新生成"
            }
            
        except Exception as e:
            logger.error(f"处理图片操作失败: {str(e)}")
            return {
                "success": False,
                "message": f"处理失败: {str(e)}"
            } 