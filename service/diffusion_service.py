import os
import time
import logging
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline, KolorsPipeline
from PIL import Image
import base64
import io
import yaml
from dotenv import load_dotenv
import ffmpeg
import shutil
from typing import List
from typing import Callable, Optional

load_dotenv()

logger = logging.getLogger(__name__)


class LocalDiffusionService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.model_config = None

        # 如果指定了模型名称，立即加载配置
        if model_name:
            self.model_config = self._load_model_config()

    def set_model(self, model_name: str):
        """设置要使用的模型（不立即加载）"""
        try:
            # 如果设置的是不同的模型，清除当前模型
            if self.model_name != model_name:
                logger.info(f"切换模型从 {self.model_name} 到 {model_name}")
                self._unload_current_model()

            self.model_name = model_name
            self.model_config = self._load_model_config()
            logger.info(f"模型已设置为: {model_name}")
        except Exception as e:
            logger.error(f"设置模型失败: {str(e)}")
            raise

    def _unload_current_model(self):
        """卸载当前模型，释放内存"""
        if self.pipeline is not None:
            logger.info("正在卸载当前模型...")
            del self.pipeline
            self.pipeline = None

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("当前模型已卸载")

    def is_model_loaded(self):
        """检查模型是否已加载"""
        return self.pipeline is not None

    def preload_model(self):
        """预加载模型"""
        if self.model_name is None:
            raise ValueError("未设置模型名称，请先调用 set_model() 方法")

        # 如果有不同的模型已加载，先卸载
        if self.pipeline is not None:
            current_model_path = getattr(self.pipeline, '_model_path', None)
            new_model_path = self.model_config['path']

            if current_model_path != new_model_path:
                logger.info("检测到模型变更，卸载当前模型")
                self._unload_current_model()
            else:
                logger.info("模型已经加载，无需重复加载")
                return

        logger.info(f"开始预加载模型: {self.model_name}")
        self._load_model()
        logger.info(f"模型预加载完成: {self.model_name}")

    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self.pipeline is None:
            if self.model_name is None:
                raise ValueError("未设置模型名称，请先调用 set_model() 方法")
            self._load_model()

    def _load_model_config(self):
        """从YAML文件加载模型配置"""
        try:
            config_path = "config/model.yaml"
            logger.info(f"尝试读取配置文件: {config_path}")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logger.info(f"成功读取配置文件，包含模型: {list(config.keys()) if config else '无'}")

            if not config:
                raise ValueError("配置文件为空")

            if self.model_name not in config:
                available_models = list(config.keys())
                raise ValueError(f"模型 '{self.model_name}' 不存在。可用模型: {available_models}")

            model_config = config[self.model_name]
            logger.info(f"加载模型配置: {self.model_name}")
            logger.info(f"模型路径: {model_config['path']}")
            logger.info(f"完整配置: {model_config}")

            return model_config

        except Exception as e:
            logger.error(f"加载模型配置失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise

    def _load_model(self):
        """加载diffusion模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            logger.info(f"使用设备: {self.device}")

            model_path = self.model_config['path']
            single_files = self.model_config.get('single_files', False)
            use_safetensors = self.model_config.get('use_safetensors', True)
            model_type = self.model_config.get('type', 'sd')  # 默认为SD模型

            logger.info(f"模型路径: {model_path}")
            logger.info(f"模型类型: {model_type}")
            logger.info(f"单文件模式: {single_files}")
            logger.info(f"使用safetensors: {use_safetensors}")

            if model_type.lower() == 'sdxl':
                # 加载SDXL模型
                if single_files:
                    logger.info(f"加载SDXL单文件模型: {model_path}")
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False  # 允许下载必要的文件
                    )
                else:
                    logger.info(f"加载SDXL目录模型: {model_path}")
                    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False  # 允许下载必要的文件
                    )
            elif model_type.lower() == 'sd':
                # 加载标准SD模型
                if single_files:
                    logger.info(f"加载SD单文件模型: {model_path}")
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False  # 允许下载必要的文件
                    )
                else:
                    logger.info(f"加载SD目录模型: {model_path}")
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False  # 允许下载必要的文件
                    )

            elif model_type.lower() == 'kolors':
                self.pipeline = KolorsPipeline.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16, 
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

            # 记录模型路径用于后续比较
            setattr(self.pipeline, '_model_path', model_path)

            logger.info("模型加载完成")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise

    def _get_available_models_from_config(self):
        """从配置文件中获取可用的模型列表"""
        try:
            config_path = "config/model.yaml"
            logger.info(f"尝试读取配置文件获取可用模型: {config_path}")

            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return []

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                logger.warning("配置文件为空")
                return []

            models = list(config.keys())
            logger.info(f"从配置文件中读取到 {len(models)} 个模型: {models}")
            return models

        except Exception as e:
            logger.error(f"读取配置文件失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return []

    def get_available_models(self):
        """获取可用的模型列表（公共方法）"""
        return self._get_available_models_from_config()

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        callback: Optional[Callable] = None
    ) -> str:
        """生成图片并返回base64编码的图片数据"""
        try:
            logger.info(f"开始生成图片，提示词: {prompt}")
            logger.info(f"参数: steps={steps}, guidance={guidance_scale}, size={width}x{height}")

            self._ensure_model_loaded()

            # 设置默认的负面提示词
            if negative_prompt is None:
                negative_prompt = "low quality, bad quality, sketches, blurry, blur, out of focus, grainy, text, watermark, logo, banner, extra digits, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, disgusting, amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed limbs, missing arms, missing legs, floating limbs, disconnected limbs, long neck, long body, mutated hands and fingers, out of frame, double, two heads, blurred, ugly, disfigured, too many limbs, deformed, repetitive, black and white, grainy, extra limbs, bad anatomy, high pass filter, airbrush, portrait, zoomed, soft light, smooth skin, closeup, deformed, extra limbs, extra faces, mutated hands, bad anatomy, bad proportions, blind, bad eyes, ugly eyes, dead eyes, blur, vignette, out of shot, out of focus, gaussian, closeup, monochrome, grainy, noisy, text, writing, watermark, logo, overexposed, underexposed, over-saturated, under-saturated"

            # 生成图片
            with torch.no_grad():
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    callback=callback,         # 回调函数
                    callback_steps=1           # 确保每一步都回调
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
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise

    def generate_image_with_style(
        self,
        prompt: str,
        style: str = "comic",
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        callback: Optional[Callable] = None
    ) -> str:
        # 开关：设置为 True 使用长提示词，False 使用精简版
        USE_LONG_NEGATIVE_PROMPT = False 

        # 长而全面的负向提示词 (适用于高 Token 上限模型)
        long_negative_prompt = (
            # 文字和签名移除
            "text, letters, words, numbers, digits, writing, signature, watermark, logo, username, artist name, copyright, url, stamp, error, inscription, "
            # 质量和解剖结构
            "low quality, worst quality, bad quality, jpeg artifacts, blurry, noisy, grainy, ugly, disgusting, deformed, mutated, extra limbs, extra fingers, fewer fingers, bad anatomy, malformed limbs, mutated hands, poorly drawn hands, poorly drawn face, missing arms, missing legs, long neck, long body, disfigured, gross proportions, "
            # 风格和构图
            "out of frame, body out of frame, cropped, cut off, duplicate, two heads, black and white, monochrome, sketches, boring, dull"
        )
        
        # 精简版负向提示词 (适用于低 Token 上限模型，如 SD 1.5)
        short_negative_prompt = (
            "text, letters, writing, signature, watermark, logo, " # 主要针对文字
            "low quality, worst quality, bad quality, blurry, ugly, deformed, mutated hands, poorly drawn face" # 主要针对质量和常见崩坏点
            "blurry face, blurred face, poorly drawn face, bad face, distorted face, extra eyes, missing eyes, asymmetrical eyes, cross-eyed, lazy eye, extra mouth, missing mouth, deformed mouth, extra nose, missing nose, deformed nose"
        )

        # 根据开关选择要使用的版本
        negative_prompt_to_use = long_negative_prompt if USE_LONG_NEGATIVE_PROMPT else short_negative_prompt
        logger.info(f"Using {'Long' if USE_LONG_NEGATIVE_PROMPT else 'Short'} Negative Prompt.")

        """根据风格生成图片"""
        try:
            # 根据风格调整提示词
            style_prompts = {
                "comic": "comic style, manga style, anime style, vibrant colors, bold lines, dynamic composition",
                "realistic": "photorealistic, highly detailed, professional photography, sharp focus",
                "artistic": "artistic style, painterly, oil painting, masterpiece, beautiful composition",
                "cartoon": "cartoon style, cute, colorful, simple lines, friendly",
                "sketch": "sketch style, pencil drawing, black and white, artistic sketch",
                "pixel": "pixel art style, 8-bit, retro gaming, pixelated, chibi",
                "pixel_comic": "pixel art style, comic panel, comic book layout, 8-bit, 16-bit, low resolution, blocky, pixelated, clear lines, dynamic composition, retro gaming aesthetic"
            }

            style_prompt = style_prompts.get(style, style_prompts["pixel_comic"])
            full_prompt = f"{style_prompt}, {prompt}, high quality, detailed"

            return self.generate_image(
                prompt=full_prompt,
                negative_prompt=negative_prompt_to_use,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                callback=callback
            )

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

    def create_video_from_images(self, image_paths: List[str], output_path: str, fps: float = 0.5):
        """将图片拼接成视频"""
        try:
            if not image_paths:
                raise ValueError("图片列表不能为空")
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 使用ffmpeg将图片拼接成视频
            (
                ffmpeg
                .input('pipe:', r=str(fps), f='image2pipe')
                .output(output_path, vcodec='libx264', pix_fmt='yuv420p', y='-y')
                .run(input=b''.join(open(p, 'rb').read() for p in image_paths))
            )
            
            logger.info(f"视频创建成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"创建视频失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise


def generate_img2img(self, prompt: str, image: Image.Image, strength: float = 0.8,
                     guidance_scale: float = 7.5, num_inference_steps: int = 20,
                     negative_prompt: str = "", width: int = 512, height: int = 512):
    """使用当前加载的模型进行图生图生成"""
    self._ensure_model_loaded()

    # 调整输入图片大小
    image = image.resize((width, height))

    # 检查是否为SDXL模型
    if hasattr(self.pipeline, 'unet') and hasattr(self.pipeline.unet.config, 'in_channels'):
        in_channels = self.pipeline.unet.config.in_channels
        is_sdxl = in_channels == 4 and hasattr(self.pipeline, 'text_encoder_2')
    else:
        is_sdxl = False

    if is_sdxl:
        # SDXL图生图
        from diffusers import StableDiffusionXLImg2ImgPipeline
        # 这里需要根据实际情况调整
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
    else:
        # 标准SD图生图
        from diffusers import StableDiffusionImg2ImgPipeline
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )

    return result.images[0]