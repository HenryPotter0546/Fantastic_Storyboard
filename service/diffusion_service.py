# service/diffusion_service.py
import os
import logging
import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.loaders import LoraLoaderMixin
from PIL import Image
import base64
import io
import yaml
from dotenv import load_dotenv
import ffmpeg
import shutil
from typing import List, Callable, Optional
import threading
import copy

load_dotenv()

logger = logging.getLogger(__name__)



class LocalDiffusionService:
    _lora_lock = threading.Lock()

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.lora_weights_loaded = False
        self.lora_weights_path = None
        self.lora_scale = 1.0
        self.model_config = None
        self.current_lora_name = None

        # 保存原始权重的字典，支持多个组件
        self._original_state_dicts = {}
        
        # IP-Adapter和ControlNet相关属性
        self.ip_adapter_ref_image = None
        self.control_image = None

        if model_name:
            self.model_config = self._load_model_config()

    def set_model(self, model_name: str):
        """设置要使用的模型（不立即加载）"""
        try:
            if self.model_name != model_name:
                logger.info(f"切换模型从 {self.model_name} 到 {model_name}")
                self._unload_current_model()
                self.lora_weights_loaded = False
                self.lora_weights_path = None
                self.lora_scale = 1.0
                self.current_lora_name = None
                self._original_state_dicts = {}

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
            logger.error(traceback.format_exc())
            raise

    def _load_model(self):
        """加载diffusion模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            logger.info(f"使用设备: {self.device}")

            model_path = self.model_config['path']
            single_files = self.model_config.get('single_files', False)
            use_safetensors = self.model_config.get('use_safetensors', True)
            model_type = self.model_config.get('type', 'sd')

            logger.info(f"模型路径: {model_path}")
            logger.info(f"模型类型: {model_type}")
            logger.info(f"单文件模式: {single_files}")
            logger.info(f"使用safetensors: {use_safetensors}")

            if model_type.lower() == 'sdxl':
                if single_files:
                    logger.info(f"加载SDXL单文件模型: {model_path}")
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False,
                    )
                else:
                    logger.info(f"加载SDXL目录模型: {model_path}")
                    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False,
                    )
            elif model_type.lower() == 'sd':
                if single_files:
                    logger.info(f"加载SD单文件模型: {model_path}")
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False,
                    )
                else:
                    logger.info(f"加载SD目录模型: {model_path}")
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=use_safetensors,
                        local_files_only=False,
                    )
            elif model_type.lower() == 'kolors':
                from diffusers import KolorsPipeline
                self.pipeline = KolorsPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                )
            elif model_type.lower() == 'ip-adapter':

                # copy from https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter_sdxl_controlnet_demo.ipynb
                
                from diffusers import (
                    StableDiffusionXLControlNetPipeline, 
                    ControlNetModel, 
                    StableDiffusionControlNetPipeline, 
                    AutoencoderKL, 
                    DDIMScheduler,
                )
                from service.ip_adapter import IPAdapterXL,IPAdapter

                # controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
                controlnet_path = self.model_config['controlnet_path']
                image_encoder_path = self.model_config['image_encoder_path']
                ip_ckpt = self.model_config['ip_ckpt']
                vae_model_path = self.model_config['vae_model_path']

                # load SD pipeline
                noise_scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                    steps_offset=1,
                )
                vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
                controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_path,
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=torch.float16,
                    scheduler=noise_scheduler,
                    feature_extractor=None,
                    safety_checker=None,
                )

                # load ip-adapter
                self.pipeline = IPAdapter(pipe, image_encoder_path, ip_ckpt, device=self.device)

                # image = Image.open("assets/images/statue.png")
                # depth_map = Image.open("assets/structure_controls/depth.png").resize((1024, 1024))

                # images = ip_model.generate(pil_image=image, image=depth_map, controlnet_conditioning_scale=0.7, num_inference_steps=30, seed=42)

            else:
                raise ValueError(f"未知模型类型: {model_type}")

            if model_type.lower() != 'ip-adapter':
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                self.pipeline = self.pipeline.to(self.device)

                if self.device == "cuda":
                    self.pipeline.enable_attention_slicing()
                    self.pipeline.enable_vae_slicing()

            setattr(self.pipeline, '_model_path', model_path)

            logger.info("模型加载完成")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _get_available_models_from_config(self):
        """从配置文件中获取非LoRA的可用模型列表"""
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

            # 过滤掉type为lora的模型
            models = [name for name, conf in config.items() if conf.get('type', '').lower() != 'lora']
            logger.info(f"从配置文件中读取到 {len(models)} 个非LoRA模型: {models}")
            return models

        except Exception as e:
            logger.error(f"读取配置文件失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_available_models(self):
        """获取可用的模型列表（公共方法）"""
        return self._get_available_models_from_config()

    def _get_available_loras_from_config(self):
        try:
            config_path = "config/model.yaml"
            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return []

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                logger.warning("配置文件为空")
                return []

            loras = []
            for name, conf in config.items():
                if conf.get('type', '').lower() == 'lora':
                    loras.append({
                        'name': name,
                        'path': conf['path'],
                        'description': conf.get('description', ''),
                        'prompt': conf.get('prompt', '')
                    })
            logger.info(f"加载到 {len(loras)} 个LoRA风格")
            return loras
        except Exception as e:
            logger.error(f"读取LoRA配置失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_available_loras(self):
        """公共接口获取LoRA风格"""
        return self._get_available_loras_from_config()

    def _save_original_weights(self):
        """保存原始权重到内存"""
        if hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
            self._original_state_dicts['unet'] = copy.deepcopy(self.pipeline.unet.state_dict())
            logger.info("已保存UNet原始权重")
        
        if hasattr(self.pipeline, 'text_encoder') and self.pipeline.text_encoder is not None:
            self._original_state_dicts['text_encoder'] = copy.deepcopy(self.pipeline.text_encoder.state_dict())
            logger.info("已保存TextEncoder原始权重")

    def load_lora_weights(self, lora_name: str, scale: float = 1.0):
        """改进的LoRA权重加载方法"""
        with self._lora_lock:
            if self.pipeline is None:
                raise RuntimeError("请先加载主模型，再加载LoRA权重")

            loras = self.get_available_loras()
            lora_conf = next((l for l in loras if l['name'] == lora_name), None)
            if not lora_conf:
                raise FileNotFoundError(f"LoRA风格 {lora_name} 未在配置中找到")

            lora_path = lora_conf['path']
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA权重文件不存在: {lora_path}")

            # 检查是否已经加载了相同的LoRA
            if (self.lora_weights_loaded and 
                self.lora_weights_path == lora_path and 
                abs(self.lora_scale - scale) < 1e-4):
                logger.info(f"LoRA权重 {lora_name} 已加载且scale相同，无需重复加载")
                return

            try:
                logger.info(f"加载LoRA权重: {lora_name}，路径: {lora_path}，scale: {scale}")

                # 如果已经加载了其他LoRA，先卸载
                if self.lora_weights_loaded:
                    self.unload_lora_weights()

                # 首次加载LoRA时保存原始权重
                if not self._original_state_dicts:
                    self._save_original_weights()

                # 使用正确的API加载LoRA权重
                try:
                    # 方法1：使用load_lora_weights（推荐）
                    self.pipeline.load_lora_weights(lora_path)
                    logger.info("使用 load_lora_weights 方法加载成功")
                except Exception as e1:
                    logger.warning(f"load_lora_weights 方法失败: {e1}")
                    try:
                        # 方法2：直接加载到unet
                        from diffusers.utils import convert_state_dict_to_diffusers
                        
                        # 加载LoRA权重文件
                        if lora_path.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            lora_state_dict = load_file(lora_path)
                        else:
                            lora_state_dict = torch.load(lora_path, map_location=self.device)
                        
                        # 转换为diffusers格式
                        converted_state_dict = convert_state_dict_to_diffusers(lora_state_dict)
                        
                        # 加载到pipeline
                        self.pipeline.load_lora_weights(converted_state_dict)
                        logger.info("使用转换方法加载成功")
                    except Exception as e2:
                        logger.warning(f"转换方法也失败: {e2}")
                        try:
                            # 方法3：直接使用unet的load_attn_procs方法
                            self.pipeline.unet.load_attn_procs(lora_path)
                            logger.info("使用 load_attn_procs 方法加载成功")
                        except Exception as e3:
                            logger.error(f"所有LoRA加载方法都失败: {e3}")
                            raise e3

                # 设置LoRA scale
                if hasattr(self.pipeline, 'set_lora_scale'):
                    self.pipeline.set_lora_scale(scale)
                else:
                    # 手动设置scale
                    self._manual_set_lora_scale(scale)

                self.lora_weights_loaded = True
                self.lora_weights_path = lora_path
                self.lora_scale = scale
                self.current_lora_name = lora_name

                logger.info(f"LoRA权重 {lora_name} 加载成功，scale: {scale}")

            except Exception as e:
                logger.error(f"加载LoRA失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise

    def _manual_set_lora_scale(self, scale: float):
        """手动设置LoRA scale的方法"""
        try:
            # 如果pipeline有cross_attention_kwargs参数，可以通过这种方式设置
            if hasattr(self.pipeline, 'unet'):
                # 遍历unet的所有attention processor
                for name, processor in self.pipeline.unet.attn_processors.items():
                    if hasattr(processor, 'scale'):
                        processor.scale = scale
                    elif hasattr(processor, 'lora_scale'):
                        processor.lora_scale = scale
                logger.info(f"手动设置LoRA scale为: {scale}")
        except Exception as e:
            logger.warning(f"手动设置LoRA scale失败: {e}")

    def unload_lora_weights(self):
        """卸载LoRA权重，恢复主模型权重"""
        with self._lora_lock:
            if self.lora_weights_loaded:
                logger.info("卸载LoRA权重，恢复主模型权重")
                
                try:
                    # 方法1：使用unload_lora_weights
                    if hasattr(self.pipeline, 'unload_lora_weights'):
                        self.pipeline.unload_lora_weights()
                        logger.info("使用 unload_lora_weights 方法卸载成功")
                    else:
                        # 方法2：恢复原始权重
                        self._restore_original_weights()
                        logger.info("使用权重恢复方法卸载成功")
                    
                    self.lora_weights_loaded = False
                    self.lora_weights_path = None
                    self.lora_scale = 1.0
                    self.current_lora_name = None
                    
                    logger.info("LoRA权重已卸载")
                except Exception as e:
                    logger.error(f"卸载LoRA权重失败: {e}")
                    # 强制恢复原始权重
                    self._restore_original_weights()
                    self.lora_weights_loaded = False
                    self.lora_weights_path = None
                    self.lora_scale = 1.0
                    self.current_lora_name = None
            else:
                logger.info("未加载任何LoRA权重，无需卸载")

    def _restore_original_weights(self):
        """恢复原始权重"""
        try:
            if 'unet' in self._original_state_dicts:
                self.pipeline.unet.load_state_dict(self._original_state_dicts['unet'], strict=False)
                self.pipeline.unet.to(self.device)
                logger.info("已恢复UNet原始权重")
            
            if 'text_encoder' in self._original_state_dicts:
                self.pipeline.text_encoder.load_state_dict(self._original_state_dicts['text_encoder'], strict=False)
                self.pipeline.text_encoder.to(self.device)
                logger.info("已恢复TextEncoder原始权重")
        except Exception as e:
            logger.error(f"恢复原始权重失败: {e}")

    def set_lora_scale(self, scale: float):
        """动态设置LoRA权重scale"""
        with self._lora_lock:
            if not self.lora_weights_loaded:
                raise RuntimeError("未加载任何LoRA权重，无法设置scale")

            logger.info(f"设置LoRA权重scale: {scale}")
            try:
                if hasattr(self.pipeline, 'set_lora_scale'):
                    self.pipeline.set_lora_scale(scale)
                else:
                    self._manual_set_lora_scale(scale)
                
                self.lora_scale = scale
                logger.info(f"LoRA scale设置为: {scale}")
            except Exception as e:
                logger.error(f"设置LoRA权重scale失败: {str(e)}")
                raise

    def load_sdxl_ip_adapter_model(self):
        """加载SDXL-IP-Adapter模型用于一键式生成"""
        try:
            logger.info("开始加载SDXL-IP-Adapter模型...")
            
            # 设置模型为SDXL-IP-Adapter（需要在config/model.yaml中配置）
            self.set_model("sdxl-ip-adapter")
            
            # 预加载模型
            self.preload_model()
            
            logger.info("SDXL-IP-Adapter模型加载完成")
            
        except Exception as e:
            logger.error(f"加载SDXL-IP-Adapter模型失败: {str(e)}")
            raise

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        mode: str = "custom",
        callback: Optional[Callable] = None,
    ) -> str:
        """生成图片并返回base64编码的图片数据"""
        try:
            logger.info(f"开始生成图片，提示词: {prompt}")
            logger.info(f"参数: steps={steps}, guidance={guidance_scale}, size={width}x{height}")
            
            # 记录LoRA状态
            if self.lora_weights_loaded:
                logger.info(f"当前已加载LoRA: {self.current_lora_name}, scale: {self.lora_scale}")

            self._ensure_model_loaded()

            if negative_prompt is None:
                negative_prompt = (
                    "low quality, bad quality, sketches, blurry, blur, out of focus, grainy, text, watermark, logo, banner, "
                    "extra digits, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, disgusting, "
                    "amputation, extra limbs, extra arms, extra legs, disfigured, gross proportions, malformed limbs, missing arms, "
                    "missing legs, floating limbs, disconnected limbs, long neck, long body, mutated hands and fingers, out of frame, "
                    "double, two heads, blurred, ugly, disfigured, too many limbs, deformed, repetitive, black and white, grainy, "
                    "extra limbs, bad anatomy, high pass filter, airbrush, portrait, zoomed, soft light, smooth skin, closeup, "
                    "deformed, extra limbs, extra faces, mutated hands, bad anatomy, bad proportions, blind, bad eyes, ugly eyes, "
                    "dead eyes, blur, vignette, out of shot, out of focus, gaussian, closeup, monochrome, grainy, noisy, text, writing, "
                    "watermark, logo, overexposed, underexposed, over-saturated, under-saturated"
                )

            with torch.no_grad():
                # 准备生成参数
                if mode == "custom":
                    generate_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance_scale,
                        "width": width,
                        "height": height,
                    }
                elif mode == "quick":
                    # 使用IP-Adapter和ControlNet进行一键式生成
                    generate_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance_scale,
                        "width": width,
                        "height": height,
                    }
                    
                    # 添加IP-Adapter参考图像（如果可用）
                    if self.ip_adapter_ref_image is not None:
                        generate_kwargs["pil_image"] = self.ip_adapter_ref_image
                        logger.info("使用IP-Adapter参考图像")
                    
                    # 添加ControlNet控制图像（如果可用）
                    if self.control_image is not None:
                        generate_kwargs["image"] = self.control_image
                        generate_kwargs["controlnet_conditioning_scale"] = 0.7
                        logger.info("使用ControlNet控制图像")
                    else:
                        # 如果没有ControlNet图像，创建一个空白图像作为控制图像
                        from PIL import Image
                        import numpy as np
                        blank_image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
                        generate_kwargs["image"] = blank_image
                        generate_kwargs["controlnet_conditioning_scale"] = 0.0  # 设置为0表示不使用控制
                        logger.info("使用空白ControlNet控制图像")
                    
                    if self.ip_adapter_ref_image is None and self.control_image is None:
                        logger.warning("一键式生成模式：未找到参考图像或控制图像，使用标准生成")
                
                # 添加callback参数（如果支持）
                if callback is not None:
                    generate_kwargs.update({
                        "callback": callback,
                        "callback_steps": 1,
                    })

                # 检查是否是IP-Adapter模型
                if hasattr(self.pipeline, 'generate') and callable(getattr(self.pipeline, 'generate')):
                    # IP-Adapter模型使用generate方法
                    logger.info("使用IP-Adapter生成方法")
                    
                    # 调整参数以匹配IP-Adapter的generate方法签名
                    ip_adapter_kwargs = {
                        "prompt": generate_kwargs.get("prompt"),
                        "negative_prompt": generate_kwargs.get("negative_prompt"),
                        "num_inference_steps": generate_kwargs.get("num_inference_steps", 30),
                        "guidance_scale": generate_kwargs.get("guidance_scale", 7.5),
                        "scale": 1.0,  # IP-Adapter的scale参数
                    }
                    
                    # 添加IP-Adapter特定参数
                    if "pil_image" in generate_kwargs:
                        ip_adapter_kwargs["pil_image"] = generate_kwargs["pil_image"]
                    
                    # 添加其他kwargs参数
                    for key, value in generate_kwargs.items():
                        if key not in ["prompt", "negative_prompt", "num_inference_steps", "guidance_scale", "scale", "pil_image"]:
                            ip_adapter_kwargs[key] = value
                    
                    images = self.pipeline.generate(**ip_adapter_kwargs)
                    image = images[0] if isinstance(images, list) else images
                else:
                    # 标准pipeline直接调用
                    image = self.pipeline(**generate_kwargs).images[0]

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_url = f"data:image/png;base64,{img_str}"

            logger.info("图片生成完成")
            return image_url

        except Exception as e:
            logger.error(f"生成图片时发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def generate_image_with_style(
        self,
        prompt: str,
        style: str = "comic",
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        mode: str = "custom",
        callback: Optional[Callable] = None,
    ) -> str:
        USE_LONG_NEGATIVE_PROMPT = False

        long_negative_prompt = (
            "text, letters, words, numbers, digits, writing, signature, watermark, logo, username, artist name, copyright, url, stamp, error, inscription, "
            "low quality, worst quality, bad quality, jpeg artifacts, blurry, noisy, grainy, ugly, disgusting, deformed, mutated, extra limbs, extra fingers, fewer fingers, bad anatomy, malformed limbs, mutated hands, poorly drawn hands, poorly drawn face, missing arms, missing legs, long neck, long body, disfigured, gross proportions, "
            "out of frame, body out of frame, cropped, cut off, duplicate, two heads, black and white, monochrome, sketches, boring, dull"
        )

        short_negative_prompt = (
            "text, letters, writing, signature, watermark, logo, "
            "low quality, worst quality, bad quality, blurry, ugly, deformed, mutated hands, poorly drawn face"
            "blurry face, blurred face, poorly drawn face, bad face, distorted face, extra eyes, missing eyes, asymmetrical eyes, cross-eyed, lazy eye, extra mouth, missing mouth, deformed mouth, extra nose, missing nose, deformed nose"
        )

        negative_prompt_to_use = long_negative_prompt if USE_LONG_NEGATIVE_PROMPT else short_negative_prompt
        logger.info(f"Using {'Long' if USE_LONG_NEGATIVE_PROMPT else 'Short'} Negative Prompt.")

        try:
            style_prompts = {
                "comic": "comic style",
                "realistic": "photorealistic, highly detailed, professional photography, sharp focus",
                "artistic": "artistic style, painterly, oil painting, masterpiece, beautiful composition",
                "cartoon": "cartoon style, cute, colorful, simple lines, friendly",
                "sketch": "sketch style, pencil drawing, black and white, artistic sketch",
                "pixel": "pixel art style, 8-bit, retro gaming, pixelated, chibi",
                "pixel_comic": "pixel art style, comic panel, comic book layout, 8-bit, 16-bit, low resolution, blocky, pixelated, clear lines, dynamic composition, retro gaming aesthetic",
            }

            style_prompt = style_prompts.get(style, style_prompts["comic"])
            full_prompt = f"{style_prompt}, {prompt}, high quality, detailed"

            return self.generate_image(
                prompt=full_prompt,
                negative_prompt=negative_prompt_to_use,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                mode=mode,
                callback=callback,
            )
        except Exception as e:
            logger.error(f"生成风格化图片时发生错误: {str(e)}")
            raise

    def create_video_from_images(self, image_paths: List[str], output_path: str, fps: float = 0.5):
        """将图片拼接成视频"""
        try:
            if not image_paths:
                raise ValueError("图片列表不能为空")

            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            (
                ffmpeg.input('pipe:', r=str(fps), f='image2pipe')
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

