import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, StableDiffusionUpscalePipeline
from typing import Optional
from .lora_manager import LoRAManager
import os


class InferencePipeline:
    """Stable Diffusion 推理封装，支持多模型切换、LoRA 动态加载和 Hires.fix 高清修复"""

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.device = device
        self.fp16 = fp16
        self.current_model_path = None
        self.pipe = None
        self.lora_manager = None
        self.vae = None

        # 默认使用 SD v1.5
        if model_path is None:
            model_path = "C:/Users/acer/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5"
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载指定路径的模型"""
        if self.current_model_path == model_path and self.pipe is not None:
            return  # 已加载相同模型

        print(f"加载模型: {model_path}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.fp16 and self.device == "cuda" else torch.float32,
            safety_checker=None,
            local_files_only=True,
        )
        # 使用 Euler a 采样器 - 更快更清晰
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(self.device)
        self.lora_manager = LoRAManager(self.pipe)
        self.current_model_path = model_path
        print("模型加载完成")

    def get_available_models(self) -> list:
        """获取可选模型列表"""
        models = []
        models_dir = "E:/桌面/claude/text2img-project/models"
        if os.path.exists(models_dir):
            for folder in os.listdir(models_dir):
                folder_path = os.path.join(models_dir, folder)
                if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "model_index.json")):
                    models.append(folder)
        # 添加默认 SD v1.5
        models.append("stable-diffusion-v1-5")
        return models

    def get_model_path(self, model_name: str) -> str:
        """获取模型路径"""
        if model_name == "stable-diffusion-v1-5":
            return "C:/Users/acer/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5"
        return f"E:/桌面/claude/text2img-project/models/{model_name}"

    # 针对手指和透视问题的专用负面提示词
    HAND_FIX_NEGATIVE = "low quality, blurry, bad anatomy, deformed hands, deformed fingers, extra fingers, six fingers, missing fingers, malformed hands, bad hands, poorly drawn hands, extra limbs, poorly drawn face, bad proportions, wrong perspective, distorted perspective, perspective error"

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        hires_fix: bool = False,
        upscale_steps: int = 20,
        fix_hands: bool = True,
    ):
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 默认高质量负面提示词（手指修复优先）
        if not negative_prompt:
            if fix_hands:
                negative_prompt = self.HAND_FIX_NEGATIVE
            else:
                negative_prompt = "low quality, blurry, bad anatomy, deformed hands, deformed face, extra limbs, poorly drawn face"

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = result.images[0]

        # Hires.fix: 低分辨率生成后放大
        if hires_fix and image is not None:
            image = self._hires_fix(image, prompt, negative_prompt, upscale_steps, generator)

        return image

    def _hires_fix(self, low_res_img, prompt, negative_prompt, steps, generator):
        """高清修复 - 先放大再细化"""
        from diffusers import DPMSolverMultistepScheduler

        # 加载超分辨率 pipeline (使用现有的 VAE)
        try:
            # 4x 放大
            new_width = low_res_img.width * 2
            new_height = low_res_img.height * 2

            # 使用同一 pipe 但调整采样器进行去噪放大
            pipe = self.pipe
            original_scheduler = pipe.scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

            # 创建低分辨率图像的 latent 表示
            from PIL import Image
            import numpy as np

            # 简单的双线性放大 + 去噪
            upscaled = low_res_img.resize((new_width, new_height), Image.LANCZOS)

            # 使用 img2img 模式细化
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled,
                num_inference_steps=steps,
                guidance_scale=7.5,
                strength=0.3,  # 低强度保持原图结构
                generator=generator,
            )
            pipe.scheduler = original_scheduler
            return result.images[0]
        except Exception as e:
            print(f"Hires.fix 失败: {e}")
            return low_res_img
