from __future__ import annotations

import gc
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .config import GenerationRequest, MODELS_DIR
from .history import GenerationHistory, HistoryRecord
from .lora_manager import LoRAManager
from .model_registry import available_model_names, resolve_model


class InferencePipeline:
    """Lazy Stable Diffusion service for a 6GB GPU workflow."""

    def __init__(self, device: str | None = None, fp16: bool = True, models_dir: str | Path = MODELS_DIR):
        self.device = device or _default_device()
        self.fp16 = fp16
        self.models_dir = Path(models_dir)
        self.current_model_name: str | None = None
        self.pipe = None
        self.img2img_pipe = None
        self.lora_manager: LoRAManager | None = None
        self.history = GenerationHistory()
        self.cancel_requested = False

    def get_available_models(self) -> list[str]:
        return available_model_names(self.models_dir)

    def get_model_path(self, model_name: str) -> str:
        return str(resolve_model(self.models_dir, model_name).path)

    def load_model(self, model_name: str) -> None:
        if self.pipe is not None and self.current_model_name == model_name:
            return

        self.release()
        torch = _import_torch()
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

        dtype = torch.float16 if self.fp16 and self.device == "cuda" else torch.float32
        model = resolve_model(self.models_dir, model_name)
        if model.kind == "single_file":
            self.pipe = StableDiffusionPipeline.from_single_file(
                str(model.path),
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=True,
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                str(model.path),
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=True,
            )

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        _enable_memory_savers(self.pipe)

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipe.components)
        self.img2img_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.img2img_pipe = self.img2img_pipe.to(self.device)
        _enable_memory_savers(self.img2img_pipe)

        self.lora_manager = LoRAManager(self.pipe)
        self.current_model_name = model_name

    def release(self) -> None:
        if self.lora_manager is not None:
            try:
                self.lora_manager.clear()
            except Exception:
                pass
        self.pipe = None
        self.img2img_pipe = None
        self.lora_manager = None
        gc.collect()
        try:
            torch = _import_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def cancel(self) -> None:
        self.cancel_requested = True

    def generate_batch(self, request: GenerationRequest) -> Iterator[tuple[int, object, Path | None, str]]:
        request = request.normalized()
        self.cancel_requested = False
        self.load_model(request.model_name)
        assert self.pipe is not None
        assert self.lora_manager is not None

        self.lora_manager.apply(request.lora_name, request.lora_weight)
        torch = _import_torch()

        for index in range(request.batch_count):
            if self.cancel_requested:
                yield index, None, None, "已取消，已生成的图片已保留。"
                break

            seed = request.seed_for_index(index)
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
            image = self.pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                generator=generator,
            ).images[0]

            if request.hires_fix:
                image = self._hires_fix(image, request, generator)

            path = self.history.save_image(image, seed, index)
            self.history.append(
                HistoryRecord(
                    created_at=datetime.now().isoformat(timespec="seconds"),
                    image_path=str(path),
                    seed=seed,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    settings=asdict(request),
                )
            )
            yield index, image, path, f"完成 {index + 1}/{request.batch_count}：{path.name}"

    def _hires_fix(self, image, request: GenerationRequest, generator):
        if self.img2img_pipe is None:
            return image
        target_width = int(round(image.width * request.hires_scale / 8) * 8)
        target_height = int(round(image.height * request.hires_scale / 8) * 8)
        upscaled = image.resize((target_width, target_height))
        return self.img2img_pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image=upscaled,
            strength=request.denoise_strength,
            num_inference_steps=request.hires_steps,
            guidance_scale=request.guidance,
            generator=generator,
        ).images[0]


def _default_device() -> str:
    try:
        torch = _import_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少 PyTorch，请先按 README 安装依赖。") from exc
    return torch


def _enable_memory_savers(pipe) -> None:
    for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_xformers_memory_efficient_attention"):
        method = getattr(pipe, method_name, None)
        if method is None:
            continue
        try:
            method()
        except Exception:
            continue
