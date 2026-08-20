from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HISTORY_FILE = OUTPUTS_DIR / "history.jsonl"
FAVORITES_FILE = OUTPUTS_DIR / "favorites.json"
LORA_DIR = PROJECT_ROOT / "lora_output"


QUALITY_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, noisy, jpeg artifacts, watermark, "
    "bad anatomy, bad hands, deformed fingers, extra fingers, missing fingers, "
    "extra limbs, poorly drawn face, distorted perspective"
)


@dataclass(frozen=True)
class QualityPreset:
    name: str
    description: str
    width: int
    height: int
    steps: int
    guidance: float
    hires_fix: bool
    hires_scale: float
    hires_steps: int
    denoise_strength: float
    batch_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, QualityPreset] = {
    "3060_6gb_portrait": QualityPreset(
        name="RTX 3060 6GB 人像",
        description="默认推荐。先用 640x960 顺序生成 4 张，再对满意结果开高清修复。",
        width=640,
        height=960,
        steps=28,
        guidance=7.0,
        hires_fix=False,
        hires_scale=1.2,
        hires_steps=12,
        denoise_strength=0.28,
        batch_count=4,
    ),
    "3060_6gb_quality": QualityPreset(
        name="RTX 3060 6GB 高清",
        description="质量优先。适合单张精修，目标约 768x1152。",
        width=640,
        height=960,
        steps=34,
        guidance=7.5,
        hires_fix=True,
        hires_scale=1.2,
        hires_steps=16,
        denoise_strength=0.25,
        batch_count=1,
    ),
    "safe_square": QualityPreset(
        name="省显存方图",
        description="显存紧张或先试风格时使用。",
        width=512,
        height=512,
        steps=24,
        guidance=7.0,
        hires_fix=False,
        hires_scale=1.0,
        hires_steps=10,
        denoise_strength=0.25,
        batch_count=2,
    ),
}


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 640
    height: int = 960
    steps: int = 28
    guidance: float = 7.0
    seed: int | None = None
    batch_count: int = 4
    hires_fix: bool = False
    hires_scale: float = 1.2
    hires_steps: int = 12
    denoise_strength: float = 0.28
    model_name: str = "stable-diffusion-v1-5"
    lora_name: str | None = None
    lora_weight: float = 0.8

    def normalized(self) -> "GenerationRequest":
        prompt = self.prompt.strip()
        negative_prompt = self.negative_prompt.strip() or QUALITY_NEGATIVE_PROMPT
        width = _round_to_multiple(int(self.width), 8)
        height = _round_to_multiple(int(self.height), 8)
        steps = _clamp(int(self.steps), 1, 80)
        guidance = _clamp(float(self.guidance), 1.0, 15.0)
        batch_count = _clamp(int(self.batch_count), 1, 8)
        hires_scale = _clamp(float(self.hires_scale), 1.0, 1.6)
        hires_steps = _clamp(int(self.hires_steps), 1, 40)
        denoise_strength = _clamp(float(self.denoise_strength), 0.05, 0.65)

        if not prompt:
            raise ValueError("请输入提示词。")
        if width < 256 or height < 256:
            raise ValueError("宽度和高度不能低于 256。")
        if width * height > 1024 * 1536:
            raise ValueError("分辨率过高，RTX 3060 6GB 容易显存不足。建议先用 640x960 或 768x1152。")

        return GenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=self.seed,
            batch_count=batch_count,
            hires_fix=bool(self.hires_fix),
            hires_scale=hires_scale,
            hires_steps=hires_steps,
            denoise_strength=denoise_strength,
            model_name=self.model_name,
            lora_name=self.lora_name or None,
            lora_weight=_clamp(float(self.lora_weight), 0.0, 1.5),
        )

    def seed_for_index(self, index: int) -> int | None:
        if self.seed is None:
            return None
        return int(self.seed) + index

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_choices() -> list[str]:
    return list(PRESETS.keys())


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))
