from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SINGLE_FILE_EXTENSIONS = {".safetensors", ".ckpt"}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path
    kind: str
    complete: bool
    message: str


def discover_models(models_dir: str | Path) -> list[ModelInfo]:
    root = Path(models_dir)
    if not root.exists():
        return []

    models: list[ModelInfo] = []
    for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if item.is_dir() and (item / "model_index.json").exists():
            complete = is_complete_diffusers_model(item)
            models.append(
                ModelInfo(
                    name=item.name,
                    path=item,
                    kind="diffusers",
                    complete=complete,
                    message="完整 Diffusers 模型" if complete else "缺少 unet/vae/text_encoder 权重文件",
                )
            )
        elif item.is_file() and item.suffix.lower() in SINGLE_FILE_EXTENSIONS:
            complete = item.stat().st_size > 1024 * 1024
            models.append(
                ModelInfo(
                    name=item.stem,
                    path=item,
                    kind="single_file",
                    complete=complete,
                    message="单文件模型" if complete else "文件太小，可能是 Git LFS 指针或下载不完整",
                )
            )
    return models


def available_model_names(models_dir: str | Path, include_default: bool = False) -> list[str]:
    names = [model.name for model in discover_models(models_dir) if model.complete]
    if include_default and "stable-diffusion-v1-5" not in names:
        names.append("stable-diffusion-v1-5")
    return names


def resolve_model(models_dir: str | Path, model_name: str) -> ModelInfo:
    for model in discover_models(models_dir):
        if model.name == model_name:
            if not model.complete:
                raise FileNotFoundError(f"模型不完整：{model.name}。{model.message}")
            return model
    raise FileNotFoundError(f"未找到模型：{model_name}")


def is_complete_diffusers_model(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or not (path / "model_index.json").exists():
        return False
    for folder_name in ("unet", "vae", "text_encoder"):
        folder = path / folder_name
        if not folder.exists():
            return False
        if not any(folder.glob("*.bin")) and not any(folder.glob("*.safetensors")):
            return False
    return True
