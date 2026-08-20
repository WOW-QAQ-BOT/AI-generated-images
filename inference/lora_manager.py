from __future__ import annotations

from pathlib import Path

from .config import LORA_DIR


class LoRAManager:
    """Small wrapper around Diffusers LoRA loading.

    Diffusers already knows how to merge/unmerge adapters. This class keeps the
    UI state honest and makes failures visible instead of silently ignoring them.
    """

    def __init__(self, pipe, lora_dir: str | Path = LORA_DIR):
        self.pipe = pipe
        self.lora_dir = Path(lora_dir)
        self.active_name: str | None = None
        self.active_path: Path | None = None
        self.active_weight: float | None = None

    @staticmethod
    def list_available_loras(lora_dir: str | Path = LORA_DIR) -> list[str]:
        root = Path(lora_dir)
        if not root.exists():
            return []
        names = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and _looks_like_lora(child):
                names.append(child.name)
        return names

    @staticmethod
    def resolve_lora(name: str, lora_dir: str | Path = LORA_DIR) -> Path:
        path = Path(lora_dir) / name
        if not path.exists() or not _looks_like_lora(path):
            raise FileNotFoundError(f"未找到 LoRA：{name}")
        return path

    def apply(self, name: str | None, weight: float = 0.8) -> None:
        if not name:
            self.clear()
            return
        path = self.resolve_lora(name, self.lora_dir)
        if self.active_name == name and self.active_weight == weight:
            return
        self.clear()
        self.pipe.load_lora_weights(str(path), adapter_name=name)
        self.pipe.set_adapters([name], adapter_weights=[float(weight)])
        self.active_name = name
        self.active_path = path
        self.active_weight = float(weight)

    def clear(self) -> None:
        if self.active_name is None:
            return
        if hasattr(self.pipe, "delete_adapters"):
            self.pipe.delete_adapters([self.active_name])
        elif hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()
        self.active_name = None
        self.active_path = None
        self.active_weight = None


def _looks_like_lora(path: Path) -> bool:
    names = {child.name for child in path.iterdir()} if path.exists() else set()
    return bool(
        {"adapter_config.json", "adapter_model.safetensors"} <= names
        or {"pytorch_lora_weights.safetensors"} & names
        or (path / "unet_lora").exists()
    )
