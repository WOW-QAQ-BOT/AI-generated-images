from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, OSError):
        return False


@dataclass
class LoRATrainingConfig:
    """LoRA training config tuned for RTX 3060 6GB."""

    pretrained_model: str = "models/anything-v5"
    output_dir: str = "./lora_output"
    resolution: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    seed: int = 42
    fp16: bool = True
    train_text_encoder: bool = False
    rank: int = 8


class LoRATrainer:
    """LoRA training launcher with dataset validation and command preview."""

    def __init__(
        self,
        config: Optional[LoRATrainingConfig] = None,
        runner: Callable = subprocess.run,
        cuda_probe: Callable[[], bool] = cuda_is_available,
    ):
        self.config = config or LoRATrainingConfig()
        self._runner = runner
        self._cuda_probe = cuda_probe

    def resolve_dataset_dir(self, train_data_dir: str) -> Path:
        """Resolve a common single nested directory produced by ZIP extraction."""
        root = Path(train_data_dir)
        if not root.is_dir():
            return root
        if any(path.suffix.lower() in IMAGE_EXTENSIONS for path in root.iterdir() if path.is_file()):
            return root
        candidates = [
            directory
            for directory in root.rglob("*")
            if directory.is_dir()
            and any(
                path.suffix.lower() in IMAGE_EXTENSIONS
                for path in directory.iterdir()
                if path.is_file()
            )
        ]
        return candidates[0] if len(candidates) == 1 else root

    def validate_dataset(self, train_data_dir: str, caption_extension: str = ".txt") -> list[str]:
        root = self.resolve_dataset_dir(train_data_dir)
        errors: list[str] = []
        if not root.exists() or not root.is_dir():
            return ["训练集路径不存在或不是文件夹。"]

        images = [path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            errors.append("训练集里没有图片文件。")

        missing_captions = [path.name for path in images if not path.with_suffix(caption_extension).exists()]
        if missing_captions:
            sample = ", ".join(missing_captions[:5])
            errors.append(f"有 {len(missing_captions)} 张图片缺少同名描述文件：{sample}")

        if self.config.resolution > 768:
            errors.append("RTX 3060 6GB 不建议训练分辨率超过 768。")
        if self.config.batch_size != 1:
            errors.append("RTX 3060 6GB 建议 batch_size 固定为 1。")
        if self.config.train_text_encoder:
            errors.append("RTX 3060 6GB 不建议训练 text encoder，容易显存不足。")
        if not 0 < self.config.learning_rate <= 0.001:
            errors.append("学习率必须大于 0 且不超过 0.001；风格 LoRA 推荐 0.0001。")
        return errors

    def train(
        self,
        train_data_dir: str,
        caption_extension: str = ".txt",
        output_name: str = "my_lora",
    ):
        errors = self.validate_dataset(train_data_dir, caption_extension)
        if errors:
            raise ValueError("\n".join(errors))
        if not self._cuda_probe():
            raise RuntimeError(
                "CUDA 不可用：当前 PyTorch 可能是 CPU 版本。"
                "请安装支持 CUDA 的 PyTorch 后再开始训练。"
            )

        dataset_dir = self.resolve_dataset_dir(train_data_dir)
        output_dir = os.path.join(self.config.output_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)
        cmd = self.build_training_command(str(dataset_dir), output_name, as_list=True)
        result = self._runner(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "没有错误输出").strip()
            raise RuntimeError(f"训练进程退出码 {result.returncode}：\n{details}")
        return output_dir

    def build_training_command(self, train_data_dir: str, output_name: str = "my_lora", as_list: bool = False):
        output_dir = os.path.join(self.config.output_dir, output_name)
        dataset_dir = self.resolve_dataset_dir(train_data_dir)
        script_path = Path(__file__).with_name("train_text_to_image_lora.py").resolve()
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=1",
            "--num_machines=1",
            f"--mixed_precision={'fp16' if self.config.fp16 else 'no'}",
            "--dynamo_backend=no",
            str(script_path),
            f"--pretrained_model_name_or_path={self.config.pretrained_model}",
            f"--train_data_dir={dataset_dir}",
            f"--output_dir={output_dir}",
            f"--resolution={self.config.resolution}",
            f"--train_batch_size={self.config.batch_size}",
            f"--gradient_accumulation_steps={self.config.gradient_accumulation_steps}",
            f"--learning_rate={self.config.learning_rate}",
            f"--max_train_steps={self.config.max_train_steps}",
            "--lr_scheduler=cosine",
            f"--rank={self.config.rank}",
            "--gradient_checkpointing",
        ]
        if self.config.fp16:
            cmd.append("--mixed_precision=fp16")
        if self.config.train_text_encoder:
            cmd.append("--train_text_encoder")
        if self.config.seed:
            cmd.append(f"--seed={self.config.seed}")
        if as_list:
            return cmd
        return " ".join(cmd)

    def get_training_command(self, train_data_dir: str, output_name: str = "my_lora"):
        return self.build_training_command(train_data_dir, output_name, as_list=False)
