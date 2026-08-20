from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class LoRATrainingConfig:
    """LoRA training config tuned for RTX 3060 6GB."""

    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "./lora_output"
    resolution: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    seed: int = 42
    fp16: bool = True
    train_text_encoder: bool = False
    rank: int = 4


class LoRATrainer:
    """LoRA training launcher with dataset validation and command preview."""

    def __init__(self, config: Optional[LoRATrainingConfig] = None):
        self.config = config or LoRATrainingConfig()

    def validate_dataset(self, train_data_dir: str, caption_extension: str = ".txt") -> list[str]:
        root = Path(train_data_dir)
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

        output_dir = os.path.join(self.config.output_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)
        cmd = self.build_training_command(train_data_dir, output_name, as_list=True)
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with return code {result.returncode}")
        return output_dir

    def build_training_command(self, train_data_dir: str, output_name: str = "my_lora", as_list: bool = False):
        output_dir = os.path.join(self.config.output_dir, output_name)
        cmd = [
            "accelerate",
            "launch",
            "diffusers/examples/text_to_image/train_text_to_image_lora.py",
            f"--pretrained_model_name_or_path={self.config.pretrained_model}",
            f"--dataset_name={train_data_dir}",
            "--caption_column=file_name",
            f"--output_dir={output_dir}",
            f"--resolution={self.config.resolution}",
            f"--batch_size={self.config.batch_size}",
            f"--gradient_accumulation_steps={self.config.gradient_accumulation_steps}",
            f"--learning_rate={self.config.learning_rate}",
            f"--max_train_steps={self.config.max_train_steps}",
            "--lr_scheduler=cosine",
            f"--lora_r={self.config.rank}",
        ]
        if self.config.fp16:
            cmd.append("--fp16")
        if self.config.train_text_encoder:
            cmd.append("--train_text_encoder")
        if self.config.seed:
            cmd.append(f"--seed={self.config.seed}")
        if as_list:
            return cmd
        return " ".join(cmd)

    def get_training_command(self, train_data_dir: str, output_name: str = "my_lora"):
        return self.build_training_command(train_data_dir, output_name, as_list=False)
