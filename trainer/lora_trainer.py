import os
import subprocess
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRATrainingConfig:
    """LoRA 训练配置（适配 RTX 3060）"""
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
    """LoRA 微调训练器"""

    def __init__(self, config: Optional[LoRATrainingConfig] = None):
        self.config = config or LoRATrainingConfig()

    def train(
        self,
        train_data_dir: str,
        caption_extension: str = ".txt",
        output_name: str = "my_lora",
    ):
        """
        执行 LoRA 训练

        Args:
            train_data_dir: 训练图片文件夹路径，图片和同名文本文件配对
            caption_extension: 描述文件扩展名，默认 .txt
            output_name: 输出 LoRA 名称
        """
        output_dir = os.path.join(self.config.output_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "accelerate", "launch",
            "diffusers/examples/text_to_image/train_text_to_image_lora.py",
            "--pretrained_model_name_or_path=" + self.config.pretrained_model,
            "--dataset_name=" + train_data_dir,
            "--caption_column=file_name",
            "--output_dir=" + output_dir,
            "--resolution=" + str(self.config.resolution),
            "--batch_size=" + str(self.config.batch_size),
            "--gradient_accumulation_steps=" + str(self.config.gradient_accumulation_steps),
            "--learning_rate=" + str(self.config.learning_rate),
            "--max_train_steps=" + str(self.config.max_train_steps),
            "--lr_scheduler=" + "cosine",
            "--lora_r=" + str(self.config.rank),
        ]

        if self.config.fp16:
            cmd.append("--fp16")

        if self.config.seed:
            cmd.append("--seed=" + str(self.config.seed))

        print("Running command:")
        print(" ".join(cmd))

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with return code {result.returncode}")

        print(f"LoRA saved to {output_dir}")
        return output_dir

    def get_training_command(self, train_data_dir: str, output_name: str = "my_lora"):
        """
        返回训练命令（供手动执行或调试用）
        """
        output_dir = os.path.join(self.config.output_dir, output_name)
        return (
            f"accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py "
            f"--pretrained_model_name_or_path={self.config.pretrained_model} "
            f"--dataset_name={train_data_dir} "
            f"--caption_column=file_name "
            f"--output_dir={output_dir} "
            f"--resolution={self.config.resolution} "
            f"--batch_size={self.config.batch_size} "
            f"--gradient_accumulation_steps={self.config.gradient_accumulation_steps} "
            f"--learning_rate={self.config.learning_rate} "
            f"--max_train_steps={self.config.max_train_steps} "
            f"--lr_scheduler=cosine "
            f"--lora_r={self.config.rank} "
            + ("--fp16 " if self.config.fp16 else "")
            + (f"--seed={self.config.seed} " if self.config.seed else "")
        )
