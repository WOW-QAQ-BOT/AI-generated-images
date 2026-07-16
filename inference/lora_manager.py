import os
from typing import List
from peft import PeftModel


class LoRAManager:
    """LoRA 权重管理器，负责加载/卸载/切换 LoRA"""

    def __init__(self, pipe):
        self.pipe = pipe
        self.active_loras: dict[str, str] = {}  # name -> path

    @staticmethod
    def list_available_loras(lora_dir: str = "./lora_output") -> List[str]:
        """列出所有可用的 LoRA 权重"""
        if not os.path.exists(lora_dir):
            return []
        return [
            d for d in os.listdir(lora_dir)
            if os.path.isdir(os.path.join(lora_dir, d))
        ]

    def load_lora(self, name: str, lora_path: str):
        """加载 LoRA 权重到 pipeline"""
        if name in self.active_loras:
            print(f"LoRA '{name}' already loaded, skipping")
            return

        try:
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet,
                lora_path,
            )
            self.active_loras[name] = lora_path
            print(f"LoRA '{name}' loaded from {lora_path}")
        except Exception as e:
            print(f"Failed to load LoRA: {e}")

    def unload_lora(self, name: str):
        """卸载指定 LoRA"""
        if name not in self.active_loras:
            print(f"LoRA '{name}' not loaded")
            return
        del self.active_loras[name]
        print(f"Unload LoRA '{name}': reload base model to remove adapter")

    def get_active_loras(self) -> List[str]:
        """获取当前已加载的 LoRA 列表"""
        return list(self.active_loras.keys())
