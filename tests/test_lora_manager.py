from pathlib import Path

import pytest

from inference.lora_manager import LoRAManager


class FakePipe:
    def __init__(self):
        self.loaded = []
        self.adapters = []
        self.deleted = []

    def load_lora_weights(self, path, adapter_name):
        self.loaded.append((Path(path).name, adapter_name))

    def set_adapters(self, adapters, adapter_weights):
        self.adapters.append((adapters, adapter_weights))

    def delete_adapters(self, adapters):
        self.deleted.append(adapters)


def test_list_available_loras_filters_non_lora_folders(tmp_path):
    (tmp_path / "style-a").mkdir()
    (tmp_path / "style-a" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "style-a" / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "notes").mkdir()

    assert LoRAManager.list_available_loras(tmp_path) == ["style-a"]


def test_apply_lora_switches_adapter(tmp_path):
    lora = tmp_path / "style-a"
    lora.mkdir()
    (lora / "pytorch_lora_weights.safetensors").write_bytes(b"")
    pipe = FakePipe()
    manager = LoRAManager(pipe, lora_dir=tmp_path)

    manager.apply("style-a", 0.75)

    assert pipe.loaded == [("style-a", "style-a")]
    assert pipe.adapters == [(["style-a"], [0.75])]
    assert manager.active_name == "style-a"


def test_missing_lora_raises(tmp_path):
    manager = LoRAManager(FakePipe())

    with pytest.raises(FileNotFoundError):
        manager.resolve_lora("missing", tmp_path)
