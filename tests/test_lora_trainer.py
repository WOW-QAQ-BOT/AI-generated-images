from pathlib import Path
from subprocess import CompletedProcess

import pytest

from trainer.lora_trainer import LoRATrainer, LoRATrainingConfig
from trainer.train_text_to_image_lora import local_weight_loading_options


def test_validate_dataset_reports_missing_folder(tmp_path):
    trainer = LoRATrainer()

    errors = trainer.validate_dataset(str(tmp_path / "missing"))

    assert errors == ["训练集路径不存在或不是文件夹。"]


def test_validate_dataset_reports_missing_captions(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    trainer = LoRATrainer()

    errors = trainer.validate_dataset(str(tmp_path))

    assert "缺少同名描述文件" in errors[0]


def test_training_command_uses_safe_3060_defaults(tmp_path):
    trainer = LoRATrainer(LoRATrainingConfig(pretrained_model="model", max_train_steps=10, rank=4))

    command = trainer.get_training_command(str(tmp_path), "style")

    assert "--train_batch_size=1" in command
    assert "--gradient_accumulation_steps=4" in command
    assert "--mixed_precision=fp16" in command
    assert "--fp16" not in command
    assert "--train_text_encoder" not in command
    assert "--num_processes=1" in command
    assert "--num_machines=1" in command
    assert "--dynamo_backend=no" in command


def test_nested_single_dataset_directory_is_resolved(tmp_path):
    nested = tmp_path / "1_qizhu"
    nested.mkdir()
    (nested / "auto_00000.png").write_bytes(b"image")
    (nested / "auto_00000.txt").write_text("qizhu_style, 1girl", encoding="utf-8")

    trainer = LoRATrainer()

    assert trainer.resolve_dataset_dir(str(tmp_path)) == nested
    assert trainer.validate_dataset(str(tmp_path)) == []


def test_validation_rejects_dangerous_learning_rate(tmp_path):
    (tmp_path / "a.png").write_bytes(b"image")
    (tmp_path / "a.txt").write_text("style", encoding="utf-8")
    trainer = LoRATrainer(LoRATrainingConfig(learning_rate=0.05))

    errors = trainer.validate_dataset(str(tmp_path))

    assert any("学习率" in error and "0.001" in error for error in errors)


def test_training_command_uses_project_script_and_local_image_folder(tmp_path):
    trainer = LoRATrainer(LoRATrainingConfig(pretrained_model="models/anything-v5"))

    command = trainer.build_training_command(str(tmp_path), "style", as_list=True)

    script = next(Path(part) for part in command if part.endswith("train_text_to_image_lora.py"))
    assert script.name == "train_text_to_image_lora.py"
    assert script.is_file()
    assert f"--train_data_dir={tmp_path}" in command
    assert "--train_batch_size=1" in command
    assert "--mixed_precision=fp16" in command
    assert not any(part.startswith("--dataset_name=") for part in command)


def test_default_training_model_is_the_local_anything_v5():
    trainer = LoRATrainer()

    command = trainer.build_training_command("train_data", "style", as_list=True)

    assert "--pretrained_model_name_or_path=models/anything-v5" in command


def test_training_failure_reports_subprocess_output(tmp_path):
    (tmp_path / "a.png").write_bytes(b"image")
    (tmp_path / "a.txt").write_text("style", encoding="utf-8")

    def failing_runner(*args, **kwargs):
        return CompletedProcess(args[0], 1, stdout="loading model", stderr="CUDA out of memory")

    trainer = LoRATrainer(runner=failing_runner, cuda_probe=lambda: True)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        trainer.train(str(tmp_path), output_name="style")


def test_training_stops_before_launch_when_cuda_is_unavailable(tmp_path):
    (tmp_path / "a.png").write_bytes(b"image")
    (tmp_path / "a.txt").write_text("style", encoding="utf-8")
    launched = False

    def runner(*args, **kwargs):
        nonlocal launched
        launched = True

    trainer = LoRATrainer(runner=runner, cuda_probe=lambda: False)

    with pytest.raises(RuntimeError, match="CUDA.*PyTorch"):
        trainer.train(str(tmp_path), output_name="style")

    assert launched is False


def test_local_bin_weights_are_preferred_over_safetensors(tmp_path):
    unet_dir = tmp_path / "unet"
    unet_dir.mkdir()
    (unet_dir / "diffusion_pytorch_model.bin").write_bytes(b"weights")
    (unet_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    options = local_weight_loading_options(tmp_path, "unet")

    assert options == {"use_safetensors": False, "low_cpu_mem_usage": True}
