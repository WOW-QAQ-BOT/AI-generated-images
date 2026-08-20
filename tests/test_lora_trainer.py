from trainer.lora_trainer import LoRATrainer, LoRATrainingConfig


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

    assert "--batch_size=1" in command
    assert "--gradient_accumulation_steps=4" in command
    assert "--fp16" in command
    assert "--train_text_encoder" not in command
