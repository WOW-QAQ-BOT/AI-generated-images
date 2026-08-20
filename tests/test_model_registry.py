from inference.model_registry import available_model_names, discover_models, resolve_model


def _complete_model(root, name="complete-model"):
    model = root / name
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    for folder_name in ("unet", "vae", "text_encoder"):
        folder = model / folder_name
        folder.mkdir()
        (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    return model


def test_available_names_are_local_only_by_default(tmp_path):
    assert available_model_names(tmp_path) == []
    assert "stable-diffusion-v1-5" not in available_model_names(tmp_path)


def test_discovers_complete_diffusers_folder(tmp_path):
    _complete_model(tmp_path, "anything-v5")

    models = discover_models(tmp_path)

    assert models[0].name == "anything-v5"
    assert models[0].kind == "diffusers"
    assert models[0].complete is True
    assert "anything-v5" in available_model_names(tmp_path, include_default=False)


def test_hides_incomplete_diffusers_folder_from_available_names(tmp_path):
    model = tmp_path / "broken"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / "unet").mkdir()
    (model / "unet" / "config.json").write_text("{}", encoding="utf-8")

    models = discover_models(tmp_path)

    assert models[0].complete is False
    assert "broken" not in available_model_names(tmp_path, include_default=False)


def test_discovers_large_single_file_checkpoint(tmp_path):
    checkpoint = tmp_path / "dream.safetensors"
    checkpoint.write_bytes(b"0" * (1024 * 1024 + 1))

    model = resolve_model(tmp_path, "dream")

    assert model.kind == "single_file"
    assert model.complete is True


def test_rejects_tiny_single_file_checkpoint(tmp_path):
    checkpoint = tmp_path / "pointer.safetensors"
    checkpoint.write_text("version https://git-lfs.github.com/spec/v1", encoding="utf-8")

    models = discover_models(tmp_path)

    assert models[0].complete is False
    assert "pointer" not in available_model_names(tmp_path, include_default=False)
