from inference.pipeline import InferencePipeline


def test_model_discovery_uses_project_models_dir(tmp_path):
    model = tmp_path / "anything-v5"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    for folder_name in ("unet", "vae", "text_encoder"):
        folder = model / folder_name
        folder.mkdir()
        (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    service = InferencePipeline(device="cpu", models_dir=tmp_path)

    assert service.get_available_models()[0] == "anything-v5"
    assert service.get_model_path("anything-v5") == str(model)


def test_pipeline_never_resolves_implicit_remote_default(tmp_path):
    service = InferencePipeline(device="cpu", models_dir=tmp_path)

    assert service.get_available_models() == []
    try:
        service.get_model_path("stable-diffusion-v1-5")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("remote fallback must not be available offline")


def test_local_stable_diffusion_name_resolves_to_its_local_path(tmp_path):
    model = tmp_path / "stable-diffusion-v1-5"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    for folder_name in ("unet", "vae", "text_encoder"):
        folder = model / folder_name
        folder.mkdir()
        (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    service = InferencePipeline(device="cpu", models_dir=tmp_path)

    assert service.get_model_path("stable-diffusion-v1-5") == str(model)


def test_incomplete_local_model_is_hidden_and_rejected(tmp_path):
    model = tmp_path / "anything-v5"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    (model / "unet").mkdir()
    (model / "unet" / "config.json").write_text("{}", encoding="utf-8")

    service = InferencePipeline(device="cpu", models_dir=tmp_path)

    assert "anything-v5" not in service.get_available_models()
    try:
        service.get_model_path("anything-v5")
    except FileNotFoundError as exc:
        assert "模型不完整" in str(exc)
    else:
        raise AssertionError("incomplete model should be rejected")


def test_cancel_sets_flag_without_loading_model(tmp_path):
    service = InferencePipeline(device="cpu", models_dir=tmp_path)

    service.cancel()

    assert service.cancel_requested is True
