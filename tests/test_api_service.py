import base64
import json

import pytest
from PIL import Image

from inference.api_providers.base import ApiImageRequest, ApiImageResult, ApiProviderError
from inference.api_service import ApiImageService
from inference.history import GenerationHistory


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
BROKEN_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2nXcAAAAASUVORK5CYII="
)


class FakeProvider:
    def generate(self, request, session=None):
        return [
            ApiImageResult(
                image_bytes=PNG_BYTES,
                mime_type="image/png",
                revised_prompt="revised lake",
                notes=("模型自动选择了部分参数。",),
            )
        ]


class BrokenImageProvider:
    def generate(self, request, session=None):
        return [ApiImageResult(image_bytes=BROKEN_PNG_BYTES)]


class EchoSecretProvider:
    def generate(self, request, session=None):
        return [
            ApiImageResult(
                image_bytes=PNG_BYTES,
                revised_prompt=f"provider echoed {request.api_key}",
                notes=(f"provider note {request.api_key}",),
            )
        ]


class FailingHistory(GenerationHistory):
    def append_many_atomic(self, records):
        raise OSError("private local path and raw operating-system detail")


def test_service_saves_valid_image_and_history_without_secrets(tmp_path):
    history = GenerationHistory(
        output_dir=tmp_path / "api",
        history_file=tmp_path / "history.jsonl",
        favorites_file=tmp_path / "favorites.json",
    )
    service = ApiImageService(
        history=history,
        provider_factory=lambda _name: FakeProvider(),
    )
    request = ApiImageRequest(
        api_key="never-persist-this-key",
        model="vendor/free-image-model",
        prompt="quiet lake",
        base_url="https://private-api.example/v1",
    )

    outcome = service.generate("OpenAI 兼容接口", request)

    assert len(outcome.paths) == 1
    assert outcome.paths[0].parent == tmp_path / "api"
    with Image.open(outcome.paths[0]) as image:
        assert image.size == (1, 1)

    persisted = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    assert "never-persist-this-key" not in persisted
    assert "private-api.example" not in persisted
    record = json.loads(persisted)
    assert record["settings"]["source"] == "api"
    assert record["settings"]["provider"] == "OpenAI 兼容接口"
    assert record["settings"]["model"] == "vendor/free-image-model"
    assert "模型自动选择了部分参数" in outcome.status


def test_service_converts_corrupt_image_parser_errors_to_safe_api_error(tmp_path):
    history = GenerationHistory(
        output_dir=tmp_path / "api",
        history_file=tmp_path / "history.jsonl",
        favorites_file=tmp_path / "favorites.json",
    )
    service = ApiImageService(
        history=history,
        provider_factory=lambda _name: BrokenImageProvider(),
    )
    request = ApiImageRequest(api_key="secret", model="model", prompt="prompt")

    with pytest.raises(ApiProviderError, match="有效图片"):
        service.generate("OpenAI", request)

    assert list((tmp_path / "api").glob("*")) == []
    assert not (tmp_path / "history.jsonl").exists()


def test_service_redacts_key_from_every_persisted_string_and_status(tmp_path):
    secret = "unit-never-write-this-anywhere"
    history = GenerationHistory(
        output_dir=tmp_path / "api",
        history_file=tmp_path / "history.jsonl",
        favorites_file=tmp_path / "favorites.json",
    )
    service = ApiImageService(
        history=history,
        provider_factory=lambda _name: EchoSecretProvider(),
    )
    request = ApiImageRequest(
        api_key=secret,
        model=f"vendor/{secret}",
        prompt=f"paint a lake {secret}",
        negative_prompt=f"avoid {secret}",
        output_format="jpeg",
    )

    outcome = service.generate("OpenAI 兼容接口", request)

    persisted = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    record = json.loads(persisted)
    assert secret not in persisted
    assert secret not in outcome.status
    assert record["settings"]["output_format"] == "png"
    assert record["settings"]["requested_output_format"] == "jpeg"
    assert "revised_prompt" not in record["settings"]


def test_service_rolls_back_image_and_returns_safe_history_write_error(tmp_path):
    history = FailingHistory(
        output_dir=tmp_path / "api",
        history_file=tmp_path / "history.jsonl",
        favorites_file=tmp_path / "favorites.json",
    )
    service = ApiImageService(
        history=history,
        provider_factory=lambda _name: FakeProvider(),
    )
    request = ApiImageRequest(api_key="secret", model="model", prompt="prompt")

    with pytest.raises(ApiProviderError, match="历史记录写入失败") as captured:
        service.generate("OpenAI", request)

    assert "private local path" not in str(captured.value)
    assert list((tmp_path / "api").glob("*")) == []


def test_service_returns_safe_local_output_write_error(tmp_path):
    history = GenerationHistory(
        output_dir=tmp_path / "api",
        history_file=tmp_path / "history.jsonl",
        favorites_file=tmp_path / "favorites.json",
    )
    history.output_dir.rmdir()
    history.output_dir.write_text("this path is now a file", encoding="utf-8")
    service = ApiImageService(
        history=history,
        provider_factory=lambda _name: FakeProvider(),
    )
    request = ApiImageRequest(api_key="secret", model="model", prompt="prompt")

    with pytest.raises(ApiProviderError, match="本地输出写入失败") as captured:
        service.generate("OpenAI", request)

    assert "File exists" not in str(captured.value)
    assert not (tmp_path / "history.jsonl").exists()
