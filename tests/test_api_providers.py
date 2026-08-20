import base64
import json

import pytest

from inference.api_providers.base import ApiImageRequest, ApiProviderError
from inference.api_providers.gemini_provider import GeminiProvider
from inference.api_providers.openai_compatible_provider import OpenAICompatibleProvider
from inference.api_providers.openai_provider import OpenAIProvider
from inference.api_providers.registry import (
    create_provider,
    provider_defaults,
    provider_names,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, content=b"", headers=None):
        self._payload = payload
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.text = json.dumps(payload or {}, ensure_ascii=False)

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=65536):
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self.responses.pop(0)


def test_registry_returns_editable_suggestions_without_model_allowlist():
    assert provider_names() == ["OpenAI", "Gemini", "OpenAI 兼容接口"]
    assert provider_defaults("OpenAI").model == "gpt-image-2"
    assert provider_defaults("Gemini").model == "gemini-3.1-flash-image"

    compatible = provider_defaults("OpenAI 兼容接口")

    assert compatible.base_url_visible is True
    assert compatible.base_url == "http://127.0.0.1:8000/v1"


def test_registry_constructs_fresh_stateless_provider_instances():
    first = create_provider("OpenAI")
    second = create_provider("OpenAI")

    assert isinstance(first, OpenAIProvider)
    assert isinstance(create_provider("Gemini"), GeminiProvider)
    assert isinstance(create_provider("OpenAI 兼容接口"), OpenAICompatibleProvider)
    assert first is not second
    assert first.__dict__ == {}


def test_openai_passes_free_form_model_and_decodes_every_image():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "created": 1784900000,
                    "background": "opaque",
                    "data": [
                        {
                            "b64_json": base64.b64encode(b"image-one").decode(),
                            "revised_prompt": None,
                            "url": None,
                        },
                        {
                            "b64_json": base64.b64encode(b"image-two").decode(),
                            "revised_prompt": "revised",
                            "url": None,
                        },
                    ],
                    "output_format": "png",
                    "quality": "high",
                    "size": "1536x1024",
                }
            )
        ]
    )
    provider = OpenAIProvider()
    request = ApiImageRequest(
        api_key="secret-value",
        model="vendor/free-form-model:beta",
        prompt="cat astronaut",
        negative_prompt="watermark",
        size="1536x1024",
        quality="high",
        count=2,
        output_format="png",
    )

    results = provider.generate(request, session=session)

    call = session.calls[0]
    assert call["url"] == "https://api.openai.com/v1/images/generations"
    assert call["json"]["model"] == "vendor/free-form-model:beta"
    assert call["json"]["prompt"] == "cat astronaut\n\nAvoid: watermark"
    assert call["json"]["output_format"] == "png"
    assert call["headers"]["Authorization"] == "Bearer secret-value"
    assert [result.image_bytes for result in results] == [b"image-one", b"image-two"]
    assert provider.__dict__ == {}


def test_openai_translates_legacy_dalle_response_format():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "created": 1784900000,
                    "data": [
                        {
                            "b64_json": base64.b64encode(b"dalle-image").decode(),
                            "revised_prompt": "revised",
                            "url": None,
                        }
                    ],
                }
            )
        ]
    )
    request = ApiImageRequest(
        api_key="secret",
        model="dall-e-3",
        prompt="vintage train",
        output_format="jpeg",
    )

    OpenAIProvider().generate(request, session=session)

    payload = session.calls[0]["json"]
    assert payload["response_format"] == "b64_json"
    assert "output_format" not in payload
    assert payload["quality"] == "standard"


def test_openai_translates_generic_high_quality_for_dalle3():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "created": 1784900000,
                    "data": [
                        {
                            "b64_json": base64.b64encode(b"dalle-hd").decode(),
                            "revised_prompt": None,
                            "url": None,
                        }
                    ],
                }
            )
        ]
    )
    request = ApiImageRequest(
        api_key="secret",
        model="dall-e-3",
        prompt="mountain",
        quality="high",
    )

    OpenAIProvider().generate(request, session=session)

    assert session.calls[0]["json"]["quality"] == "hd"


def test_openai_repeats_dalle3_requests_and_reports_ignored_output_format():
    payload = {
        "created": 1784900000,
        "data": [
            {
                "b64_json": base64.b64encode(b"dalle-image").decode(),
                "revised_prompt": None,
                "url": None,
            }
        ],
    }
    session = FakeSession([FakeResponse(payload), FakeResponse(payload)])
    request = ApiImageRequest(
        api_key="secret",
        model="dall-e-3",
        prompt="mountain",
        count=2,
        output_format="webp",
    )

    results = OpenAIProvider().generate(request, session=session)

    assert len(session.calls) == 2
    assert all(call["json"]["n"] == 1 for call in session.calls)
    assert len(results) == 2
    assert all("PNG" in " ".join(result.notes) for result in results)


def test_openai_compatible_normalizes_base_url_and_downloads_same_origin_url():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "created": 1784900000,
                    "data": [
                        {
                            "b64_json": None,
                            "revised_prompt": None,
                            "url": "https://images.example/generated.png",
                        }
                    ],
                }
            ),
            FakeResponse(content=b"downloaded-image"),
        ]
    )
    provider = OpenAICompatibleProvider()
    request = ApiImageRequest(
        api_key="compatible-secret",
        model="flux-custom/latest",
        prompt="misty forest",
        base_url="https://images.example/v1/",
    )

    results = provider.generate(request, session=session)

    post_call, get_call = session.calls
    assert post_call["url"] == "https://images.example/v1/images/generations"
    assert post_call["json"]["model"] == "flux-custom/latest"
    assert post_call["json"]["response_format"] == "b64_json"
    assert get_call["url"] == "https://images.example/generated.png"
    assert "headers" not in get_call
    assert get_call["stream"] is True
    assert get_call["allow_redirects"] is False
    assert results[0].image_bytes == b"downloaded-image"
    assert "输出格式" in " ".join(results[0].notes)
    assert provider.__dict__ == {}


def test_openai_compatible_rejects_cross_origin_image_url():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "data": [
                        {
                            "b64_json": None,
                            "url": "http://127.0.0.1/private-metadata",
                        }
                    ]
                }
            )
        ]
    )
    request = ApiImageRequest(
        api_key="compatible-secret",
        model="custom",
        prompt="forest",
        base_url="https://images.example/v1",
    )

    with pytest.raises(ApiProviderError, match="同源"):
        OpenAICompatibleProvider().generate(request, session=session)

    assert len(session.calls) == 1


def test_url_download_rejects_declared_image_over_size_limit():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "data": [
                        {
                            "b64_json": None,
                            "url": "https://images.example/generated.png",
                        }
                    ]
                }
            ),
            FakeResponse(
                content=b"",
                headers={"Content-Length": str(33 * 1024 * 1024)},
            ),
        ]
    )
    request = ApiImageRequest(
        api_key="compatible-secret",
        model="custom",
        prompt="forest",
        base_url="https://images.example/v1",
    )

    with pytest.raises(ApiProviderError, match="32MB"):
        OpenAICompatibleProvider().generate(request, session=session)


def test_openai_compatible_rejects_credentials_embedded_in_base_url():
    request = ApiImageRequest(
        api_key="secret",
        model="custom",
        prompt="forest",
        base_url="https://user:password@images.example/v1",
    )

    with pytest.raises(ApiProviderError, match="用户名"):
        OpenAICompatibleProvider().generate(request, session=FakeSession([]))


def test_gemini_url_encodes_free_form_model_and_parses_inline_image():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": base64.b64encode(b"gemini-image").decode(),
                                        }
                                    }
                                ],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                            "index": 0,
                        }
                    ]
                }
            )
        ]
    )
    provider = GeminiProvider()
    request = ApiImageRequest(
        api_key="gemini-secret",
        model="gemini-custom/image beta",
        prompt="paper dragon",
        size="2048x1152",
    )

    results = provider.generate(request, session=session)

    call = session.calls[0]
    assert call["url"].endswith(
        "/v1/models/gemini-custom%2Fimage%20beta:generateContent"
    )
    assert call["headers"]["x-goog-api-key"] == "gemini-secret"
    assert call["json"]["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert "responseFormat" not in call["json"]["generationConfig"]
    assert results[0].image_bytes == b"gemini-image"
    assert results[0].mime_type == "image/png"
    assert results[0].notes
    assert provider.__dict__ == {}


def test_gemini_3_maps_requested_size_and_repeats_for_count():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(b"gemini-3-image").decode(),
                            }
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ]
    }
    session = FakeSession([FakeResponse(payload), FakeResponse(payload)])
    request = ApiImageRequest(
        api_key="secret",
        model="gemini-3.1-flash-image",
        prompt="glass city",
        size="2048x1152",
        count=2,
    )

    results = GeminiProvider().generate(request, session=session)

    image_config = session.calls[0]["json"]["generationConfig"]["responseFormat"]["image"]
    assert image_config == {"aspectRatio": "16:9", "imageSize": "2K"}
    assert len(session.calls) == 2
    assert len(results) == 2
