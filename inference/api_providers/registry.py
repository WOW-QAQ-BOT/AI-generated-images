from __future__ import annotations

from dataclasses import dataclass

from .base import ApiProviderError


OPENAI = "OpenAI"
GEMINI = "Gemini"
OPENAI_COMPATIBLE = "OpenAI 兼容接口"


@dataclass(frozen=True)
class ProviderDefaults:
    model: str
    base_url: str = ""
    base_url_visible: bool = False


_DEFAULTS = {
    OPENAI: ProviderDefaults(model="gpt-image-2"),
    GEMINI: ProviderDefaults(model="gemini-3.1-flash-image"),
    OPENAI_COMPATIBLE: ProviderDefaults(
        model="",
        base_url="http://127.0.0.1:8000/v1",
        base_url_visible=True,
    ),
}


def provider_names() -> list[str]:
    return list(_DEFAULTS)


def provider_defaults(name: str) -> ProviderDefaults:
    try:
        return _DEFAULTS[name]
    except KeyError as exc:
        raise ApiProviderError(f"不支持的 API 服务商：{name}") from exc


def create_provider(name: str):
    if name == OPENAI:
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    if name == GEMINI:
        from .gemini_provider import GeminiProvider

        return GeminiProvider()
    if name == OPENAI_COMPATIBLE:
        from .openai_compatible_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider()
    raise ApiProviderError(f"不支持的 API 服务商：{name}")
