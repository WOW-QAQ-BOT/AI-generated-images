from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from .base import (
    ApiImageRequest,
    ApiImageResult,
    ApiProviderError,
    compose_prompt,
)
from .openai_provider import _http_session, _parse_openai_images, _url_origin


def _compatible_endpoint(base_url: str) -> str:
    base_url = str(base_url).strip()
    if not base_url:
        raise ApiProviderError("OpenAI 兼容接口必须填写 Base URL。")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ApiProviderError("Base URL 必须是完整的 http 或 https 地址。")
    if parsed.username or parsed.password:
        raise ApiProviderError("Base URL 不能包含用户名或密码。")
    if parsed.query or parsed.fragment:
        raise ApiProviderError("Base URL 不能包含查询参数或锚点。")
    return base_url.rstrip("/") + "/images/generations"


class OpenAICompatibleProvider:
    def generate(
        self,
        request: ApiImageRequest,
        session: Any | None = None,
    ) -> list[ApiImageResult]:
        normalized = request.normalized()
        endpoint = _compatible_endpoint(normalized.base_url)
        http = _http_session(session)
        payload = {
            "model": normalized.model,
            "prompt": compose_prompt(normalized),
            "n": normalized.count,
            "size": normalized.size,
            "quality": normalized.quality,
            "response_format": "b64_json",
        }
        notes = (
            "兼容接口未统一支持输出格式参数，已按实际返回的图片格式保存。",
        )
        try:
            response = http.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {normalized.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(10, 180),
            )
            return _parse_openai_images(
                response,
                normalized,
                http,
                allowed_image_origin=_url_origin(endpoint),
                notes=notes,
            )
        except ApiProviderError:
            raise
        except Exception as exc:
            raise ApiProviderError(
                "连接 OpenAI 兼容图片接口失败，请检查 Base URL、网络或服务状态。"
            ) from exc
