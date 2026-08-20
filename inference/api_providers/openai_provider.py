from __future__ import annotations

import base64
import binascii
from typing import Any
from urllib.parse import urlsplit

from .base import (
    ApiImageRequest,
    ApiImageResult,
    ApiProviderError,
    compose_prompt,
    raise_for_api_error,
)


OPENAI_IMAGES_ENDPOINT = "https://api.openai.com/v1/images/generations"
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def _http_session(session: Any | None) -> Any:
    if session is not None:
        return session
    try:
        import requests
    except ImportError as exc:
        raise ApiProviderError(
            "缺少 requests 依赖，请运行 python -m pip install -r requirements.txt。"
        ) from exc
    return requests.Session()


def _mime_type(output_format: str) -> str:
    if output_format == "jpeg":
        return "image/jpeg"
    if output_format == "webp":
        return "image/webp"
    return "image/png"


def _decode_base64_image(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ApiProviderError("API 返回了空的图片数据。")
    try:
        image_bytes = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ApiProviderError("API 返回的图片数据不是有效的 Base64。") from exc
    if not image_bytes:
        raise ApiProviderError("API 返回了空的图片数据。")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ApiProviderError("API 返回的单张图片超过 32MB，已停止处理。")
    return image_bytes


def _response_payload(response: Any, secrets: list[str]) -> dict[str, Any]:
    raise_for_api_error(response, secrets)
    try:
        payload = response.json()
    except Exception as exc:
        raise ApiProviderError("API 返回的 JSON 无法解析。") from exc
    if not isinstance(payload, dict):
        raise ApiProviderError("API 返回格式无效：顶层内容不是对象。")
    return payload


def _url_origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(str(url))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ApiProviderError("API 返回了不安全或无效的图片地址。")
    if parsed.username or parsed.password:
        raise ApiProviderError("API 返回的图片地址不能包含用户名或密码。")
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise ApiProviderError("API 返回的图片地址端口无效。") from exc
    return parsed.scheme.lower(), parsed.hostname.lower(), port


def _read_limited_image_response(response: Any) -> bytes:
    headers = getattr(response, "headers", {}) or {}
    declared_length = headers.get("Content-Length") or headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > MAX_IMAGE_BYTES:
                raise ApiProviderError(
                    "API 返回的单张图片超过 32MB，已停止处理。"
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        source = iterator(chunk_size=64 * 1024)
    else:
        source = (bytes(getattr(response, "content", b"")),)
    for chunk in source:
        if not chunk:
            continue
        chunk = bytes(chunk)
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise ApiProviderError("API 返回的单张图片超过 32MB，已停止处理。")
        chunks.append(chunk)
    image_bytes = b"".join(chunks)
    if not image_bytes:
        raise ApiProviderError("API 图片下载结果为空。")
    return image_bytes


def _download_image(
    url: object,
    session: Any,
    request: ApiImageRequest,
    allowed_origin: tuple[str, str, int] | None,
) -> bytes:
    if not isinstance(url, str) or not url:
        raise ApiProviderError("API 返回了无效的图片地址。")
    origin = _url_origin(url)
    if allowed_origin is not None:
        if origin != allowed_origin:
            raise ApiProviderError(
                "兼容接口返回了非同源图片地址，已阻止自动下载。"
            )
    elif origin[0] != "https":
        raise ApiProviderError("OpenAI 返回了非 HTTPS 图片地址，已阻止自动下载。")

    try:
        response = session.get(
            url,
            timeout=(10, 180),
            stream=True,
            allow_redirects=False,
        )
    except Exception as exc:
        raise ApiProviderError("下载 API 图片失败，请检查网络连接。") from exc
    try:
        status_code = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status_code < 400:
            raise ApiProviderError("API 图片下载发生重定向，已按安全策略停止。")
        raise_for_api_error(response, [request.api_key])
        return _read_limited_image_response(response)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _parse_openai_images(
    response: Any,
    request: ApiImageRequest,
    session: Any,
    *,
    allowed_image_origin: tuple[str, str, int] | None = None,
    notes: tuple[str, ...] = (),
) -> list[ApiImageResult]:
    payload = _response_payload(response, [request.api_key])
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ApiProviderError("API 返回结果中没有图片。")

    output_format = str(payload.get("output_format") or request.output_format).lower()
    results: list[ApiImageResult] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            image_bytes = _decode_base64_image(item["b64_json"])
        elif item.get("url"):
            image_bytes = _download_image(
                item["url"],
                session,
                request,
                allowed_image_origin,
            )
        else:
            continue
        results.append(
            ApiImageResult(
                image_bytes=image_bytes,
                mime_type=_mime_type(output_format),
                revised_prompt=item.get("revised_prompt"),
                notes=notes,
            )
        )
    if not results:
        raise ApiProviderError("API 返回结果中没有可读取的图片数据。")
    return results


class OpenAIProvider:
    def generate(
        self,
        request: ApiImageRequest,
        session: Any | None = None,
    ) -> list[ApiImageResult]:
        normalized = request.normalized()
        http = _http_session(session)
        payload: dict[str, Any] = {
            "model": normalized.model,
            "prompt": compose_prompt(normalized),
            "n": normalized.count,
            "size": normalized.size,
            "quality": normalized.quality,
        }
        model_lower = normalized.model.lower()
        notes: list[str] = []
        if model_lower.startswith("dall-e-"):
            payload["response_format"] = "b64_json"
            if normalized.output_format != "png":
                notes.append(
                    "DALL-E 接口不应用 JPEG/WebP 选项，已按实际返回的 PNG 保存。"
                )
            if model_lower == "dall-e-3" and payload["quality"] == "high":
                payload["quality"] = "hd"
            elif payload["quality"] not in {"standard", "hd"}:
                payload["quality"] = "standard"
        else:
            payload["output_format"] = normalized.output_format
        try:
            request_count = normalized.count if model_lower == "dall-e-3" else 1
            if model_lower == "dall-e-3":
                payload["n"] = 1
                if normalized.count > 1:
                    notes.append(
                        "DALL-E 3 每次仅支持 1 张，已自动分次完成请求。"
                    )
            results: list[ApiImageResult] = []
            for _ in range(request_count):
                response = http.post(
                    OPENAI_IMAGES_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {normalized.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=(10, 180),
                )
                results.extend(
                    _parse_openai_images(
                        response,
                        normalized,
                        http,
                        notes=tuple(notes),
                    )
                )
            return results
        except ApiProviderError:
            raise
        except Exception as exc:
            raise ApiProviderError(
                "连接 OpenAI 图片 API 失败，请检查网络、代理或服务状态。"
            ) from exc
