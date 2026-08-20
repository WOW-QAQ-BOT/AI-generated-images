from __future__ import annotations

import math
import re
from typing import Any
from urllib.parse import quote

from .base import (
    ApiImageRequest,
    ApiImageResult,
    ApiProviderError,
    compose_prompt,
    raise_for_api_error,
)
from .openai_provider import _decode_base64_image, _http_session, _response_payload


GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1/models"
_ASPECT_RATIOS = (
    ("1:1", 1 / 1),
    ("1:4", 1 / 4),
    ("1:8", 1 / 8),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:1", 4 / 1),
    ("4:3", 4 / 3),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("8:1", 8 / 1),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
)


def _gemini_image_config(size: str) -> dict[str, str] | None:
    match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", str(size).strip().lower())
    if not match:
        return None
    width, height = (int(match.group(1)), int(match.group(2)))
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    aspect_ratio = min(
        _ASPECT_RATIOS,
        key=lambda item: abs(math.log(ratio / item[1])),
    )[0]
    longest_edge = max(width, height)
    if longest_edge <= 1536:
        image_size = "1K"
    elif longest_edge <= 2560:
        image_size = "2K"
    else:
        image_size = "4K"
    return {"aspectRatio": aspect_ratio, "imageSize": image_size}


def _parse_gemini_images(
    response: Any,
    request: ApiImageRequest,
    notes: tuple[str, ...],
) -> list[ApiImageResult]:
    payload = _response_payload(response, [request.api_key])
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ApiProviderError("Gemini 返回结果中没有候选图片。")

    results: list[ApiImageResult] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline, dict):
                continue
            data = inline.get("data")
            mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            results.append(
                ApiImageResult(
                    image_bytes=_decode_base64_image(data),
                    mime_type=str(mime_type),
                    notes=notes,
                )
            )
    if not results:
        raise ApiProviderError("Gemini 返回结果中没有可读取的图片数据。")
    return results


class GeminiProvider:
    def generate(
        self,
        request: ApiImageRequest,
        session: Any | None = None,
    ) -> list[ApiImageResult]:
        normalized = request.normalized()
        http = _http_session(session)
        encoded_model = quote(normalized.model, safe="")
        endpoint = f"{GEMINI_API_ROOT}/{encoded_model}:generateContent"
        generation_config: dict[str, Any] = {"responseModalities": ["IMAGE"]}
        notes: list[str] = []
        if normalized.model.lower().startswith("gemini-3"):
            image_config = _gemini_image_config(normalized.size)
            if image_config:
                generation_config["responseFormat"] = {"image": image_config}
            else:
                notes.append("填写的尺寸无法转换，Gemini 将自动选择图片尺寸。")
        else:
            notes.append("该模型未应用自定义尺寸，由 Gemini 模型决定输出尺寸。")
        if normalized.quality != "auto":
            notes.append("Gemini 接口不使用通用质量参数，已由模型自动控制质量。")
        if normalized.output_format != "png":
            notes.append("Gemini 返回格式由模型决定，保存时会识别实际图片格式。")

        payload = {
            "contents": [{"parts": [{"text": compose_prompt(normalized)}]}],
            "generationConfig": generation_config,
        }
        results: list[ApiImageResult] = []
        try:
            for _ in range(normalized.count):
                response = http.post(
                    endpoint,
                    headers={
                        "x-goog-api-key": normalized.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=(10, 180),
                )
                raise_for_api_error(response, [normalized.api_key])
                results.extend(
                    _parse_gemini_images(response, normalized, tuple(notes))
                )
            return results
        except ApiProviderError:
            raise
        except Exception as exc:
            raise ApiProviderError(
                "连接 Gemini 图片 API 失败，请检查网络、代理或服务状态。"
            ) from exc
