from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Protocol


class ApiProviderError(RuntimeError):
    """A sanitized error that is safe to show in the web UI."""


@dataclass(frozen=True)
class ApiImageRequest:
    api_key: str
    model: str
    prompt: str
    negative_prompt: str = ""
    size: str = "1024x1024"
    quality: str = "auto"
    count: int = 1
    output_format: str = "png"
    base_url: str = ""

    def normalized(self) -> "ApiImageRequest":
        api_key = str(self.api_key).strip()
        model = str(self.model).strip()
        prompt = str(self.prompt).strip()
        if not api_key:
            raise ApiProviderError("请填写 API Key。")
        if not model:
            raise ApiProviderError("请填写模型名称。")
        if not prompt:
            raise ApiProviderError("请填写提示词。")
        try:
            count = int(self.count)
        except (TypeError, ValueError) as exc:
            raise ApiProviderError("生成数量必须是整数。") from exc
        if not 1 <= count <= 10:
            raise ApiProviderError("生成数量必须在 1 到 10 之间。")
        output_format = str(self.output_format).strip().lower() or "png"
        if output_format not in {"png", "jpeg", "webp"}:
            raise ApiProviderError("输出格式必须是 png、jpeg 或 webp。")

        return replace(
            self,
            api_key=api_key,
            model=model,
            prompt=prompt,
            negative_prompt=str(self.negative_prompt).strip(),
            size=str(self.size).strip() or "auto",
            quality=str(self.quality).strip() or "auto",
            count=count,
            output_format=output_format,
            base_url=str(self.base_url).strip(),
        )


@dataclass(frozen=True)
class ApiImageResult:
    image_bytes: bytes
    mime_type: str = "image/png"
    revised_prompt: str | None = None
    notes: tuple[str, ...] = ()


class ApiProvider(Protocol):
    def generate(
        self,
        request: ApiImageRequest,
        session: Any | None = None,
    ) -> list[ApiImageResult]:
        """Generate one or more images without retaining request secrets."""


def compose_prompt(request: ApiImageRequest) -> str:
    prompt = str(request.prompt).strip()
    negative_prompt = str(request.negative_prompt).strip()
    if not negative_prompt:
        return prompt
    return f"{prompt}\n\nAvoid: {negative_prompt}"


def redact_secrets(message: object, secrets: list[str] | tuple[str, ...] = ()) -> str:
    cleaned = str(message)
    for secret in secrets:
        secret = str(secret)
        if secret:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    cleaned = re.sub(
        r"(?i)(authorization\s*[:=]?\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)(\bbearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)(\b(?:api[_-]?key|key)\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        cleaned,
    )
    return cleaned


def raise_for_api_error(
    response: Any,
    secrets: list[str] | tuple[str, ...] = (),
) -> None:
    status_code = int(getattr(response, "status_code", 0) or 0)
    if 200 <= status_code < 300:
        return

    if status_code in {401, 403}:
        message = "API 认证或权限失败，请检查密钥、账户权限和模型访问权限。"
    elif status_code == 429:
        message = "API 请求过于频繁或额度已用完，请稍后再试或检查账户额度。"
    elif status_code in {400, 404, 405, 422}:
        message = "API 请求被服务商拒绝，所选模型可能不支持图片生成或参数不兼容。"
    elif status_code >= 500:
        message = "远程 API 服务暂时不可用，请稍后再试。"
    else:
        message = f"API 请求失败（HTTP {status_code or '未知'}）。"
    raise ApiProviderError(message)
