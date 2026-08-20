from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError

from .api_providers.base import (
    ApiImageRequest,
    ApiImageResult,
    ApiProvider,
    ApiProviderError,
    redact_secrets,
)
from .api_providers.registry import create_provider
from .config import FAVORITES_FILE, HISTORY_FILE, OUTPUTS_DIR
from .history import GenerationHistory, HistoryRecord


MAX_IMAGE_BYTES = 32 * 1024 * 1024
_FORMAT_EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
_EXTENSION_FORMATS = {".png": "png", ".jpg": "jpeg", ".webp": "webp"}


@dataclass(frozen=True)
class ApiGenerationOutcome:
    paths: tuple[Path, ...]
    status: str


class ApiImageService:
    def __init__(
        self,
        history: GenerationHistory | None = None,
        provider_factory: Callable[[str], ApiProvider] = create_provider,
    ):
        self.history = history or GenerationHistory(
            output_dir=OUTPUTS_DIR / "api",
            history_file=HISTORY_FILE,
            favorites_file=FAVORITES_FILE,
        )
        self._provider_factory = provider_factory

    def generate(
        self,
        provider_name: str,
        request: ApiImageRequest,
    ) -> ApiGenerationOutcome:
        normalized = request.normalized()
        provider = self._provider_factory(provider_name)
        results = provider.generate(normalized)
        if not results:
            raise ApiProviderError("API 没有返回图片。")

        paths: list[Path] = []
        records: list[HistoryRecord] = []
        notes: list[str] = []
        safe = lambda value: redact_secrets(value, [normalized.api_key])
        try:
            for index, result in enumerate(results):
                path = self._save_image(result, index)
                paths.append(path)
                for note in result.notes:
                    sanitized_note = safe(note)
                    if sanitized_note and sanitized_note not in notes:
                        notes.append(sanitized_note)
                records.append(
                    HistoryRecord(
                        created_at=datetime.now().isoformat(timespec="seconds"),
                        image_path=str(path),
                        seed=None,
                        prompt=safe(normalized.prompt),
                        negative_prompt=safe(normalized.negative_prompt),
                        settings={
                            "source": "api",
                            "provider": safe(provider_name),
                            "model": safe(normalized.model),
                            "size": safe(normalized.size),
                            "quality": safe(normalized.quality),
                            "output_format": _EXTENSION_FORMATS[path.suffix.lower()],
                            "requested_output_format": safe(
                                normalized.output_format
                            ),
                            "batch_count": normalized.count,
                        },
                    )
                )
        except ApiProviderError:
            self._remove_paths(paths)
            raise

        try:
            self.history.append_many_atomic(records)
        except Exception as exc:
            self._remove_paths(paths)
            raise ApiProviderError(
                "本地历史记录写入失败，已回滚本次保存的图片。"
            ) from exc

        status_lines = [
            f"API 生成完成：{safe(provider_name)} / {safe(normalized.model)}",
            f"已保存 {len(paths)} 张图片到 {self.history.output_dir}",
        ]
        status_lines.extend(f"提示：{note}" for note in notes)
        return ApiGenerationOutcome(paths=tuple(paths), status="\n".join(status_lines))

    @staticmethod
    def _remove_paths(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass

    def _save_image(self, result: ApiImageResult, index: int) -> Path:
        image_bytes = bytes(result.image_bytes)
        if not image_bytes:
            raise ApiProviderError("API 返回了空图片。")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ApiProviderError("API 返回的单张图片超过 32MB，已停止保存。")

        try:
            with Image.open(BytesIO(image_bytes)) as verification_image:
                verification_image.verify()
            with Image.open(BytesIO(image_bytes)) as loaded_image:
                loaded_image.load()
                image_format = str(loaded_image.format or "PNG").upper()
                image = loaded_image.copy()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise ApiProviderError("API 返回的数据不是有效图片。") from exc

        if image_format not in _FORMAT_EXTENSIONS:
            image_format = "PNG"
        extension = _FORMAT_EXTENSIONS[image_format]
        temporary_name = ""
        try:
            self.history.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            final_path = (
                self.history.output_dir
                / f"{timestamp}-api-{index + 1}{extension}"
            )
            fd, temporary_name = tempfile.mkstemp(
                prefix=".api-image-",
                suffix=".tmp",
                dir=self.history.output_dir,
            )
            os.close(fd)
            image.save(temporary_name, format=image_format)
            os.replace(temporary_name, final_path)
            return final_path
        except Exception as exc:
            if temporary_name:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
            raise ApiProviderError(
                "本地输出写入失败，请检查磁盘空间和目录权限。"
            ) from exc
        finally:
            image.close()
