from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL.Image import Image

from .config import FAVORITES_FILE, HISTORY_FILE, OUTPUTS_DIR


_HISTORY_LOCK = threading.RLock()


@dataclass(frozen=True)
class HistoryRecord:
    created_at: str
    image_path: str
    seed: int | None
    prompt: str
    negative_prompt: str
    settings: dict[str, Any]
    status: str = "success"
    error: str | None = None


class GenerationHistory:
    def __init__(
        self,
        output_dir: Path = OUTPUTS_DIR,
        history_file: Path = HISTORY_FILE,
        favorites_file: Path = FAVORITES_FILE,
    ):
        self.output_dir = Path(output_dir)
        self.history_file = Path(history_file)
        self.favorites_file = Path(favorites_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, image: Image, seed: int | None, index: int) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed_part = "random" if seed is None else str(seed)
        path = self.output_dir / f"{timestamp}-seed-{seed_part}-{index + 1}.png"
        image.save(path)
        return path

    def append(self, record: HistoryRecord) -> None:
        with _HISTORY_LOCK:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def append_many_atomic(self, records: list[HistoryRecord]) -> None:
        if not records:
            return
        with _HISTORY_LOCK:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            existing = b""
            if self.history_file.exists():
                existing = self.history_file.read_bytes()
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{self.history_file.name}.",
                suffix=".tmp",
                dir=self.history_file.parent,
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(existing)
                    if existing and not existing.endswith(b"\n"):
                        handle.write(b"\n")
                    for record in records:
                        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
                        handle.write(line.encode("utf-8"))
                os.replace(temp_name, self.history_file)
            except Exception:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise

    def list_recent(self, limit: int = 20, favorites_only: bool = False) -> list[dict[str, Any]]:
        with _HISTORY_LOCK:
            favorites = self._read_favorites()
            if not self.history_file.exists():
                return []
            lines = self.history_file.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["favorite"] = record.get("image_path") in favorites
            if favorites_only and not record["favorite"]:
                continue
            records.append(record)
        return list(reversed(records[-limit:]))

    def toggle_favorite(self, image_path: str) -> bool:
        image_path = str(image_path).strip()
        if not image_path:
            raise ValueError("请先选择一张历史图片。")
        favorites = self._read_favorites()
        if image_path in favorites:
            favorites.remove(image_path)
            enabled = False
        else:
            favorites.add(image_path)
            enabled = True
        write_json_atomic(self.favorites_file, sorted(favorites))
        return enabled

    def _read_favorites(self) -> set[str]:
        if not self.favorites_file.exists():
            return set()
        try:
            payload = json.loads(self.favorites_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return set()
        if not isinstance(payload, list):
            return set()
        return {str(item) for item in payload}


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
