import json
from pathlib import Path
import threading

from inference.history import GenerationHistory, HistoryRecord, write_json_atomic


class FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake-png")


def test_history_saves_image_and_appends_jsonl(tmp_path):
    history = GenerationHistory(output_dir=tmp_path, history_file=tmp_path / "history.jsonl")
    image_path = history.save_image(FakeImage(), seed=42, index=0)

    history.append(
        HistoryRecord(
            created_at="2026-07-21T11:00:00",
            image_path=str(image_path),
            seed=42,
            prompt="prompt",
            negative_prompt="negative",
            settings={"width": 640},
        )
    )

    assert image_path.exists()
    records = history.list_recent()
    assert records[0]["seed"] == 42
    assert records[0]["settings"]["width"] == 640


def test_write_json_atomic_replaces_file(tmp_path):
    path = tmp_path / "data.json"

    write_json_atomic(path, {"ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_toggle_favorite_marks_history_record(tmp_path):
    history = GenerationHistory(output_dir=tmp_path, history_file=tmp_path / "history.jsonl", favorites_file=tmp_path / "favorites.json")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")
    history.append(
        HistoryRecord(
            created_at="2026-07-25T12:00:00",
            image_path=str(image_path),
            seed=1,
            prompt="prompt",
            negative_prompt="negative",
            settings={},
        )
    )

    enabled = history.toggle_favorite(str(image_path))

    assert enabled is True
    assert history.list_recent()[0]["favorite"] is True
    assert history.list_recent(favorites_only=True)[0]["image_path"] == str(image_path)


def test_append_and_atomic_batch_append_do_not_lose_concurrent_records(
    tmp_path,
    monkeypatch,
):
    history_file = tmp_path / "history.jsonl"
    history = GenerationHistory(
        output_dir=tmp_path / "images",
        history_file=history_file,
        favorites_file=tmp_path / "favorites.json",
    )

    def record(name):
        return HistoryRecord(
            created_at="2026-07-25T12:00:00",
            image_path=str(tmp_path / f"{name}.png"),
            seed=None,
            prompt=name,
            negative_prompt="",
            settings={},
        )

    history.append(record("seed"))
    original_read_bytes = Path.read_bytes
    batch_read_started = threading.Event()
    allow_batch_replace = threading.Event()
    append_finished = threading.Event()
    errors = []

    def controlled_read_bytes(path):
        data = original_read_bytes(path)
        if path == history_file and not batch_read_started.is_set():
            batch_read_started.set()
            if not allow_batch_replace.wait(timeout=2):
                raise RuntimeError("test synchronization timed out")
        return data

    monkeypatch.setattr(Path, "read_bytes", controlled_read_bytes)

    def run_batch():
        try:
            history.append_many_atomic([record("api")])
        except Exception as exc:
            errors.append(exc)

    def run_single():
        try:
            history.append(record("local"))
        except Exception as exc:
            errors.append(exc)
        finally:
            append_finished.set()

    batch_thread = threading.Thread(target=run_batch)
    batch_thread.start()
    assert batch_read_started.wait(timeout=2)
    single_thread = threading.Thread(target=run_single)
    single_thread.start()
    append_finished.wait(timeout=0.1)
    allow_batch_replace.set()
    batch_thread.join(timeout=2)
    single_thread.join(timeout=2)

    assert errors == []
    prompts = {row["prompt"] for row in history.list_recent(limit=10)}
    assert prompts == {"seed", "api", "local"}
