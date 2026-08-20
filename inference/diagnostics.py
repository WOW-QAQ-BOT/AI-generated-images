from __future__ import annotations

import importlib.util
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from .model_registry import discover_models


@dataclass(frozen=True)
class DiagnosticItem:
    name: str
    status: str
    message: str


def run_diagnostics(models_dir: str | Path, ports: list[int] | None = None) -> list[DiagnosticItem]:
    ports = ports or [7860, 7861, 7862]
    items: list[DiagnosticItem] = [_python_check()]
    for package in ("gradio", "torch", "diffusers", "transformers", "safetensors", "requests"):
        items.append(_package_check(package))
    items.append(
        DiagnosticItem(
            "API 密钥",
            "ok",
            "由使用者在 API 作画页面临时输入；诊断不会读取或保存密钥。",
        )
    )
    items.append(_cuda_check())
    for port in ports:
        items.append(_port_check(port))
    items.extend(_model_checks(models_dir))
    return items


def diagnostics_markdown(items: list[DiagnosticItem]) -> str:
    if not items:
        return "暂无诊断结果。"
    icons = {"ok": "✅", "warn": "⚠️", "error": "❌"}
    lines = ["| 项目 | 状态 | 说明 |", "|---|---|---|"]
    for item in items:
        lines.append(f"| {item.name} | {icons.get(item.status, item.status)} | {item.message} |")
    return "\n".join(lines)


def _python_check() -> DiagnosticItem:
    version = sys.version_info
    if (version.major, version.minor) in ((3, 10), (3, 11)):
        return DiagnosticItem("Python", "ok", sys.version.split()[0])
    if version.major == 3 and version.minor >= 12:
        return DiagnosticItem("Python", "warn", f"{sys.version.split()[0]} 可运行界面，但 AI 依赖更推荐 3.10 或 3.11")
    return DiagnosticItem("Python", "error", "建议使用 Python 3.10 或 3.11")


def _package_check(package: str) -> DiagnosticItem:
    if importlib.util.find_spec(package) is not None:
        return DiagnosticItem(package, "ok", "已安装")
    return DiagnosticItem(package, "error", "未安装，运行 python -m pip install -r requirements.txt")


def _cuda_check() -> DiagnosticItem:
    try:
        import torch
    except Exception:
        return DiagnosticItem("CUDA", "warn", "未安装 torch，无法检查 CUDA")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return DiagnosticItem("CUDA", "ok", f"{name}，约 {memory_gb:.1f}GB 显存")
    return DiagnosticItem("CUDA", "warn", "当前未检测到 CUDA，将使用 CPU，生成会很慢")


def _port_check(port: int) -> DiagnosticItem:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        in_use = sock.connect_ex(("127.0.0.1", int(port))) == 0
    if in_use:
        return DiagnosticItem(f"端口 {port}", "warn", "已被占用，可换 7861/7862")
    return DiagnosticItem(f"端口 {port}", "ok", "可用")


def _model_checks(models_dir: str | Path) -> list[DiagnosticItem]:
    models = discover_models(models_dir)
    if not models:
        return [DiagnosticItem("模型", "warn", "models 目录下没有可用模型")]
    return [
        DiagnosticItem(f"模型 {model.name}", "ok" if model.complete else "warn", f"{model.kind}：{model.message}")
        for model in models
    ]
