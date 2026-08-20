from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from inference.model_registry import discover_models


RUNTIME_PACKAGES = (
    "gradio",
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
    "requests",
    "PIL",
    "safetensors",
)

_OUTPUT_ERROR_SUMMARY = "输出目录不可写"
_OUTPUT_ERROR_REMEDY = "请检查 outputs 目录是否存在、为目录且具有写入权限"
_PYTHON_ERROR_REMEDY = "请安装 Python 3.10 或 3.11 后重新运行启动器"


@dataclass(frozen=True)
class CheckResult:
    code: str
    title: str
    level: str
    summary: str
    remedy: str = ""

    def __post_init__(self) -> None:
        if self.level not in {"ok", "warn", "error"}:
            raise ValueError(f"未知检查级别：{self.level}")


@dataclass(frozen=True)
class PreflightReport:
    items: tuple[CheckResult, ...]

    @property
    def can_start(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> tuple[CheckResult, ...]:
        return tuple(item for item in self.items if item.level == "error")


def _python_check() -> CheckResult:
    version = sys.version_info
    summary = f"Python {version.major}.{version.minor}.{version.micro}"
    if version < (3, 10):
        return CheckResult(
            "python",
            "Python",
            "error",
            f"{summary}；最低要求为 Python 3.10",
            _PYTHON_ERROR_REMEDY,
        )
    if version >= (3, 12):
        return CheckResult(
            "python", "Python", "warn", f"{summary}；3.12+ 版本尚未充分验证"
        )
    return CheckResult("python", "Python", "ok", f"{summary} 可用")


def _dependency_checks(package_finder) -> list[CheckResult]:
    missing = []
    for package in RUNTIME_PACKAGES:
        try:
            found = package_finder(package)
        except Exception:
            found = None
        if found is None:
            missing.append(package)

    if not missing:
        return [CheckResult("dependencies", "运行依赖", "ok", "运行依赖已就绪")]

    items = [
        CheckResult(
            "dependencies",
            "运行依赖",
            "error",
            f"缺少 {len(missing)} 个运行依赖",
            "请在本地离线环境中安装缺失依赖",
        )
    ]
    items.extend(
        CheckResult(
            f"dependency.{package}",
            package,
            "error",
            f"缺少运行依赖：{package}",
            "请在本地离线环境中安装该依赖",
        )
        for package in missing
    )
    return items


def _default_torch_probe() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        return "warn", "CUDA 不可用"
    properties = torch.cuda.get_device_properties(0)
    memory_gb = properties.total_memory / (1024**3)
    return "ok", f"{properties.name}，显存 {memory_gb:.1f} GB"


def _gpu_check(torch_probe) -> CheckResult:
    probe = _default_torch_probe if torch_probe is None else torch_probe
    try:
        level, summary = probe()
        if level not in {"ok", "warn"}:
            raise ValueError("invalid probe result")
        return CheckResult("gpu", "GPU / CUDA", level, str(summary))
    except Exception:
        return CheckResult("gpu", "GPU / CUDA", "warn", "CUDA 探测失败或不可用")


def _models_check(project_root: Path) -> CheckResult:
    models_dir = project_root / "models"
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        models = discover_models(models_dir)
    except OSError:
        return CheckResult("models", "本地模型", "warn", "无法读取本地模型目录")

    complete_count = sum(model.complete for model in models)
    incomplete_count = len(models) - complete_count
    if complete_count:
        summary = f"发现 {complete_count} 个完整本地模型"
        if incomplete_count:
            summary += f"，另有 {incomplete_count} 个不完整模型"
        return CheckResult("models", "本地模型", "ok", summary)

    summary = "未发现完整本地模型"
    if incomplete_count:
        summary += f"，发现 {incomplete_count} 个不完整模型"
    return CheckResult("models", "本地模型", "warn", summary)


def _outputs_check(project_root: Path) -> CheckResult:
    outputs = (project_root / "outputs").resolve()
    temporary_path: str | None = None
    descriptor: int | None = None
    try:
        outputs.mkdir(parents=True, exist_ok=True)
        if not outputs.is_dir():
            raise OSError("outputs is not a directory")
        descriptor, temporary_path = tempfile.mkstemp(dir=outputs)
        os.close(descriptor)
        descriptor = None
        Path(temporary_path).unlink()
        temporary_path = None
    except OSError:
        return CheckResult(
            "outputs", "输出目录", "error", _OUTPUT_ERROR_SUMMARY, _OUTPUT_ERROR_REMEDY
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink()
            except OSError:
                pass
    return CheckResult("outputs", "输出目录", "ok", "输出目录可写")


def run_preflight(
    project_root: Path,
    package_finder=importlib.util.find_spec,
    torch_probe=None,
) -> PreflightReport:
    """Run local-only launcher checks without starting services or installing packages."""
    root = Path(project_root)
    items: list[CheckResult] = [_python_check()]
    items.extend(_dependency_checks(package_finder))
    items.append(_gpu_check(torch_probe))
    items.append(_models_check(root))
    items.append(CheckResult("port", "端口", "ok", "将在启动时选择可用端口"))
    items.append(_outputs_check(root))
    return PreflightReport(tuple(items))
