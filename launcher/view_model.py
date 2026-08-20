from __future__ import annotations

from dataclasses import dataclass

from launcher.checks import CheckResult, PreflightReport


_PROCESS_TEXT = {
    "stopped": "已停止",
    "starting": "正在启动",
    "running": "正在运行",
    "stopping": "正在停止",
    "failed": "启动失败",
}


@dataclass(frozen=True)
class LauncherUiState:
    can_start: bool
    can_stop: bool
    can_open_web: bool
    overall_text: str
    rows: tuple[CheckResult, ...]


def _overall_text(report: PreflightReport, process_state: str) -> str:
    prefix = _PROCESS_TEXT[process_state]
    error_count = len(report.errors)
    if error_count:
        return f"{prefix}；体检发现 {error_count} 项错误"
    warning_count = sum(item.level == "warn" for item in report.items)
    if warning_count:
        return f"{prefix}；体检完成，{warning_count} 项警告"
    return f"{prefix}；体检通过"


def derive_ui_state(
    report: PreflightReport,
    process_state: str,
    url: str = "",
) -> LauncherUiState:
    if process_state not in _PROCESS_TEXT:
        raise ValueError(f"未知进程状态：{process_state}")
    return LauncherUiState(
        can_start=report.can_start and process_state in {"stopped", "failed"},
        can_stop=process_state in {"starting", "running"},
        can_open_web=process_state == "running" and bool(url.strip()),
        overall_text=_overall_text(report, process_state),
        rows=report.items,
    )
