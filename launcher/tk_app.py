"""Tkinter adapter for the strict-offline AI workbench launcher."""

from __future__ import annotations

import os
import queue
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import tkinter as tk
from tkinter import messagebox, ttk

from launcher.checks import PreflightReport, run_preflight
from launcher.process import LauncherProcessError, WorkbenchProcessController
from launcher.redaction import redact_launcher_text
from launcher.view_model import derive_ui_state


_WINDOW_TITLE = "AI 绘画专业工作台 v4.2"
_OFFLINE_LABEL = "严格离线"
_TK_FAILURE = "无法创建桌面启动窗口，请确认 Windows Python 已安装 Tkinter。"
_READY_FAILURE = "本地 Gradio 服务未能就绪，请查看已脱敏运行日志。"
_START_FAILURE = "无法启动本地 Gradio 服务，请查看已脱敏运行日志。"
_CLOSE_FAILURE = (
    "无法确认本地工作台已停止。启动器将保持打开；"
    "请点击“停止工作台”重试，或关闭时选择保留工作台运行。"
)
_DIRECTORY_FAILURE = "无法打开项目内目录。"
_README_FAILURE = "未找到项目内 README.md 安装说明。"
_WEB_FAILURE = "本地工作台地址无效，暂时无法打开网页。"
_NON_WINDOWS_FAILURE = "此功能仅支持 Windows 桌面环境。"
_ALLOWED_PROJECT_ITEMS = frozenset({"models", "outputs", "README.md"})

# Narrow module seams keep the real adapter testable without a display or child.
CONTROLLER_FACTORY = WorkbenchProcessController
PREFLIGHT_RUNNER = run_preflight
THREAD_FACTORY = threading.Thread


@dataclass(frozen=True)
class LauncherEvent:
    """An immutable value passed from a worker to the Tk main thread."""

    kind: str
    payload: object = None


class EventBridge:
    """Queue-only worker bridge; widgets are deliberately not referenced here."""

    def __init__(self) -> None:
        self._events: queue.Queue[LauncherEvent] = queue.Queue()

    def enqueue(self, kind: str, payload: object = None) -> None:
        self._events.put(LauncherEvent(str(kind), payload))

    def drain(self, handler) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if handler(event) is False:
                return

    def discard(self) -> None:
        """Drop queued events after the desktop root has begun destruction."""
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                return


def resolve_project_path(project_root: Path, item: str) -> Path:
    """Resolve an approved project-local item and reject traversal or symlinks."""
    if item not in _ALLOWED_PROJECT_ITEMS:
        raise ValueError("不允许访问该项目路径")
    root = Path(project_root).resolve()
    target = (root / item).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("项目路径不能超出项目根目录") from None
    return target


def is_safe_workbench_url(value: object) -> bool:
    """Allow only a concrete local Gradio URL on the IPv4 loopback address."""
    try:
        parsed = urlparse(str(value).strip())
        port = parsed.port
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.username is None
        and parsed.password is None
        and port is not None
        and 1 <= port <= 65535
    )


def build_diagnostic_snapshot(
    report: PreflightReport, process_state: str, url: str
) -> str:
    """Build a useful, in-memory, credential-safe diagnostics snapshot."""
    lines = [
        "AI 绘画专业工作台 v4.2",
        "模式：严格离线启动",
        f"进程状态：{process_state}",
        f"本地地址：{url}",
        "环境检查：",
    ]
    for item in report.items:
        line = f"- [{item.level}] {item.title}：{item.summary}"
        if item.remedy:
            line += f"；建议：{item.remedy}"
        lines.append(line)
    return redact_launcher_text("\n".join(lines))


class LauncherApplication:
    """Thin Tkinter view layer over the launcher service modules."""

    def __init__(self, root: tk.Tk, project_root: Path):
        self.root = root
        self.project_root = Path(project_root).resolve()
        self.events = EventBridge()
        self.report = PreflightReport(())
        self._preflight_generation = 0
        self._preflight_status = "unchecked"
        self._start_in_flight = False
        self._close_requested = threading.Event()
        self._close_worker_in_flight = False
        self._closing = False
        self._closed = False
        self._row_labels: list[ttk.Label] = []
        self.controller = CONTROLLER_FACTORY(
            self.project_root,
            log_callback=lambda value: self.events.enqueue("log", value),
            state_callback=lambda state: self.events.enqueue("state", state),
        )

        self._build_window()
        self.root.after(100, self.drain_events)

    def _build_window(self) -> None:
        self.root.title(_WINDOW_TITLE)
        self.root.minsize(760, 650)
        self.root.geometry("860x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        container = ttk.Frame(self.root, padding=14)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(5, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=_WINDOW_TITLE, font=("Microsoft YaHei UI", 15, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header, text=f"[{_OFFLINE_LABEL}]").grid(row=0, column=1, sticky="e")

        self.overall_var = tk.StringVar(value="环境状态：正在等待检测")
        ttk.Label(container, textvariable=self.overall_var).grid(
            row=1, column=0, sticky="w", pady=(10, 3)
        )

        self.address_var = tk.StringVar(value="本地地址：尚未就绪")
        ttk.Label(container, textvariable=self.address_var).grid(
            row=2, column=0, sticky="w", pady=(0, 3)
        )

        self.rows_frame = ttk.Frame(container)
        self.rows_frame.grid(row=3, column=0, sticky="ew")
        self.rows_frame.columnconfigure(0, weight=1)

        controls = ttk.Frame(container)
        controls.grid(row=4, column=0, sticky="ew", pady=(12, 10))
        for column in range(4):
            controls.columnconfigure(column, weight=1)
        self.buttons = {
            "preflight": ttk.Button(controls, text="重新检测", command=self.start_preflight),
            "start": ttk.Button(controls, text="启动工作台", command=self.start_workbench),
            "stop": ttk.Button(controls, text="停止工作台", command=self.stop_workbench),
            "web": ttk.Button(controls, text="打开网页", command=self.open_workbench),
            "models": ttk.Button(controls, text="打开模型目录", command=lambda: self.open_project_item("models")),
            "outputs": ttk.Button(controls, text="打开输出目录", command=lambda: self.open_project_item("outputs")),
            "readme": ttk.Button(controls, text="查看安装说明", command=self.open_readme),
            "diagnostic": ttk.Button(controls, text="复制诊断信息", command=self.copy_diagnostic),
        }
        for index, button in enumerate(self.buttons.values()):
            button.grid(row=index // 4, column=index % 4, sticky="ew", padx=3, pady=3)

        logs = ttk.LabelFrame(container, text="运行日志", padding=6)
        logs.grid(row=5, column=0, sticky="nsew")
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)
        self.log_widget = tk.Text(logs, height=16, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(logs, orient="vertical", command=self.log_widget.yview)
        self.log_widget.configure(yscrollcommand=scrollbar.set)
        self.log_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._render_report()
        self._update_view_state()

    def _render_report(self) -> None:
        for label in self._row_labels:
            label.destroy()
        self._row_labels.clear()
        indicator = {"ok": "●", "warn": "●", "error": "●"}
        colour = {"ok": "#208040", "warn": "#9a6700", "error": "#b42318"}
        for index, item in enumerate(self.report.items):
            text = redact_launcher_text(
                f"{indicator[item.level]} {item.title}    {item.summary}"
                + (f"（{item.remedy}）" if item.remedy else "")
            )
            label = ttk.Label(self.rows_frame, text=text, foreground=colour[item.level])
            label.grid(row=index, column=0, sticky="w", pady=1)
            self._row_labels.append(label)

    def _update_view_state(self) -> None:
        state = derive_ui_state(self.report, self.controller.state, self.controller.url)
        if self._preflight_status == "unchecked":
            overall = "环境状态：尚未检测，不能启动工作台"
        elif self._preflight_status == "checking":
            overall = "环境状态：正在检测，不能启动工作台"
        elif self._preflight_status == "failed":
            overall = "环境状态：检测未完成，不能启动工作台"
        else:
            overall = state.overall_text
        safe_url = self.controller.url if is_safe_workbench_url(self.controller.url) else ""
        self.overall_var.set(redact_launcher_text(overall))
        self.address_var.set(redact_launcher_text(f"本地地址：{safe_url or '尚未就绪'}"))
        has_controlled_child = self.controller.process is not None
        can_start = (
            self._preflight_status == "ready"
            and state.can_start
            and not has_controlled_child
            and not self._start_in_flight
            and not self._closing
        )
        self.buttons["start"].configure(state="normal" if can_start else "disabled")
        self.buttons["stop"].configure(
            state="normal" if state.can_stop or has_controlled_child else "disabled"
        )
        self.buttons["web"].configure(
            state="normal" if state.can_open_web and bool(safe_url) else "disabled"
        )

    def _append_log(self, value: object) -> None:
        text = redact_launcher_text(value)
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text.rstrip() + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _handle_event(self, event: LauncherEvent) -> None:
        if event.kind == "report":
            generation, report = event.payload
            if generation == self._preflight_generation and isinstance(report, PreflightReport):
                self._preflight_status = "ready"
                self.report = report
                self._render_report()
        elif event.kind == "log":
            self._append_log(event.payload)
        elif event.kind == "preflight_failed":
            if event.payload == self._preflight_generation:
                self._preflight_status = "failed"
                self._append_log("环境检测未完成，请检查本地项目目录和 Python 环境。")
        elif event.kind == "start_failed":
            self._start_in_flight = False
            if self._closing and self._close_requested.is_set():
                if self.controller.process is None:
                    self._closed = True
                    return False
                self._start_close_worker()
            else:
                self._append_log(_START_FAILURE)
        elif event.kind == "start_complete":
            self._start_in_flight = False
            if self._closing and self._close_requested.is_set():
                if self.controller.process is None:
                    self._closed = True
                    return False
                self._start_close_worker()
        elif event.kind == "ready_failed":
            self._start_in_flight = False
            if self._closing and self._close_requested.is_set():
                if self.controller.process is None:
                    self._closed = True
                    return False
                self._start_close_worker()
            else:
                self._append_log(_READY_FAILURE)
        elif event.kind == "child_exited":
            self._append_log(f"本地 Gradio 服务已退出，退出码：{event.payload}。")
        elif event.kind == "closed":
            self._start_in_flight = False
            self._close_worker_in_flight = False
            self._closed = True
            return False
        elif event.kind == "close_failed":
            self._start_in_flight = False
            self._close_worker_in_flight = False
            self._closing = False
            self._close_requested.clear()
            self._append_log(_CLOSE_FAILURE)
            self._show_error(_CLOSE_FAILURE)
        self._update_view_state()

    def drain_events(self) -> None:
        self.events.drain(self._handle_event)
        if self._closed:
            self.events.discard()
            self.root.destroy()
            return
        self._monitor_controller()
        if not self._closed:
            self.root.after(100, self.drain_events)

    def _monitor_controller(self) -> None:
        try:
            exit_code = self.controller.poll()
        except Exception:
            return
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            self._handle_event(LauncherEvent("child_exited", exit_code))

    def start_initial_preflight(self) -> None:
        self.start_preflight()

    def start_preflight(self) -> None:
        self._preflight_generation += 1
        generation = self._preflight_generation
        self._preflight_status = "checking"
        self._update_view_state()

        def worker() -> None:
            try:
                report = PREFLIGHT_RUNNER(self.project_root)
            except Exception:
                self.events.enqueue("preflight_failed", generation)
                return
            self.events.enqueue("report", (generation, report))

        THREAD_FACTORY(target=worker, daemon=True).start()

    def start_workbench(self) -> None:
        if self._start_in_flight:
            return
        if self._preflight_status != "ready":
            return
        if self.controller.process is not None:
            return
        if not derive_ui_state(self.report, self.controller.state, self.controller.url).can_start:
            return
        self._start_in_flight = True
        self._update_view_state()

        def worker() -> None:
            if self._close_requested.is_set():
                self._finish_close_from_worker()
                return
            try:
                self.controller.start()
            except LauncherProcessError:
                if self._close_requested.is_set():
                    self._finish_close_from_worker()
                else:
                    self.events.enqueue("start_failed")
                return
            except Exception:
                if self._close_requested.is_set():
                    self._finish_close_from_worker()
                else:
                    self.events.enqueue("start_failed")
                return
            if self._close_requested.is_set():
                self._finish_close_from_worker()
                return
            try:
                ready = self.controller.wait_until_ready()
            except Exception:
                if self._close_requested.is_set():
                    self._finish_close_from_worker()
                else:
                    self.events.enqueue("start_failed")
                return
            if self._close_requested.is_set():
                self._finish_close_from_worker()
                return
            if not ready:
                try:
                    self.controller.stop()
                except Exception:
                    pass
                try:
                    self.controller.poll()
                except Exception:
                    pass
                self.events.enqueue("ready_failed")
                return
            self.events.enqueue("start_complete")

        THREAD_FACTORY(target=worker, daemon=True).start()

    def _finish_close_from_worker(self) -> None:
        """Stop any registered child and report closure only after ownership clears."""
        if self.controller.process is not None:
            try:
                self.controller.stop()
            except Exception:
                pass
            try:
                self.controller.poll()
            except Exception:
                pass
        self.events.enqueue(
            "closed" if self.controller.process is None else "close_failed"
        )

    def _start_close_worker(self) -> None:
        if self._close_worker_in_flight:
            return
        self._close_worker_in_flight = True
        THREAD_FACTORY(target=self._finish_close_from_worker, daemon=True).start()

    def stop_workbench(self) -> None:
        def worker() -> None:
            try:
                self.controller.stop()
            except Exception:
                self.events.enqueue("log", "本地 Gradio 服务停止未完成，请稍后重新检测。")

        THREAD_FACTORY(target=worker, daemon=True).start()

    def open_workbench(self) -> None:
        url = self.controller.url
        if not is_safe_workbench_url(url):
            self._show_error(_WEB_FAILURE)
            return
        try:
            webbrowser.open(url)
        except Exception:
            self._show_error(_WEB_FAILURE)

    def open_project_item(self, item: str) -> None:
        try:
            target = resolve_project_path(self.project_root, item)
            if item in {"models", "outputs"}:
                target.mkdir(parents=True, exist_ok=True)
                target = resolve_project_path(self.project_root, item)
            if os.name != "nt" or not hasattr(os, "startfile"):
                self._show_error(_NON_WINDOWS_FAILURE)
                return
            os.startfile(str(target))
        except (OSError, ValueError):
            self._show_error(_DIRECTORY_FAILURE)

    def open_readme(self) -> None:
        try:
            readme = resolve_project_path(self.project_root, "README.md")
            if not readme.is_file():
                raise OSError("README missing")
            if os.name != "nt" or not hasattr(os, "startfile"):
                self._show_error(_NON_WINDOWS_FAILURE)
                return
            os.startfile(str(readme))
        except (OSError, ValueError):
            self._show_error(_README_FAILURE)

    def copy_diagnostic(self) -> None:
        snapshot = build_diagnostic_snapshot(self.report, self.controller.state, self.controller.url)
        self.root.clipboard_clear()
        self.root.clipboard_append(redact_launcher_text(snapshot))
        self._append_log("已复制脱敏诊断信息。")

    def _show_error(self, text: str) -> None:
        messagebox.showerror(_WINDOW_TITLE, redact_launcher_text(text), parent=self.root)

    def on_close(self) -> None:
        if self._closing:
            return
        child = self.controller.process
        if child is None:
            self._closing = True
            self._close_requested.set()
            if self._start_in_flight:
                self._update_view_state()
                return
            self._closed = True
            self.root.destroy()
            return
        stop_child = messagebox.askyesno(
            _WINDOW_TITLE,
            "工作台仍在运行，是否同时停止本地工作台？",
            parent=self.root,
        )
        self._closing = True
        self._close_requested.set()
        if not stop_child:
            self._closed = True
            self.root.destroy()
            return
        self._start_close_worker()


def main(project_root: Path | None = None) -> int:
    """Create the desktop window only on explicit launch, never at import time."""
    root_path = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    try:
        root = tk.Tk()
    except Exception:
        print(_TK_FAILURE, file=sys.stderr)
        return 1
    try:
        application = LauncherApplication(root, root_path)
        application.start_initial_preflight()
        root.mainloop()
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass
        print(_TK_FAILURE, file=sys.stderr)
        return 1
    return 0
