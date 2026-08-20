"""Local Gradio child-process lifecycle management."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from launcher.offline import build_offline_environment
from launcher.redaction import redact_launcher_text


class LauncherProcessError(RuntimeError):
    pass


_PORT_ERROR = "端口必须是 1 到 65535 之间的整数"
_START_ERROR = "无法启动本地 Gradio 服务"
_READY_TIMEOUT_LOG = "本地 Gradio 服务启动超时"
_STOP_ERROR_LOG = "本地 Gradio 服务无法停止"


def _valid_port(port: object) -> bool:
    return isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535


def is_port_available(port: int) -> bool:
    """Return whether a loopback bind to this port succeeds."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True


def find_available_port(
    start: int = 7860,
    attempts: int = 20,
    is_available=None,
) -> int:
    """Find the first available loopback port in a bounded consecutive range."""
    if not _valid_port(start) or isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise LauncherProcessError(_PORT_ERROR)

    checker = is_available or is_port_available
    for port in range(start, min(65535, start + attempts - 1) + 1):
        if checker(port):
            return port
    raise LauncherProcessError("未找到可用端口")


def is_local_port_open(port: int, timeout: float = 0.2) -> bool:
    """Return whether the loopback TCP port accepts a connection."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


class WorkbenchProcessController:
    def __init__(
        self,
        project_root: str | Path,
        *,
        popen_factory=subprocess.Popen,
        port_available=is_port_available,
        port_probe=is_local_port_open,
        sleep=time.sleep,
        monotonic=time.monotonic,
        log_callback=None,
        state_callback=None,
        lifecycle_lock=None,
    ):
        self._project_root = Path(project_root)
        self._popen_factory = popen_factory
        self._port_available = port_available
        self._port_probe = port_probe
        self._sleep = sleep
        self._monotonic = monotonic
        self._log_callback = log_callback
        self._state_callback = state_callback
        self._state = "stopped"
        self._url = ""
        self._process = None
        self._port: int | None = None
        self._last_exit_code: int | None = None
        self._generation = 0
        self._lock = lifecycle_lock if lifecycle_lock is not None else threading.RLock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def url(self) -> str:
        with self._lock:
            return self._url

    @property
    def process(self):
        with self._lock:
            return self._process

    @property
    def last_exit_code(self) -> int | None:
        with self._lock:
            return self._last_exit_code

    def _set_state(self, state: str) -> None:
        if self._state == state:
            return
        self._state = state
        if self._state_callback is not None:
            self._state_callback(state)

    def _is_current(self, child, generation: int) -> bool:
        return self._process is child and self._generation == generation

    def _clear_current(self, exit_code: int | None = None) -> None:
        if exit_code is not None:
            self._last_exit_code = exit_code
        self._process = None
        self._url = ""
        self._port = None
        self._generation += 1

    def _finish_current(self, child, generation: int, exit_code: int) -> bool:
        if not self._is_current(child, generation):
            return False
        self._last_exit_code = exit_code
        self._set_state("stopped" if exit_code == 0 else "failed")
        return True

    def _emit_log(self, value: object) -> None:
        if self._log_callback is not None:
            self._log_callback(redact_launcher_text(value))

    def _read_stream(self, stream, prefix: str) -> None:
        while True:
            line = stream.readline()
            if not line:
                return
            self._emit_log(prefix + line.rstrip("\r\n"))

    def _start_reader(self, stream, prefix: str) -> None:
        if stream is not None:
            threading.Thread(
                target=self._read_stream,
                args=(stream, prefix),
                daemon=True,
            ).start()

    def _preferred_port(self, preferred_port: int | None) -> int:
        if preferred_port is not None:
            if not _valid_port(preferred_port):
                raise LauncherProcessError(_PORT_ERROR)
            return preferred_port

        environment_port = os.environ.get("GRADIO_SERVER_PORT")
        if environment_port is None:
            return 7860
        try:
            port = int(environment_port)
        except (TypeError, ValueError):
            raise LauncherProcessError(_PORT_ERROR) from None
        if not _valid_port(port):
            raise LauncherProcessError(_PORT_ERROR)
        return port

    def start(self, preferred_port: int | None = None) -> str:
        with self._lock:
            if self._process is not None:
                exit_code = self._process.poll()
                if exit_code is None:
                    raise LauncherProcessError("本地 Gradio 服务已经在运行")
                self._last_exit_code = exit_code
                self._clear_current()

            start_port = self._preferred_port(preferred_port)
            app_path = self._project_root / "app.py"
            if not app_path.is_file():
                raise LauncherProcessError("未找到 app.py")
            chosen_port = find_available_port(start_port, is_available=self._port_available)
            intended_url = f"http://127.0.0.1:{chosen_port}"
            try:
                child = self._popen_factory(
                    [sys.executable, str(app_path)],
                    shell=False,
                    cwd=str(self._project_root),
                    env=build_offline_environment(os.environ, chosen_port),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **self._creationflags(),
                )
            except Exception:
                self._set_state("failed")
                raise LauncherProcessError(_START_ERROR) from None

            self._generation += 1
            self._process = child
            self._port = chosen_port
            self._url = intended_url
            self._last_exit_code = None
            self._set_state("starting")
        self._start_reader(child.stdout, "")
        self._start_reader(child.stderr, "[stderr] ")
        return intended_url

    @staticmethod
    def _creationflags() -> dict[str, int]:
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            return {"creationflags": subprocess.CREATE_NO_WINDOW}
        return {}

    def wait_until_ready(self, timeout: float = 60.0, interval: float = 0.2) -> bool:
        deadline = self._monotonic() + timeout
        while True:
            with self._lock:
                child = self._process
                generation = self._generation
                port = self._port
                if child is None or self._state not in ("starting", "running"):
                    return False

            exit_code = child.poll()
            with self._lock:
                if not self._is_current(child, generation):
                    return False
                if self._state not in ("starting", "running"):
                    return False
                if exit_code is not None:
                    self._finish_current(child, generation, exit_code)
                    return False

            if self._monotonic() >= deadline:
                return self._mark_readiness_timeout(child, generation)
            probe_open = self._port_probe(port)
            if self._monotonic() >= deadline:
                return self._mark_readiness_timeout(child, generation)
            if probe_open:
                with self._lock:
                    if not self._is_current(child, generation):
                        return False
                    if self._state not in ("starting", "running"):
                        return False
                    self._set_state("running")
                    return True
            self._sleep(interval)

    def _mark_readiness_timeout(self, child, generation: int) -> bool:
        should_log = False
        with self._lock:
            if self._is_current(child, generation) and self._state in ("starting", "running"):
                self._set_state("failed")
                should_log = True
        if should_log:
            self._emit_log(_READY_TIMEOUT_LOG)
        return False

    def poll(self) -> int | None:
        with self._lock:
            child = self._process
            generation = self._generation
        if child is None:
            return None
        exit_code = child.poll()
        if exit_code is None:
            return None
        with self._lock:
            if not self._is_current(child, generation):
                return None
            if self._state == "stopping":
                return exit_code
            self._clear_current(exit_code)
            self._set_state("stopped" if exit_code == 0 else "failed")
            return exit_code

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            child = self._process
            if child is None:
                self._url = ""
                self._port = None
                self._set_state("stopped")
                return
            if self._state == "stopping":
                return
            generation = self._generation
            finished_code = child.poll()
            if finished_code is not None:
                self._clear_current(finished_code)
                self._set_state("stopped")
                return
            self._set_state("stopping")

        try:
            child.terminate()
            try:
                exit_code = child.wait(timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                exit_code = child.wait(timeout)
        except Exception:
            try:
                confirmed_code = child.poll()
            except Exception:
                confirmed_code = None
            with self._lock:
                if not self._is_current(child, generation):
                    return
                if confirmed_code is not None:
                    self._clear_current(confirmed_code)
                    self._set_state("stopped")
                    return
                self._set_state("failed")
            self._emit_log(_STOP_ERROR_LOG)
            return

        with self._lock:
            if not self._is_current(child, generation):
                return
            self._clear_current(exit_code)
            self._set_state("stopped")
