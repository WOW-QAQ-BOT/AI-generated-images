import contextlib
import importlib
import io
import sys
import threading
from pathlib import Path

from launcher.checks import CheckResult, PreflightReport


def _fresh_tk_app():
    sys.modules.pop("launcher.tk_app", None)
    return importlib.import_module("launcher.tk_app")


class FakeWidget:
    active_root = None

    def __init__(self, *_args, **kwargs):
        self.options = dict(kwargs)
        self.destroyed = False
        self.lines = []

    def grid(self, *_args, **_kwargs):
        return None

    def columnconfigure(self, *_args, **_kwargs):
        return None

    def rowconfigure(self, *_args, **_kwargs):
        return None

    def configure(self, **kwargs):
        self._assert_alive()
        self.options.update(kwargs)

    def destroy(self):
        self.destroyed = True

    def insert(self, _where, value):
        self._assert_alive()
        self.lines.append(value)

    def see(self, _where):
        self._assert_alive()
        return None

    def yview(self, *_args):
        return None

    def set(self, *_args):
        return None

    @classmethod
    def _assert_alive(cls):
        if cls.active_root is not None and cls.active_root.destroyed:
            raise AssertionError("widget mutation after root.destroy")


class FakeStringVar:
    active_root = None
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        if self.active_root is not None and self.active_root.destroyed:
            raise AssertionError("variable mutation after root.destroy")
        self.value = value

    def get(self):
        return self.value


class FakeRoot:
    def __init__(self):
        self.callbacks = []
        self.destroyed = False
        self.protocols = {}
        self.clipboard = ""

    def title(self, _value):
        return None

    def minsize(self, *_args):
        return None

    def geometry(self, _value):
        return None

    def protocol(self, name, callback):
        self.protocols[name] = callback

    def columnconfigure(self, *_args, **_kwargs):
        return None

    def rowconfigure(self, *_args, **_kwargs):
        return None

    def after(self, milliseconds, callback):
        self.callbacks.append((milliseconds, callback))

    def run_next_callback(self):
        _milliseconds, callback = self.callbacks.pop(0)
        callback()

    def destroy(self):
        self.destroyed = True

    def clipboard_clear(self):
        self.clipboard = ""

    def clipboard_append(self, value):
        self.clipboard += value


class DeferredThread:
    queued = []

    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.queued.append(self.target)


class FakeController:
    def __init__(self, **callbacks):
        self.log_callback = callbacks["log_callback"]
        self.state_callback = callbacks["state_callback"]
        self._state = "stopped"
        self._url = ""
        self._process = None
        self.ready_result = True
        self.stop_raises = False
        self.stop_retains_child = False
        self.poll_results = []
        self.start_calls = 0
        self.stop_calls = 0
        self.stop_started = threading.Event()
        self.stop_release = None

    @property
    def state(self):
        return self._state

    @property
    def url(self):
        return self._url

    @property
    def process(self):
        return self._process

    def _set_state(self, value):
        self._state = value
        self.state_callback(value)

    def start(self):
        self.start_calls += 1
        self._process = object()
        self._url = "http://127.0.0.1:7860"
        self._set_state("starting")
        return self._url

    def wait_until_ready(self):
        if self.ready_result:
            self._set_state("running")
            return True
        self._set_state("failed")
        return False

    def stop(self):
        self.stop_calls += 1
        self.stop_started.set()
        if self.stop_release is not None:
            assert self.stop_release.wait(1.0)
        if self.stop_raises:
            raise RuntimeError("api_key=stop-secret")
        if self.stop_retains_child:
            return
        self._process = None
        self._url = ""
        self._set_state("stopped")

    def poll(self):
        if not self.poll_results:
            return None
        exit_code = self.poll_results.pop(0)
        if exit_code is None:
            return None
        self._process = None
        self._url = ""
        self._set_state("stopped" if exit_code == 0 else "failed")
        return exit_code


def _headless_application(tk_app, monkeypatch, tmp_path, *, controller=None, preflight=None):
    DeferredThread.queued = []
    controller = controller or FakeController.__new__(FakeController)
    if not hasattr(controller, "_state"):
        FakeController.__init__(controller, log_callback=lambda _value: None, state_callback=lambda _value: None)

    def factory(_project_root, **callbacks):
        controller.log_callback = callbacks["log_callback"]
        controller.state_callback = callbacks["state_callback"]
        return controller

    for name in ("Frame", "Label", "Button", "LabelFrame", "Scrollbar"):
        monkeypatch.setattr(tk_app.ttk, name, FakeWidget)
    monkeypatch.setattr(tk_app.tk, "Text", FakeWidget)
    monkeypatch.setattr(tk_app.tk, "StringVar", FakeStringVar)
    monkeypatch.setattr(tk_app, "CONTROLLER_FACTORY", factory)
    monkeypatch.setattr(tk_app, "THREAD_FACTORY", DeferredThread)
    monkeypatch.setattr(tk_app, "PREFLIGHT_RUNNER", preflight or (lambda _root: PreflightReport(())))
    root = FakeRoot()
    FakeWidget.active_root = root
    FakeStringVar.active_root = root
    application = tk_app.LauncherApplication(root, tmp_path)
    return application, root, controller


def _run_one_worker():
    worker = DeferredThread.queued.pop(0)
    worker()


def _finish_preflight(application):
    application.start_preflight()
    _run_one_worker()
    application.drain_events()


def test_launcher_module_exposes_public_interfaces_without_creating_window(monkeypatch):
    import tkinter

    calls = []

    def forbidden_tk(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("Tk must not be created during import")

    monkeypatch.setattr(tkinter, "Tk", forbidden_tk)
    tk_app = _fresh_tk_app()

    assert callable(tk_app.main)
    assert hasattr(tk_app, "LauncherApplication")
    assert calls == []


def test_pyw_entrypoint_is_thin_and_has_no_process_or_environment_logic():
    entrypoint = Path(__file__).resolve().parents[1] / "launcher.pyw"
    source = entrypoint.read_text(encoding="utf-8")

    assert source == (
        "from pathlib import Path\n"
        "from launcher.tk_app import main\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main(Path(__file__).resolve().parent))\n"
    )
    for forbidden in ("pip", "subprocess", "os.environ", "requests"):
        assert forbidden not in source


def test_project_local_path_resolution_rejects_escape(tmp_path):
    tk_app = _fresh_tk_app()
    root = tmp_path / "project"
    root.mkdir()

    assert tk_app.resolve_project_path(root, "models") == (root / "models").resolve()
    assert tk_app.resolve_project_path(root, "outputs") == (root / "outputs").resolve()
    assert tk_app.resolve_project_path(root, "README.md") == (root / "README.md").resolve()

    try:
        tk_app.resolve_project_path(root, "../outside")
    except ValueError:
        pass
    else:
        raise AssertionError("an escaped path must be rejected")


def test_web_url_validation_allows_only_loopback_http_port():
    tk_app = _fresh_tk_app()

    assert tk_app.is_safe_workbench_url("http://127.0.0.1:7860")
    for value in (
        "https://127.0.0.1:7860", "http://localhost:7860", "http://127.0.0.1",
        "http://127.0.0.1:0", "http://127.0.0.1:70000", "http://example.com:7860",
        "http://user@127.0.0.1:7860",
    ):
        assert not tk_app.is_safe_workbench_url(value)


def test_diagnostic_snapshot_redacts_credentials():
    tk_app = _fresh_tk_app()
    report = PreflightReport(
        (CheckResult("models", "本地模型", "warn", "api_key=very-secret Bearer token"),)
    )
    diagnostic = tk_app.build_diagnostic_snapshot(
        report, "failed", "http://127.0.0.1:7860?api_key=also-secret"
    )

    assert "very-secret" not in diagnostic
    assert "also-secret" not in diagnostic
    assert "token" not in diagnostic
    assert "[REDACTED]" in diagnostic


def test_event_bridge_queues_worker_events_until_main_thread_drain():
    tk_app = _fresh_tk_app()
    bridge = tk_app.EventBridge()
    seen = []

    worker = threading.Thread(target=lambda: bridge.enqueue("log", "api_key=secret"))
    worker.start()
    worker.join()

    assert seen == []
    bridge.drain(lambda event: seen.append(event))
    assert [(event.kind, event.payload) for event in seen] == [("log", "api_key=secret")]


def test_adapter_gates_initial_and_overlapping_preflight_results(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    reports = iter((
        PreflightReport((CheckResult("new", "新检测", "ok", "new"),)),
        PreflightReport((CheckResult("old", "旧检测", "error", "old"),)),
    ))
    app, _root, _controller = _headless_application(
        tk_app, monkeypatch, tmp_path, preflight=lambda _root: next(reports)
    )

    assert app.buttons["start"].options["state"] == "disabled"
    assert "尚未" in app.address_var.get()
    app.start_preflight()
    app.start_preflight()
    assert app.buttons["start"].options["state"] == "disabled"
    DeferredThread.queued.pop(1)()
    app.drain_events()
    assert app.report.items[0].code == "new"
    assert app.buttons["start"].options["state"] == "normal"
    _run_one_worker()
    app.drain_events()
    assert app.report.items[0].code == "new"


def test_adapter_failed_preflight_revokes_start_permission(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()

    def failing_preflight(_root):
        raise RuntimeError("api_key=must-not-leak")

    app, _root, _controller = _headless_application(
        tk_app, monkeypatch, tmp_path, preflight=failing_preflight
    )
    app.start_preflight()
    _run_one_worker()
    app.drain_events()

    assert app.buttons["start"].options["state"] == "disabled"
    assert "未完成" in app.overall_var.get()
    assert "must-not-leak" not in "".join(app.log_widget.lines)


def test_adapter_start_command_requires_current_ready_preflight_and_no_child(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    app, _root, controller = _headless_application(tk_app, monkeypatch, tmp_path)

    app.start_workbench()
    assert DeferredThread.queued == []
    assert controller.start_calls == 0

    app.start_preflight()
    app.start_workbench()
    assert len(DeferredThread.queued) == 1
    _run_one_worker()
    app.drain_events()
    app.start_workbench()
    assert len(DeferredThread.queued) == 1
    _run_one_worker()
    assert controller.start_calls == 1

    controller._state = "failed"
    controller._process = object()
    app._update_view_state()
    assert app.buttons["start"].options["state"] == "disabled"
    assert app.buttons["stop"].options["state"] == "normal"
    app.start_workbench()
    assert DeferredThread.queued == []


def test_adapter_rapid_double_start_synchronously_queues_only_one_worker(
    monkeypatch, tmp_path
):
    tk_app = _fresh_tk_app()
    app, _root, controller = _headless_application(tk_app, monkeypatch, tmp_path)
    _finish_preflight(app)

    app.start_workbench()
    app.start_workbench()

    assert len(DeferredThread.queued) == 1
    assert controller.start_calls == 0
    assert app.buttons["start"].options["state"] == "disabled"


def test_adapter_clears_start_gate_on_success_event(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    app, _root, _controller = _headless_application(tk_app, monkeypatch, tmp_path)
    _finish_preflight(app)

    app.start_workbench()
    _run_one_worker()
    assert app._start_in_flight is True

    app.drain_events()

    assert app._start_in_flight is False


def test_adapter_clears_start_gate_on_fixed_start_failure(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(
        log_callback=lambda _value: None, state_callback=lambda _value: None
    )

    def failing_start():
        controller.start_calls += 1
        raise tk_app.LauncherProcessError("fixed")

    controller.start = failing_start
    app, _root, controller = _headless_application(
        tk_app, monkeypatch, tmp_path, controller=controller
    )
    _finish_preflight(app)

    app.start_workbench()
    _run_one_worker()
    assert app._start_in_flight is True

    app.drain_events()

    assert app._start_in_flight is False
    assert controller.start_calls == 1


def test_adapter_clears_start_gate_after_readiness_cleanup_event(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(
        log_callback=lambda _value: None, state_callback=lambda _value: None
    )
    controller.ready_result = False
    app, _root, controller = _headless_application(
        tk_app, monkeypatch, tmp_path, controller=controller
    )
    _finish_preflight(app)

    app.start_workbench()
    _run_one_worker()
    assert app._start_in_flight is True

    app.drain_events()

    assert app._start_in_flight is False
    assert controller.stop_calls == 1


def test_adapter_close_cancels_queued_start_before_child_creation(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    app, root, controller = _headless_application(tk_app, monkeypatch, tmp_path)
    _finish_preflight(app)

    app.start_workbench()
    assert app._start_in_flight is True
    assert len(DeferredThread.queued) == 1

    app.on_close()

    assert root.destroyed is False
    _run_one_worker()
    assert controller.start_calls == 0
    app.drain_events()
    assert root.destroyed is True


def test_adapter_close_during_delayed_start_stops_registered_child(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(
        log_callback=lambda _value: None, state_callback=lambda _value: None
    )
    entered_start = threading.Event()
    release_start = threading.Event()
    original_start = controller.start

    def delayed_start():
        entered_start.set()
        assert release_start.wait(1.0)
        return original_start()

    controller.start = delayed_start
    app, root, controller = _headless_application(
        tk_app, monkeypatch, tmp_path, controller=controller
    )
    _finish_preflight(app)
    app.start_workbench()
    worker = threading.Thread(target=DeferredThread.queued.pop(0))
    worker.start()
    assert entered_start.wait(1.0)

    app.on_close()

    assert root.destroyed is False
    release_start.set()
    worker.join(1.0)
    assert worker.is_alive() is False
    assert controller.start_calls == 1
    assert controller.stop_calls == 1
    assert controller.process is None
    app.drain_events()
    assert root.destroyed is True


def test_adapter_start_command_rejects_failed_and_stale_preflight(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()

    def failing_preflight(_root):
        raise RuntimeError("api_key=preflight-secret")

    app, _root, controller = _headless_application(
        tk_app, monkeypatch, tmp_path, preflight=failing_preflight
    )
    app.start_preflight()
    _run_one_worker()
    app.drain_events()
    app.start_workbench()
    assert DeferredThread.queued == []
    assert controller.start_calls == 0

    app._preflight_status = "ready"
    app.start_preflight()
    app.events.enqueue("report", (app._preflight_generation - 1, PreflightReport(())))
    app.drain_events()
    assert app._preflight_status == "checking"
    app.start_workbench()
    assert len(DeferredThread.queued) == 1
    assert controller.start_calls == 0


def test_adapter_readiness_failure_stops_child_before_exposing_retry(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(log_callback=lambda _value: None, state_callback=lambda _value: None)
    controller.ready_result = False
    app, _root, controller = _headless_application(tk_app, monkeypatch, tmp_path, controller=controller)
    _finish_preflight(app)

    app.start_workbench()
    _run_one_worker()
    app.drain_events()

    assert controller.stop_calls == 1
    assert controller.process is None
    assert app.buttons["start"].options["state"] == "normal"
    assert "未能就绪" in "".join(app.log_widget.lines)


def test_adapter_readiness_cleanup_exception_keeps_owned_child_stoppable(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(log_callback=lambda _value: None, state_callback=lambda _value: None)
    controller.ready_result = False
    controller.stop_raises = True
    app, _root, controller = _headless_application(tk_app, monkeypatch, tmp_path, controller=controller)
    _finish_preflight(app)

    app.start_workbench()
    _run_one_worker()
    app.drain_events()

    assert controller.process is not None
    assert app.buttons["start"].options["state"] == "disabled"
    assert app.buttons["stop"].options["state"] == "normal"
    assert "未能就绪" in "".join(app.log_widget.lines)


def test_adapter_readiness_retained_live_child_stays_stoppable(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(log_callback=lambda _value: None, state_callback=lambda _value: None)
    controller.ready_result = False
    controller.stop_retains_child = True
    app, _root, controller = _headless_application(tk_app, monkeypatch, tmp_path, controller=controller)
    _finish_preflight(app)

    app.start_workbench()
    _run_one_worker()
    app.drain_events()

    assert controller.process is not None
    assert app.buttons["start"].options["state"] == "disabled"
    assert app.buttons["stop"].options["state"] == "normal"
    assert "未能就绪" in "".join(app.log_widget.lines)


def test_adapter_polls_unexpected_exit_updates_address_and_buttons(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(log_callback=lambda _value: None, state_callback=lambda _value: None)
    app, _root, controller = _headless_application(tk_app, monkeypatch, tmp_path, controller=controller)
    _finish_preflight(app)
    controller.start()
    app.drain_events()
    assert "127.0.0.1:7860" in app.address_var.get()
    assert app.buttons["web"].options["state"] == "disabled"
    controller.wait_until_ready()
    app.drain_events()
    assert "127.0.0.1:7860" in app.address_var.get()
    assert app.buttons["web"].options["state"] == "normal"

    controller.poll_results.append(6)
    app.drain_events()

    assert app.buttons["start"].options["state"] == "normal"
    assert app.buttons["web"].options["state"] == "disabled"
    assert "尚未就绪" in app.address_var.get()
    assert "退出码：6" in "".join(app.log_widget.lines)


def test_adapter_polls_zero_exit_and_restores_stopped_controls(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(log_callback=lambda _value: None, state_callback=lambda _value: None)
    app, _root, controller = _headless_application(tk_app, monkeypatch, tmp_path, controller=controller)
    _finish_preflight(app)
    controller.start()
    controller.wait_until_ready()
    app.drain_events()
    controller.poll_results.append(0)

    app.drain_events()

    assert app.buttons["start"].options["state"] == "normal"
    assert app.buttons["stop"].options["state"] == "disabled"
    assert "尚未就绪" in app.address_var.get()
    assert "退出码：0" in "".join(app.log_widget.lines)


def test_adapter_keeps_draining_during_slow_controlled_close(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    controller = FakeController(log_callback=lambda _value: None, state_callback=lambda _value: None)
    controller.start()
    controller.wait_until_ready()
    controller.stop_release = threading.Event()
    app, root, controller = _headless_application(tk_app, monkeypatch, tmp_path, controller=controller)
    _finish_preflight(app)
    monkeypatch.setattr(tk_app.messagebox, "askyesno", lambda *_args, **_kwargs: True)

    app.on_close()
    worker = threading.Thread(target=DeferredThread.queued.pop(0))
    worker.start()
    assert controller.stop_started.wait(1.0)
    root.run_next_callback()
    assert root.destroyed is False
    assert root.callbacks

    controller.stop_release.set()
    worker.join(1.0)
    root.run_next_callback()
    assert root.destroyed is True


def test_adapter_close_stop_exception_keeps_control_and_shows_fixed_failure(
    monkeypatch, tmp_path
):
    tk_app = _fresh_tk_app()
    controller = FakeController(
        log_callback=lambda _value: None, state_callback=lambda _value: None
    )
    controller.start()
    controller.wait_until_ready()
    controller.stop_raises = True
    app, root, controller = _headless_application(
        tk_app, monkeypatch, tmp_path, controller=controller
    )
    shown = []
    monkeypatch.setattr(tk_app.messagebox, "askyesno", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        tk_app.messagebox,
        "showerror",
        lambda _title, text, **_kwargs: shown.append(text),
    )

    app.on_close()
    _run_one_worker()
    app.drain_events()

    assert root.destroyed is False
    assert app._closing is False
    assert controller.process is not None
    assert app.buttons["stop"].options["state"] == "normal"
    assert shown == [tk_app._CLOSE_FAILURE]
    assert tk_app._CLOSE_FAILURE in "".join(app.log_widget.lines)
    assert "stop-secret" not in "".join(app.log_widget.lines)


def test_adapter_close_normal_return_with_retained_child_keeps_control(
    monkeypatch, tmp_path
):
    tk_app = _fresh_tk_app()
    controller = FakeController(
        log_callback=lambda _value: None, state_callback=lambda _value: None
    )
    controller.start()
    controller.wait_until_ready()
    controller.stop_retains_child = True
    app, root, controller = _headless_application(
        tk_app, monkeypatch, tmp_path, controller=controller
    )
    shown = []
    monkeypatch.setattr(tk_app.messagebox, "askyesno", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        tk_app.messagebox,
        "showerror",
        lambda _title, text, **_kwargs: shown.append(text),
    )

    app.on_close()
    _run_one_worker()
    app.drain_events()

    assert root.destroyed is False
    assert app._closing is False
    assert controller.process is not None
    assert app.buttons["stop"].options["state"] == "normal"
    assert shown == [tk_app._CLOSE_FAILURE]


def test_adapter_discards_events_after_closed_before_destroying_root(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()
    app, root, _controller = _headless_application(tk_app, monkeypatch, tmp_path)
    app.events.enqueue("closed")
    app.events.enqueue("log", "this must not touch a destroyed widget")
    app.events.enqueue("state", "running")

    app.drain_events()

    assert root.destroyed is True
    assert app.log_widget.lines == []


def test_main_returns_fixed_stderr_without_secret_when_tk_creation_fails(monkeypatch, tmp_path):
    tk_app = _fresh_tk_app()

    def failing_tk():
        raise RuntimeError("api_key=should-not-leak")

    monkeypatch.setattr(tk_app.tk, "Tk", failing_tk)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        assert tk_app.main(tmp_path) == 1
    assert "无法创建桌面启动窗口" in stderr.getvalue()
    assert "should-not-leak" not in stderr.getvalue()


def _read_windows_entry_script(name):
    return (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8")


def test_windows_entry_scripts_use_utf8_project_directory_and_resolved_python():
    start = _read_windows_entry_script("启动工作台.bat")
    install = _read_windows_entry_script("安装依赖.bat")

    for source in (start, install):
        assert "chcp 65001" in source
        assert 'pushd "%~dp0"' in source
        assert "popd" in source
        positions = [source.index(candidate) for candidate in ("python", "py -3", "python3")]
        assert positions == sorted(positions)
        assert "sys.executable" in source
        assert '"%PYTHON_EXE%"' in source
        assert "C:\\Users\\" not in source


def test_start_script_is_tkinter_only_and_contains_no_install_or_download_action():
    start = _read_windows_entry_script("启动工作台.bat")
    lowered = start.lower()

    assert "import tkinter" in start
    assert "launcher.pyw" in start
    for forbidden in (
        "pip", "curl", "wget", "huggingface", "snapshot_download",
        "http://", "https://", "requirements", "set http_proxy",
        "set https_proxy", "set no_proxy", "hf_hub_offline",
        "transformers_offline",
    ):
        assert forbidden not in lowered


def test_install_script_has_explicit_network_menu_and_only_requirements_install_variants():
    install = _read_windows_entry_script("安装依赖.bat")

    for label in (
        "1. 使用当前 pip 配置", "2. 官方 PyPI", "3. 清华镜像", "0. 取消",
    ):
        assert label in install
    assert "联网" in install
    assert "requirements.txt" in install
    assert "requirements-dev.txt" not in install
    for command in (
        '"%PYTHON_EXE%" -m pip install -r requirements.txt',
        '"%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.org/simple',
        '"%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple',
    ):
        assert command in install
    assert ":MENU" in install
    assert "输入无效" in install
    assert "已取消" in install
    assert "pause" in install.lower()
    assert "exit /b %PIP_EXIT_CODE%" in install


def _batch_label_block(source, label):
    normalized = source.replace("\r\n", "\n")
    marker = f"\n:{label}\n"
    start = normalized.index(marker) + 1
    next_label = normalized.find("\n:", start + len(marker) - 1)
    return normalized[start:] if next_label == -1 else normalized[start:next_label]


def _batch_nonempty_lines(block):
    return [line.strip() for line in block.replace("\r\n", "\n").splitlines() if line.strip()]


def test_windows_entry_scripts_probe_each_candidate_with_sentinel_and_existing_path_gate():
    start = _read_windows_entry_script("启动工作台.bat")
    install = _read_windows_entry_script("安装依赖.bat")
    expected = (
        ("TRY_PYTHON", "python"),
        ("TRY_PY_LAUNCHER", "py -3"),
        ("TRY_PYTHON3", "python3"),
    )

    for source, imports_tkinter in ((start, True), (install, False)):
        label_positions = [source.index(f":{label}") for label, _candidate in expected]
        assert label_positions == sorted(label_positions)
        assert source.count("CODEX_PYTHON_OK#") == 3
        assert source.count('if defined CANDIDATE_PYTHON if exist "%CANDIDATE_PYTHON%" (') == 3
        assert source.count('set "PYTHON_EXE=%CANDIDATE_PYTHON%"') == 3
        assert "PYTHONIOENCODING" not in source

        for label, candidate in expected:
            block = _batch_label_block(source, label)
            probe = (
                f'for /f "tokens=1,* delims=#" %%A in (`{candidate} -c '
                + ('"import tkinter; import sys; ' if imports_tkinter else '"import sys; ')
                + "sys.exit(3) if sys.version_info < (3, 10) else None; "
                + "sys.stdout.reconfigure(encoding='utf-8', errors='strict'); "
                + 'print(\'CODEX_PYTHON_OK#\' + sys.executable)" 2^>nul`) do ('
            )
            assert source.count(probe) == 1
            assert 'set "CANDIDATE_PYTHON="' in block
            assert probe in block
            assert 'if "%%A"=="CODEX_PYTHON_OK" (' in block
            assert 'set "CANDIDATE_PYTHON=%%B"' in block
            assert 'if defined CANDIDATE_PYTHON if exist "%CANDIDATE_PYTHON%" (' in block
            assert 'set "PYTHON_EXE=%CANDIDATE_PYTHON%"' in block
            assert 'goto :PYTHON_FOUND' in block
            assert block.index('set "CANDIDATE_PYTHON="') < block.index(probe)
            assert block.index(probe) < block.index('if defined CANDIDATE_PYTHON if exist "%CANDIDATE_PYTHON%" (')


def test_process_module_defers_annotations_for_python_entry_defense():
    source = _read_windows_entry_script("launcher/process.py")

    assert source.startswith(
        '"""Local Gradio child-process lifecycle management."""\n\n'
        "from __future__ import annotations\n"
    )


def test_windows_entry_scripts_keep_utf8_probe_output_and_explicit_success_fallthroughs():
    start = _read_windows_entry_script("启动工作台.bat").replace("\r\n", "\n")
    install = _read_windows_entry_script("安装依赖.bat").replace("\r\n", "\n")

    assert start.count("sys.stdout.reconfigure(encoding='utf-8', errors='strict')") == 3
    assert install.count("sys.stdout.reconfigure(encoding='utf-8', errors='strict')") == 3
    assert (
        '\n:INSTALL_RESULT\nif not "%PIP_EXIT_CODE%"=="0" goto :INSTALL_FAILED\n\n:INSTALL_SUCCESS\n'
        in install
    )

    python_found = _batch_nonempty_lines(_batch_label_block(start, "PYTHON_FOUND"))
    assert python_found.count('start "" /b "%PYTHON_DIR%pythonw.exe" "%CD%\\launcher.pyw"') == 1
    assert python_found.count('start "" /b "%PYTHON_EXE%" "%CD%\\launcher.pyw"') == 1
    assert python_found[-1] == "if errorlevel 1 goto :LAUNCH_FAILED"
    assert "if errorlevel 1 goto :LAUNCH_FAILED\n\n:START_SUCCESS\n" in start


def test_install_script_control_flow_has_unique_choice_commands_and_safe_terminal_cleanup():
    install = _read_windows_entry_script("安装依赖.bat")
    normalized = install.replace("\r\n", "\n")
    menu = _batch_label_block(install, "MENU")
    mappings = (
        ("1", "INSTALL_CURRENT", '"%PYTHON_EXE%" -m pip install -r requirements.txt'),
        ("2", "INSTALL_OFFICIAL", '"%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.org/simple'),
        ("3", "INSTALL_MIRROR", '"%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple'),
    )

    for choice, label, command in mappings:
        assert normalized.count(f'if "%CHOICE%"=="{choice}" goto :{label}') == 1
        assert normalized.count(f"\n:{label}\n") == 1
        lines = _batch_nonempty_lines(_batch_label_block(install, label))
        command_index = lines.index(command)
        assert sum("-m pip install" in line for line in lines) == 1
        assert lines[command_index + 1] == 'set "PIP_EXIT_CODE=%ERRORLEVEL%"'
        assert lines[command_index + 2] == "goto :INSTALL_RESULT"

    assert normalized.count('if "%CHOICE%"=="0" goto :CANCELLED') == 1
    assert menu.rstrip().endswith("goto :MENU")
    cancel = _batch_label_block(install, "CANCELLED")
    assert "-m pip" not in cancel

    terminal_expectations = (
        ("NO_PYTHON", "exit /b 1", True),
        ("INSTALL_SUCCESS", "exit /b 0", False),
        ("INSTALL_FAILED", "exit /b %PIP_EXIT_CODE%", True),
        ("CANCELLED", "exit /b 2", True),
        ("MISSING_REQUIREMENTS", "exit /b 3", True),
    )
    for label, exit_line, requires_pause in terminal_expectations:
        block = _batch_label_block(install, label)
        lines = _batch_nonempty_lines(block)
        assert "popd" in lines
        assert f"endlocal & {exit_line}" in lines
        assert ("pause" in lines) is requires_pause


def test_start_script_terminal_paths_are_explicit_and_balanced():
    start = _read_windows_entry_script("启动工作台.bat")
    terminal_expectations = (
        ("NO_PYTHON", "exit /b 1", True),
        ("START_SUCCESS", "exit /b 0", False),
        ("MISSING_LAUNCHER", "exit /b 2", True),
        ("LAUNCH_FAILED", "exit /b 3", True),
    )

    for label, exit_line, requires_pause in terminal_expectations:
        block = _batch_label_block(start, label)
        lines = _batch_nonempty_lines(block)
        assert "popd" in lines
        assert f"endlocal & {exit_line}" in lines
        assert ("pause" in lines) is requires_pause
