import os
import subprocess
import sys
import threading

import pytest

import launcher.process as process_module
from launcher.process import (
    LauncherProcessError,
    WorkbenchProcessController,
    find_available_port,
)


class FakeStream:
    def __init__(self, lines=()):
        self.lines = list(lines)

    def readline(self):
        return self.lines.pop(0) if self.lines else ""


class FakeProcess:
    def __init__(self, *, poll_results=None, wait_results=None, stdout=None, stderr=None):
        self.poll_results = list(poll_results or [None])
        self.wait_results = list(wait_results or [0])
        self.stdout = stdout
        self.stderr = stderr
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.poll_results.pop(0) if self.poll_results else None

    def wait(self, timeout):
        result = self.wait_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class ObservedLifecycleLock:
    """A real reentrant lock which exposes the second acquisition attempt."""

    def __init__(self):
        self._lock = threading.RLock()
        self._attempt_count = 0
        self.second_attempt = threading.Event()

    def __enter__(self):
        self._attempt_count += 1
        if self._attempt_count == 2:
            self.second_attempt.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


def make_factory(process=None, error=None):
    calls = []

    def factory(*args, **kwargs):
        calls.append({"args": list(args[0]), **kwargs})
        if error is not None:
            raise error
        return process or FakeProcess()

    return factory, calls


def make_controller(tmp_path, *, process=None, **kwargs):
    (tmp_path / "app.py").write_text("pass", encoding="utf-8")
    factory, calls = make_factory(process, kwargs.pop("popen_error", None))
    controller = WorkbenchProcessController(
        tmp_path,
        popen_factory=factory,
        port_available=kwargs.pop("port_available", lambda _port: True),
        port_probe=kwargs.pop("port_probe", lambda _port: False),
        sleep=kwargs.pop("sleep", lambda _seconds: None),
        monotonic=kwargs.pop("monotonic", lambda: 0.0),
        **kwargs,
    )
    return controller, calls


def test_find_available_port_skips_occupied_ports():
    port = find_available_port(
        start=7860,
        attempts=3,
        is_available=lambda value: value == 7862,
    )
    assert port == 7862


def test_find_available_port_raises_when_range_is_full():
    with pytest.raises(LauncherProcessError, match="可用端口"):
        find_available_port(7860, 2, is_available=lambda _value: False)


def test_find_available_port_rejects_invalid_bounds():
    for start, attempts in ((0, 1), (65536, 1), (7860, 0), (7860, -1)):
        with pytest.raises(LauncherProcessError):
            find_available_port(start, attempts, is_available=lambda _value: True)


def test_find_available_port_does_not_exceed_maximum_port():
    with pytest.raises(LauncherProcessError, match="可用端口"):
        find_available_port(65535, 2, is_available=lambda _value: False)


def test_controller_rejects_duplicate_start(tmp_path):
    controller, _calls = make_controller(tmp_path)
    controller.start(7860)
    with pytest.raises(LauncherProcessError, match="已经在运行"):
        controller.start(7861)


def test_controller_uses_current_interpreter_and_offline_environment(tmp_path):
    controller, calls = make_controller(tmp_path)
    controller.start(7862)
    call = calls[0]
    assert call["args"] == [sys.executable, str(tmp_path / "app.py")]
    assert call["cwd"] == str(tmp_path)
    assert call["env"]["HF_HUB_OFFLINE"] == "1"
    assert call["env"]["GRADIO_SERVER_PORT"] == "7862"
    assert call["stdout"] is subprocess.PIPE
    assert call["stderr"] is subprocess.PIPE
    assert call["text"] is True
    assert call["encoding"] == "utf-8"
    assert call["errors"] == "replace"
    assert call["shell"] is False
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        assert call["creationflags"] == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in call


def test_start_reader_creates_and_starts_daemon_threads_for_both_streams(
    monkeypatch, tmp_path
):
    created = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(process_module.threading, "Thread", FakeThread)
    stdout = FakeStream()
    stderr = FakeStream()
    process = FakeProcess(stdout=stdout, stderr=stderr)
    controller, _calls = make_controller(tmp_path, process=process)

    controller.start(7862)

    assert [(thread.args, thread.daemon, thread.started) for thread in created] == [
        ((stdout, ""), True, True),
        ((stderr, "[stderr] "), True, True),
    ]


def test_controller_rejects_missing_application_without_starting_child(tmp_path):
    controller, calls = make_controller(tmp_path)
    (tmp_path / "app.py").unlink()
    with pytest.raises(LauncherProcessError):
        controller.start()
    assert calls == []


def test_controller_rejects_invalid_explicit_port_with_fixed_error(tmp_path):
    controller, _calls = make_controller(tmp_path)
    for value in (0, 65536, "7862", True):
        with pytest.raises(LauncherProcessError, match="端口必须是 1 到 65535 之间的整数"):
            controller.start(value)


def test_controller_uses_environment_port_when_no_explicit_port(tmp_path):
    previous = os.environ.get("GRADIO_SERVER_PORT")
    os.environ["GRADIO_SERVER_PORT"] = "7979"
    try:
        controller, _calls = make_controller(tmp_path)
        assert controller.start() == "http://127.0.0.1:7979"
    finally:
        if previous is None:
            os.environ.pop("GRADIO_SERVER_PORT", None)
        else:
            os.environ["GRADIO_SERVER_PORT"] = previous


def test_controller_rejects_invalid_environment_port_with_fixed_error(tmp_path):
    previous = os.environ.get("GRADIO_SERVER_PORT")
    os.environ["GRADIO_SERVER_PORT"] = "not-a-port"
    try:
        controller, _calls = make_controller(tmp_path)
        with pytest.raises(LauncherProcessError, match="端口必须是 1 到 65535 之间的整数"):
            controller.start()
    finally:
        if previous is None:
            os.environ.pop("GRADIO_SERVER_PORT", None)
        else:
            os.environ["GRADIO_SERVER_PORT"] = previous


def test_readiness_waits_for_loopback_probe_then_runs(tmp_path):
    probes = iter([False, False, True])
    sleeps = []
    clock = iter([0.0] * 10)
    controller, _calls = make_controller(
        tmp_path,
        port_probe=lambda _port: next(probes),
        sleep=sleeps.append,
        monotonic=lambda: next(clock),
    )
    controller.start(7860)
    assert controller.wait_until_ready(timeout=1.0, interval=0.2) is True
    assert controller.state == "running"
    assert sleeps == [0.2, 0.2]


def test_readiness_records_zero_and_abnormal_child_exit(tmp_path):
    for exit_code, state in ((0, "stopped"), (3, "failed")):
        process = FakeProcess(poll_results=[exit_code])
        controller, _calls = make_controller(tmp_path, process=process)
        controller.start(7860)
        assert controller.wait_until_ready() is False
        assert controller.last_exit_code == exit_code
        assert controller.state == state


def test_readiness_timeout_leaves_child_for_explicit_stop(tmp_path):
    messages = []
    clock = iter([0.0, 0.0, 1.0])
    process = FakeProcess()
    controller, _calls = make_controller(
        tmp_path,
        process=process,
        log_callback=messages.append,
        monotonic=lambda: next(clock),
    )
    controller.start(7860)
    assert controller.wait_until_ready(timeout=1.0) is False
    assert controller.state == "failed"
    assert controller.process is process
    assert messages == ["本地 Gradio 服务启动超时"]


def test_stop_gracefully_terminates_and_clears_process(tmp_path):
    process = FakeProcess(wait_results=[0])
    controller, _calls = make_controller(tmp_path, process=process)
    controller.start(7860)
    controller.stop(timeout=2.0)
    assert process.terminated is True
    assert process.killed is False
    assert controller.process is None
    assert controller.url == ""
    assert controller.last_exit_code == 0
    assert controller.state == "stopped"


def test_stop_kills_after_terminate_timeout(tmp_path):
    process = FakeProcess(wait_results=[subprocess.TimeoutExpired("app.py", 1), 9])
    controller, _calls = make_controller(tmp_path, process=process)
    controller.start(7860)
    controller.stop(timeout=1.0)
    assert process.terminated is True
    assert process.killed is True
    assert controller.last_exit_code == 9


def test_stop_without_process_normalizes_state(tmp_path):
    controller, _calls = make_controller(tmp_path)
    controller.stop()
    assert controller.state == "stopped"
    assert controller.url == ""


def test_stop_does_not_terminate_an_already_finished_child(tmp_path):
    process = FakeProcess(poll_results=[0])
    controller, _calls = make_controller(tmp_path, process=process)
    controller.start(7860)
    controller.stop()
    assert process.terminated is False
    assert controller.last_exit_code == 0
    assert controller.process is None
    assert controller.state == "stopped"


def test_poll_transitions_abnormal_child_exit(tmp_path):
    process = FakeProcess(poll_results=[4])
    controller, _calls = make_controller(tmp_path, process=process)
    controller.start(7860)
    assert controller.poll() == 4
    assert controller.last_exit_code == 4
    assert controller.state == "failed"


def test_reader_redacts_stdout_and_stderr_before_callback(tmp_path):
    messages = []
    controller, _calls = make_controller(tmp_path, log_callback=messages.append)
    controller._read_stream(FakeStream(["api_key=very-secret\\n"]), "")
    controller._read_stream(FakeStream(["Bearer token-value\\n"]), "[stderr] ")
    assert messages == ["api_key=[REDACTED]", "[stderr] Bearer [REDACTED]"]


def test_state_callback_receives_actual_transitions_only(tmp_path):
    states = []
    controller, _calls = make_controller(tmp_path, state_callback=states.append)
    controller.start(7860)
    controller.stop()
    controller.stop()
    assert states == ["starting", "stopping", "stopped"]


def test_popen_exception_becomes_fixed_error_and_failed_state(tmp_path):
    states = []
    controller, _calls = make_controller(
        tmp_path,
        popen_error=RuntimeError("do not expose this detail"),
        state_callback=states.append,
    )
    with pytest.raises(LauncherProcessError, match="无法启动本地 Gradio 服务") as error:
        controller.start(7860)
    assert "do not expose" not in str(error.value)
    assert controller.state == "failed"
    assert states == ["failed"]


def test_concurrent_start_records_only_one_live_child(tmp_path):
    (tmp_path / "app.py").write_text("pass", encoding="utf-8")
    first_factory_call = threading.Event()
    lifecycle_lock = ObservedLifecycleLock()
    calls = []
    results = []

    def factory(*_args, **_kwargs):
        calls.append(object())
        if len(calls) == 1:
            first_factory_call.set()
            assert lifecycle_lock.second_attempt.wait(1.0)
        return FakeProcess()

    controller = WorkbenchProcessController(
        tmp_path,
        popen_factory=factory,
        port_available=lambda _port: True,
        lifecycle_lock=lifecycle_lock,
    )

    def start_in_thread(port):
        try:
            results.append(controller.start(port))
        except LauncherProcessError:
            results.append("duplicate")

    first = threading.Thread(target=start_in_thread, args=(7860,))
    second = threading.Thread(target=start_in_thread, args=(7861,))
    first.start()
    assert first_factory_call.wait(1.0)
    second.start()
    first.join(1.0)
    second.join(1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(calls) == 1
    assert sorted(results) == ["duplicate", "http://127.0.0.1:7860"]


def test_readiness_cannot_overwrite_state_after_stop(tmp_path):
    probe_entered = threading.Event()
    release_probe = threading.Event()
    result = []
    states = []
    process = FakeProcess(wait_results=[0])

    def probe(_port):
        probe_entered.set()
        assert release_probe.wait(1.0)
        return True

    controller, _calls = make_controller(
        tmp_path,
        process=process,
        port_probe=probe,
        monotonic=lambda: 0.0,
        state_callback=states.append,
    )
    controller.start(7860)
    worker = threading.Thread(target=lambda: result.append(controller.wait_until_ready()))
    worker.start()
    assert probe_entered.wait(1.0)
    controller.stop()
    release_probe.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert result == [False]
    assert controller.state == "stopped"
    assert "running" not in states[states.index("stopped") + 1 :]


def test_readiness_times_out_before_a_probe_at_deadline(tmp_path):
    probes = []
    clock = iter([0.0, 1.0])
    controller, _calls = make_controller(
        tmp_path,
        port_probe=lambda port: probes.append(port) or True,
        monotonic=lambda: next(clock),
    )
    controller.start(7860)
    assert controller.wait_until_ready(timeout=1.0) is False
    assert controller.state == "failed"
    assert probes == []


def test_readiness_times_out_when_successful_probe_crosses_deadline(tmp_path):
    clock = iter([0.0, 0.0, 1.0])
    controller, _calls = make_controller(
        tmp_path,
        port_probe=lambda _port: True,
        monotonic=lambda: next(clock),
    )
    controller.start(7860)
    assert controller.wait_until_ready(timeout=1.0) is False
    assert controller.state == "failed"


def test_stop_keeps_child_when_post_kill_wait_times_out(tmp_path):
    messages = []
    process = FakeProcess(
        wait_results=[
            subprocess.TimeoutExpired("app.py", 1),
            subprocess.TimeoutExpired("app.py", 1),
        ]
    )
    controller, _calls = make_controller(tmp_path, process=process, log_callback=messages.append)
    controller.start(7860)
    controller.stop(timeout=1.0)

    assert process.terminated is True
    assert process.killed is True
    assert controller.process is process
    assert controller.url == "http://127.0.0.1:7860"
    assert controller.last_exit_code is None
    assert controller.state == "failed"
    assert messages == ["本地 Gradio 服务无法停止"]


def test_poll_releases_retained_failed_child_after_later_zero_exit(tmp_path):
    process = FakeProcess(
        poll_results=[None, None],
        wait_results=[
            subprocess.TimeoutExpired("app.py", 1),
            subprocess.TimeoutExpired("app.py", 1),
        ],
    )
    controller, _calls = make_controller(tmp_path, process=process)
    controller.start(7860)
    controller.stop(timeout=1.0)
    process.poll_results.append(0)

    assert controller.poll() == 0
    assert controller.last_exit_code == 0
    assert controller.state == "stopped"
    assert controller.process is None
    assert controller.url == ""


def test_poll_releases_retained_failed_child_after_later_nonzero_exit(tmp_path):
    process = FakeProcess(
        poll_results=[None, None],
        wait_results=[
            subprocess.TimeoutExpired("app.py", 1),
            subprocess.TimeoutExpired("app.py", 1),
        ],
    )
    controller, _calls = make_controller(tmp_path, process=process)
    controller.start(7860)
    controller.stop(timeout=1.0)
    process.poll_results.append(6)

    assert controller.poll() == 6
    assert controller.last_exit_code == 6
    assert controller.state == "failed"
    assert controller.process is None
    assert controller.url == ""
