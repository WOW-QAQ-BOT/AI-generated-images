import pytest

from launcher.checks import CheckResult, PreflightReport
from launcher.view_model import derive_ui_state


def test_view_model_disables_start_on_red_error():
    report = PreflightReport(
        (CheckResult("dependency.gradio", "Gradio", "error", "未安装"),)
    )

    state = derive_ui_state(report, process_state="stopped")

    assert state.can_start is False
    assert state.can_stop is False


def test_view_model_allows_model_warning():
    report = PreflightReport(
        (CheckResult("models", "本地模型", "warn", "没有完整模型"),)
    )

    state = derive_ui_state(report, process_state="stopped")

    assert state.can_start is True


def test_view_model_has_expected_buttons_for_every_process_state():
    expectations = {
        "stopped": (True, False),
        "starting": (False, True),
        "running": (False, True),
        "stopping": (False, False),
        "failed": (True, False),
    }

    for process_state, expected_buttons in expectations.items():
        state = derive_ui_state(PreflightReport(()), process_state)
        assert (state.can_start, state.can_stop) == expected_buttons


def test_view_model_opens_web_only_for_running_nonempty_url():
    report = PreflightReport(())

    assert derive_ui_state(report, "running", " http://127.0.0.1:7860 ").can_open_web
    assert not derive_ui_state(report, "running", "   ").can_open_web
    assert not derive_ui_state(report, "starting", "http://127.0.0.1:7860").can_open_web


def test_view_model_rejects_unknown_process_state():
    with pytest.raises(ValueError, match="未知进程状态"):
        derive_ui_state(PreflightReport(()), "paused")


def test_check_result_rejects_unknown_level():
    with pytest.raises(ValueError, match="未知检查级别"):
        CheckResult("bad", "Bad", "info", "not supported")


def test_view_model_keeps_report_rows_in_order_and_is_immutable():
    rows = (
        CheckResult("python", "Python", "ok", "ready"),
        CheckResult("models", "本地模型", "warn", "missing"),
    )
    state = derive_ui_state(PreflightReport(rows), "stopped")

    assert state.rows == rows
    with pytest.raises(AttributeError):
        state.can_start = False
