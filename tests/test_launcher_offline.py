import pytest

from launcher.offline import build_offline_environment
from launcher.redaction import redact_launcher_text


def test_offline_environment_preserves_proxy_and_adds_local_bypass():
    base = {
        "HTTPS_PROXY": "http://proxy.example:8080",
        "NO_PROXY": "internal.example",
    }

    env = build_offline_environment(base, 7862)

    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert env["GRADIO_SHARE"] == "False"
    assert env["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert {"internal.example", "127.0.0.1", "localhost"} <= set(env["NO_PROXY"].split(","))
    assert {"127.0.0.1", "localhost"} <= set(env["no_proxy"].split(","))
    assert env["GRADIO_SERVER_PORT"] == "7862"
    assert base == {
        "HTTPS_PROXY": "http://proxy.example:8080",
        "NO_PROXY": "internal.example",
    }


def test_offline_environment_overrides_inherited_gradio_share_and_analytics():
    base = {
        "GRADIO_SHARE": "True",
        "GRADIO_ANALYTICS_ENABLED": "True",
    }

    env = build_offline_environment(base, 7862)

    assert env["GRADIO_SHARE"] == "False"
    assert env["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert base == {
        "GRADIO_SHARE": "True",
        "GRADIO_ANALYTICS_ENABLED": "True",
    }


def test_offline_environment_normalizes_each_bypass_spelling_without_removing_proxies():
    base = {
        "http_proxy": "http://lowercase-proxy.example:8080",
        "NO_PROXY": " internal.example,localhost,,internal.example, ",
        "no_proxy": "127.0.0.1,service.local,127.0.0.1",
    }

    env = build_offline_environment(base, 1)

    assert env["http_proxy"] == "http://lowercase-proxy.example:8080"
    assert env["NO_PROXY"] == "internal.example,localhost,127.0.0.1"
    assert env["no_proxy"] == "127.0.0.1,service.local,localhost"
    assert base["NO_PROXY"] == " internal.example,localhost,,internal.example, "
    assert base["no_proxy"] == "127.0.0.1,service.local,127.0.0.1"


def test_offline_environment_rejects_out_of_range_ports():
    with pytest.raises(ValueError, match="端口"):
        build_offline_environment({}, 0)
    with pytest.raises(ValueError, match="端口"):
        build_offline_environment({}, 65536)


def test_offline_environment_rejects_non_integer_port_values():
    with pytest.raises(ValueError, match="端口"):
        build_offline_environment({}, "7862")
    with pytest.raises(ValueError, match="端口"):
        build_offline_environment({}, True)


def test_launcher_redaction_removes_common_credentials():
    value = "Authorization: Bearer top-secret-token api_key=another-secret"

    cleaned = redact_launcher_text(value)

    assert "top-secret-token" not in cleaned
    assert "another-secret" not in cleaned
    assert "[REDACTED]" in cleaned


def test_launcher_redaction_removes_supplied_and_provider_prefix_secrets():
    value = "token=unit-secret sk-live-abcdefghijklmnop AIza12345678901234567890"

    cleaned = redact_launcher_text(value, ["unit-secret"])

    assert "unit-secret" not in cleaned
    assert "sk-live-" not in cleaned
    assert "AIza" not in cleaned


def test_launcher_redaction_removes_github_and_slack_style_secrets():
    value = "ghp_abcdefghijklmnopqrstuvwxyz1234567890 xoxb-1234567890-abcdef"

    cleaned = redact_launcher_text(value)

    assert "ghp_" not in cleaned
    assert "xoxb-" not in cleaned
    assert cleaned == "[REDACTED] [REDACTED]"


def test_launcher_redaction_removes_assignments_and_long_generic_tokens():
    fixtures = (
        "OPENAI_API_KEY=openai-unit-secret",
        "GEMINI_API_KEY=gemini-unit-secret",
        "token=token-unit-secret",
        "access_token=access-unit-secret",
        "sk-unitGenericCredential123456789",
        "github_pat_unitGenericCredential123456789",
    )

    for value in fixtures:
        cleaned = redact_launcher_text(value)
        assert "[REDACTED]" in cleaned
        assert value.split("=", 1)[-1] not in cleaned


def test_launcher_redaction_does_not_hide_ordinary_text_or_paths():
    value = (
        "模型路径 C:\\models\\sketch-token\\github_pat_notes.txt；"
        "token count=12；sk-short；状态正常"
    )

    assert redact_launcher_text(value) == value


def test_launcher_redaction_keeps_ordinary_chinese_and_paths_readable():
    value = "模型路径 C:\\models\\portrait，状态正常；Bearer session-token"

    cleaned = redact_launcher_text(value)

    assert cleaned == "模型路径 C:\\models\\portrait，状态正常；Bearer [REDACTED]"
