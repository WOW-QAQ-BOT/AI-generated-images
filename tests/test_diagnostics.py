from inference.diagnostics import run_diagnostics


def test_diagnostics_reports_api_dependency_and_temporary_key_policy(tmp_path):
    items = run_diagnostics(tmp_path, ports=[65530])

    assert any(item.name == "requests" for item in items)
    key_items = [item for item in items if item.name == "API 密钥"]
    assert len(key_items) == 1
    assert "不会读取或保存" in key_items[0].message
