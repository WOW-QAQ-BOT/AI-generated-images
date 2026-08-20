from pathlib import Path

import launcher.checks as checks
from launcher.checks import RUNTIME_PACKAGES, run_preflight


def _all_packages_present(_name):
    return object()


def _torch_warning():
    return "warn", "CUDA 不可用"


def _item(report, code):
    return next(item for item in report.items if item.code == code)


def test_missing_dependency_blocks_start(tmp_path):
    report = run_preflight(
        tmp_path,
        package_finder=lambda name: None if name == "gradio" else object(),
        torch_probe=_torch_warning,
    )

    assert report.can_start is False
    assert any(
        item.code == "dependency.gradio" and item.level == "error"
        for item in report.items
    )


def test_dependency_success_has_ok_aggregate_without_detail_errors(tmp_path):
    report = run_preflight(tmp_path, _all_packages_present, _torch_warning)

    assert _item(report, "dependencies").level == "ok"
    assert not any(item.code.startswith("dependency.") for item in report.items)
    assert set(RUNTIME_PACKAGES) == {
        "gradio", "torch", "diffusers", "transformers", "peft", "accelerate",
        "requests", "PIL", "safetensors",
    }


def test_python_result_is_reported(tmp_path):
    report = run_preflight(tmp_path, _all_packages_present, _torch_warning)

    assert _item(report, "python").level in {"ok", "warn"}
    assert _item(report, "python").summary


class _VersionInfo(tuple):
    @property
    def major(self):
        return self[0]

    @property
    def minor(self):
        return self[1]

    @property
    def micro(self):
        return self[2]


def test_python_39_is_blocking_error_with_fixed_remedy(monkeypatch, tmp_path):
    monkeypatch.setattr(checks.sys, "version_info", _VersionInfo((3, 9, 18)))

    report = run_preflight(tmp_path, _all_packages_present, _torch_warning)
    python = _item(report, "python")

    assert python.level == "error"
    assert python.summary == "Python 3.9.18；最低要求为 Python 3.10"
    assert python.remedy == "请安装 Python 3.10 或 3.11 后重新运行启动器"
    assert report.can_start is False


def test_python_310_and_311_are_supported(monkeypatch, tmp_path):
    for version in ((3, 10, 0), (3, 11, 9)):
        monkeypatch.setattr(checks.sys, "version_info", _VersionInfo(version))
        python = _item(
            run_preflight(tmp_path, _all_packages_present, _torch_warning),
            "python",
        )
        assert python.level == "ok"
        assert python.summary == f"Python {version[0]}.{version[1]}.{version[2]} 可用"
        assert python.remedy == ""


def test_python_312_plus_keeps_existing_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(checks.sys, "version_info", _VersionInfo((3, 13, 2)))

    python = _item(
        run_preflight(tmp_path, _all_packages_present, _torch_warning),
        "python",
    )

    assert python.level == "warn"
    assert python.summary == "Python 3.13.2；3.12+ 版本尚未充分验证"
    assert python.remedy == ""


def test_torch_probe_exception_becomes_warning(tmp_path):
    def exploding_probe():
        raise RuntimeError("probe must not escape")

    report = run_preflight(tmp_path, _all_packages_present, exploding_probe)

    gpu = _item(report, "gpu")
    assert gpu.level == "warn"
    assert "probe must not escape" not in gpu.summary


def test_missing_model_is_warning_and_does_not_block(tmp_path):
    report = run_preflight(tmp_path, _all_packages_present, _torch_warning)

    model = _item(report, "models")
    assert model.level == "warn"
    assert report.can_start is True
    assert (tmp_path / "models").is_dir()


def test_models_report_complete_and_incomplete_counts(tmp_path):
    models_dir = tmp_path / "models"
    complete = models_dir / "complete"
    incomplete = models_dir / "incomplete"
    for part in ("unet", "vae", "text_encoder"):
        path = complete / part
        path.mkdir(parents=True)
        (path / "weights.bin").write_bytes(b"x")
    (complete / "model_index.json").write_text("{}", encoding="utf-8")
    incomplete.mkdir(parents=True)
    (incomplete / "model_index.json").write_text("{}", encoding="utf-8")

    report = run_preflight(tmp_path, _all_packages_present, _torch_warning)

    model = _item(report, "models")
    assert model.level == "ok"
    assert "1" in model.summary
    assert "不完整" in model.summary


def test_output_path_as_file_blocks_start(tmp_path):
    (tmp_path / "outputs").write_text("not a directory", encoding="utf-8")
    report = run_preflight(
        tmp_path,
        package_finder=_all_packages_present,
        torch_probe=lambda: ("ok", "RTX 3060"),
    )

    output = _item(report, "outputs")
    assert output.level == "error"
    assert report.can_start is False
    assert output.summary == "输出目录不可写"
    assert output.remedy == "请检查 outputs 目录是否存在、为目录且具有写入权限"


def test_output_check_creates_directory_and_cleans_temporary_file(tmp_path):
    report = run_preflight(tmp_path, _all_packages_present, _torch_warning)
    outputs = tmp_path / "outputs"

    assert _item(report, "outputs").level == "ok"
    assert outputs.is_dir()
    assert list(outputs.iterdir()) == []


def test_port_row_is_stable_informational_result(tmp_path):
    report = run_preflight(tmp_path, _all_packages_present, _torch_warning)

    port = _item(report, "port")
    assert port.level == "ok"
    assert port.summary == "将在启动时选择可用端口"
