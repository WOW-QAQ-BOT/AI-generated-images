import ast
from pathlib import Path

from inference.model_registry import available_model_names
from launcher.checks import CheckResult, PreflightReport
from launcher.offline import build_offline_environment
from launcher.redaction import redact_launcher_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_FLAGS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
)


def test_empty_models_directory_has_no_implicit_remote_model(tmp_path):
    models = tmp_path / "models"
    models.mkdir()

    assert available_model_names(models) == []


def test_offline_environment_preserves_proxy_pairs_and_adds_loopback_bypass():
    base = {
        "HTTP_PROXY": "http://upper-http.example:8000",
        "HTTPS_PROXY": "http://upper-https.example:8443",
        "http_proxy": "http://lower-http.example:8001",
        "https_proxy": "http://lower-https.example:8444",
        "NO_PROXY": "internal.example",
        "no_proxy": "other.example",
    }

    environment = build_offline_environment(base, 7862)

    for name in OFFLINE_FLAGS:
        assert environment[name] == "1"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert environment[name] == base[name]
    assert environment["NO_PROXY"] == "internal.example,127.0.0.1,localhost"
    assert environment["no_proxy"] == "other.example,127.0.0.1,localhost"


def test_create_ui_blocks_ast_disables_gradio_analytics():
    tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))
    create_ui = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_ui"
    )
    blocks_calls = [
        item.context_expr
        for node in ast.walk(create_ui)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and isinstance(item.context_expr.func.value, ast.Name)
        and item.context_expr.func.value.id == "gr"
        and item.context_expr.func.attr == "Blocks"
    ]

    assert len(blocks_calls) == 1
    keywords = {item.arg: item.value for item in blocks_calls[0].keywords}
    assert isinstance(keywords.get("analytics_enabled"), ast.Constant)
    assert keywords["analytics_enabled"].value is False


def test_actual_gradio_launch_ast_explicitly_disables_share():
    tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]

    assert len(launch_calls) == 1
    keywords = {item.arg: item.value for item in launch_calls[0].keywords}
    assert isinstance(keywords.get("share"), ast.Constant)
    assert keywords["share"].value is False


def test_launcher_modules_do_not_import_network_or_install_clients():
    forbidden_roots = {
        "requests",
        "huggingface_hub",
        "pip",
        "openai",
        "google",
        "inference.api_providers",
    }
    violations = []
    for path in sorted((PROJECT_ROOT / "launcher").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            for name in imported:
                if any(name == root or name.startswith(root + ".") for root in forbidden_roots):
                    violations.append((path.name, name))

    assert violations == []


def test_primary_start_script_has_no_install_download_or_proxy_mutation():
    source = (PROJECT_ROOT / "启动工作台.bat").read_text(encoding="utf-8").lower()

    for forbidden in (
        " pip ",
        "-m pip",
        "install",
        "download",
        "curl",
        "wget",
        "huggingface",
        "http://",
        "https://",
        "set http_proxy",
        "set https_proxy",
        "set no_proxy",
    ):
        assert forbidden not in source


def test_missing_model_only_preflight_remains_startable():
    report = PreflightReport(
        (
            CheckResult(
                "models",
                "本地模型",
                "warn",
                "没有完整本地模型",
                "请将完整模型放入 models 目录",
            ),
        )
    )

    assert report.can_start is True


def test_release_redaction_covers_representative_secret_forms():
    fixtures = {
        "api_key=unit-release-secret": "unit-release-secret",
        "Bearer unit-bearer-secret": "unit-bearer-secret",
        "Authorization: Bearer unit-auth-secret": "unit-auth-secret",
        "key=sk-proj-releaseDummy123": "sk-proj-releaseDummy123",
        "token=AIzaReleaseDummy123": "AIzaReleaseDummy123",
        "token=ghp_releaseDummy123": "ghp_releaseDummy123",
        "token=xoxb-releaseDummy123": "xoxb-releaseDummy123",
        "http://127.0.0.1:8000/v1?api_key=query-secret": "query-secret",
    }

    for source, secret in fixtures.items():
        assert secret not in redact_launcher_text(source)


def test_readme_has_complete_compile_command_and_primary_script_split():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        "python -m compileall app.py inference launcher trainer tests launcher.pyw"
        in readme
    )
    assert "启动工作台.bat" in readme
    assert "安装依赖.bat" in readme
    assert readme.index("启动工作台.bat") < readme.index("python app.py")


def test_task6_report_has_balanced_inline_code_markers():
    report = (
        PROJECT_ROOT
        / ".superpowers"
        / "sdd"
        / "2026-07-25-v4.2-offline-desktop-launcher"
        / "task-6-report.md"
    ).read_text(encoding="utf-8")

    assert all(line.count("`") % 2 == 0 for line in report.splitlines())
