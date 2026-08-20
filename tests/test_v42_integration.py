from __future__ import annotations

import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Model:
    def __init__(self, name: str, kind: str, complete: bool, message: str) -> None:
        self.name = name
        self.kind = kind
        self.complete = complete
        self.message = message
        self.path = Path("models") / name


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _markdown_section(markdown: str, heading: str, level: int) -> str:
    if level < 1:
        raise ValueError("heading level must be positive")
    marker = f"{'#' * level} {heading}"
    start = re.search(rf"^{re.escape(marker)}$", markdown, flags=re.MULTILINE)
    if start is None:
        raise ValueError(f"missing heading: {marker}")
    following = re.search(
        rf"^#{{1,{level}}} [^\n]+$",
        markdown[start.end() :],
        flags=re.MULTILINE,
    )
    end = start.end() + following.start() if following else len(markdown)
    return markdown[start.start() : end]


def test_markdown_section_does_not_swallow_sibling_headings() -> None:
    """Catch a heading helper that treats ### as a substring match for ##."""
    fixture = """## Parent
parent body
### Child A
child a body
### Child B
child b body
## Next
next body
"""

    parent = _markdown_section(fixture, "Parent", level=2)
    child_a = _markdown_section(fixture, "Child A", level=3)
    child_b = _markdown_section(fixture, "Child B", level=3)

    assert "Child A" in parent
    assert "Child B" in parent
    assert "Next" not in parent
    assert "child a body" in child_a
    assert "Child B" not in child_a
    assert "child b body" not in child_a
    assert "Next" not in child_a
    assert "child b body" in child_b
    assert "Next" not in child_b
    try:
        _markdown_section(fixture, "Child A", level=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a ## query must not match a ### heading substring")


def _function_node(source: str, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _function_source(source: str, name: str) -> str:
    text = ast.get_source_segment(source, _function_node(source, name))
    assert text is not None
    return text


def _execute_model_report(models: list[_Model]) -> str:
    source = _read("app.py")
    node = _function_node(source, "model_report_ui")
    namespace = {
        "MODELS_DIR": PROJECT_ROOT / "models",
        "discover_models": lambda _models_dir: models,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace["model_report_ui"]()


def test_v42_titles_and_offline_analytics_are_bound_to_create_ui_blocks() -> None:
    """Catch a v4 title or telemetry-enabled Blocks call outside the actual UI factory."""
    readme = _read("README.md")
    app_source = _read("app.py")
    create_ui = _function_node(app_source, "create_ui")
    create_ui_source = _function_source(app_source, "create_ui")
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

    assert readme.startswith("# AI 绘画专业工作台 v4.2\n")
    assert "AI 绘画专业工作台 v4.2" in create_ui_source
    assert "严格离线启动" in create_ui_source
    assert "仅在用户主动使用 API 生成时访问网络" in create_ui_source
    assert not re.search(r"AI 绘画专业工作台 v4(?!\.2)", readme)
    assert not re.search(r"AI 绘画专业工作台 v4(?!\.2)", app_source)
    assert len(blocks_calls) == 1
    analytics = {
        keyword.arg: keyword.value
        for keyword in blocks_calls[0].keywords
        if keyword.arg is not None
    }
    assert isinstance(analytics.get("analytics_enabled"), ast.Constant)
    assert analytics["analytics_enabled"].value is False


def test_runtime_launch_is_loopback_only_and_never_inherits_share() -> None:
    source = _read("app.py")
    launch_calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]

    assert len(launch_calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in launch_calls[0].keywords
        if keyword.arg is not None
    }
    assert isinstance(keywords.get("server_name"), ast.Constant)
    assert keywords["server_name"].value == "127.0.0.1"
    assert isinstance(keywords.get("share"), ast.Constant)
    assert keywords["share"].value is False


def test_model_report_empty_and_incomplete_only_keep_api_drawing_available() -> None:
    """Catch incomplete downloads being mistaken for a usable local model."""
    empty = _execute_model_report([])
    incomplete = _execute_model_report(
        [_Model("broken-v5", "diffusers", False, "缺少 unet 权重")]
    )

    for report in (empty, incomplete):
        assert "本地生成不可用，但 API 作画仍可使用" in report
    assert "broken-v5" in incomplete
    assert "不完整：缺少 unet 权重" in incomplete
    assert incomplete.index("本地生成不可用，但 API 作画仍可使用") < incomplete.index(
        "| 名称 | 类型 | 状态 | 路径 |"
    )


def test_model_report_with_a_complete_model_does_not_show_no_model_warning() -> None:
    """Catch a warning branch that incorrectly disables an actually usable model."""
    complete = _execute_model_report(
        [_Model("ready-v5", "single-file", True, "权重完整")]
    )

    assert "ready-v5" in complete
    assert "可用：权重完整" in complete
    assert "本地生成不可用，但 API 作画仍可使用" not in complete


def test_recommended_windows_path_has_the_exact_early_first_run_block_and_primary_semantics() -> None:
    """Catch setup steps moved away from the recommended startup path or split apart."""
    readme = _read("README.md")
    windows = _markdown_section(readme, "Windows 启动（推荐）", level=2)
    exact_steps = "\n".join(
        (
            "1. 解压项目。",
            "2. 如缺少依赖，单独运行“安装依赖.bat”。",
            "3. 双击“启动工作台.bat”。",
            "4. 查看体检结果并点击“启动工作台”。",
        )
    )

    assert readme.index("## Windows 启动（推荐）") < 2000
    assert exact_steps in windows
    assert windows.index(exact_steps) < 700
    assert "“启动工作台.bat”是 Windows 的主要启动入口。" in windows
    assert "## 高级命令行备用方式" not in windows
    for text in (
        "[严格离线]",
        "Python、运行依赖、CUDA/GPU、本地模型、端口和输出目录",
        "“启动工作台”“停止工作台”“打开网页”“重新检测”“打开模型目录”“打开输出目录”",
        "“查看安装说明”和“复制诊断信息”",
        "没有完整本地模型只是警告",
        "API 作画仍可使用",
        "不会自动下载模型",
    ):
        assert text in windows


def test_install_and_proxy_contracts_stay_in_their_respective_sections() -> None:
    """Catch offline/install safeguards being detached from the action they govern."""
    readme = _read("README.md")
    install = _markdown_section(readme, "缺少运行依赖时", level=3)
    proxy = _markdown_section(readme, "严格离线与代理", level=3)

    for text in ("明确的联网操作", "使用当前 pip 配置", "官方 PyPI", "清华镜像", "仅安装 requirements.txt", "不会自动修复依赖", "不提供 .exe"):
        assert text in install
    assert "不会清空" in proxy
    assert "保留现有的外部代理变量" in proxy
    assert "localhost" in proxy
    assert "### 严格离线与代理" not in install
    assert "## 高级命令行备用方式" not in proxy
    assert '$env:HTTP_PROXY=""' not in readme
    assert '$env:HTTPS_PROXY=""' not in readme


def test_advanced_cli_fallback_is_portable_and_keeps_the_port_command_in_its_code_block() -> None:
    """Catch a machine-specific fallback command that fails after unpacking elsewhere."""
    cli = _markdown_section(_read("README.md"), "高级命令行备用方式", level=2)
    code_blocks = re.findall(r"```powershell\n(.*?)```", cli, flags=re.DOTALL)

    assert "只有需要命令行排错时才使用下面的备用方式" in cli
    assert "一般 Windows 使用者应继续双击“启动工作台.bat”" in cli
    assert "先在项目根目录打开 PowerShell" in cli
    assert not re.search(r"\b[A-Za-z]:[\\/]", cli)
    assert code_blocks == ['$env:GRADIO_SERVER_PORT="7862"\npython app.py\n']
    assert "## API 作画" not in cli


def test_api_network_boundary_is_documented_in_the_api_section() -> None:
    """Catch API networking language moved outside the user action it describes."""
    api = _markdown_section(_read("README.md"), "API 作画", level=2)

    for text in (
        "用户输入自己的临时 API Key",
        "主动提交生成",
        "API 网络访问只会",
        "可能产生服务商费用",
    ):
        assert text in api
    assert "## 模型放置" not in api


def test_design_records_final_verification_after_task8_evidence() -> None:
    """Catch a release package whose design still claims verification is pending."""
    design = _read(
        "docs/superpowers/specs/2026-07-25-v4.2-offline-desktop-launcher-design.md"
    )

    assert re.search(r"^状态：已实施并验证$", design, flags=re.MULTILINE)
    assert "状态：实施中，等待最终验证" not in design
