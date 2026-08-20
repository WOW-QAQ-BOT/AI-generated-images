from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from inference.api_providers.base import ApiImageRequest, ApiProviderError, redact_secrets
from inference.api_providers.registry import OPENAI, provider_names
from inference.api_service import ApiImageService
from inference.api_ui import provider_ui_state
from inference.config import MODELS_DIR, PRESETS, GenerationRequest
from inference.diagnostics import diagnostics_markdown, run_diagnostics
from inference.lora_manager import LoRAManager
from inference.local_ui import local_model_ui_state
from inference.model_registry import discover_models
from inference.pipeline import InferencePipeline
from inference.prompt_tools import PROMPT_STYLES, enhance_prompt
from trainer.lora_trainer import LoRATrainer, LoRATrainingConfig


NO_LORA = "不使用 LoRA"
service = InferencePipeline()
api_service = ApiImageService()
trainer = LoRATrainer()


def _preset_label(key: str) -> str:
    preset = PRESETS[key]
    return f"{preset.name} | {preset.description}"


PRESET_LABELS = {_preset_label(key): key for key in PRESETS}
DEFAULT_PRESET_LABEL = _preset_label("3060_6gb_portrait")


def apply_preset(label: str):
    preset = PRESETS[PRESET_LABELS[label]]
    return (
        preset.width,
        preset.height,
        preset.steps,
        preset.guidance,
        preset.batch_count,
        preset.hires_fix,
        preset.hires_scale,
        preset.hires_steps,
        preset.denoise_strength,
        preset.description,
    )


def refresh_models():
    models = service.get_available_models()
    state = local_model_ui_state(models)
    return (
        gr.update(choices=list(state.choices), value=state.value),
        gr.update(interactive=state.can_generate),
        state.message,
    )


def refresh_loras():
    loras = LoRAManager.list_available_loras()
    choices = [NO_LORA] + loras
    return gr.update(choices=choices, value=NO_LORA), f"找到 {len(loras)} 个 LoRA。"


def enhance_prompt_ui(prompt, style):
    enhanced, negative = enhance_prompt(prompt, style)
    return enhanced, negative, f"已应用：{style}"


def generate_images(
    prompt,
    negative_prompt,
    preset_label,
    model_name,
    lora_name,
    lora_weight,
    width,
    height,
    steps,
    guidance,
    batch_count,
    seed,
    hires_fix,
    hires_scale,
    hires_steps,
    denoise_strength,
):
    gallery = []
    status_lines = []
    resolved_seed = None if seed in (None, "") else int(seed)
    resolved_lora = None if lora_name in (None, "", NO_LORA) else lora_name
    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        width=int(width),
        height=int(height),
        steps=int(steps),
        guidance=float(guidance),
        seed=resolved_seed,
        batch_count=int(batch_count),
        hires_fix=bool(hires_fix),
        hires_scale=float(hires_scale),
        hires_steps=int(hires_steps),
        denoise_strength=float(denoise_strength),
        model_name=model_name,
        lora_name=resolved_lora,
        lora_weight=float(lora_weight),
    )

    try:
        normalized = request.normalized()
        status_lines.append(f"开始生成：{PRESETS[PRESET_LABELS[preset_label]].name}")
        yield gallery, "\n".join(status_lines), _history_markdown()
        for _, image, path, message in service.generate_batch(normalized):
            if image is not None:
                gallery.append((image, path.name if path else "预览"))
            status_lines.append(message)
            yield gallery, "\n".join(status_lines), _history_markdown()
    except Exception as exc:
        status_lines.append(f"生成失败：{exc}")
        yield gallery, "\n".join(status_lines), _history_markdown()


def cancel_generation():
    service.cancel()
    return "正在取消。已经完成的图片会保留在 outputs 文件夹。"


def mark_api_model_edited(_value):
    return True


def api_provider_ui(provider, current_model, model_edited, current_base_url):
    state = provider_ui_state(
        provider=str(provider),
        current_model=str(current_model or ""),
        model_edited=bool(model_edited),
        current_base_url=str(current_base_url or ""),
    )
    return (
        gr.update(value=state.model),
        gr.update(value=state.base_url, visible=state.base_url_visible),
        state.model_edited,
    )


def generate_api_images(
    provider,
    api_key,
    model,
    base_url,
    prompt,
    negative_prompt,
    size,
    quality,
    count,
    output_format,
):
    gallery = []
    try:
        request = ApiImageRequest(
            api_key=str(api_key or ""),
            model=str(model or ""),
            prompt=str(prompt or ""),
            negative_prompt=str(negative_prompt or ""),
            size=str(size or "auto"),
            quality=str(quality or "auto"),
            count=int(count),
            output_format=str(output_format or "png"),
            base_url=str(base_url or ""),
        )
        outcome = api_service.generate(str(provider), request)
        gallery = [(str(path), path.name) for path in outcome.paths]
        return gallery, outcome.status, _history_markdown()
    except ApiProviderError as exc:
        message = redact_secrets(exc, [str(api_key or "")])
        return gallery, f"API 生成失败：{message}", _history_markdown()
    except Exception:
        return (
            gallery,
            "API 生成发生意外错误，请检查本地依赖、输出目录权限后重试。",
            _history_markdown(),
        )


def run_diagnostics_ui():
    return diagnostics_markdown(run_diagnostics(MODELS_DIR))


def model_report_ui():
    models = discover_models(MODELS_DIR)
    has_complete_model = any(model.complete for model in models)
    sections = []
    if not has_complete_model:
        sections.append("未发现完整本地模型：本地生成不可用，但 API 作画仍可使用。请将完整模型放入 models 目录后重新检查。")
    if not models:
        return "\n\n".join(sections)
    lines = ["| 名称 | 类型 | 状态 | 路径 |", "|---|---|---|---|"]
    for model in models:
        status = "可用" if model.complete else "不完整"
        lines.append(f"| {model.name} | {model.kind} | {status}：{model.message} | `{model.path}` |")
    sections.append("\n".join(lines))
    return "\n\n".join(sections)


def history_choices(favorites_only=False):
    rows = service.history.list_recent(limit=50, favorites_only=favorites_only)
    choices = [row.get("image_path", "") for row in rows if row.get("image_path")]
    return gr.update(choices=choices, value=choices[0] if choices else None), _history_markdown(20, favorites_only)


def toggle_favorite(image_path):
    enabled = service.history.toggle_favorite(image_path)
    state = "已收藏" if enabled else "已取消收藏"
    return state, _history_markdown(20)


def copy_history_settings(image_path):
    rows = service.history.list_recent(limit=200)
    for row in rows:
        if row.get("image_path") == image_path:
            settings = row.get("settings", {})
            return (
                row.get("prompt", ""),
                row.get("negative_prompt", ""),
                settings.get("width", 640),
                settings.get("height", 960),
                settings.get("steps", 28),
                settings.get("guidance", 7.0),
                settings.get("batch_count", 1),
                row.get("seed"),
                settings.get("hires_fix", False),
                settings.get("hires_scale", 1.2),
                settings.get("hires_steps", 12),
                settings.get("denoise_strength", 0.28),
                "已复制历史参数到生成页。",
            )
    raise gr.Error("没有找到这条历史记录。")


def open_outputs_folder_hint():
    return f"输出目录：{service.history.output_dir}"


def train_lora(train_data_dir, output_name, pretrained_model, max_steps, learning_rate, rank, resolution):
    if not str(train_data_dir).strip() or not str(output_name).strip():
        return "请填写训练集路径和输出名称。"
    try:
        trainer.config = LoRATrainingConfig(
            pretrained_model=str(pretrained_model).strip() or "runwayml/stable-diffusion-v1-5",
            max_train_steps=int(max_steps),
            learning_rate=float(learning_rate),
            rank=int(rank),
            resolution=int(resolution),
            batch_size=1,
            gradient_accumulation_steps=4,
            train_text_encoder=False,
            fp16=True,
        )
        errors = trainer.validate_dataset(str(train_data_dir).strip())
        command = trainer.get_training_command(str(train_data_dir).strip(), str(output_name).strip())
        if errors:
            return "训练前检查未通过：\n" + "\n".join(f"- {err}" for err in errors) + "\n\n命令预览：\n" + command
        output_path = trainer.train(train_data_dir=str(train_data_dir).strip(), output_name=str(output_name).strip())
        return f"训练完成，LoRA 已保存到：{output_path}"
    except Exception as exc:
        return f"训练失败：{exc}"


def preview_lora_command(train_data_dir, output_name, pretrained_model, max_steps, learning_rate, rank, resolution):
    trainer.config = LoRATrainingConfig(
        pretrained_model=str(pretrained_model).strip() or "runwayml/stable-diffusion-v1-5",
        max_train_steps=int(max_steps),
        learning_rate=float(learning_rate),
        rank=int(rank),
        resolution=int(resolution),
        batch_size=1,
        gradient_accumulation_steps=4,
        train_text_encoder=False,
        fp16=True,
    )
    errors = trainer.validate_dataset(str(train_data_dir).strip()) if str(train_data_dir).strip() else ["请先填写训练集路径。"]
    command = trainer.get_training_command(str(train_data_dir).strip() or "<训练集路径>", str(output_name).strip() or "my_lora")
    if errors:
        return "检查提示：\n" + "\n".join(f"- {err}" for err in errors) + "\n\n命令预览：\n" + command
    return "检查通过。\n\n命令预览：\n" + command


def _history_markdown(limit: int = 8, favorites_only: bool = False) -> str:
    rows = service.history.list_recent(limit=limit, favorites_only=favorites_only)
    if not rows:
        return "暂无生成历史。"
    lines = []
    for row in rows:
        seed = row.get("seed")
        path = row.get("image_path", "")
        prompt = row.get("prompt", "")[:90]
        star = "★ " if row.get("favorite") else ""
        lines.append(f"- {star}`{seed if seed is not None else 'random'}` {prompt}  \n  `{path}`")
    return "\n".join(lines)


def create_ui() -> gr.Blocks:
    models = service.get_available_models()
    local_state = local_model_ui_state(models)
    default_model = local_state.value
    loras = [NO_LORA] + LoRAManager.list_available_loras()
    preset = PRESETS["3060_6gb_portrait"]

    with gr.Blocks(
        title="AI 绘画专业工作台 v4.2",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# AI 绘画专业工作台 v4.2")
        gr.Markdown(
            "严格离线启动：本地模型发现与本地生成不会自动联网；"
            "仅在用户主动使用 API 生成时访问网络。支持自由模型名称、临时 API Key、历史收藏、诊断和 LoRA 训练检查。"
        )

        with gr.Tabs():
            with gr.TabItem("生成工作台"):
                with gr.Row():
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(label="提示词", lines=5, placeholder="例如：银发魔法少女，蓝眼睛，魔法阵，电影光")
                        negative_prompt = gr.Textbox(label="反向提示词", lines=3, placeholder="留空会自动使用质量优化反向词")
                        with gr.Row():
                            prompt_style = gr.Dropdown(label="提示词助手", choices=list(PROMPT_STYLES.keys()), value="画质增强")
                            enhance_btn = gr.Button("增强提示词")
                        prompt_status = gr.Textbox(label="提示词状态", lines=1)
                        with gr.Row():
                            generate_btn = gr.Button("生成", variant="primary", interactive=local_state.can_generate)
                            cancel_btn = gr.Button("取消")

                        gallery = gr.Gallery(label="生成结果", columns=2, height=640, object_fit="contain")
                        status = gr.Textbox(label="状态", lines=8)

                    with gr.Column(scale=3):
                        preset_dropdown = gr.Dropdown(label="质量预设", choices=list(PRESET_LABELS.keys()), value=DEFAULT_PRESET_LABEL)
                        preset_note = gr.Markdown(preset.description)
                        gr.Markdown(local_state.message)
                        with gr.Row():
                            model_selector = gr.Dropdown(label="模型", choices=list(local_state.choices), value=local_state.value)
                            refresh_models_btn = gr.Button("刷新模型")
                        with gr.Row():
                            lora_selector = gr.Dropdown(label="LoRA", choices=loras, value=NO_LORA)
                            refresh_loras_btn = gr.Button("刷新 LoRA")
                        lora_weight = gr.Slider(0, 1.5, value=0.8, step=0.05, label="LoRA 权重")

                        with gr.Accordion("尺寸与质量", open=True):
                            with gr.Row():
                                width = gr.Slider(256, 1024, value=preset.width, step=64, label="宽度")
                                height = gr.Slider(256, 1536, value=preset.height, step=64, label="高度")
                            steps = gr.Slider(10, 60, value=preset.steps, step=1, label="采样步数")
                            guidance = gr.Slider(1, 15, value=preset.guidance, step=0.5, label="提示词强度")
                            batch_count = gr.Slider(1, 8, value=preset.batch_count, step=1, label="顺序生成张数")
                            seed = gr.Number(label="种子，留空为随机", value=None, precision=0)

                        with gr.Accordion("高清修复", open=False):
                            hires_fix = gr.Checkbox(label="启用高清修复", value=preset.hires_fix)
                            hires_scale = gr.Slider(1.0, 1.6, value=preset.hires_scale, step=0.05, label="放大倍率")
                            hires_steps = gr.Slider(4, 30, value=preset.hires_steps, step=1, label="高清修复步数")
                            denoise_strength = gr.Slider(0.05, 0.65, value=preset.denoise_strength, step=0.01, label="重绘强度")

                history = gr.Markdown(_history_markdown())

            with gr.TabItem("API 作画"):
                gr.Markdown(
                    "API Key 只在当前网页中临时保留，刷新、关闭页面或重启应用后清除；"
                    "不会写入配置、历史记录或日志。API 调用可能产生服务商费用。"
                )
                api_model_edited = gr.State(False)
                with gr.Row():
                    with gr.Column(scale=5):
                        api_prompt = gr.Textbox(
                            label="提示词",
                            lines=5,
                            placeholder="描述你想生成的图片",
                        )
                        api_negative_prompt = gr.Textbox(
                            label="反向提示词（可选）",
                            lines=3,
                            placeholder="例如：模糊、水印、低画质",
                        )
                        api_generate_btn = gr.Button("使用 API 生成", variant="primary")
                        api_gallery = gr.Gallery(
                            label="API 生成结果",
                            columns=2,
                            height=640,
                            object_fit="contain",
                        )
                        api_status = gr.Textbox(label="API 状态", lines=8)
                    with gr.Column(scale=3):
                        api_provider = gr.Dropdown(
                            label="API 服务商",
                            choices=provider_names(),
                            value=OPENAI,
                        )
                        api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="仅在当前网页会话临时使用",
                        )
                        api_model = gr.Textbox(
                            label="模型名称",
                            value="gpt-image-2",
                            placeholder="可自由填写服务商支持的图片模型名称",
                        )
                        api_base_url = gr.Textbox(
                            label="兼容接口 Base URL",
                            value="",
                            placeholder="例如：http://127.0.0.1:8000/v1",
                            visible=False,
                        )
                        api_size = gr.Textbox(
                            label="图片尺寸",
                            value="1024x1024",
                            placeholder="例如：1024x1024、2048x1152 或 auto",
                        )
                        api_quality = gr.Dropdown(
                            label="质量",
                            choices=["auto", "low", "medium", "high"],
                            value="auto",
                        )
                        api_count = gr.Slider(
                            1,
                            4,
                            value=1,
                            step=1,
                            label="生成数量",
                        )
                        api_output_format = gr.Dropdown(
                            label="输出格式",
                            choices=["png", "jpeg", "webp"],
                            value="png",
                        )
                api_history = gr.Markdown(_history_markdown())

            with gr.TabItem("历史与收藏"):
                with gr.Row():
                    history_selector = gr.Dropdown(label="历史图片", choices=[])
                    refresh_history_btn = gr.Button("刷新历史")
                    refresh_favorites_btn = gr.Button("只看收藏")
                with gr.Row():
                    favorite_btn = gr.Button("收藏/取消收藏")
                    copy_settings_btn = gr.Button("复用这张参数")
                    outputs_btn = gr.Button("显示输出目录")
                favorite_status = gr.Textbox(label="操作结果", lines=2)
                history_panel = gr.Markdown(_history_markdown(20))

            with gr.TabItem("诊断"):
                gr.Markdown("检查依赖、CUDA、端口和模型完整性。")
                with gr.Row():
                    diagnostics_btn = gr.Button("运行环境体检", variant="primary")
                    model_report_btn = gr.Button("检查模型")
                diagnostics_output = gr.Markdown(run_diagnostics_ui())
                model_output = gr.Markdown(model_report_ui())

            with gr.TabItem("LoRA 训练"):
                gr.Markdown("RTX 3060 6GB 建议 batch size 1、不开 text encoder，先用 512 分辨率训练。")
                with gr.Row():
                    with gr.Column():
                        train_data_dir = gr.Textbox(label="训练集路径", placeholder="图片和同名 .txt 描述文件所在文件夹")
                        output_name = gr.Textbox(label="输出名称", value="my_lora")
                        pretrained_model = gr.Textbox(label="基础模型路径或名称", value="runwayml/stable-diffusion-v1-5")
                        max_steps = gr.Number(label="训练步数", value=1000, precision=0)
                        learning_rate = gr.Number(label="学习率", value=1e-4)
                        rank = gr.Slider(2, 32, value=4, step=1, label="LoRA rank")
                        resolution = gr.Slider(384, 768, value=512, step=64, label="训练分辨率")
                        with gr.Row():
                            preview_train_btn = gr.Button("训练前检查")
                            train_btn = gr.Button("开始训练", variant="primary")
                    train_output = gr.Textbox(label="训练日志", lines=18)

        enhance_btn.click(fn=enhance_prompt_ui, inputs=[prompt, prompt_style], outputs=[prompt, negative_prompt, prompt_status])
        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[width, height, steps, guidance, batch_count, hires_fix, hires_scale, hires_steps, denoise_strength, preset_note],
        )
        refresh_models_btn.click(fn=refresh_models, inputs=[], outputs=[model_selector, generate_btn, status])
        refresh_loras_btn.click(fn=refresh_loras, inputs=[], outputs=[lora_selector, status])
        cancel_btn.click(fn=cancel_generation, inputs=[], outputs=[status])
        generate_btn.click(
            fn=generate_images,
            inputs=[
                prompt, negative_prompt, preset_dropdown, model_selector, lora_selector, lora_weight,
                width, height, steps, guidance, batch_count, seed, hires_fix, hires_scale, hires_steps, denoise_strength,
            ],
            outputs=[gallery, status, history],
        )
        api_model.input(
            fn=mark_api_model_edited,
            inputs=[api_model],
            outputs=[api_model_edited],
        )
        api_provider.change(
            fn=api_provider_ui,
            inputs=[api_provider, api_model, api_model_edited, api_base_url],
            outputs=[api_model, api_base_url, api_model_edited],
        )
        api_generate_btn.click(
            fn=generate_api_images,
            inputs=[
                api_provider,
                api_key,
                api_model,
                api_base_url,
                api_prompt,
                api_negative_prompt,
                api_size,
                api_quality,
                api_count,
                api_output_format,
            ],
            outputs=[api_gallery, api_status, api_history],
        )
        refresh_history_btn.click(fn=lambda: history_choices(False), inputs=[], outputs=[history_selector, history_panel])
        refresh_favorites_btn.click(fn=lambda: history_choices(True), inputs=[], outputs=[history_selector, history_panel])
        favorite_btn.click(fn=toggle_favorite, inputs=[history_selector], outputs=[favorite_status, history_panel])
        outputs_btn.click(fn=open_outputs_folder_hint, inputs=[], outputs=[favorite_status])
        copy_settings_btn.click(
            fn=copy_history_settings,
            inputs=[history_selector],
            outputs=[
                prompt, negative_prompt, width, height, steps, guidance, batch_count, seed,
                hires_fix, hires_scale, hires_steps, denoise_strength, favorite_status,
            ],
        )
        diagnostics_btn.click(fn=run_diagnostics_ui, inputs=[], outputs=[diagnostics_output])
        model_report_btn.click(fn=model_report_ui, inputs=[], outputs=[model_output])
        preview_train_btn.click(
            fn=preview_lora_command,
            inputs=[train_data_dir, output_name, pretrained_model, max_steps, learning_rate, rank, resolution],
            outputs=[train_output],
        )
        train_btn.click(
            fn=train_lora,
            inputs=[train_data_dir, output_name, pretrained_model, max_steps, learning_rate, rank, resolution],
            outputs=[train_output],
        )

    return demo


demo = create_ui()


if __name__ == "__main__":
    no_proxy_items = {
        item.strip()
        for item in os.environ.get("NO_PROXY", "").split(",")
        if item.strip()
    }
    no_proxy_items.update({"127.0.0.1", "localhost"})
    os.environ["NO_PROXY"] = ",".join(sorted(no_proxy_items))
    configured_port = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    port = int(configured_port) if configured_port else None
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=True,
        show_error=True,
    )
