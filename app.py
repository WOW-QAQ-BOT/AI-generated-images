import gradio as gr
import torch
import os
from inference.pipeline import InferencePipeline
from trainer.lora_trainer import LoRATrainer, LoRATrainingConfig

# 全局实例
pipe = InferencePipeline()
trainer = LoRATrainer()

def generate_image(prompt, negative_prompt, steps, guidance, seed, width, height, model_name, hires_fix, upscale_steps):
    if not prompt.strip():
        return None, "请输入描述文字"

    # 切换模型
    model_path = pipe.get_model_path(model_name)
    if pipe.current_model_path != model_path:
        pipe.load_model(model_path)

    try:
        img = pipe.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            seed=int(seed) if seed else None,
            width=int(width),
            height=int(height),
            hires_fix=hires_fix,
            upscale_steps=int(upscale_steps),
        )
        return img, "生成成功" + (" (Hires.fix)" if hires_fix else "")
    except Exception as e:
        return None, f"错误: {str(e)}"

def train_lora(train_data_dir, output_name, max_steps, learning_rate):
    if not train_data_dir.strip() or not output_name.strip():
        return "错误: 请填写训练集路径和输出名称"
    try:
        config = LoRATrainingConfig(
            max_train_steps=int(max_steps),
            learning_rate=float(learning_rate),
        )
        trainer.config = config
        output_path = trainer.train(
            train_data_dir=train_data_dir,
            output_name=output_name,
        )
        return f"训练完成！LoRA 保存至: {output_path}"
    except Exception as e:
        return f"训练错误: {str(e)}"

def refresh_loras():
    from inference.lora_manager import LoRAManager
    loras = LoRAManager.list_available_loras()
    return gr.update(choices=loras), f"发现 {len(loras)} 个 LoRA"

def load_lora_to_pipeline(lora_name):
    from inference.lora_manager import LoRAManager
    loras = LoRAManager.list_available_loras()
    if lora_name and lora_name in loras:
        lora_path = os.path.join("./lora_output", lora_name, "unet_lora")
        if os.path.exists(lora_path):
            pipe.lora_manager.load_lora(lora_name, lora_path)
            return f"已加载 LoRA: {lora_name}"
    return "未找到指定 LoRA"

def refresh_models():
    models = pipe.get_available_models()
    return gr.update(choices=models), f"发现 {len(models)} 个模型"

# 获取可用模型
available_models = pipe.get_available_models()
default_model = "anything-v5" if "anything-v5" in available_models else available_models[0] if available_models else "stable-diffusion-v1-5"

with gr.Blocks(title="AI 作画") as demo:
    gr.Markdown("# AI 文本生成绘画")
    gr.Markdown("基于 Stable Diffusion + LoRA 微调 | 支持多模型切换和 Hires.fix 高清修复")

    with gr.Tabs():
        with gr.TabItem("文生图"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(
                        label="选择模型",
                        choices=available_models,
                        value=default_model,
                        interactive=True,
                    )
                    with gr.Row():
                        refresh_models_btn = gr.Button("刷新模型")

                    prompt = gr.Textbox(label="描述文字", placeholder="输入图片描述")
                    negative_prompt = gr.Textbox(label="反向描述", placeholder="不希望出现的元素（留空使用默认）")

                    with gr.Accordion("参数设置", open=True):
                        steps = gr.Slider(10, 100, value=30, step=1, label="推理步数")
                        guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="引导系数")
                        width = gr.Slider(256, 1024, value=512, step=64, label="宽度")
                        height = gr.Slider(256, 1024, value=512, step=64, label="高度")
                        seed = gr.Number(label="随机种子（留空则随机）", value=None)
                        with gr.Accordion("高清修复 (Hires.fix)", open=False):
                            hires_fix = gr.Checkbox(label="启用 Hires.fix（先生成低分辨率再放大细化）", value=False)
                            upscale_steps = gr.Slider(10, 50, value=20, step=1, label="放大细化步数")

                    generate_btn = gr.Button("生成图片", variant="primary")

                    gr.Markdown("---")
                    lora_selector = gr.Dropdown(label="加载 LoRA", choices=[], interactive=True)
                    with gr.Row():
                        refresh_btn = gr.Button("刷新 LoRA")
                        load_lora_btn = gr.Button("应用 LoRA")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="生成的图片", type="pil")
                    status = gr.Textbox(label="状态")

            generate_btn.click(
                fn=generate_image,
                inputs=[prompt, negative_prompt, steps, guidance, seed, width, height, model_selector, hires_fix, upscale_steps],
                outputs=[output_image, status],
            )
            refresh_models_btn.click(fn=refresh_models, inputs=[], outputs=[model_selector, status])
            refresh_btn.click(fn=refresh_loras, inputs=[], outputs=[lora_selector, status])
            load_lora_btn.click(fn=load_lora_to_pipeline, inputs=[lora_selector], outputs=[status])

        with gr.TabItem("LoRA 训练"):
            gr.Markdown("## 训练自己的 LoRA 风格")
            gr.Markdown("准备训练集：图片文件夹，每张图片配同名 .txt 描述文件")
            with gr.Row():
                with gr.Column(scale=1):
                    train_data_dir = gr.Textbox(label="训练集路径", placeholder="图片文件夹路径")
                    output_name = gr.Textbox(label="输出名称", placeholder="my_style")
                    max_steps = gr.Number(label="训练步数", value=1000)
                    learning_rate = gr.Number(label="学习率", value=1e-4)
                    train_btn = gr.Button("开始训练", variant="primary")
                with gr.Column(scale=1):
                    train_output = gr.Textbox(label="训练日志", lines=20)

            train_btn.click(
                fn=train_lora,
                inputs=[train_data_dir, output_name, max_steps, learning_rate],
                outputs=[train_output],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
