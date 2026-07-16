# AI 文本生成绘画

基于 Stable Diffusion + LoRA 微调的文本到图像生成系统。

## 环境要求

- Python 3.10+（本项目使用 Python 3.13）
- NVIDIA GPU (RTX 3060 12GB 验证通过)
- CUDA 12.4

## 快速开始

### 1. 安装依赖

```bash
cd text2img-project
python -m venv venv
# Windows:
venv\Scripts\activate
# 或 Linux: source venv/bin/activate
pip install -r requirements.txt
```

**注意：** 安装 PyTorch CUDA 版本时，如遇网络问题，可使用镜像：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. 首次运行 - 下载模型

首次运行会从 Hugging Face 下载 Stable Diffusion v1.5 模型（约 4GB）：

```bash
python test_pipeline.py
```

确保网络畅通，或提前配置 `HF_ENDPOINT` 镜像。

### 3. 启动 Web 界面

```bash
python app.py
```

访问 http://localhost:7860

### 4. 生成图片

在"文生图"标签页输入描述文字，点击"生成图片"。

### 5. 训练 LoRA

1. 准备训练集：图片文件夹，每张图片配同名 `.txt` 描述文件
2. 在"LoRA 训练"标签页填写路径和参数
3. 点击"开始训练"
4. 训练完成后在"文生图"页面加载并使用

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| 推理步数 | 去噪步数，越高质量越好但越慢 | 20-50 |
| 引导系数 | 对描述的遵循程度 | 7-8 |
| LoRA rank | 微调参数量，越大效果越好但训练越慢 | 4-8 |
| 学习率 | 训练速度 | 1e-4 |

## 项目结构

```
text2img-project/
├── app.py                  # Gradio 主界面
├── inference/
│   ├── pipeline.py         # SD 推理引擎
│   └── lora_manager.py     # LoRA 权重管理
├── trainer/
│   └── lora_trainer.py    # LoRA 训练器
└── requirements.txt
```

## 常见问题

**Q: 提示 `CUDA available: False`**
A: 确认安装了 CUDA 版 PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/cu124`

**Q: 模型下载失败**
A: 配置 Hugging Face 镜像或手动下载模型到 `~/.cache/huggingface/` 目录

**Q: 显存不足 (OOM)**
A: 减小图片分辨率（width/height 调低到 384），或降低 batch_size
