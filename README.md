# AI 绘画专业工作台 v4.2

这是一个同时支持本地 Stable Diffusion 和在线图片 API 的 Gradio 工作台。本地生成默认按 RTX 3060 6GB 显存使用保守参数。v4.2 提供严格离线的 Windows 桌面启动器：启动、体检、本地模型发现和本地生成不会自动联网。

## v4.2 主要功能

- API 作画：支持 OpenAI、Gemini 和 OpenAI 兼容图片接口。
- 使用者在网页密码框中填写自己的 API Key，不使用统一密钥。
- API Key 只在当前页面临时保留，刷新、关闭页面或重启应用后清除。
- API Key 不写入环境变量、配置、历史、收藏、图片信息或日志。
- 模型名称可以自由填写，不使用固定模型白名单。
- 服务商默认模型仅作为可编辑建议：OpenAI 为 `gpt-image-2`，Gemini 为 `gemini-3.1-flash-image`。
- API 图片保存在 `outputs/api/`，并进入现有历史与收藏系统。

## 原有功能

- 启动/模型体检：检查 Python、依赖、CUDA、端口和模型完整性。
- 支持单文件模型：`models/xxx.safetensors` 和 `models/xxx.ckpt` 可以直接显示在模型下拉框。
- 支持 Diffusers 文件夹模型：`models/anything-v5/` 仍然可用。
- 图片历史与收藏：生成记录保存在 `outputs/history.jsonl`，收藏保存在 `outputs/favorites.json`。
- 复用历史参数：从历史里选择图片，一键把提示词、种子、尺寸和高清修复参数带回生成页。
- 提示词助手：内置画质、人物、光影、背景、修手、二次元、写实增强。
- LoRA 训练检查：训练前检查数据集、描述文件和 RTX 3060 6GB 安全参数。

## Windows 启动（推荐）

首次使用请按下面顺序操作：

1. 解压项目。
2. 如缺少依赖，单独运行“安装依赖.bat”。
3. 双击“启动工作台.bat”。
4. 查看体检结果并点击“启动工作台”。

“启动工作台.bat”是 Windows 的主要启动入口。它只寻找可用的现有 Python
和 Tkinter 并打开桌面启动器；它不会安装依赖、下载模型或访问 Hugging Face。
最低要求为 Python 3.10，推荐使用 Python 3.10 或 3.11；Python 3.12 及以上版本
仍可启动，但预检会显示“尚未充分验证”的警告。
启动器以 **[严格离线]** 模式运行：桌面启动、预检、本地模型发现和本地生成均不
会自动安装、下载或查询网络资源。

启动器会显示以下六项检查：Python、运行依赖、CUDA/GPU、本地模型、端口和输出目录。
随后可使用“启动工作台”“停止工作台”“打开网页”“重新检测”“打开模型目录”“打开输出目录”
“查看安装说明”和“复制诊断信息”。

没有完整本地模型只是警告：网页工作台仍可启动，**API 作画仍可使用**，但本地生成会
被禁用。将完整 Diffusers 文件夹模型或单文件模型放入 `models/` 后，点击“重新检测”
即可更新状态；v4.2 不会自动下载模型。

### 缺少运行依赖时

“安装依赖.bat”是明确的联网操作，不会被启动器自动调用。脚本会询问安装来源：

1. 使用当前 pip 配置
2. 官方 PyPI
3. 清华镜像

它仅安装 requirements.txt 中的运行依赖，不安装开发测试依赖。若安装失败，请检查
网络或重新运行脚本后选择其他来源；v4.2 不会自动修复依赖，也不提供 .exe。

### 严格离线与代理

严格离线不会清空你的 `HTTP_PROXY`、`HTTPS_PROXY` 或其他外部代理变量；它会保留现有的外部代理变量，并为 localhost 补充绕过代理设置。这样当你明确进行 API 作画时，
仍可按自己的网络配置使用代理。

当前桌面界面使用 Tkinter，以避免额外安装桌面框架；检查、离线环境和进程管理与界面
分离，正式版可替换为 PySide6 适配器。

## 高级命令行备用方式

只有需要命令行排错时才使用下面的备用方式；一般 Windows 使用者应继续双击“启动工作台.bat”。先在项目根目录打开 PowerShell，可以用 `GRADIO_SERVER_PORT` 指定端口：

```powershell
$env:GRADIO_SERVER_PORT="7862"
python app.py
```

## API 作画

打开网页里的“API 作画”页面，然后：

1. 选择 `OpenAI`、`Gemini` 或 `OpenAI 兼容接口`。
2. 在密码框中填写你自己的 API Key。
3. 填写模型名称。可以使用建议值，也可以输入服务商支持的其他图片模型。
4. 填写提示词、尺寸、质量和生成数量。
5. 点击“使用 API 生成”。

注意：

- API 网络访问只会在用户输入自己的临时 API Key 并主动提交生成后发生；启动和本地
  生成不会代表用户联网。
- 模型名称不受本项目限制，但模型本身必须支持所选服务商的图片生成协议。
- 不要把 API Key 填入提示词、模型名称或 Base URL。
- 在线 API 调用可能产生服务商费用，也可能要求账户充值、组织验证或单独开通模型权限。
- API 图片会保存在 `outputs/api/`。

### OpenAI

建议模型为 `gpt-image-2`，也可以填写其他支持 Image API 的模型名称。项目调用 `/v1/images/generations`，支持 Base64 图片，也兼容旧模型的 URL/Base64 返回方式。

OpenAI 官方资料：

- <https://developers.openai.com/api/docs/guides/image-generation>
- <https://developers.openai.com/api/reference/resources/images/methods/generate>

### Gemini

建议模型为 `gemini-3.1-flash-image`。模型名称可以自由填写；如果填写的模型不支持图片输出，页面会显示服务商错误。

Gemini 3 图片模型会根据 `宽度x高度` 自动转换为接近的宽高比和 `1K`、`2K` 或 `4K` 输出设置。其他 Gemini 模型由服务商决定实际尺寸。

Gemini 官方资料：

- <https://ai.google.dev/gemini-api/docs/generate-content/image-generation>

### OpenAI 兼容接口

填写接口根地址，例如：

```text
http://127.0.0.1:8000/v1
```

程序会自动请求：

```text
http://127.0.0.1:8000/v1/images/generations
```

兼容接口必须使用 `http` 或 `https`，Base URL 中不能包含用户名、密码、查询参数或锚点。不同服务对尺寸、质量和数量参数的支持可能不同。

为防止兼容服务返回任意内网地址，URL 形式的图片结果只允许从与
Base URL 相同的来源下载；跨域图片 URL 会被拒绝。建议兼容服务直接
返回 `b64_json`。远程图片采用流式限量下载，单张上限为 32MB。

DALL-E 3 选择多张时，工作台会自动拆成每次 1 张的请求；DALL-E
不支持的 JPEG/WebP 选项会显示提示，并按实际返回的 PNG 保存。

### API 常见报错

- `401/403`：Key 错误、账户权限不足或没有模型访问权限。
- `429`：请求过于频繁、余额不足或额度已经用完。
- `400/404/422`：模型不支持图片生成，或该服务不支持某个尺寸/质量参数。
- 超时、DNS、代理错误：检查网络和代理设置；高质量大图可能需要接近两分钟。
- 返回内容不是图片：所选模型可能只支持文字输出。

## 模型放置

### Diffusers 文件夹模型

```text
models/
  anything-v5/
    model_index.json
    unet/
    vae/
    text_encoder/
    tokenizer/
    scheduler/
```

`unet`、`vae`、`text_encoder` 里必须有 `.bin` 或 `.safetensors` 权重。

### 单文件模型

```text
models/
  dreamshaper.safetensors
  realistic-vision.ckpt
```

文件太小的 `.safetensors` 会被标记为不完整，因为它可能只是 Git LFS 指针。

## 推荐首次测试

```text
模型：anything-v5 或你的单文件模型
尺寸：512 x 768
张数：1
步数：28
提示词强度：7
高清修复：关闭
LoRA：不使用
```

提示词：

```text
masterpiece, best quality, 1girl, silver hair, blue eyes, fantasy mage girl, glowing magic circle, soft lighting, anime illustration
```

第一张成功后再改成 `640 x 960`、张数 `4`。满意后再开高清修复，放大倍率 `1.2`，重绘强度 `0.25 - 0.30`。

## LoRA 训练数据集

本仓库附带一个已经训练完成的动漫绘画风格 LoRA：

- LoRA 名称：`my_lora`
- 权重文件：[`lora_output/my_lora/pytorch_lora_weights.safetensors`](lora_output/my_lora/pytorch_lora_weights.safetensors)
- 基础模型：`Anything v5`（Stable Diffusion 1.5 系列）
- 训练分辨率：`512`
- LoRA rank：`8`
- 推荐推理权重：`0.6 - 0.9`，建议先使用 `0.8`
- 适用方向：动漫人物、动漫头像和二次元插画风格

### 训练集来源

本次训练使用 Hugging Face 数据集
[`tenshitenshi/my_train_data`](https://huggingface.co/datasets/tenshitenshi/my_train_data)
中的 `1_qizhu` 子集。本项目不会把原始训练图片提交到 GitHub；需要复现训练时，
请从上述 Hugging Face 页面获取数据集，并自行检查数据集页面标注的许可证、图片来源
和使用限制。该数据集当前标记为 `other` 许可证，不应在未确认授权范围时用于商业用途。

### 使用仓库附带的 LoRA

1. 安装项目依赖并准备完整的 `Anything v5` Diffusers 基础模型。
2. 确认权重位于：

   ```text
   lora_output/
     my_lora/
       pytorch_lora_weights.safetensors
   ```

3. 双击“启动工作台.bat”，完成检查后启动网页工作台。
4. 进入“工作生成台”，在“LoRA”下拉框选择 `my_lora`。如果没有显示，点击“刷新 LoRA”。
5. 将“LoRA 权重”先设置为 `0.8`，输入提示词后生成图片。
6. 风格过强或人物细节失真时，将权重降低到 `0.5 - 0.7`；风格不明显时，可逐步提高到 `0.9 - 1.0`。

推荐首次测试参数：

```text
基础模型：anything-v5
LoRA：my_lora
LoRA 权重：0.8
尺寸：512 x 768
采样步数：28
提示词强度：7
```

推荐提示词：

```text
masterpiece, best quality, 1girl, anime portrait, detailed eyes, soft lighting
```

### 使用自己的训练集继续训练

训练集文件夹建议这样放：

```text
train_data/
  001.png
  001.txt
  002.png
  002.txt
```

每张图片必须有同名 `.txt` 描述文件。“训练前检查”会先检查这些问题，再给出命令预览。

在网页的“LoRA 训练”页面中：

1. 填写训练集文件夹路径。程序可以自动识别只有一个图片子目录的常见解压结构。
2. 输出名称填写新的名称；不要覆盖希望保留的 LoRA。
3. 基础模型填写 `models/anything-v5`。
4. RTX 3060 6GB 建议使用分辨率 `512`、batch size `1`、rank `8`、学习率 `0.0001`。
5. 首次先训练 `100 - 200` 步检查效果，再根据图片数量增加到 `1000` 步左右。
6. 点击“训练前检查”，确认没有数据集或参数错误后再开始训练。

训练期间不要同时进行本地图片生成；两者会争用显存。训练成功后，点击绘画页面的
“刷新 LoRA”，即可在下拉框中选择新模型。

## 验证

普通运行不需要安装 `pytest`。只有需要执行开发测试时，再安装可选的
开发依赖：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m compileall app.py inference launcher trainer tests launcher.pyw
```

如果镜像站暂时无法下载 `pytest`，不影响 `python app.py` 启动和绘画；
跳过上述开发测试依赖即可。
