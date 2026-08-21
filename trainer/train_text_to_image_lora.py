"""Small SD 1.x LoRA trainer for local image + sibling caption datasets.

The implementation follows the Diffusers training recipe while keeping the
surface area intentionally small for the desktop workbench. Heavy imports are
deferred until after argument parsing so ``--help`` remains useful when a
training dependency is missing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def local_weight_loading_options(model_path: str | Path, subfolder: str) -> dict:
    """Avoid Windows safetensors mmap crashes when equivalent local .bin weights exist."""
    component_dir = Path(model_path) / subfolder
    if component_dir.is_dir() and any(component_dir.glob("*.bin")):
        return {"use_safetensors": False, "low_cpu_mem_usage": True}
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SD 1.x UNet LoRA adapter.")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def image_caption_pairs(root: Path) -> list[tuple[Path, str]]:
    pairs = []
    for image_path in sorted(root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.is_file():
            raise FileNotFoundError(f"图片缺少同名描述文件：{image_path.name}")
        caption = caption_path.read_text(encoding="utf-8-sig").strip()
        if not caption:
            raise ValueError(f"描述文件为空：{caption_path.name}")
        pairs.append((image_path, caption))
    if not pairs:
        raise ValueError(f"训练目录中没有图片：{root}")
    return pairs


def main():
    args = parse_args()
    if not 0 < args.learning_rate <= 0.001:
        raise ValueError("学习率必须大于 0 且不超过 0.001。")

    import torch
    import torch.nn.functional as functional
    from accelerate import Accelerator
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import CLIPTextModel, CLIPTokenizer

    train_root = Path(args.train_data_dir).resolve()
    pairs = image_caption_pairs(train_root)
    model_path = args.pretrained_model_name_or_path
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    print(
        f"device={accelerator.device} mixed_precision={accelerator.mixed_precision} "
        f"images={len(pairs)}",
        flush=True,
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError(
            "CUDA 不可用：训练进程正在使用 CPU。请安装支持 CUDA 的 PyTorch。"
        )
    torch.manual_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        **local_weight_loading_options(model_path, "unet"),
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    cast_training_params(unet, dtype=torch.float32)

    preprocess = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class CaptionDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, index):
            image_path, caption = pairs[index]
            with Image.open(image_path) as image:
                pixels = preprocess(image.convert("RGB"))
            input_ids = tokenizer(
                caption,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
            return {"pixel_values": pixels, "input_ids": input_ids}

    dataloader = DataLoader(
        CaptionDataset(),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    trainable_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    epochs = math.ceil(args.max_train_steps / updates_per_epoch)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=min(100, args.max_train_steps // 10),
        num_training_steps=args.max_train_steps,
    )
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    print(
        f"training_ready batches={len(dataloader)} trainable_parameters="
        f"{sum(parameter.numel() for parameter in trainable_parameters)}",
        flush=True,
    )

    global_step = 0
    unet.train()
    for _epoch in range(epochs):
        for batch in dataloader:
            if global_step == 0 and accelerator.is_main_process:
                print("first_batch_started", flush=True)
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(
                        batch["input_ids"].to(accelerator.device)
                    )[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = (
                    noise_scheduler.get_velocity(latents, noise, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction"
                    else noise
                )
                loss = functional.mse_loss(prediction.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and (global_step == 1 or global_step % 10 == 0):
                    print(
                        f"step {global_step}/{args.max_train_steps} loss={loss.detach().item():.6f}",
                        flush=True,
                    )
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            str(output_dir),
            unet_lora_layers=lora_state,
            safe_serialization=True,
        )
        print(f"LoRA saved to {output_dir}", flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
