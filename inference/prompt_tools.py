from __future__ import annotations


BASE_NEGATIVE = (
    "low quality, worst quality, blurry, noisy, jpeg artifacts, watermark, "
    "bad anatomy, bad hands, deformed fingers, extra fingers, missing fingers"
)


PROMPT_STYLES: dict[str, tuple[str, str]] = {
    "画质增强": (
        "masterpiece, best quality, ultra detailed, sharp focus, rich details",
        BASE_NEGATIVE,
    ),
    "人物增强": (
        "delicate face, detailed eyes, soft skin, elegant pose, beautiful composition",
        BASE_NEGATIVE + ", poorly drawn face, bad proportions",
    ),
    "光影增强": (
        "cinematic lighting, soft rim light, volumetric light, depth of field",
        BASE_NEGATIVE + ", flat lighting, overexposed, underexposed",
    ),
    "背景增强": (
        "detailed background, atmospheric perspective, layered scenery, immersive environment",
        BASE_NEGATIVE + ", empty background, messy background",
    ),
    "修手": (
        "beautiful hands, natural fingers, detailed hands",
        BASE_NEGATIVE + ", bad hands, fused fingers, broken fingers, extra fingers",
    ),
    "二次元": (
        "anime illustration, clean lineart, vibrant colors, expressive eyes",
        BASE_NEGATIVE + ", realistic skin texture, photo noise",
    ),
    "写实": (
        "photorealistic, natural skin texture, realistic lighting, 85mm lens, high detail",
        BASE_NEGATIVE + ", anime, cartoon, doll-like face",
    ),
}


def enhance_prompt(prompt: str, style: str) -> tuple[str, str]:
    prompt = (prompt or "").strip() or "1girl, silver hair, blue eyes"
    style_prompt, negative = PROMPT_STYLES.get(style, PROMPT_STYLES["画质增强"])
    if style_prompt.lower() in prompt.lower():
        return prompt, negative
    return f"{style_prompt}, {prompt}", negative
