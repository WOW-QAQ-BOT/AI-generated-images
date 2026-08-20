# V3 Workbench Enhancements Design

Goal: build a safer and more useful local AI drawing workbench on top of the running v2 project.

Scope:
- Add a diagnostics page for Python, dependencies, CUDA, ports, and model completeness.
- Support both Diffusers folder models and single-file `.safetensors` / `.ckpt` checkpoints.
- Improve image history with favorites, reusable settings, and clearer recent output display.
- Add a prompt assistant that expands short Chinese or English ideas into Stable Diffusion prompts.
- Improve LoRA training validation and expose safer RTX 3060 6GB defaults.

Architecture:
- `inference.model_registry` owns model discovery and model loading descriptors.
- `inference.diagnostics` owns environment and model checks.
- `inference.prompt_tools` owns prompt expansion presets.
- `inference.history` owns image history and favorites metadata.
- `app.py` stays as the Gradio composition layer.

Constraints:
- Keep RTX 3060 6GB defaults conservative.
- Do not require internet at app startup.
- Do not remove the existing Diffusers folder workflow.
- Keep tests lightweight and independent of real GPU/model weights.
