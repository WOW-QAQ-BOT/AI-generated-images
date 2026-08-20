# V3 Workbench Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add diagnostics, single-file model support, history/favorites, prompt assistant, and LoRA training safety improvements.

**Architecture:** Keep the app small but split reusable logic into focused inference helper modules. `app.py` only wires Gradio controls to model registry, diagnostics, prompt tools, history, and training services.

**Tech Stack:** Python, Gradio, Diffusers, PyTorch, pytest-style lightweight tests.

## Global Constraints

- RTX 3060 6GB defaults remain conservative.
- Startup must not download models or load GPU weights.
- Tests must not require real model weights or CUDA.
- v3 must be packaged as a new zip without overwriting the user's working copy.

---

### Task 1: Model Registry And Diagnostics

Add tests for complete folder, incomplete folder, and single-file checkpoint discovery. Implement `ModelInfo`, `discover_models`, `available_model_names`, and `resolve_model`. Update pipeline model loading to use `from_pretrained` for folders and `from_single_file` for checkpoint files.

### Task 2: History And Favorites

Add favorites metadata in `outputs/favorites.json`, keep `history.jsonl` readable, and expose toggle/list methods for the UI.

### Task 3: Prompt Assistant

Add deterministic prompt enhancement presets for quality, portrait, lighting, background, hands, anime, and realistic styles.

### Task 4: LoRA Training Safety

Validate training folders before launching training. Expose safer RTX 3060 defaults and clearer command previews.

### Task 5: Gradio V3 UI And Packaging

Rewrite UI labels as normal Chinese, add diagnostics/history/prompt tabs, run verification, and package `AI-generated-images-professional-restart-v3.zip`.
