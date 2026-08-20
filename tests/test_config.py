import pytest

from inference.config import GenerationRequest, PRESETS, QUALITY_NEGATIVE_PROMPT


def test_default_rtx_3060_preset_is_portrait_batch_workflow():
    preset = PRESETS["3060_6gb_portrait"]

    assert preset.width == 640
    assert preset.height == 960
    assert preset.batch_count == 4
    assert preset.hires_fix is False


def test_generation_request_normalizes_quality_defaults():
    request = GenerationRequest(prompt="  a silver hair mage  ", width=641, height=959, seed=100)

    normalized = request.normalized()

    assert normalized.prompt == "a silver hair mage"
    assert normalized.negative_prompt == QUALITY_NEGATIVE_PROMPT
    assert normalized.width % 8 == 0
    assert normalized.height % 8 == 0
    assert normalized.seed_for_index(3) == 103


def test_generation_request_rejects_empty_prompt():
    with pytest.raises(ValueError, match="提示词"):
        GenerationRequest(prompt=" ").normalized()


def test_generation_request_rejects_too_large_resolution():
    with pytest.raises(ValueError, match="显存不足"):
        GenerationRequest(prompt="test", width=2048, height=2048).normalized()
