from inference.prompt_tools import enhance_prompt


def test_enhance_prompt_adds_style_prefix():
    prompt, negative = enhance_prompt("silver hair mage", "光影增强")

    assert "cinematic lighting" in prompt
    assert "silver hair mage" in prompt
    assert "flat lighting" in negative


def test_enhance_prompt_uses_default_subject_when_empty():
    prompt, _ = enhance_prompt("", "二次元")

    assert "anime illustration" in prompt
    assert "1girl" in prompt
