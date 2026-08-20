from inference.api_ui import provider_ui_state


def test_provider_change_does_not_overwrite_manually_edited_model():
    state = provider_ui_state(
        provider="Gemini",
        current_model="my-private-image-model",
        model_edited=True,
        current_base_url="",
    )

    assert state.model == "my-private-image-model"
    assert state.model_edited is True
    assert state.base_url_visible is False


def test_provider_change_supplies_suggestion_before_manual_edit():
    state = provider_ui_state(
        provider="Gemini",
        current_model="gpt-image-2",
        model_edited=False,
        current_base_url="",
    )

    assert state.model == "gemini-3.1-flash-image"
    assert state.model_edited is False


def test_compatible_provider_shows_default_base_url_without_storing_a_key():
    state = provider_ui_state(
        provider="OpenAI 兼容接口",
        current_model="",
        model_edited=False,
        current_base_url="",
    )

    assert state.base_url == "http://127.0.0.1:8000/v1"
    assert state.base_url_visible is True
    assert not hasattr(state, "api_key")
