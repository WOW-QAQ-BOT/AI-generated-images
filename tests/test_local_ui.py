from inference.local_ui import local_model_ui_state


def test_no_local_model_disables_only_local_generation():
    state = local_model_ui_state([])

    assert state.choices == ()
    assert state.value is None
    assert state.can_generate is False
    assert "API 作画" in state.message


def test_complete_local_model_enables_generation():
    state = local_model_ui_state(["anything-v5"])

    assert state.value == "anything-v5"
    assert state.can_generate is True
