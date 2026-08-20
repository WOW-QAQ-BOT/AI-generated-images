from __future__ import annotations

from dataclasses import dataclass

from .api_providers.registry import provider_defaults


@dataclass(frozen=True)
class ApiUiState:
    model: str
    base_url: str
    base_url_visible: bool
    model_edited: bool


def provider_ui_state(
    provider: str,
    current_model: str,
    model_edited: bool,
    current_base_url: str,
) -> ApiUiState:
    defaults = provider_defaults(provider)
    model = str(current_model)
    if not model_edited:
        model = defaults.model
    base_url = str(current_base_url)
    if defaults.base_url_visible and not base_url.strip():
        base_url = defaults.base_url
    return ApiUiState(
        model=model,
        base_url=base_url,
        base_url_visible=defaults.base_url_visible,
        model_edited=bool(model_edited),
    )
