from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalModelUiState:
    choices: tuple[str, ...]
    value: str | None
    can_generate: bool
    message: str


def local_model_ui_state(
    models: list[str] | tuple[str, ...],
) -> LocalModelUiState:
    choices = tuple(models)
    if choices:
        return LocalModelUiState(
            choices=choices,
            value=choices[0],
            can_generate=True,
            message=f"已发现 {len(choices)} 个完整本地模型，可进行本地生成。",
        )
    return LocalModelUiState(
        choices=(),
        value=None,
        can_generate=False,
        message="未发现完整本地模型；本地生成已禁用，仍可使用 API 作画。",
    )
