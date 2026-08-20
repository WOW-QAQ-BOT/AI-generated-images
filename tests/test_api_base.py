import pytest

from inference.api_providers.base import (
    ApiImageRequest,
    ApiProviderError,
    compose_prompt,
    raise_for_api_error,
    redact_secrets,
)


def test_request_preserves_free_form_model_identifier():
    request = ApiImageRequest(
        api_key="  test-key  ",
        model="  vendor/new-image-model:beta  ",
        prompt="  moonlit city  ",
    ).normalized()

    assert request.api_key == "test-key"
    assert request.model == "vendor/new-image-model:beta"
    assert request.prompt == "moonlit city"
    assert request.count == 1


def test_request_rejects_missing_key_model_and_prompt():
    with pytest.raises(ApiProviderError, match="API Key"):
        ApiImageRequest(api_key="", model="model", prompt="prompt").normalized()
    with pytest.raises(ApiProviderError, match="模型"):
        ApiImageRequest(api_key="key", model="", prompt="prompt").normalized()
    with pytest.raises(ApiProviderError, match="提示词"):
        ApiImageRequest(api_key="key", model="model", prompt="").normalized()


def test_request_rejects_out_of_range_count_and_output_format():
    with pytest.raises(ApiProviderError, match="数量"):
        ApiImageRequest(api_key="key", model="model", prompt="prompt", count=0).normalized()
    with pytest.raises(ApiProviderError, match="格式"):
        ApiImageRequest(
            api_key="key",
            model="model",
            prompt="prompt",
            output_format="gif",
        ).normalized()


def test_compose_prompt_adds_negative_guidance_only_when_present():
    request = ApiImageRequest(
        api_key="key",
        model="model",
        prompt="portrait",
        negative_prompt="blur, watermark",
    )
    assert compose_prompt(request) == "portrait\n\nAvoid: blur, watermark"
    assert compose_prompt(
        ApiImageRequest(api_key="key", model="model", prompt="portrait")
    ) == "portrait"


def test_redaction_removes_exact_key_and_authorization_forms():
    secret = "unit-test-secret-value"
    message = (
        f"Bearer {secret}; Authorization: Bearer secondary-token; "
        f"api_key={secret}; key={secret}"
    )

    cleaned = redact_secrets(message, [secret])

    assert secret not in cleaned
    assert "secondary-token" not in cleaned
    assert "[REDACTED]" in cleaned


def test_http_error_translation_is_chinese_and_never_leaks_key():
    class Response:
        status_code = 401
        text = '{"error":"invalid key unit-test-secret-value"}'

    with pytest.raises(ApiProviderError, match="认证") as captured:
        raise_for_api_error(Response(), ["unit-test-secret-value"])

    assert "unit-test-secret-value" not in str(captured.value)


def test_http_error_translation_never_exposes_raw_provider_body():
    class Response:
        status_code = 500
        text = (
            '{"error":"internal trace",'
            '"authorization":"Bearer another-customer-secret"}'
        )

    with pytest.raises(ApiProviderError, match="暂时不可用") as captured:
        raise_for_api_error(Response(), [])

    message = str(captured.value)
    assert "internal trace" not in message
    assert "another-customer-secret" not in message
