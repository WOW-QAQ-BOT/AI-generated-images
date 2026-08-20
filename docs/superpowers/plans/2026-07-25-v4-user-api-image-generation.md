# V4 User-Supplied API Image Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add secure, session-temporary API image generation for OpenAI, Gemini, and OpenAI-compatible services without restricting the user's model identifier.

**Architecture:** Stateless provider adapters receive the API key as a request argument, perform one protocol-specific HTTP request, and return image bytes without retaining secrets. A separate API image service validates inputs, saves images through the existing history layer, and exposes sanitized results to a new Gradio tab.

**Tech Stack:** Python 3.10+, Gradio 4+, Requests 2.31+, Pillow, pytest, mocked HTTP responses

## Global Constraints

- API keys are entered in a password textbox and remain only in the current page session.
- API keys must never enter environment variables, files, history, favorites, diagnostics, logs, exceptions shown to users, or the release archive.
- No process-global key, shared key cache, or provider-instance key attribute is allowed.
- Model identifiers are arbitrary non-empty strings; provider defaults are editable suggestions, not an allowlist.
- V4 supports OpenAI, Gemini, and OpenAI-compatible `/images/generations` services.
- Real API calls, billing, user accounts, and persistent key storage are outside automated tests and release verification.
- The project is not a Git repository. Do not initialize Git without user authorization; replace commit steps with verified local checkpoints.
- Current official default suggestions: OpenAI `gpt-image-2`; Gemini `gemini-3.1-flash-image`.
- Official protocol references:
  - `https://developers.openai.com/api/reference/resources/images/methods/generate`
  - `https://developers.openai.com/api/docs/guides/image-generation`
  - `https://ai.google.dev/gemini-api/docs/generate-content/image-generation`

## File Structure

- Create `inference/api_providers/__init__.py`: public exports.
- Create `inference/api_providers/base.py`: request/result contracts, validation, redaction, HTTP error translation.
- Create `inference/api_providers/openai_provider.py`: OpenAI Images API adapter and base64/URL response parsing.
- Create `inference/api_providers/gemini_provider.py`: Gemini `generateContent` adapter and inline-image parsing.
- Create `inference/api_providers/openai_compatible_provider.py`: configurable compatible endpoint adapter.
- Create `inference/api_providers/registry.py`: provider names, editable suggestions, and stateless provider construction.
- Create `inference/api_service.py`: orchestration, image validation, saving, and history integration.
- Create `inference/api_ui.py`: pure provider-field state transitions, independent of Gradio imports.
- Create `tests/test_api_base.py`: validation and secret-redaction tests.
- Create `tests/test_api_providers.py`: mocked provider request/response tests.
- Create `tests/test_api_service.py`: saving, history, and secret-isolation tests.
- Create `tests/test_api_ui.py`: provider-control behavior and free-form model pass-through tests.
- Modify `inference/diagnostics.py`: add Requests/API capability checks without reading keys.
- Modify `app.py`: add the API tab and event handlers.
- Modify `requirements.txt`: add explicit Requests dependency.
- Modify `README.md`: add temporary-key usage, provider setup, charges, and troubleshooting.

---

### Task 1: Provider contracts, validation, and secret redaction

**Files:**
- Create: `inference/api_providers/__init__.py`
- Create: `inference/api_providers/base.py`
- Create: `tests/test_api_base.py`

**Interfaces:**
- Produces: `ApiImageRequest`, `ApiImageResult`, `ApiProviderError`, `ApiProvider`, `compose_prompt()`, `redact_secrets()`, `raise_for_api_error()`.
- `ApiImageRequest` fields: `api_key`, `model`, `prompt`, `negative_prompt`, `size`, `quality`, `count`, `output_format`, `base_url`.
- `ApiImageResult` fields: `image_bytes`, `mime_type`, `revised_prompt`, `notes`.

- [ ] **Step 1: Write failing contract and validation tests**

```python
def test_request_keeps_free_form_model_and_validates_required_fields():
    request = ApiImageRequest(
        api_key="unit-test-secret",
        model="vendor/new-image-model:beta",
        prompt="moonlit city",
    ).normalized()
    assert request.model == "vendor/new-image-model:beta"
    assert request.count == 1

def test_redaction_removes_key_and_authorization_forms():
    message = "Bearer unit-test-secret api_key=unit-test-secret"
    cleaned = redact_secrets(message, ["unit-test-secret"])
    assert "unit-test-secret" not in cleaned
    assert "[REDACTED]" in cleaned
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run:

```powershell
python -m pytest tests/test_api_base.py -q
```

Expected: collection fails because `inference.api_providers.base` does not exist.

- [ ] **Step 3: Implement immutable request/result contracts**

```python
@dataclass(frozen=True)
class ApiImageRequest:
    api_key: str
    model: str
    prompt: str
    negative_prompt: str = ""
    size: str = "1024x1024"
    quality: str = "auto"
    count: int = 1
    output_format: str = "png"
    base_url: str = ""

    def normalized(self) -> "ApiImageRequest":
        key = self.api_key.strip()
        model = self.model.strip()
        prompt = self.prompt.strip()
        if not key:
            raise ApiProviderError("请填写 API Key。")
        if not model:
            raise ApiProviderError("请填写模型名称。")
        if not prompt:
            raise ApiProviderError("请填写提示词。")
        if not 1 <= int(self.count) <= 10:
            raise ApiProviderError("生成数量必须在 1 到 10 之间。")
        return replace(
            self,
            api_key=key,
            model=model,
            prompt=prompt,
            negative_prompt=self.negative_prompt.strip(),
            size=self.size.strip() or "auto",
            quality=self.quality.strip() or "auto",
            count=int(self.count),
            output_format=self.output_format.strip().lower() or "png",
            base_url=self.base_url.strip(),
        )

@dataclass(frozen=True)
class ApiImageResult:
    image_bytes: bytes
    mime_type: str = "image/png"
    revised_prompt: str | None = None
    notes: tuple[str, ...] = ()
```

Define the stateless provider protocol explicitly:

```python
class ApiProvider(Protocol):
    def generate(
        self,
        request: ApiImageRequest,
        session: Any | None = None,
    ) -> list[ApiImageResult]: ...
```

`compose_prompt()` appends `"\n\nAvoid: <negative prompt>"` only when a negative prompt is present. `redact_secrets()` replaces exact submitted secrets, bearer-token patterns, and common `api_key=` forms. `raise_for_api_error()` maps status codes 401/403, 429, 400/404, and 5xx to short Chinese messages and redacts response text before raising `ApiProviderError`.

- [ ] **Step 4: Run base tests**

Run:

```powershell
python -m pytest tests/test_api_base.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Record the local checkpoint**

Run:

```powershell
python -m compileall inference/api_providers tests/test_api_base.py
```

Expected: compilation succeeds. Git commit is skipped because this project has no Git repository.

---

### Task 2: Provider registry and editable defaults

**Files:**
- Create: `inference/api_providers/registry.py`
- Modify: `inference/api_providers/__init__.py`
- Create: `tests/test_api_providers.py`

**Interfaces:**
- Produces: `ProviderDefaults(model: str, base_url: str, base_url_visible: bool)`.
- Produces: `provider_names() -> list[str]`, `provider_defaults(name: str) -> ProviderDefaults`, `create_provider(name: str) -> ApiProvider`.
- Provider display names are exactly `OpenAI`, `Gemini`, and `OpenAI 兼容接口`.

- [ ] **Step 1: Write failing registry tests**

```python
def test_registry_returns_editable_suggestions_not_model_validation():
    assert provider_defaults("OpenAI").model == "gpt-image-2"
    assert provider_defaults("Gemini").model == "gemini-3.1-flash-image"
    compatible = provider_defaults("OpenAI 兼容接口")
    assert compatible.base_url_visible is True
    assert compatible.base_url == "http://127.0.0.1:8000/v1"
```

- [ ] **Step 2: Run the registry test and confirm failure**

Run:

```powershell
python -m pytest tests/test_api_providers.py::test_registry_returns_editable_suggestions_not_model_validation -q
```

Expected: import or symbol failure.

- [ ] **Step 3: Implement the registry**

Use constant metadata only for suggestions and UI visibility. `create_provider()` returns a new stateless adapter and rejects only unknown provider display names. It never validates a model string against defaults.

- [ ] **Step 4: Run registry tests**

Run:

```powershell
python -m pytest tests/test_api_providers.py::test_registry_returns_editable_suggestions_not_model_validation -q
```

Expected: pass.

- [ ] **Step 5: Record the local checkpoint**

Run:

```powershell
python -m compileall inference/api_providers
```

Expected: compilation succeeds.

---

### Task 3: OpenAI and OpenAI-compatible adapters

**Files:**
- Create: `inference/api_providers/openai_provider.py`
- Create: `inference/api_providers/openai_compatible_provider.py`
- Modify: `inference/api_providers/registry.py`
- Modify: `tests/test_api_providers.py`

**Interfaces:**
- `OpenAIProvider.generate(request: ApiImageRequest, session=None) -> list[ApiImageResult]`.
- `OpenAICompatibleProvider.generate(request: ApiImageRequest, session=None) -> list[ApiImageResult]`.
- Both accept an injected Requests-like session for tests and construct no state containing a key.

- [ ] **Step 1: Write failing OpenAI request and response tests**

```python
def test_openai_passes_free_form_model_and_parses_base64(fake_session):
    request = ApiImageRequest(
        api_key="secret-value",
        model="vendor/free-form-model",
        prompt="cat astronaut",
        size="1536x1024",
        quality="high",
        count=2,
        output_format="png",
    )
    results = OpenAIProvider().generate(request, session=fake_session)
    sent = fake_session.calls[0]
    assert sent["url"] == "https://api.openai.com/v1/images/generations"
    assert sent["json"]["model"] == "vendor/free-form-model"
    assert sent["headers"]["Authorization"] == "Bearer secret-value"
    assert len(results) == 2

def test_compatible_appends_images_path_and_parses_url(fake_session):
    request = ApiImageRequest(
        api_key="secret-value",
        model="flux-custom",
        prompt="forest",
        base_url="https://images.example/v1/",
    )
    OpenAICompatibleProvider().generate(request, session=fake_session)
    assert fake_session.calls[0]["url"] == "https://images.example/v1/images/generations"
```

- [ ] **Step 2: Run provider tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_api_providers.py -k "openai or compatible" -q
```

Expected: missing provider classes.

- [ ] **Step 3: Implement OpenAI request construction**

Send `model`, composed `prompt`, `n`, `size`, and `quality`. For GPT Image-style models, send `output_format`; for `dall-e-*`, request `response_format="b64_json"` and omit unsupported `output_format`. This compatibility branch changes optional parameters only and never rejects an arbitrary model identifier.

Use:

```python
response = http.post(
    endpoint,
    headers={"Authorization": f"Bearer {request.api_key}", "Content-Type": "application/json"},
    json=payload,
    timeout=(10, 180),
)
```

Parse every `data[]` item. Decode `b64_json`; if an item contains `url`, download it with the same session without forwarding the Authorization header. Reject empty, malformed, or oversized responses with a sanitized `ApiProviderError`.

- [ ] **Step 4: Implement compatible URL validation**

Require an `http` or `https` base URL with a host, reject embedded usernames/passwords, normalize one trailing slash, then append `/images/generations`. Use the same response parser as OpenAI but send `response_format="b64_json"` for widest compatible-server support.

- [ ] **Step 5: Run provider tests**

Run:

```powershell
python -m pytest tests/test_api_providers.py -k "openai or compatible" -q
```

Expected: all selected tests pass and no assertion output contains `secret-value`.

- [ ] **Step 6: Record the local checkpoint**

Run:

```powershell
python -m compileall inference/api_providers tests/test_api_providers.py
```

Expected: compilation succeeds.

---

### Task 4: Gemini adapter

**Files:**
- Create: `inference/api_providers/gemini_provider.py`
- Modify: `inference/api_providers/registry.py`
- Modify: `tests/test_api_providers.py`

**Interfaces:**
- `GeminiProvider.generate(request: ApiImageRequest, session=None) -> list[ApiImageResult]`.
- Endpoint shape: `https://generativelanguage.googleapis.com/v1/models/{URL_ENCODED_MODEL}:generateContent`.

- [ ] **Step 1: Write failing Gemini tests**

```python
def test_gemini_uses_submitted_model_and_parses_inline_data(fake_session):
    request = ApiImageRequest(
        api_key="gemini-secret",
        model="gemini-custom/image beta",
        prompt="paper dragon",
        count=1,
    )
    results = GeminiProvider().generate(request, session=fake_session)
    call = fake_session.calls[0]
    assert "gemini-custom%2Fimage%20beta:generateContent" in call["url"]
    assert call["headers"]["x-goog-api-key"] == "gemini-secret"
    assert call["json"]["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert results[0].mime_type == "image/png"
```

- [ ] **Step 2: Run the Gemini tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_api_providers.py -k gemini -q
```

Expected: missing provider class.

- [ ] **Step 3: Implement Gemini request and parsing**

Send:

```python
payload = {
    "contents": [{"parts": [{"text": compose_prompt(request)}]}],
    "generationConfig": {"responseModalities": ["IMAGE"]},
}
```

For model identifiers beginning with `gemini-3`, translate `WIDTHxHEIGHT` into the nearest documented aspect ratio and map the longest edge to `1K`, `2K`, or `4K` under `generationConfig.responseFormat.image`. For other arbitrary model identifiers, omit image-size configuration and add a result note that the provider controls size. Repeat the request `count` times because Gemini response counts are model-dependent.

Parse `candidates[].content.parts[]` and accept both REST snake-case `inline_data`/`mime_type` and SDK-style camel-case `inlineData`/`mimeType`. Skip thought/text parts and raise a sanitized error if no image appears.

- [ ] **Step 4: Run Gemini tests**

Run:

```powershell
python -m pytest tests/test_api_providers.py -k gemini -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Record the local checkpoint**

Run:

```powershell
python -m compileall inference/api_providers
```

Expected: compilation succeeds.

---

### Task 5: API image service, file validation, and history integration

**Files:**
- Create: `inference/api_service.py`
- Create: `tests/test_api_service.py`

**Interfaces:**
- `ApiGenerationOutcome(paths: tuple[Path, ...], status: str)`.
- `ApiImageService.__init__(history: GenerationHistory | None = None, provider_factory: Callable[[str], ApiProvider] = create_provider)`.
- `ApiImageService.generate(provider_name: str, request: ApiImageRequest) -> ApiGenerationOutcome`.

- [ ] **Step 1: Write failing service tests**

```python
def test_service_saves_images_and_history_without_key(tmp_path, monkeypatch):
    history = GenerationHistory(
        output_dir=tmp_path / "api",
        history_file=tmp_path / "history.jsonl",
        favorites_file=tmp_path / "favorites.json",
    )
    service = ApiImageService(
        history=history,
        provider_factory=lambda _: FakeProvider(PNG_BYTES),
    )
    outcome = service.generate(
        "OpenAI",
        ApiImageRequest(api_key="never-persist-this", model="custom-model", prompt="lake"),
    )
    assert outcome.paths[0].parent == tmp_path / "api"
    persisted = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    assert "custom-model" in persisted
    assert "never-persist-this" not in persisted
```

- [ ] **Step 2: Run service tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_api_service.py -q
```

Expected: missing `ApiImageService`.

- [ ] **Step 3: Implement validated, atomic image saving**

Open bytes with Pillow, call `verify()`, reopen, load fully, and save to a temporary file in `outputs/api/` before `os.replace()`. Choose the extension from the validated actual image format rather than trusting the provider response. Enforce a 32 MB decoded-image limit.

- [ ] **Step 4: Append non-secret history**

For each output, append:

```python
HistoryRecord(
    created_at=datetime.now().isoformat(timespec="seconds"),
    image_path=str(path),
    seed=None,
    prompt=request.prompt,
    negative_prompt=request.negative_prompt,
    settings={
        "source": "api",
        "provider": provider_name,
        "model": request.model,
        "size": request.size,
        "quality": request.quality,
        "output_format": request.output_format,
        "batch_count": request.count,
    },
)
```

Do not serialize `asdict(request)` because it contains the key and base URL may contain sensitive service details.

- [ ] **Step 5: Run service and existing history tests**

Run:

```powershell
python -m pytest tests/test_api_service.py tests/test_history.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Record the local checkpoint**

Run:

```powershell
python -m compileall inference/api_service.py tests/test_api_service.py
```

Expected: compilation succeeds.

---

### Task 6: Gradio API tab and session-temporary key flow

**Files:**
- Create: `inference/api_ui.py`
- Modify: `app.py`
- Create: `tests/test_api_ui.py`

**Interfaces:**
- `ApiUiState(model: str, base_url: str, base_url_visible: bool, model_edited: bool)`.
- `provider_ui_state(provider: str, current_model: str, model_edited: bool) -> ApiUiState`.
- `api_provider_ui()` in `app.py` converts `ApiUiState` to Gradio updates.
- `mark_api_model_edited() -> bool`.
- `generate_api_images(provider, api_key, model, base_url, prompt, negative_prompt, size, quality, count, output_format)`.

- [ ] **Step 1: Write failing UI-helper tests**

```python
def test_provider_change_does_not_overwrite_manually_edited_model():
    state = provider_ui_state(
        "Gemini", "my-private-image-model", True
    )
    assert state.model == "my-private-image-model"
    assert state.model_edited is True

def test_provider_change_supplies_default_before_manual_edit():
    state = provider_ui_state("Gemini", "gpt-image-2", False)
    assert state.model == "gemini-3.1-flash-image"
    assert state.model_edited is False
```

- [ ] **Step 2: Run UI-helper tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_api_ui.py -q
```

Expected: missing helper functions or tab integration.

- [ ] **Step 3: Add application service and safe handlers**

Implement `provider_ui_state()` without importing Gradio, then create one stateless-orchestration `ApiImageService` instance in `app.py`. The handler constructs `ApiImageRequest` from submitted component values, calls the service, returns gallery file paths, status, and refreshed history, and catches `ApiProviderError` separately from unexpected exceptions. Every displayed error passes through `redact_secrets(..., [api_key])`.

- [ ] **Step 4: Add the API tab**

Add:

```python
api_key = gr.Textbox(label="API Key", type="password")
api_model = gr.Textbox(label="模型名称", value="gpt-image-2")
api_model_edited = gr.State(False)
```

Also add provider dropdown, conditionally visible compatible base URL, prompt, negative prompt, editable size textbox, quality dropdown, count slider, output-format dropdown, generate button, gallery, and status. Do not add a “remember key” control in v4. Do not place the key in `gr.State`.

Bind `.input()` on the model textbox to `mark_api_model_edited`, provider `.change()` to `api_provider_ui`, and generate `.click()` directly with the password textbox as an input.

- [ ] **Step 5: Run UI tests and compile the app**

Run:

```powershell
python -m pytest tests/test_api_ui.py -q
python -m compileall app.py
```

Expected: tests pass and compilation succeeds.

- [ ] **Step 6: Record the local checkpoint**

Run:

```powershell
rg -n "API_KEY|api_key|Authorization" app.py inference
```

Expected: key references appear only as request parameters/local variables and header construction; no environment reads, global key assignment, file writes, or logging calls contain the key.

---

### Task 7: Diagnostics, dependencies, and user documentation

**Files:**
- Modify: `inference/diagnostics.py`
- Modify: `requirements.txt`
- Modify: `README.md`
- Modify: `tests/test_diagnostics.py`

**Interfaces:**
- Diagnostics check the `requests` package and display API capability guidance without inspecting any key.

- [ ] **Step 1: Write a failing Requests diagnostic test**

```python
def test_diagnostics_includes_requests_without_key_checks(tmp_path):
    items = run_diagnostics(tmp_path, ports=[65530])
    assert any(item.name == "requests" for item in items)
    assert all("API Key" not in item.name for item in items)
```

- [ ] **Step 2: Run the diagnostic test and confirm failure**

Run:

```powershell
python -m pytest tests/test_diagnostics.py -q
```

Expected: Requests item is missing.

- [ ] **Step 3: Add dependency and diagnostic copy**

Add `requests>=2.31.0` to `requirements.txt`. Add `requests` to the package checks and one informational item: “API 密钥由使用者在 API 作画页面临时输入，诊断不会读取或保存密钥。”

- [ ] **Step 4: Update README**

Document:

- Opening the `API 作画` tab.
- Entering a personal key in the password field.
- Refresh/close clears the key.
- OpenAI default suggestion `gpt-image-2`.
- Gemini default suggestion `gemini-3.1-flash-image`.
- Any non-empty model ID can be entered, but it must support the selected protocol and image generation.
- Compatible base URL format ending at `/v1`; the app appends `/images/generations`.
- API calls may cost money and require provider account permissions.
- Network/proxy, 401/403, 429, unsupported-model, and timeout troubleshooting.

- [ ] **Step 5: Run diagnostics and full lightweight tests**

Run:

```powershell
python -m pytest tests/test_diagnostics.py tests/test_api_base.py tests/test_api_providers.py tests/test_api_service.py tests/test_api_ui.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Record the local checkpoint**

Run:

```powershell
python -m compileall inference app.py tests
```

Expected: compilation succeeds.

---

### Task 8: Full regression, secret audit, and v4 package

**Files:**
- Verify: all Python sources and tests.
- Create: `outputs/AI-generated-images-professional-restart-v4.zip`

**Interfaces:**
- Produces a clean release archive containing project source, tests, docs, and startup files but no caches, generated images, API keys, or local history.

- [ ] **Step 1: Run the full test suite**

Run:

```powershell
python -m pytest tests -q --basetemp C:\tmp\pytest-v4-final
```

Expected: all tests pass.

- [ ] **Step 2: Run compilation verification**

Run:

```powershell
python -m compileall app.py inference trainer tests
```

Expected: no syntax errors.

- [ ] **Step 3: Audit for retained secrets**

Run:

```powershell
rg -n --hidden -g "!*.zip" -g "!__pycache__/**" -g "!.pytest_cache/**" "(sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,}|Authorization\\s*[:=]\\s*Bearer\\s+[A-Za-z0-9_-]+)" .
```

Expected: no actual credentials. Test-only dummy strings may appear only in `tests/` and must be clearly nonfunctional.

- [ ] **Step 4: Build the archive**

Create `outputs/AI-generated-images-professional-restart-v4.zip` while excluding:

- `__pycache__/`
- `.pytest_cache/`
- `*.pyc`
- local `outputs/` generated images and history
- downloaded model weights
- temporary files

- [ ] **Step 5: Inspect archive contents**

Verify the archive contains `app.py`, `requirements.txt`, `README.md`, all provider modules, API service, tests, design, and plan. Verify no cache, model weight, generated image, history, favorite file, or credential-like string is present.

- [ ] **Step 6: Report the verified result**

Report exact test count, compilation status, archive size, package path, and the limitation that no paid external API call was made because no user credential is stored or requested during development.
