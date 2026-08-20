# V4 User-Supplied API Image Generation Design

Date: 2026-07-25
Status: Approved design, awaiting written-spec review

## Goal

Add API-based image generation to the existing local Gradio workbench. Each visitor supplies their own API key in the web page, can use a freely entered image-model identifier, and receives results through the same output, history, and favorites workflow as local generation.

## Scope

- Add an `API 作画` tab to `app.py`.
- Support OpenAI, Gemini, and OpenAI-compatible image APIs in v4.
- Let the user enter the model identifier as free text. The application must not enforce a fixed model allowlist.
- Provide provider-specific model suggestions as editable defaults only.
- Let the user enter a custom base URL for OpenAI-compatible services.
- Save successful API images under `outputs/api/`.
- Add successful API generations to the existing history and favorites system.
- Add API dependency and connectivity guidance to diagnostics and the README.
- Keep API keys temporary in v4.

## Non-goals

- Persisting, synchronizing, encrypting, or centrally managing API keys.
- Creating user accounts or an authentication system.
- Supporting every proprietary API protocol. Services not using the OpenAI-compatible protocol require a future provider adapter.
- Verifying a real paid API call during automated tests.

## User Interface

The new `API 作画` tab contains:

- Provider: `OpenAI`, `Gemini`, or `OpenAI 兼容接口`.
- API key: a password-style textbox.
- Model: an editable textbox with a provider-specific suggestion. Any non-empty model identifier is accepted and passed to the selected provider.
- Base URL: shown only for `OpenAI 兼容接口`, with an editable API root such as `https://example.com/v1`. The adapter appends `/images/generations`.
- Prompt and optional negative prompt.
- Image size, quality, output count, and output format controls.
- Generate button, gallery, and a Chinese status panel.

Changing the provider updates suggestions but never overwrites a model value that the user has manually edited during the session.

## API Key Lifecycle and Isolation

- The API key is entered by each visitor in the web page.
- The textbox uses password display.
- The key is passed to the backend only when that visitor starts an API request.
- It remains available only in the current browser page session and is cleared by refresh, page close, or application restart.
- It is never written to environment variables, configuration files, history JSON, image metadata, favorites, diagnostics, or logs.
- Provider errors are sanitized before display. Error messages must not include request headers, raw request bodies, or the API key.
- Concurrent visitor requests must receive the key from their own submitted component value; no process-global key variable or shared key cache is allowed.

Persistent key storage is deferred to a future formal release and must be an explicit opt-in feature with a separate security design.

## Architecture

### Provider boundary

Create `inference/api_providers/`:

- `base.py`: request/result data structures, validation, common sanitized errors, and the provider interface.
- `openai_provider.py`: OpenAI image-generation requests.
- `gemini_provider.py`: Gemini image-generation requests and inline-image response parsing.
- `openai_compatible_provider.py`: configurable OpenAI-compatible base URL and model.
- `registry.py`: maps provider names to stateless provider implementations and returns editable defaults.

Provider implementations are stateless. The API key is a request argument and is not stored on an instance.

### Application service

Create `inference/api_service.py` to:

1. Validate provider-independent inputs.
2. Resolve the selected stateless provider.
3. Send the request using the submitted key.
4. Decode and validate returned image bytes.
5. Save images atomically to `outputs/api/`.
6. Add non-secret generation settings to history.
7. Return gallery paths and a sanitized Chinese status message.

`app.py` only composes controls and event handlers. It must not contain provider-specific HTTP parsing.

## Data Flow

1. A visitor enters their key, model, and prompt.
2. Gradio submits those component values to the API-generation handler.
3. The handler creates an in-memory request object.
4. The selected provider sends the external request.
5. Returned image bytes are validated and saved.
6. History records provider, model, prompt, size, quality, and output path, but never the key or authorization header.
7. The gallery displays the saved files.
8. The key remains only in that visitor's password textbox until the page session ends.

## Model and Protocol Compatibility

- Model identifiers are never restricted to a hardcoded list.
- Provider suggestions are convenience values and may be edited.
- OpenAI and Gemini use dedicated adapters because their request and response protocols differ.
- Other services can use `OpenAI 兼容接口` when they implement the relevant compatible image endpoint.
- An arbitrary model name does not imply protocol compatibility. A clear Chinese error explains when the selected service or model does not support image generation.
- Each adapter translates the shared controls to parameters supported by its protocol. Optional controls that a provider does not support are omitted, and the status panel reports that they were not applied.

## Error Handling

User-facing errors distinguish:

- Missing API key, model, prompt, or custom base URL.
- Authentication or permission failure.
- Unsupported model or image capability.
- Rate limit or quota exhaustion.
- Network timeout, proxy, DNS, or remote service failure.
- Invalid response or missing image data.
- Local output write failure.

Raw provider responses may be summarized for debugging only after redaction. Automated redaction must remove the submitted key and common authorization-header forms.

## Testing

Automated tests use mocked HTTP responses and do not require network access, paid API credentials, GPU access, or model files.

Required coverage:

- Provider registry and editable default behavior.
- Free-form model identifiers pass through unchanged.
- OpenAI, Gemini, and OpenAI-compatible request construction.
- Base64 and inline-image response parsing.
- Successful image saving and history integration.
- Missing-input and malformed-response handling.
- Secret redaction from errors and history.
- Isolation: no module-level or provider-instance API key retention.
- Existing v3 tests remain passing.

## Documentation and Packaging

- Update `README.md` with temporary-key behavior, provider setup, custom base URL examples, and billing/network cautions.
- State clearly that API usage may incur provider charges.
- Package the completed project as `AI-generated-images-professional-restart-v4.zip`.
- Exclude caches, test artifacts, API keys, and generated user images from the package.

## Acceptance Criteria

- A visitor can select OpenAI, Gemini, or an OpenAI-compatible service and enter their own key in the page.
- A visitor can enter any non-empty model identifier without a local allowlist rejection.
- Successful API images are saved and appear in history/favorites.
- Refreshing or closing the page removes the entered key.
- No key appears in repository files, outputs, history, diagnostics, logs, exception messages, or the release archive.
- All unit tests and Python compilation checks pass.
