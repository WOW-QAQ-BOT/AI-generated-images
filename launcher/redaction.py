import re
from collections.abc import Iterable


_REDACTED = "[REDACTED]"
_AUTHORIZATION_BEARER = re.compile(
    r"(?i)(Authorization\s*:\s*Bearer\s+)([^\s,;]+)"
)
_GENERIC_BEARER = re.compile(r"(?i)(\bBearer\s+)([^\s,;]+)")
_KEY_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:[a-z][a-z0-9_]*_api_key|api_key|apikey|api-key|key|"
    r"access_token|token)\b\s*[:=]\s*)([^\s,;]+)"
)
_PROVIDER_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|github_pat_[A-Za-z0-9_-]{20,}|"
    r"AIza[A-Za-z0-9_-]+|ghp_[A-Za-z0-9_-]+|xox[A-Za-z0-9_-]+)"
)


def redact_launcher_text(value: object, secrets: Iterable[str] = ()) -> str:
    """Return diagnostic text with known credential forms replaced safely."""
    cleaned = str(value)
    for secret in sorted((str(item) for item in secrets if str(item)), key=len, reverse=True):
        cleaned = cleaned.replace(secret, _REDACTED)
    cleaned = _AUTHORIZATION_BEARER.sub(r"\1" + _REDACTED, cleaned)
    cleaned = _GENERIC_BEARER.sub(r"\1" + _REDACTED, cleaned)
    cleaned = _KEY_ASSIGNMENT.sub(r"\1" + _REDACTED, cleaned)
    return _PROVIDER_SECRET.sub(_REDACTED, cleaned)
