from collections.abc import Mapping


_OFFLINE_SETTINGS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "GRADIO_SHARE": "False",
    "GRADIO_ANALYTICS_ENABLED": "False",
}
_LOCAL_BYPASSES = ("127.0.0.1", "localhost")


def _normalize_bypasses(value: str) -> str:
    entries: list[str] = []
    for entry in value.split(","):
        entry = entry.strip()
        if entry and entry not in entries:
            entries.append(entry)
    for local_value in _LOCAL_BYPASSES:
        if local_value not in entries:
            entries.append(local_value)
    return ",".join(entries)


def build_offline_environment(base: Mapping[str, str], port: int) -> dict[str, str]:
    """Build an isolated child-process environment for the local Gradio server."""
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("端口必须是 1 到 65535 之间的整数")

    environment = dict(base)
    environment.update(_OFFLINE_SETTINGS)
    environment["NO_PROXY"] = _normalize_bypasses(environment.get("NO_PROXY", ""))
    environment["no_proxy"] = _normalize_bypasses(environment.get("no_proxy", ""))
    environment["GRADIO_SERVER_PORT"] = str(port)
    return environment
