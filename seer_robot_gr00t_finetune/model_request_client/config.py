"""Configuration for the independent model request process."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "model_request.json"


@dataclass(frozen=True)
class BackendConfig:
    name: str
    url: str
    timeout_ms: int


@dataclass(frozen=True)
class ModelRequestConfig:
    config_file: Path
    collection_config: Path
    backend: str
    task: str
    request_hz: float
    request_topic: str
    response_topic: str
    dry_run: bool
    publish_action: bool
    image_quality: int
    backends: dict[str, BackendConfig]


def load_model_request_config(path: str | Path = DEFAULT_CONFIG) -> ModelRequestConfig:
    config_file = Path(path).expanduser().resolve()
    raw: dict[str, Any] = json.loads(config_file.read_text(encoding="utf-8"))
    backend = str(raw.get("backend", "pi05")).lower()
    backends = {
        name: BackendConfig(
            name=name,
            url=str(value.get("url", "")),
            timeout_ms=int(value.get("timeout_ms", 200)),
        )
        for name, value in raw.get("backends", {}).items()
    }
    if backend not in {"pi05", "gr00t"}:
        raise ValueError("backend must be pi05 or gr00t")
    if backend not in backends:
        raise ValueError(f"missing backend configuration: {backend}")
    request_hz = float(raw.get("request_hz", 5.0))
    if request_hz <= 0:
        raise ValueError("request_hz must be positive")
    quality = int(raw.get("image_quality", 75))
    if not 1 <= quality <= 100:
        raise ValueError("image_quality must be between 1 and 100")
    collection_config = Path(raw.get("collection_config", "../config/data_collection.json"))
    if not collection_config.is_absolute():
        collection_config = (config_file.parent / collection_config).resolve()
    return ModelRequestConfig(
        config_file=config_file,
        collection_config=collection_config,
        backend=backend,
        task=str(raw.get("task", "")),
        request_hz=request_hz,
        request_topic=str(raw.get("request_topic", "/ModelInference/Request")),
        response_topic=str(raw.get("response_topic", "/ModelInference/Action")),
        dry_run=bool(raw.get("dry_run", True)),
        publish_action=bool(raw.get("publish_action", False)),
        image_quality=quality,
        backends=backends,
    )
