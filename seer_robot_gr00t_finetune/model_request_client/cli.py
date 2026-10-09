"""CLI for the independent PI05/GR00T model request service."""

from __future__ import annotations

import argparse
import signal
import threading

from .config import DEFAULT_CONFIG, load_model_request_config
from .service import ModelRequestService


def main() -> int:
    parser = argparse.ArgumentParser(description="Seer PI05/GR00T model request service")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=("pi05", "gr00t"))
    parser.add_argument("--live", action="store_true", help="allow HTTP model requests")
    parser.add_argument("--publish-action", action="store_true")
    parser.add_argument(
        "--fake-data",
        action="store_true",
        help="send deterministic fake state/images without Robokit",
    )
    parser.add_argument("--url", help="override selected backend HTTP URL")
    args = parser.parse_args()
    config = load_model_request_config(args.config)
    if args.backend:
        from dataclasses import replace

        config = replace(config, backend=args.backend)
    if args.live:
        from dataclasses import replace

        config = replace(config, dry_run=False)
    if args.publish_action:
        from dataclasses import replace

        config = replace(config, publish_action=True)
    if args.url:
        from dataclasses import replace

        config = replace(
            config,
            backends={
                **config.backends,
                config.backend: replace(config.backends[config.backend], url=args.url),
            },
        )
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    service = ModelRequestService(config, fake_data=args.fake_data)
    try:
        print(
            f"Starting {config.backend} model request service (dry_run={config.dry_run})",
            flush=True,
        )
        service.start()
        service.run(stop)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
