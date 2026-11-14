#!/usr/bin/env python3
"""Run one message through multiple Neural State OS profiles."""

from __future__ import annotations

import argparse
import json
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "experiments" / "profiles_31b.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8777")
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--message", required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "secure_attachment", "separation_anxiety", "calm_autonomy"],
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def stream_answer(base_url: str, payload: dict[str, Any]) -> str:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat/stream",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    answer = ""
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            try:
                event = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if event.get("type") == "done":
                answer = str(event.get("answer") or "")
    if not answer:
        raise RuntimeError("The server returned no final answer")
    return answer


def main() -> None:
    args = parse_args()
    document = json.loads(args.profiles_file.read_text(encoding="utf-8"))
    profiles = document["profiles"]
    missing = [name for name in args.profiles if name not in profiles]
    if missing:
        raise KeyError(f"Unknown profiles: {', '.join(missing)}")

    print(f"INPUT\n{args.message}")
    for name in args.profiles:
        payload = {
            "session_id": f"compare-{name}-{uuid.uuid4().hex[:8]}",
            "message": args.message,
            "axes": profiles[name],
            "thinking": False,
            "persona": "standard",
            "modulation_scope": "all_tokens",
            "mix_protection": {"mode": "raw", "headroom": 1.35},
            "max_tokens": args.max_tokens,
        }
        answer = stream_answer(args.base_url, payload)
        print(f"\n{'=' * 72}\n{name.upper()}\n{'=' * 72}\n{answer}")


if __name__ == "__main__":
    main()
