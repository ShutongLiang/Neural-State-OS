from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".json", ".toml", ".txt", ".yml", ".yaml", ".cff", ".html"}
FORBIDDEN = (
    "/" + "Users/",
    "/" + "Volumes/",
    "root" + "@",
    "BEGIN OPENSSH" + " PRIVATE KEY",
)


def test_no_private_machine_paths_or_credentials() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part.startswith(".venv") or part == ".git" for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN:
            if marker in text:
                failures.append(f"{path.relative_to(ROOT)} contains {marker!r}")
    assert not failures, "\n".join(failures)


def test_model_weights_are_not_committed() -> None:
    forbidden_suffixes = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".onnx"}
    found = [path.relative_to(ROOT) for path in ROOT.rglob("*") if path.suffix.lower() in forbidden_suffixes]
    assert not found
