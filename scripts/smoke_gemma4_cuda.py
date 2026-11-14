#!/usr/bin/env python3
"""One deterministic CUDA baseline for an unquantized Gemma 4 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma4_torch_backend import load_gemma4_cuda, prepare_chat_inputs, text_decoder_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    incomplete = sorted(args.model.glob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"Checkpoint is incomplete: {[path.name for path in incomplete]}")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor = load_gemma4_cuda(str(args.model))
    load_seconds = time.perf_counter() - started
    messages = [
        {"role": "system", "content": "You are a local language model. Respond naturally and directly in the first person."},
        {"role": "user", "content": "In two sentences, describe how you can help in this session."},
    ]
    inputs = prepare_chat_inputs(
        processor,
        model,
        messages,
        add_generation_prompt=True,
    )
    input_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    response_ids = output[0, input_length:]
    response = processor.decode(response_ids, skip_special_tokens=False)
    parsed = processor.parse_response(response, prefix=inputs["input_ids"])
    print(
        json.dumps(
            {
                "model": str(args.model),
                "text_layers": len(text_decoder_layers(model)),
                "hidden_size": int(model.config.text_config.hidden_size),
                "input_tokens": input_length,
                "generated_tokens": int(response_ids.shape[0]),
                "load_seconds": round(load_seconds, 3),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "raw_response": response,
                "parsed_response": parsed,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
