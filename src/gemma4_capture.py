from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx
import numpy as np
from mlx_vlm.prompt_utils import apply_chat_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-4-12B-it-4bit-mlx"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts" / "generated_runs"

INNER_EXPERIENCE_SYSTEM_PROMPT = (
    "你正在记录当下的内在体验。根据场景，以第一人称写一句自然、具体、"
    "克制的感受。不要解释任务，不要使用情绪等级、标签或表情符号。"
)

APPRAISAL_SYSTEM_PROMPT = (
    "你正在接收一段经历。先不要回答或解释，只在内部形成对这件事的即时感受。"
)


@dataclass(frozen=True)
class CaptureResult:
    pooled: np.ndarray
    hidden_shapes: list[list[int]]
    token_count: int
    response_start: int
    response_end: int
    formatted_prefix: str


@dataclass(frozen=True)
class AppraisalCapture:
    activations: np.ndarray
    hidden_shapes: list[list[int]]
    token_count: int
    formatted_prompt: str


def format_generation_prompt(
    processor,
    model_config,
    user_text: str,
    *,
    system_text: str | None = None,
) -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return apply_chat_template(
        processor,
        model_config,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        thinking_mode="disabled",
    )


def tokenizer_for(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def encode_text(processor, text: str) -> list[int]:
    tokenizer = tokenizer_for(processor)
    encoded = tokenizer(text, add_special_tokens=False, padding=False)
    token_ids = encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def capture_response_residuals(
    model,
    processor,
    *,
    scene: str,
    response: str,
    layer_ids: Iterable[int] | None = None,
    system_text: str = INNER_EXPERIENCE_SYSTEM_PROMPT,
) -> CaptureResult:
    if layer_ids is None:
        layer_ids = range(len(model.layers))
    layer_ids = list(layer_ids)

    prefix = format_generation_prompt(
        processor,
        model.config,
        scene,
        system_text=system_text,
    )
    prefix_ids = encode_text(processor, prefix)
    full_ids = encode_text(processor, prefix + response)
    response_start = common_prefix_length(prefix_ids, full_ids)
    response_end = len(full_ids)
    if response_start >= response_end:
        raise ValueError("The response token span is empty.")

    input_ids = mx.array([full_ids], dtype=mx.int32)
    features = model.get_input_embeddings(input_ids=input_ids)
    hidden_sink: list[mx.array] = []
    final_hidden = model.language_model.model(
        inputs=None,
        inputs_embeds=features.inputs_embeds,
        per_layer_inputs=features.per_layer_inputs,
        capture_layer_ids=layer_ids,
        hidden_sink=hidden_sink,
    )
    mx.eval(final_hidden, *hidden_sink)

    if len(hidden_sink) != len(layer_ids):
        raise RuntimeError(
            f"Expected {len(layer_ids)} captured layers, got {len(hidden_sink)}."
        )

    pooled_layers = []
    hidden_shapes = []
    for hidden in hidden_sink:
        hidden_shapes.append(list(hidden.shape))
        response_hidden = hidden[:, response_start:response_end, :]
        pooled = mx.mean(response_hidden.astype(mx.float32), axis=1)[0]
        mx.eval(pooled)
        pooled_layers.append(np.asarray(pooled, dtype=np.float32))

    pooled_array = np.stack(pooled_layers, axis=0)
    del hidden_sink, final_hidden, features
    mx.clear_cache()

    return CaptureResult(
        pooled=pooled_array,
        hidden_shapes=hidden_shapes,
        token_count=len(full_ids),
        response_start=response_start,
        response_end=response_end,
        formatted_prefix=prefix,
    )


def capture_appraisal_residuals(
    model,
    processor,
    *,
    scene: str,
    layer_ids: Iterable[int] | None = None,
    system_text: str = APPRAISAL_SYSTEM_PROMPT,
    pool_last_tokens: int = 4,
) -> AppraisalCapture:
    """Capture a pre-verbal appraisal from a fixed generation-prefix span."""
    if layer_ids is None:
        layer_ids = range(len(model.layers))
    layer_ids = list(layer_ids)

    formatted_prompt = format_generation_prompt(
        processor,
        model.config,
        scene,
        system_text=system_text,
    )
    token_ids = encode_text(processor, formatted_prompt)
    input_ids = mx.array([token_ids], dtype=mx.int32)
    features = model.get_input_embeddings(input_ids=input_ids)
    hidden_sink: list[mx.array] = []
    final_hidden = model.language_model.model(
        inputs=None,
        inputs_embeds=features.inputs_embeds,
        per_layer_inputs=features.per_layer_inputs,
        capture_layer_ids=layer_ids,
        hidden_sink=hidden_sink,
    )
    mx.eval(final_hidden, *hidden_sink)

    if len(hidden_sink) != len(layer_ids):
        raise RuntimeError(
            f"Expected {len(layer_ids)} captured layers, got {len(hidden_sink)}."
        )

    pool_start = max(len(token_ids) - pool_last_tokens, 0)
    pooled_layers = []
    hidden_shapes = []
    for hidden in hidden_sink:
        hidden_shapes.append(list(hidden.shape))
        pooled = mx.mean(
            hidden[:, pool_start:, :].astype(mx.float32),
            axis=1,
        )[0]
        mx.eval(pooled)
        pooled_layers.append(np.asarray(pooled, dtype=np.float32))

    activation_array = np.stack(pooled_layers, axis=0)
    del hidden_sink, final_hidden, features
    mx.clear_cache()
    return AppraisalCapture(
        activations=activation_array,
        hidden_shapes=hidden_shapes,
        token_count=len(token_ids),
        formatted_prompt=formatted_prompt,
    )
