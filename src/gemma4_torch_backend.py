"""CUDA/PyTorch support for capturing and writing Gemma 4 residual states.

The clean Direct-44 release was first implemented with MLX.  Gemma 4 31B is
run on a CUDA host, so this module deliberately mirrors only the two model
interventions that the release needs:

* collect the pooled residual output of every text decoder layer;
* add one composed residual vector after a decoder layer.

It does not change model weights, attention heads, or prompt semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForMultimodalLM, AutoProcessor


APPRAISAL_SYSTEM_PROMPT = (
    "你正在接收一段第一人称内部体验记录。不要续写、评价或解释；"
    "只在内部重建这段体验所描述的身体与情感状态。"
)


@dataclass(frozen=True)
class TorchAppraisalCapture:
    """Pooled post-block residuals for one text-only appraisal prompt."""

    activations: np.ndarray
    token_count: int
    formatted_prompt_ids: list[int]


class Gemma4ResidualWriteLayer(nn.Module):
    """Transparent decoder-layer wrapper with optional post-block residual write.

    ``inner`` remains the exact Transformers implementation.  This avoids
    duplicating Gemma 4's attention, PLE, and multimodal details while placing
    the write at the same post-block residual location used by the MLX version.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("residual_direction", torch.empty(0), persistent=False)
        self.residual_alpha = 0.0
        self.residual_scope = "all_tokens"
        self.residual_generation_started = False
        self.capture_last_tokens = 0
        self._captured_pool: torch.Tensor | None = None

    def clear_control(self) -> None:
        self.residual_direction = torch.empty(0, device=self.residual_direction.device)
        self.residual_alpha = 0.0
        self.residual_scope = "all_tokens"
        self.residual_generation_started = False

    def set_residual_control(
        self,
        direction: np.ndarray | torch.Tensor,
        alpha: float,
        *,
        scope: str = "all_tokens",
    ) -> None:
        if scope not in {"all_tokens", "last_token", "generation_only"}:
            raise ValueError(f"Unsupported residual write scope: {scope}")
        device = next(self.inner.parameters()).device
        dtype = next(self.inner.parameters()).dtype
        self.residual_direction = torch.as_tensor(direction, device=device, dtype=dtype)
        self.residual_alpha = float(alpha)
        self.residual_scope = scope
        self.residual_generation_started = False

    def arm_capture(self, pool_last_tokens: int) -> None:
        self.capture_last_tokens = max(0, int(pool_last_tokens))
        self._captured_pool = None

    def disarm_capture(self) -> None:
        self.capture_last_tokens = 0
        self._captured_pool = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.inner(hidden_states, *args, **kwargs)
        if self.residual_alpha and self.residual_direction.numel():
            should_write = self.residual_scope == "all_tokens"
            if self.residual_scope == "last_token":
                should_write = hidden_states.shape[1] == 1
            elif self.residual_scope == "generation_only" and hidden_states.shape[1] == 1:
                should_write = self.residual_generation_started
                self.residual_generation_started = True
            if should_write:
                hidden_states = hidden_states + (
                    self.residual_direction * self.residual_alpha
                ).view(1, 1, -1)
        if self.capture_last_tokens:
            count = min(self.capture_last_tokens, hidden_states.shape[1])
            self._captured_pool = (
                hidden_states[:, -count:, :].float().mean(dim=1).detach().cpu()
            )
        return hidden_states


def text_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return Gemma 4's text decoder stack for both outer-model layouts."""

    for path in (("model", "language_model", "layers"), ("language_model", "layers")):
        node: object = model
        try:
            for attribute in path:
                node = getattr(node, attribute)
        except AttributeError:
            continue
        if isinstance(node, nn.ModuleList):
            return node
    raise AttributeError("Could not locate Gemma 4 text decoder layers")


def install_residual_writers(model: nn.Module) -> dict[int, Gemma4ResidualWriteLayer]:
    """Wrap every Gemma 4 text block once and return a layer-indexed writer map."""

    layers = text_decoder_layers(model)
    writers: dict[int, Gemma4ResidualWriteLayer] = {}
    for index, layer in enumerate(layers):
        if isinstance(layer, Gemma4ResidualWriteLayer):
            writer = layer
        else:
            writer = Gemma4ResidualWriteLayer(layer)
            layers[index] = writer
        writers[index] = writer
    return writers


def clear_residual_writers(writers: dict[int, Gemma4ResidualWriteLayer]) -> None:
    for writer in writers.values():
        writer.clear_control()


def load_gemma4_cuda(model_path: str, *, device_map: str = "auto"):
    """Load an unquantized Gemma 4 checkpoint on CUDA in BF16."""

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return model, processor


def prepare_chat_inputs(
    processor,
    model: nn.Module,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> dict[str, torch.Tensor]:
    """Build a text-only Gemma 4 chat prompt and place it with the model."""

    inputs = processor.apply_chat_template(
        list(messages),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in inputs.items()}


def capture_appraisal_residuals(
    model: nn.Module,
    processor,
    writers: dict[int, Gemma4ResidualWriteLayer],
    *,
    scene: str,
    layer_ids: Iterable[int] | None = None,
    system_text: str = APPRAISAL_SYSTEM_PROMPT,
    pool_last_tokens: int = 4,
) -> TorchAppraisalCapture:
    """Run one no-generation appraisal pass and pool selected post-block states."""

    selected = list(layer_ids if layer_ids is not None else writers)
    if not selected:
        raise ValueError("At least one layer must be selected for capture")
    for layer_id in selected:
        writers[layer_id].arm_capture(pool_last_tokens)
    inputs = prepare_chat_inputs(
        processor,
        model,
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": scene},
        ],
        add_generation_prompt=True,
    )
    with torch.inference_mode():
        model(**inputs, use_cache=False, logits_to_keep=1)
    pooled = []
    for layer_id in selected:
        capture = writers[layer_id]._captured_pool
        writers[layer_id].disarm_capture()
        if capture is None:
            raise RuntimeError(f"Layer {layer_id} did not expose a captured residual")
        pooled.append(capture.numpy().astype(np.float32, copy=False)[0])
    return TorchAppraisalCapture(
        activations=np.stack(pooled),
        token_count=int(inputs["input_ids"].shape[1]),
        formatted_prompt_ids=inputs["input_ids"][0].tolist(),
    )
