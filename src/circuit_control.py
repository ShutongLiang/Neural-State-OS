"""Minimal Gemma residual writer used by the Direct-6 protocol only."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GemmaPaperCircuitLayer(nn.Module):
    """Gemma decoder block with one signed residual write at its output.

    The implementation mirrors Gemma's actual attention → branch norm → MLP
    residual path.  It deliberately exposes only Direct-6 residual addition;
    no attention/MLP editing, projection removal, or auxiliary writer exists
    in this release.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        if getattr(inner, "enable_moe", False):
            raise ValueError("Direct-6 residual wrapper does not support MoE blocks")
        if getattr(inner, "hidden_size_per_layer_input", 0):
            raise ValueError("Direct-6 residual wrapper does not support input gates")
        self.config = inner.config
        self.layer_idx = inner.layer_idx
        self.layer_type = inner.layer_type
        self.self_attn = inner.self_attn
        self.mlp = inner.mlp
        self.input_layernorm = inner.input_layernorm
        self.post_attention_layernorm = inner.post_attention_layernorm
        self.pre_feedforward_layernorm = inner.pre_feedforward_layernorm
        self.post_feedforward_layernorm = inner.post_feedforward_layernorm
        self.layer_scalar = inner.layer_scalar
        self.residual_direction: mx.array | None = None
        self.residual_alpha = 0.0
        self.residual_scope = "all_tokens"
        self.residual_generation_started = False
        # Readout is deliberately opt-in.  The live demo only arms the small
        # set of reader-optimal layers, so normal generation has no capture
        # allocation or host transfer overhead.
        self.capture_last_tokens = 0
        self.captured_pool: mx.array | None = None

    def clear_control(self) -> None:
        self.residual_direction = None
        self.residual_alpha = 0.0
        self.residual_scope = "all_tokens"
        self.residual_generation_started = False

    def set_residual_control(
        self,
        direction: mx.array,
        alpha: float,
        *,
        scope: str = "all_tokens",
    ) -> None:
        if scope not in {"all_tokens", "last_token", "generation_only"}:
            raise ValueError(f"Unsupported Direct-6 control scope: {scope}")
        self.residual_direction = direction
        self.residual_alpha = float(alpha)
        self.residual_scope = scope
        self.residual_generation_started = False

    def arm_capture(self, pool_last_tokens: int) -> None:
        """Expose a pooled post-block residual for one appraisal pass."""
        self.capture_last_tokens = max(0, int(pool_last_tokens))
        self.captured_pool = None

    def disarm_capture(self) -> None:
        self.capture_last_tokens = 0
        self.captured_pool = None

    def __call__(
        self,
        x: mx.array,
        mask=None,
        cache=None,
        per_layer_input=None,
        shared_kv=None,
        offset=None,
    ):
        del per_layer_input
        residual = x
        hidden = self.input_layernorm(x)
        hidden, shared_kv, offset = self.self_attn(
            hidden,
            mask,
            cache,
            shared_kv=shared_kv,
            offset=offset,
        )
        hidden = self.post_attention_layernorm(hidden)
        hidden = residual + hidden
        residual = hidden
        hidden = self.pre_feedforward_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = self.post_feedforward_layernorm(hidden)
        hidden = residual + hidden
        if self.layer_scalar is not None:
            hidden = hidden * self.layer_scalar

        if self.residual_alpha and self.residual_direction is not None:
            delta = (self.residual_direction * self.residual_alpha).astype(hidden.dtype)
            shaped = delta.reshape(1, 1, -1)
            if self.residual_scope == "all_tokens":
                hidden = hidden + shaped
            elif self.residual_scope == "last_token" and hidden.shape[1] == 1:
                hidden = hidden + shaped
            elif self.residual_scope == "generation_only" and hidden.shape[1] == 1:
                if self.residual_generation_started:
                    hidden = hidden + shaped
                else:
                    self.residual_generation_started = True
        if self.capture_last_tokens:
            count = min(self.capture_last_tokens, hidden.shape[1])
            self.captured_pool = mx.mean(
                hidden[:, -count:, :].astype(mx.float32), axis=1
            )[0]
        return hidden, shared_kv, offset


def install_paper_circuit_layers(model) -> dict[int, GemmaPaperCircuitLayer]:
    """Replace all dense Gemma decoder layers with Direct-6 writers."""
    wrappers = {}
    for index, layer in enumerate(model.layers):
        wrapper = GemmaPaperCircuitLayer(layer)
        model.layers[index] = wrapper
        wrappers[index] = wrapper
    return wrappers
