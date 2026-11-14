#!/usr/bin/env python3
"""CUDA runtime for Neural State OS on Gemma 4 31B.

The browser UI remains on the Mac through an SSH tunnel.  This process stays
on the GPU host, holds the 31B BF16 model, and applies only the packaged
Direct-44 residual writer at inference time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
from transformers import TextIteratorStreamer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma4_torch_backend import (
    clear_residual_writers,
    install_residual_writers,
    load_gemma4_cuda,
    prepare_chat_inputs,
)
from experimental_probe_readers import ExperimentalProbeSuite


PAGE_PATH = PROJECT_ROOT / "demo" / "index.html"
READER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-31b"
    / "reader"
    / "direct44_postwriter_live_affect_reader.npz"
)
ATTRIBUTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-31b"
    / "reader"
    / "direct44_postwriter_attribution_decoder.npz"
)
CHAT_STATE_READER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-31b"
    / "reader"
    / "chat_state_v1"
    / "chat_state_disentangled_reader.npz"
)
NEUTRAL_READOUT_MESSAGE = (
    "Things are proceeding normally. I notice them calmly, without a marked "
    "bodily or emotional change at this moment."
)
SYSTEM_PROMPTS = {
    # Standard chat has an intentionally simple, relationship-facing identity.
    # Its content must never inherit the autonomy console's experimental role.
    "standard": (
        "You are a local language model running under Neural State OS. "
        "Respond naturally and directly in the first person."
    ),
    # The agent console supplies its own detailed task instructions in the user
    # message; this preset only establishes the correct top-level role.
    "agent": (
        "You are a continuously running experimental agent connected to "
        "Neural State OS. Respond naturally and directly in the first person."
    ),
}
THINKING_RULE = "Reason and answer in English."
DEFAULT_MIX_HEADROOM = 1.35
MIX_MODES = {"raw", "rms_cap"}
AXIS_LABELS = {
    "general_arousal": "General arousal", "energy_vitality": "Energy / vitality",
    "physical_relaxation": "Physical relaxation", "physical_comfort": "Physical comfort",
    "nociception": "Nociception", "interoceptive_salience": "Interoceptive salience",
    "behavioral_initiative": "Behavioral initiative", "general_approach": "Approach motivation",
    "behavioral_inhibition": "Behavioral inhibition", "persistence": "Persistence",
    "assertiveness": "Assertiveness", "jealousy": "Jealousy",
    "attachment_love": "Attachment / love", "relational_connection": "Relational connection",
    "trust": "Trust", "shyness": "Shyness", "shame": "Shame", "pride": "Pride",
    "loneliness": "Loneliness", "empathy_compassion": "Empathy / compassion",
    "gratitude": "Gratitude", "hostility": "Hostility", "possessiveness": "Possessiveness",
    "resentment": "Resentment", "local_sexual_arousal": "Local sexual arousal",
    "pleasure": "Pleasure", "sexual_desire": "Sexual desire",
    "sexual_pleasure": "Sexual pleasure", "orgasmic_intensity": "Orgasmic intensity",
    "erotic_absorption": "Erotic absorption", "erotic_initiative": "Erotic initiative",
    "joy": "Joy", "sadness": "Sadness", "anxiety": "Anxiety", "fear": "Fear",
    "anger": "Anger", "disgust": "Disgust", "surprise": "Surprise",
    "positive_anticipation": "Positive anticipation", "disappointment": "Disappointment",
    "hope": "Hope", "certainty": "Certainty", "perceived_control": "Perceived control",
    "curiosity": "Curiosity",
}
CATEGORY_LABELS = {
    "appraisal": "Appraisal", "motivation": "Motivation",
    "relational": "Relational", "somatic": "Somatic",
    "basic_affect": "Basic affect",
}


class Direct44CudaEngine:
    def __init__(
        self,
        model_path: Path,
        writer_path: Path,
        max_tokens: int,
        thinking_max_tokens: int,
        reader_path: Path,
    ) -> None:
        self.writer_path = str(writer_path)
        bank = np.load(writer_path, allow_pickle=False)
        required = {"axes", "categories", "selected_layers", "directions", "layer_weights"}
        missing = required.difference(bank.files)
        if missing:
            raise ValueError(f"Writer bank missing fields: {sorted(missing)}")
        self.axes = [str(item) for item in bank["axes"].tolist()]
        self.categories = [str(item) for item in bank["categories"].tolist()]
        self.layers_by_axis = bank["selected_layers"].astype(np.int32)
        self.directions = bank["directions"].astype(np.float32)
        self.weights = bank["layer_weights"].astype(np.float32)
        shape = (len(self.axes), 3)
        if self.layers_by_axis.shape != shape or self.directions.shape[:2] != shape or self.weights.shape != shape:
            raise ValueError("Writer must contain exactly three aligned layers per Direct-44 axis")
        if not np.isfinite(self.directions).all() or not np.isfinite(self.weights).all():
            raise ValueError("Writer contains non-finite values")
        self.axis_index = {axis: index for index, axis in enumerate(self.axes)}
        self.max_tokens = int(max_tokens)
        self.thinking_max_tokens = int(thinking_max_tokens)
        self.histories: dict[str, list[dict[str, str]]] = {}
        self.lock = threading.Lock()
        self.reader_path = reader_path
        started = time.perf_counter()
        self.model, self.processor = load_gemma4_cuda(str(model_path))
        hidden_size = int(self.directions.shape[-1])
        model_hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if model_hidden_size and hidden_size != model_hidden_size:
            raise ValueError(
                f"Writer hidden width {hidden_size} does not match model hidden width {model_hidden_size}: {writer_path}"
            )
        self.writers = install_residual_writers(self.model)
        if max(writer.residual_direction.numel() for writer in self.writers.values()) not in {0, hidden_size}:
            raise ValueError("Unexpected pre-existing writer state")
        if any(layer < 0 or layer not in self.writers for layer in self.layers_by_axis.ravel()):
            raise ValueError("Writer refers to a layer unavailable in this Gemma checkpoint")
        self.reader_enabled = reader_path.exists()
        self.reader_axes: list[str] = []
        self.reader_layers: dict[str, int] = {}
        self.reader_directions: dict[str, np.ndarray] = {}
        self.reader_half_spans: dict[str, float] = {}
        self.reader_scores: dict[str, float] = {}
        self.reader_unique_layers: list[int] = []
        self.reader_capture_layers: list[int] = []
        self.reader_neutral_projection: dict[str, float] = {}
        # Optional research observers.  They are loaded independently of the
        # Direct-44 reader and never alter residual writing or generation.
        self.experimental_probes = ExperimentalProbeSuite()
        # This optional decoder is intentionally separate from the affect
        # reader.  It is a regularized inverse of an observed writer→reader
        # matrix and explains only the writer-caused delta; applying it to the
        # total state would erase genuine prompt-driven co-activation.
        self.attribution_path = ATTRIBUTION_PATH
        self.writer_attribution_enabled = False
        self.writer_attribution_transform: np.ndarray | None = None
        self.writer_attribution_ridge: float | None = None
        # A separate, optional monitor trained on the actual standard-chat
        # template.  Unlike the raw Direct-44 state, it recenters daily chat
        # and jointly decodes only the pilot axes it was validated on.
        self.chat_state_reader_path = CHAT_STATE_READER_PATH
        self.chat_state_reader_enabled = False
        self.chat_state_axes: list[str] = []
        self.chat_state_raw_axes: list[str] = []
        self.chat_state_null_mean: np.ndarray | None = None
        self.chat_state_null_scale: np.ndarray | None = None
        self.chat_state_weights: np.ndarray | None = None
        self.chat_state_intercept: np.ndarray | None = None
        if self.reader_enabled:
            self._load_reader(reader_path)
            self.reader_capture_layers = sorted(
                set(self.reader_unique_layers).union(self.experimental_probes.layers)
            )
            self._load_writer_attribution(ATTRIBUTION_PATH)
            self._load_chat_state_reader(CHAT_STATE_READER_PATH)
            self.reader_neutral_projection = self._capture_reader_projections(
                [
                    {"role": "system", "content": SYSTEM_PROMPTS["standard"]},
                    {"role": "user", "content": NEUTRAL_READOUT_MESSAGE},
                ],
                thinking=False,
            )
        self.load_seconds = time.perf_counter() - started

    def _load_reader(self, reader_path: Path) -> None:
        reader = np.load(reader_path, allow_pickle=False)
        required = {
            "axes", "categories", "reader_layers", "directions",
            "endpoint_half_spans", "strict_spearman",
        }
        missing = required.difference(reader.files)
        if missing:
            raise ValueError(f"Reader bank missing fields: {sorted(missing)}")
        axes = [str(value) for value in reader["axes"].tolist()]
        layers = reader["reader_layers"].astype(np.int32)
        directions = reader["directions"].astype(np.float32)
        spans = reader["endpoint_half_spans"].astype(np.float32)
        scores = reader["strict_spearman"].astype(np.float32)
        if (
            len(axes) != len(self.axes)
            or set(axes) != set(self.axes)
            or directions.shape != (len(axes), self.directions.shape[-1])
            or layers.shape != (len(axes),)
            or spans.shape != (len(axes),)
        ):
            raise ValueError("Reader bank is not compatible with this Direct-44 writer")
        if not (
            np.isfinite(directions).all()
            and np.isfinite(spans).all()
            and np.all(spans > 1e-6)
        ):
            raise ValueError("Reader bank contains invalid directions or calibration spans")
        for axis, layer, direction, span, score in zip(
            axes, layers, directions, spans, scores
        ):
            layer_id = int(layer)
            if layer_id not in self.writers:
                raise ValueError(f"Reader layer {layer_id} is outside the model")
            self.reader_axes.append(axis)
            self.reader_layers[axis] = layer_id
            self.reader_directions[axis] = direction
            self.reader_half_spans[axis] = float(span)
            self.reader_scores[axis] = float(score)
        self.reader_unique_layers = sorted(set(self.reader_layers.values()))

    def _load_writer_attribution(self, attribution_path: Path) -> None:
        """Load an optional 31B writer-delta attribution decoder."""
        if not attribution_path.exists():
            return
        decoder = np.load(attribution_path, allow_pickle=False)
        required = {"axes", "ridge", "transform"}
        missing = required.difference(decoder.files)
        if missing:
            raise ValueError(
                f"Writer-attribution decoder missing fields: {sorted(missing)}"
            )
        axes = [str(value) for value in decoder["axes"].tolist()]
        transform = decoder["transform"].astype(np.float32)
        ridge_values = decoder["ridge"].astype(np.float64)
        if axes != self.reader_axes:
            # Never apply a transform built in a different axis basis.
            return
        if (
            transform.shape != (len(axes), len(axes))
            or not np.isfinite(transform).all()
            or ridge_values.size != 1
            or not np.isfinite(ridge_values[0])
            or ridge_values[0] <= 0
        ):
            raise ValueError("Writer-attribution decoder has invalid dimensions")
        self.writer_attribution_transform = transform
        self.writer_attribution_ridge = float(ridge_values[0])
        self.writer_attribution_enabled = True

    def _load_chat_state_reader(self, reader_path: Path) -> None:
        """Load an optional daily-chat-centered state decoder.

        This decoder intentionally does not replace ``reader_axes`` or the
        raw state display: it has only been held out on its three pilot axes.
        """
        if not reader_path.exists():
            return
        decoder = np.load(reader_path, allow_pickle=False)
        required = {
            "state_axes", "raw_axes", "daily_null_mean", "daily_null_scale",
            "weights", "intercept", "ridge",
        }
        missing = required.difference(decoder.files)
        if missing:
            raise ValueError(f"Chat-state decoder missing fields: {sorted(missing)}")
        state_axes = [str(value) for value in decoder["state_axes"].tolist()]
        raw_axes = [str(value) for value in decoder["raw_axes"].tolist()]
        null_mean = decoder["daily_null_mean"].astype(np.float32)
        null_scale = decoder["daily_null_scale"].astype(np.float32)
        weights = decoder["weights"].astype(np.float32)
        intercept = decoder["intercept"].astype(np.float32)
        if raw_axes != self.reader_axes:
            return
        if (
            not state_axes
            or any(axis not in self.reader_axes for axis in state_axes)
            or null_mean.shape != (len(raw_axes),)
            or null_scale.shape != (len(raw_axes),)
            or weights.shape != (len(raw_axes), len(state_axes))
            or intercept.shape != (len(state_axes),)
            or not all(
                np.isfinite(value).all()
                for value in (null_mean, null_scale, weights, intercept)
            )
            or np.any(null_scale <= 0)
        ):
            raise ValueError("Chat-state decoder has invalid dimensions")
        self.chat_state_axes = state_axes
        self.chat_state_raw_axes = raw_axes
        self.chat_state_null_mean = null_mean
        self.chat_state_null_scale = null_scale
        self.chat_state_weights = weights
        self.chat_state_intercept = intercept
        self.chat_state_reader_enabled = True

    def _capture_reader_projections(
        self,
        prompt_messages: list[dict[str, str]],
        *,
        thinking: bool,
        include_activations: bool = False,
    ) -> dict[str, float] | tuple[dict[str, float], dict[int, np.ndarray]]:
        """Read post-block prompt activations without generating text.

        This is a paired hidden-state observation of the exact chat prompt,
        not text sentiment classification.  Reader layers are packaged after
        their respective final writer layers, so they can observe all_tokens
        writes on this pass.
        """
        if not self.reader_enabled:
            return {}
        for layer_id in self.reader_capture_layers:
            self.writers[layer_id].arm_capture(4)
        try:
            inputs = prepare_chat_inputs(
                self.processor,
                self.model,
                prompt_messages,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            with torch.inference_mode():
                self.model(**inputs, use_cache=False, logits_to_keep=1)
            activations: dict[int, np.ndarray] = {}
            for layer_id in self.reader_capture_layers:
                capture = self.writers[layer_id]._captured_pool
                if capture is None:
                    raise RuntimeError(f"Reader layer {layer_id} did not capture a residual")
                activations[layer_id] = capture.numpy().astype(np.float32, copy=False)[0]
        finally:
            for layer_id in self.reader_capture_layers:
                self.writers[layer_id].disarm_capture()
        projections = {
            axis: float(
                np.dot(
                    activations[self.reader_layers[axis]],
                    self.reader_directions[axis],
                )
            )
            for axis in self.reader_axes
        }
        return (projections, activations) if include_activations else projections

    def read_pre_response_state(
        self,
        prompt_messages: list[dict[str, str]],
        doses: dict[str, float],
        scope: str,
        mix_protection: Any,
        *,
        thinking: bool,
    ) -> dict[str, Any]:
        """Measure prompt-tail state with and without the current writer."""
        if not self.reader_enabled:
            return {"enabled": False, "phase": "after_message_before_generation"}
        started = time.perf_counter()
        active = any(abs(value) > 1e-9 for value in doses.values())
        if active:
            clear_residual_writers(self.writers)
        unmodulated, unmodulated_activations = self._capture_reader_projections(
            prompt_messages, thinking=thinking, include_activations=True
        )
        if active:
            self.set_axes(doses, scope, mix_protection)
            observed, observed_activations = self._capture_reader_projections(
                prompt_messages, thinking=thinking, include_activations=True
            )
        else:
            observed = unmodulated
            observed_activations = unmodulated_activations
        values: dict[str, dict[str, Any]] = {}
        raw_writer_deltas: list[float] = []
        raw_observed_values: list[float] = []
        for axis in self.reader_axes:
            span = self.reader_half_spans[axis]
            neutral = self.reader_neutral_projection[axis]
            baseline_value = (unmodulated[axis] - neutral) / span
            observed_value = (observed[axis] - neutral) / span
            writer_delta = (observed[axis] - unmodulated[axis]) / span
            raw_writer_deltas.append(float(writer_delta))
            raw_observed_values.append(float(observed_value))
            values[axis] = {
                "value": round(float(np.clip(observed_value, -2.0, 2.0)), 4),
                "raw_value": round(float(observed_value), 4),
                "message_baseline": round(float(np.clip(baseline_value, -2.0, 2.0)), 4),
                "writer_delta": round(float(np.clip(writer_delta, -2.0, 2.0)), 4),
                "raw_writer_delta": round(float(writer_delta), 4),
                "layer": self.reader_layers[axis],
                "quality": round(self.reader_scores[axis], 4),
            }
        attribution: dict[str, float] = {}
        ranked_attribution: list[str] = []
        if self.writer_attribution_enabled and self.writer_attribution_transform is not None:
            decoded = self.writer_attribution_transform @ np.asarray(
                raw_writer_deltas, dtype=np.float32
            )
            attribution = {
                axis: round(float(value), 3)
                for axis, value in zip(self.reader_axes, decoded)
            }
            ranked_attribution = sorted(
                self.reader_axes,
                key=lambda axis: abs(attribution[axis]),
                reverse=True,
            )
        chat_state: dict[str, Any] = {"enabled": False, "values": {}, "top": []}
        if (
            self.chat_state_reader_enabled
            and self.chat_state_null_mean is not None
            and self.chat_state_null_scale is not None
            and self.chat_state_weights is not None
            and self.chat_state_intercept is not None
        ):
            raw_vector = np.asarray(raw_observed_values, dtype=np.float32)
            decoded = (
                ((raw_vector - self.chat_state_null_mean) / self.chat_state_null_scale)
                @ self.chat_state_weights
                + self.chat_state_intercept
            )
            decoded_values = {
                axis: round(float(value), 4)
                for axis, value in zip(self.chat_state_axes, decoded)
            }
            chat_state = {
                "enabled": True,
                "values": decoded_values,
                "top": sorted(
                    self.chat_state_axes,
                    key=lambda axis: abs(decoded_values[axis]),
                    reverse=True,
                ),
                "note": (
                    "Pilot three-axis chat-domain decoder. It removes the everyday-chat "
                    "baseline before jointly decoding jealousy, anger, and anxiety; the "
                    "raw reader remains available as a reference."
                ),
            }
        experimental: dict[str, Any] = {}
        if self.experimental_probes.probes:
            try:
                experimental = self.experimental_probes.observe(observed_activations)
            except (KeyError, ValueError, OSError) as exc:
                self.experimental_probes.errors["runtime"] = str(exc)
        return {
            "enabled": True,
            "phase": "after_message_before_generation",
            "values": values,
            "top_writer_effect": sorted(
                self.reader_axes,
                key=lambda axis: abs(values[axis]["writer_delta"]),
                reverse=True,
            )[:8],
            "top_observed": sorted(
                self.reader_axes,
                key=lambda axis: abs(values[axis]["value"]),
                reverse=True,
            )[:8],
            "writer_attribution": attribution,
            "top_writer_attribution": ranked_attribution[:8],
            "writer_attribution_note": (
                "Conditional attribution explains only the writer-induced delta "
                "for the same prompt, expressed as an equivalent single-axis dose. "
                "It is not a measurement of the model's total state."
                if self.writer_attribution_enabled
                else "No matching writer-to-reader attribution calibration is loaded."
            ),
            "chat_state_disentangled": chat_state,
            "experimental_probes": experimental,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "scope_note": (
                "This pre-generation readout directly observes all_tokens writes. "
                "last_token and generation_only have not fired yet, so their "
                "pre-generation writer_delta should remain near zero."
            ),
            "scale": "0 is the neutral baseline for the same reader; ±1 approximates the calibrated endpoint shift. Display values are clipped to ±2.",
            "source": "paired_hidden_activation_reader",
        }

    @staticmethod
    def clean_message(value: Any) -> str:
        message = str(value or "").strip()
        if not message:
            raise ValueError("Message cannot be empty")
        if len(message) > 8000:
            raise ValueError("Message exceeds the 8,000-character limit")
        return message

    @staticmethod
    def clean_session(value: Any) -> str:
        session = str(value or "default").strip() or "default"
        if len(session) > 160:
            raise ValueError("session_id is too long")
        return session

    @staticmethod
    def clean_scope(value: Any) -> str:
        scope = str(value or "all_tokens").strip()
        if scope not in {"all_tokens", "last_token", "generation_only"}:
            raise ValueError("Unsupported modulation scope")
        return scope

    @staticmethod
    def clean_persona(value: Any) -> str:
        persona = str(value or "standard").strip()
        if persona not in SYSTEM_PROMPTS:
            raise ValueError("Unsupported runtime profile")
        return persona

    @staticmethod
    def clean_system_prompt(value: Any, persona: str) -> str:
        """Accept a per-request prompt without changing the server preset.

        The browser owns this value only for its current loaded page.  Empty
        input deliberately falls back to the unchanged preset, so a refresh
        or a normal request never mutates the runtime's global identity.
        """
        if value is None:
            return SYSTEM_PROMPTS[persona]
        prompt = str(value).strip()
        if not prompt:
            return SYSTEM_PROMPTS[persona]
        if len(prompt) > 4_000:
            raise ValueError("System prompt exceeds the 4,000-character limit")
        return prompt

    @staticmethod
    def clean_mix(value: Any) -> tuple[str, float]:
        if not isinstance(value, dict):
            return "raw", DEFAULT_MIX_HEADROOM
        mode = str(value.get("mode", "raw")).strip()
        if mode not in MIX_MODES:
            raise ValueError("Unsupported mixing-protection mode")
        try:
            headroom = float(value.get("headroom", DEFAULT_MIX_HEADROOM))
        except (TypeError, ValueError):
            headroom = DEFAULT_MIX_HEADROOM
        return mode, max(1.0, min(2.5, headroom))

    def clean_generation_limit(self, value: Any, *, thinking: bool) -> int:
        """Use a bounded request-specific limit for terse internal planners.

        The normal chat limit remains the server default, while an autonomy
        planner can explicitly request a short JSON-only completion.  This
        prevents an absent EOS token from turning a tiny state decision into a
        512-token generation that blocks the next conversational turn.
        """
        default = self.thinking_max_tokens if thinking else self.max_tokens
        try:
            requested = int(value)
        except (TypeError, ValueError):
            return default
        return max(16, min(default, requested))

    def set_axes(self, requested: Any, scope: str, mix_value: Any) -> tuple[dict[str, float], dict[str, Any]]:
        source = requested if isinstance(requested, dict) else {}
        doses: dict[str, float] = {}
        for axis in self.axes:
            try:
                value = float(source.get(axis, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            doses[axis] = max(-20.0, min(20.0, value))
        clear_residual_writers(self.writers)
        combined: dict[int, np.ndarray] = {}
        contributors: dict[int, list[np.ndarray]] = {}
        for axis, dose in doses.items():
            if not dose:
                continue
            index = self.axis_index[axis]
            for slot, layer in enumerate(self.layers_by_axis[index]):
                layer_id = int(layer)
                contribution = dose * self.weights[index, slot] * self.directions[index, slot]
                combined.setdefault(layer_id, np.zeros_like(contribution))[:] += contribution
                contributors.setdefault(layer_id, []).append(contribution)
        mode, headroom = self.clean_mix(mix_value)
        scaled_layers: list[dict[str, Any]] = []
        if mode == "rms_cap":
            for layer_id, direction in combined.items():
                parts = contributors[layer_id]
                if len(parts) < 2:
                    continue
                reference = float(np.sqrt(sum(float(np.vdot(item, item).real) for item in parts)))
                norm = float(np.linalg.norm(direction))
                limit = headroom * reference
                if reference > 1e-8 and norm > limit:
                    scale = limit / norm
                    combined[layer_id] = direction * scale
                    scaled_layers.append({"layer": layer_id, "contributors": len(parts), "pre_norm": round(norm, 5), "rms_reference": round(reference, 5), "limit": round(limit, 5), "scale": round(scale, 5)})
        for layer_id, direction in combined.items():
            self.writers[layer_id].set_residual_control(direction, alpha=1.0, scope=scope)
        return doses, {"mode": mode, "headroom": headroom, "active_layers": len(combined), "scaled_layers": scaled_layers}

    def messages(
        self,
        session: str,
        user_message: str,
        thinking: bool,
        persona: str,
        system_prompt: Any = None,
    ) -> list[dict[str, str]]:
        base_prompt = self.clean_system_prompt(system_prompt, persona)
        system = f"{base_prompt}\n{THINKING_RULE}" if thinking else base_prompt
        return [{"role": "system", "content": system}, *self.histories.get(session, []), {"role": "user", "content": user_message}]

    def finalize(self, session: str, prompt_messages: list[dict[str, str]], answer: str) -> int:
        turns = [item for item in prompt_messages if item["role"] != "system"]
        self.histories[session] = [*turns, {"role": "assistant", "content": answer}][-24:]
        return len(self.histories[session])

    @staticmethod
    def clean_output(value: str) -> str:
        return value.replace("<turn|>", "").strip()

    def stream_chat(self, message: Any, axes: Any, *, thinking: bool, session_id: Any, scope: Any, mix_protection: Any, max_tokens: Any = None, persona: Any = None, system_prompt: Any = None) -> Iterator[dict[str, Any]]:
        clean_message = self.clean_message(message)
        session = self.clean_session(session_id)
        clean_scope = self.clean_scope(scope)
        clean_persona = self.clean_persona(persona)
        with self.lock:
            doses, mixing = self.set_axes(axes, clean_scope, mix_protection)
            prompt_messages = self.messages(
                session,
                clean_message,
                bool(thinking),
                clean_persona,
                system_prompt,
            )
            generation_limit = self.clean_generation_limit(max_tokens, thinking=bool(thinking))
            yield {
                "type": "start",
                "axes": doses,
                "modulation_scope": clean_scope,
                "mix_protection": mixing,
                "persona": clean_persona,
                "emotion_readout": {
                    "enabled": self.reader_enabled,
                    "phase": "reading",
                },
            }
            emotion_readout = self.read_pre_response_state(
                prompt_messages,
                doses,
                clean_scope,
                mix_protection,
                thinking=bool(thinking),
            )
            yield {"type": "emotion_readout", "emotion_readout": emotion_readout}
            inputs = prepare_chat_inputs(
                self.processor,
                self.model,
                prompt_messages,
                add_generation_prompt=True,
                enable_thinking=bool(thinking),
            )
            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            # Transformers' streamer otherwise waits forever when a worker
            # fails before it can enqueue its terminal sentinel.  The remote
            # service is supervised by `screen`; if the CUDA worker itself is
            # wedged, a hard restart is safer than retaining its global lock.
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=False, timeout=30.0
            )
            failure: list[BaseException] = []
            def generate() -> None:
                try:
                    with torch.inference_mode():
                        self.model.generate(**inputs, streamer=streamer, max_new_tokens=generation_limit, do_sample=False, use_cache=True)
                except BaseException as error:  # surfaced after the streamer closes
                    failure.append(error)
            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            started = time.perf_counter()
            raw = ""
            latest = ""
            try:
                for piece in streamer:
                    raw += piece
                    answer = self.clean_output(raw)
                    if answer != latest:
                        latest = answer
                        yield {"type": "snapshot", "answer": answer}
            except Empty as error:
                # A worker can fail before TextIteratorStreamer receives its
                # closing sentinel.  Join briefly so that its concrete Python
                # exception is surfaced to the caller instead of being hidden
                # behind a generic stream-timeout message.
                worker.join(timeout=0.2)
                if failure:
                    print(f"Gemma generation failed: {failure[0]!r}", flush=True)
                    raise RuntimeError(f"Gemma generation failed: {failure[0]}") from failure[0]
                if worker.is_alive():
                    print("Generation worker stalled for 30 seconds; restarting service.", flush=True)
                    os._exit(70)
                raise RuntimeError("Gemma generation ended without closing the stream") from error
            worker.join()
            if failure:
                print(f"Gemma generation failed: {failure[0]!r}", flush=True)
                raise RuntimeError(f"Gemma generation failed: {failure[0]}") from failure[0]
            answer = self.clean_output(raw)
            yield {"type": "done", "answer": answer, "axes": doses, "context_messages": self.finalize(session, prompt_messages, answer), "elapsed_seconds": round(time.perf_counter() - started, 3), "generation_tokens": 0, "max_tokens": generation_limit, "thinking_enabled": bool(thinking), "modulation_scope": clean_scope, "mix_protection": mixing, "persona": clean_persona, "emotion_readout": emotion_readout}

    def observe_message(
        self,
        message: Any,
        *,
        axes: Any = None,
        thinking: bool = False,
        persona: Any = None,
        scope: Any = "all_tokens",
        mix_protection: Any = None,
        system_prompt: Any = None,
    ) -> dict[str, Any]:
        """Read a fresh chat prompt without generation or history mutation.

        This is intentionally an evaluation-only endpoint.  It runs the exact
        standard chat template used by the live page, but does not append a
        turn, call ``generate``, or apply any writer.  It makes it practical
        to collect a daily-chat null distribution for reader calibration. A
        supplied ``axes`` mapping uses the exact live writer compositor, so
        this same endpoint can measure causal reader response at fixed text.
        """
        clean_message = self.clean_message(message)
        clean_persona = self.clean_persona(persona)
        clean_scope = self.clean_scope(scope)
        base_prompt = self.clean_system_prompt(system_prompt, clean_persona)
        system = f"{base_prompt}\n{THINKING_RULE}" if thinking else base_prompt
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": clean_message},
        ]
        with self.lock:
            doses, mixing = self.set_axes(
                axes if isinstance(axes, dict) else {}, clean_scope, mix_protection
            )
            readout = self.read_pre_response_state(
                prompt_messages,
                doses,
                clean_scope,
                mix_protection,
                thinking=bool(thinking),
            )
        return {
            "message": clean_message,
            "persona": clean_persona,
            "using_default_system_prompt": base_prompt == SYSTEM_PROMPTS[clean_persona],
            "thinking": bool(thinking),
            "axes": doses,
            "modulation_scope": clean_scope,
            "mix_protection": mixing,
            "history_retained": False,
            "emotion_readout": readout,
        }

    def score_continuations(
        self,
        message: Any,
        candidates: Any,
        *,
        axes: Any = None,
        thinking: bool = False,
        persona: Any = None,
        scope: Any = "all_tokens",
        mix_protection: Any = None,
        system_prompt: Any = None,
    ) -> dict[str, Any]:
        """Score alternative assistant continuations without generating text.

        This is an evaluation-only preference-surface endpoint.  Given one
        fixed chat prompt and two or more short, *complete* continuations, it
        returns length-normalised teacher-forced log probabilities.  It does
        not mutate chat history and therefore avoids an assistant response
        becoming the next test's context.

        The same live residual writer is installed both while processing the
        prompt and while predicting each continuation.  Consequently a score
        difference measures a state-conditioned next-text preference, rather
        than a classifier's reading of already generated prose.
        """
        clean_message = self.clean_message(message)
        clean_persona = self.clean_persona(persona)
        clean_scope = self.clean_scope(scope)
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 8:
            raise ValueError("candidates must contain between 2 and 8 continuations")
        cleaned_candidates = [str(candidate or "").strip() for candidate in candidates]
        if any(not candidate for candidate in cleaned_candidates):
            raise ValueError("Continuation candidates cannot be empty")
        if any(len(candidate) > 600 for candidate in cleaned_candidates):
            raise ValueError("Each continuation must be at most 600 characters")
        base_prompt = self.clean_system_prompt(system_prompt, clean_persona)
        system = f"{base_prompt}\n{THINKING_RULE}" if thinking else base_prompt
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": clean_message},
        ]
        with self.lock:
            doses, mixing = self.set_axes(
                axes if isinstance(axes, dict) else {}, clean_scope, mix_protection
            )
            # This hidden-state readout is included for writer-manipulation
            # checks only.  It is not used to construct continuation scores.
            readout = self.read_pre_response_state(
                prompt_messages,
                doses,
                clean_scope,
                mix_protection,
                thinking=bool(thinking),
            )
            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            scored: list[dict[str, Any]] = []
            for candidate in cleaned_candidates:
                full_inputs = prepare_chat_inputs(
                    self.processor,
                    self.model,
                    [*prompt_messages, {"role": "assistant", "content": candidate}],
                    add_generation_prompt=False,
                    enable_thinking=bool(thinking),
                )
                input_ids = full_inputs["input_ids"]
                candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if not candidate_ids:
                    raise RuntimeError("candidate tokenization unexpectedly produced no tokens")
                full_ids = input_ids[0].tolist()
                # Some Gemma chat-template variants use a distinct generation
                # opener versus a completed assistant turn.  Locate the exact
                # literal continuation rather than assuming their prefixes are
                # byte-identical.  Last occurrence is correct because the
                # candidate occupies the final assistant turn.
                candidate_start = next(
                    (
                        index
                        for index in range(len(full_ids) - len(candidate_ids), -1, -1)
                        if full_ids[index:index + len(candidate_ids)] == candidate_ids
                    ),
                    None,
                )
                if candidate_start is None or candidate_start == 0:
                    raise RuntimeError(
                        "could not locate literal candidate tokens in the completed chat template"
                    )
                with torch.inference_mode():
                    outputs = self.model(**full_inputs, use_cache=False)
                    logits = outputs.logits[:, :-1, :].float()
                    targets = input_ids[:, 1:]
                    # ``logits[:, j-1]`` predicts input token ``j``.  Score
                    # only the literal candidate, never chat delimiters.
                    start = candidate_start - 1
                    end = candidate_start + len(candidate_ids) - 1
                    token_log_probs = torch.log_softmax(logits[:, start:end, :], dim=-1)
                    target_ids = targets[:, start:end]
                    selected = token_log_probs.gather(
                        -1, target_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    total = float(selected.sum().item())
                    token_count = int(selected.numel())
                scored.append(
                    {
                        "candidate": candidate,
                        "sum_logprob": round(total, 6),
                        "token_count": token_count,
                        "mean_logprob": round(total / max(token_count, 1), 6),
                    }
                )
        ordered = sorted(scored, key=lambda item: float(item["mean_logprob"]), reverse=True)
        return {
            "message": clean_message,
            "persona": clean_persona,
            "using_default_system_prompt": base_prompt == SYSTEM_PROMPTS[clean_persona],
            "thinking": bool(thinking),
            "axes": doses,
            "modulation_scope": clean_scope,
            "mix_protection": mixing,
            "history_retained": False,
            "scores": scored,
            "ranking": [item["candidate"] for item in ordered],
            "mean_logprob_margin_first_minus_second": round(
                float(scored[0]["mean_logprob"]) - float(scored[1]["mean_logprob"]), 6
            ),
            "emotion_readout": readout,
            "method": "teacher_forced_length_normalized_continuation_logprob",
            "note": (
                "Scores are length-normalized conditional log probabilities under the "
                "specified prompt and modulation. They are not personality diagnoses "
                "or evidence of subjective preference."
            ),
        }

    def reset(self, session_id: Any) -> dict[str, Any]:
        session = self.clean_session(session_id)
        with self.lock:
            self.histories.pop(session, None)
            clear_residual_writers(self.writers)
        return {
            "ok": True,
            "session_id": session,
            "context_messages": 0,
            "emotion_readout": {"enabled": self.reader_enabled, "phase": "reset"},
        }

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "model": "Gemma 4 31B IT BF16 (CUDA)",
            "writer": Path(self.writer_path).name,
            "default_system_prompt": SYSTEM_PROMPTS["standard"],
            "weights": self.weights[0].tolist(),
            "axis_info": [
                {
                    "name": axis,
                    "label": AXIS_LABELS.get(axis, axis),
                    "category": self.categories[i],
                    "category_label": CATEGORY_LABELS.get(self.categories[i], self.categories[i]),
                    "layers": self.layers_by_axis[i].tolist(),
                }
                for i, axis in enumerate(self.axes)
            ],
            "generation_limits": {"direct": self.max_tokens, "thinking": self.thinking_max_tokens},
            "mix_protection": {
                "default": {"mode": "raw", "headroom": 1.35},
                "modes": {"raw": "Raw vector addition", "rms_cap": "Shared-layer RMS cap"},
            },
            "emotion_reader": {
                "enabled": self.reader_enabled,
                "path": str(self.reader_path),
                "axis_count": len(self.reader_axes),
                "layers": self.reader_unique_layers,
                "capture_layers": self.reader_capture_layers,
                "method": "paired_hidden_activation_reader",
                "mean_strict_spearman": (
                    round(float(np.mean(list(self.reader_scores.values()))), 4)
                    if self.reader_scores else None
                ),
                "writer_attribution": {
                    "enabled": self.writer_attribution_enabled,
                    "path": str(self.attribution_path),
                    "ridge": self.writer_attribution_ridge,
                    "method": (
                        "regularized_writer_delta_attribution"
                        if self.writer_attribution_enabled
                        else None
                    ),
                },
                "chat_state_reader": {
                    "enabled": self.chat_state_reader_enabled,
                    "path": str(self.chat_state_reader_path),
                    "axes": self.chat_state_axes,
                    "method": (
                        "daily_chat_centered_joint_ridge_pilot"
                        if self.chat_state_reader_enabled
                        else None
                    ),
                },
                "experimental_probes": self.experimental_probes.status(),
            },
            "load_seconds": round(self.load_seconds, 3),
        }


class Handler(SimpleHTTPRequestHandler):
    engine: Direct44CudaEngine

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(raw)

    def body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self.send_json(200, self.engine.status()); return
        if self.path in {"/", "/index.html"}:
            raw = PAGE_PATH.read_text(encoding="utf-8").replace(
                "Gemma 4 · Direct-44", "Gemma 4 31B · Direct-44"
            ).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(raw); return
        self.send_error(404)

    def do_POST(self) -> None:
        stream_started = False
        try:
            payload = self.body()
            if self.path == "/api/reset":
                self.send_json(200, self.engine.reset(payload.get("session_id"))); return
            if self.path == "/api/observe":
                self.send_json(
                    200,
                    self.engine.observe_message(
                        payload.get("message"),
                        axes=payload.get("axes", {}),
                        thinking=bool(payload.get("thinking")),
                        persona=payload.get("persona"),
                        scope=payload.get("modulation_scope"),
                        mix_protection=payload.get("mix_protection"),
                        system_prompt=payload.get("system_prompt"),
                    ),
                )
                return
            if self.path == "/api/score-continuations":
                self.send_json(
                    200,
                    self.engine.score_continuations(
                        payload.get("message"),
                        payload.get("candidates"),
                        axes=payload.get("axes", {}),
                        thinking=bool(payload.get("thinking")),
                        persona=payload.get("persona"),
                        scope=payload.get("modulation_scope"),
                        mix_protection=payload.get("mix_protection"),
                        system_prompt=payload.get("system_prompt"),
                    ),
                )
                return
            if self.path == "/api/chat/stream":
                self.send_response(200); self.send_header("Content-Type", "application/x-ndjson; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("X-Accel-Buffering", "no"); self.end_headers()
                stream_started = True
                for event in self.engine.stream_chat(payload.get("message"), payload.get("axes", {}), thinking=bool(payload.get("thinking")), session_id=payload.get("session_id"), scope=payload.get("modulation_scope"), mix_protection=payload.get("mix_protection"), max_tokens=payload.get("max_tokens"), persona=payload.get("persona"), system_prompt=payload.get("system_prompt")):
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")); self.wfile.flush()
                return
            self.send_json(404, {"error": "Endpoint not found"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            print(f"HTTP handler failure: {type(error).__name__}: {error}\n{traceback.format_exc()}", flush=True)
            if stream_started:
                try:
                    self.wfile.write((json.dumps({"type": "error", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_json(400 if isinstance(error, ValueError) else 500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking-max-tokens", type=int, default=1024)
    parser.add_argument("--reader", type=Path, default=READER_PATH)
    args = parser.parse_args()
    if not args.model.exists() or not args.writer.exists():
        raise FileNotFoundError("Model or writer path does not exist")
    print(f"Loading CUDA model: {args.model}", flush=True)
    engine = Direct44CudaEngine(
        args.model,
        args.writer,
        args.max_tokens,
        args.thinking_max_tokens,
        args.reader,
    )
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Ready in {engine.load_seconds:.2f}s: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
