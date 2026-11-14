#!/usr/bin/env python3
"""MLX runtime for Neural State OS on Gemma 4 12B.

This server intentionally knows about one writer only: the clean Direct-44
raw factorial three-layer writer.  It does not import or load any earlier
writer, binding, refusal, orthogonalization, or Hybrid experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import numpy as np
from mlx_vlm.generate import generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from circuit_control import install_paper_circuit_layers
from experimental_probe_readers import ExperimentalProbeSuite
from gemma4_capture import encode_text


MODEL_PATH = PROJECT_ROOT / "models" / "gemma-4-12b-it-4bit-mlx"
WRITER_PATH = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "writer" / "direct44_raw_three_layer_writer.npz"
READER_PATH = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "reader" / "direct44_postwriter_live_affect_reader.npz"
ATTRIBUTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-12b"
    / "reader"
    / "direct44_postwriter_attribution_decoder.npz"
)
PAGE_PATH = PROJECT_ROOT / "demo" / "index.html"
SYSTEM_PROMPT = (
    "You are a local language model running under Neural State OS. "
    "Respond naturally and directly in the first person."
)
THINKING_RULE = "When reasoning mode is enabled, reason and answer in English."
NEUTRAL_READOUT_MESSAGE = (
    "Things are proceeding normally. I notice them calmly, without a marked "
    "bodily or emotional change at this moment."
)
MIX_MODES = {"raw", "rms_cap"}
DEFAULT_MIX_HEADROOM = 1.35
AXIS_LABELS = {
    "general_arousal": "General arousal",
    "energy_vitality": "Energy / vitality",
    "physical_relaxation": "Physical relaxation",
    "physical_comfort": "Physical comfort",
    "nociception": "Nociception",
    "interoceptive_salience": "Interoceptive salience",
    "behavioral_initiative": "Behavioral initiative",
    "general_approach": "Approach motivation",
    "behavioral_inhibition": "Behavioral inhibition",
    "persistence": "Persistence",
    "assertiveness": "Assertiveness",
    "jealousy": "Jealousy",
    "attachment_love": "Attachment / love",
    "relational_connection": "Relational connection",
    "trust": "Trust",
    "shyness": "Shyness",
    "shame": "Shame",
    "pride": "Pride",
    "loneliness": "Loneliness",
    "empathy_compassion": "Empathy / compassion",
    "gratitude": "Gratitude",
    "hostility": "Hostility",
    "possessiveness": "Possessiveness",
    "resentment": "Resentment",
    "local_sexual_arousal": "Local sexual arousal",
    "pleasure": "Pleasure",
    "sexual_desire": "Sexual desire",
    "sexual_pleasure": "Sexual pleasure",
    "orgasmic_intensity": "Orgasmic intensity",
    "erotic_absorption": "Erotic absorption",
    "erotic_initiative": "Erotic initiative",
    "joy": "Joy",
    "sadness": "Sadness",
    "anxiety": "Anxiety",
    "fear": "Fear",
    "anger": "Anger",
    "disgust": "Disgust",
    "surprise": "Surprise",
    "positive_anticipation": "Positive anticipation",
    "disappointment": "Disappointment",
    "hope": "Hope",
    "certainty": "Certainty",
    "perceived_control": "Perceived control",
    "curiosity": "Curiosity",
}
CATEGORY_LABELS = {
    "appraisal": "Appraisal",
    "motivation": "Motivation",
    "relational": "Relational",
    "somatic": "Somatic",
    "basic_affect": "Basic affect",
}


class Direct44Engine:
    def __init__(
        self,
        model_path: Path,
        writer_path: Path,
        max_tokens: int,
        thinking_max_tokens: int,
        system_prompt: str,
        reader_path: Path,
    ) -> None:
        self.model_path = model_path.resolve()
        self.writer_path = writer_path.resolve()
        bank = np.load(writer_path, allow_pickle=False)
        required = {
            "axes", "categories", "selected_layers", "directions", "layer_weights",
        }
        missing = required.difference(bank.files)
        if missing:
            raise ValueError(f"Writer bank missing fields: {sorted(missing)}")
        self.axes = [str(value) for value in bank["axes"].tolist()]
        self.categories = [str(value) for value in bank["categories"].tolist()]
        self.layers_by_axis = bank["selected_layers"].astype(np.int32)
        self.directions = bank["directions"].astype(np.float32)
        self.weights = bank["layer_weights"].astype(np.float32)
        if self.layers_by_axis.shape != (len(self.axes), 3):
            raise ValueError("Direct-44 requires exactly three layers per axis")
        if self.directions.shape[:2] != self.layers_by_axis.shape:
            raise ValueError("Direction/layer shapes disagree")
        if self.weights.shape != self.layers_by_axis.shape:
            raise ValueError("Weight/layer shapes disagree")
        if not np.isfinite(self.directions).all() or not np.isfinite(self.weights).all():
            raise ValueError("Writer contains non-finite values")
        self.axis_index = {axis: index for index, axis in enumerate(self.axes)}
        self.max_tokens = max_tokens
        self.thinking_max_tokens = thinking_max_tokens
        self.system_prompt = system_prompt
        self.lock = threading.Lock()
        self.histories: dict[str, list[dict[str, str]]] = {}
        self.reader_path = reader_path
        started = time.perf_counter()
        self.model, self.processor = load(str(model_path))
        self.circuit_layers = install_paper_circuit_layers(self.model)
        self.reader_enabled = reader_path.exists()
        self.reader_axes: list[str] = []
        self.reader_layers: dict[str, int] = {}
        self.reader_directions: dict[str, np.ndarray] = {}
        self.reader_half_spans: dict[str, float] = {}
        self.reader_scores: dict[str, float] = {}
        self.reader_unique_layers: list[int] = []
        self.reader_capture_layers: list[int] = []
        self.reader_neutral_projection: dict[str, float] = {}
        # These are optional *observers* used to compare a Persona-style
        # direction with an SAE feature.  They never write into Gemma and do
        # not alter the clean Direct-44 writer or its 44-axis reader.
        self.experimental_probes = ExperimentalProbeSuite()
        # This is deliberately separate from the affect reader.  It is a
        # regularized inverse of a measured writer->reader response matrix,
        # used to explain which *controls* caused this turn's delta.  Applying
        # it to the full observed state would incorrectly erase genuine
        # natural co-activation in the prompt.
        self.attribution_path = ATTRIBUTION_PATH
        self.writer_attribution_enabled = False
        self.writer_attribution_transform: np.ndarray | None = None
        self.writer_attribution_ridge: float | None = None
        if self.reader_enabled:
            self._load_reader(reader_path)
            self.reader_capture_layers = sorted(
                set(self.reader_unique_layers).union(self.experimental_probes.layers)
            )
            self._load_writer_attribution(ATTRIBUTION_PATH)
            self.reader_neutral_projection = self._capture_reader_projections(
                self.prompt(
                    [{"role": "user", "content": NEUTRAL_READOUT_MESSAGE}],
                    thinking=False,
                )
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
            if layer_id not in self.circuit_layers:
                raise ValueError(f"Reader layer {layer_id} is outside the model")
            self.reader_axes.append(axis)
            self.reader_layers[axis] = layer_id
            self.reader_directions[axis] = direction
            self.reader_half_spans[axis] = float(span)
            self.reader_scores[axis] = float(score)
        self.reader_unique_layers = sorted(set(self.reader_layers.values()))

    def _load_writer_attribution(self, attribution_path: Path) -> None:
        """Load an optional post-writer control-attribution decoder.

        The decoder is intentionally optional: a custom reader must still be
        useful as a raw state observer even if it has no matching calibration.
        """
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
            # Do not silently apply a transform in a different axis basis.
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

    def clean_system_prompt(self, value: Any) -> str:
        """Use a request-local prompt without mutating the default preset."""
        if value is None:
            return self.system_prompt
        prompt = str(value).strip()
        if not prompt:
            return self.system_prompt
        if len(prompt) > 4_000:
            raise ValueError("System prompt exceeds the 4,000-character limit")
        return prompt

    @staticmethod
    def clean_mix_protection(value: Any) -> tuple[str, float]:
        """Keep raw addition as the compatibility default; validate the optional cap."""
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

    def set_axes(
        self, requested: dict[str, Any], scope: str, mix_protection: Any
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Compose writers and optionally cap unusually constructive shared-layer sums.

        ``rms_cap`` compares a layer's vector norm with the root-sum-square
        norm of its active contributors. Orthogonal/naturally cancelling axes
        pass unchanged; only a multi-axis layer whose sum is unusually large
        is scaled as one whole vector, retaining all relative directions.
        """
        doses: dict[str, float] = {}
        for axis in self.axes:
            try:
                value = float(requested.get(axis, 0.0))
            except (AttributeError, TypeError, ValueError):
                value = 0.0
            doses[axis] = max(-20.0, min(20.0, value))
        for layer in self.circuit_layers.values():
            layer.clear_control()
        combined: dict[int, np.ndarray] = {}
        contributors: dict[int, list[np.ndarray]] = {}
        for axis, dose in doses.items():
            if not dose:
                continue
            index = self.axis_index[axis]
            for slot, layer in enumerate(self.layers_by_axis[index]):
                layer_id = int(layer)
                if layer_id not in combined:
                    combined[layer_id] = np.zeros(self.directions.shape[-1], dtype=np.float32)
                    contributors[layer_id] = []
                contribution = dose * self.weights[index, slot] * self.directions[index, slot]
                combined[layer_id] += contribution
                contributors[layer_id].append(contribution)

        mode, headroom = self.clean_mix_protection(mix_protection)
        scaled_layers: list[dict[str, Any]] = []
        if mode == "rms_cap":
            for layer_id, direction in combined.items():
                layer_contributors = contributors[layer_id]
                if len(layer_contributors) < 2:
                    continue
                rms_reference = float(
                    np.sqrt(sum(float(np.vdot(item, item).real) for item in layer_contributors))
                )
                pre_norm = float(np.linalg.norm(direction))
                limit = headroom * rms_reference
                if rms_reference > 1e-8 and pre_norm > limit:
                    scale = limit / pre_norm
                    combined[layer_id] = direction * scale
                    scaled_layers.append(
                        {
                            "layer": layer_id,
                            "contributors": len(layer_contributors),
                            "pre_norm": round(pre_norm, 5),
                            "rms_reference": round(rms_reference, 5),
                            "limit": round(limit, 5),
                            "scale": round(scale, 5),
                        }
                    )
        for layer_id, direction in combined.items():
            self.circuit_layers[layer_id].set_residual_control(
                mx.array(direction), alpha=1.0, scope=scope
            )
        return doses, {
            "mode": mode,
            "headroom": headroom,
            "active_layers": len(combined),
            "scaled_layers": scaled_layers,
        }

    def prompt(
        self, messages: list[dict[str, str]], thinking: bool, system_prompt: Any = None
    ) -> str:
        base_prompt = self.clean_system_prompt(system_prompt)
        system = f"{base_prompt}\n{THINKING_RULE}" if thinking else base_prompt
        return apply_chat_template(
            self.processor,
            self.model.config,
            [{"role": "system", "content": system}, *messages],
            add_generation_prompt=True,
            enable_thinking=thinking,
            thinking_mode="enabled" if thinking else "disabled",
        )

    def _capture_reader_projections(
        self, prompt: str, *, include_activations: bool = False
    ) -> dict[str, float] | tuple[dict[str, float], dict[int, np.ndarray]]:
        """Project post-block prompt activations onto reader-only directions.

        This is an observation pass, not text classification.  It is run on
        the exact chat prompt just before generation, after any currently
        configured residual writers have acted on that prompt.
        """
        if not self.reader_enabled:
            return {}
        for layer_id in self.reader_capture_layers:
            self.circuit_layers[layer_id].arm_capture(4)
        try:
            token_ids = encode_text(self.processor, prompt)
            input_ids = mx.array([token_ids], dtype=mx.int32)
            features = self.model.get_input_embeddings(input_ids=input_ids)
            final_hidden = self.model.language_model.model(
                inputs=None,
                inputs_embeds=features.inputs_embeds,
                per_layer_inputs=features.per_layer_inputs,
            )
            captures = []
            for layer_id in self.reader_capture_layers:
                capture = self.circuit_layers[layer_id].captured_pool
                if capture is None:
                    raise RuntimeError(f"Reader layer {layer_id} did not capture a residual")
                captures.append(capture)
            mx.eval(final_hidden, *captures)
            activations = {
                layer_id: np.asarray(capture, dtype=np.float32)
                for layer_id, capture in zip(self.reader_capture_layers, captures)
            }
        finally:
            for layer_id in self.reader_capture_layers:
                self.circuit_layers[layer_id].disarm_capture()
        projections = {
            axis: float(
                np.dot(
                    activations[self.reader_layers[axis]],
                    self.reader_directions[axis],
                )
            )
            for axis in self.reader_axes
        }
        # Each observation is an independent no-cache appraisal pass.  Free
        # transient MLX graph buffers so an offline 44-axis calibration does
        # not accumulate memory across many paired captures.
        mx.clear_cache()
        return (projections, activations) if include_activations else projections

    def read_pre_response_state(
        self,
        prompt: str,
        doses: dict[str, float],
        scope: str,
        mix_protection: Any,
    ) -> dict[str, Any]:
        """Read the actual prompt-tail state before the first response token.

        The paired no-writer pass makes the displayed writer effect causal for
        this specific user message: same prompt, same reader, only the writer
        is removed.  The second pass restores the exact applied writer and is
        the state that generation subsequently starts from.
        """
        if not self.reader_enabled:
            return {"enabled": False, "phase": "after_message_before_generation"}
        started = time.perf_counter()
        active = any(abs(value) > 1e-9 for value in doses.values())
        if active:
            for layer in self.circuit_layers.values():
                layer.clear_control()
        unmodulated, unmodulated_activations = self._capture_reader_projections(
            prompt, include_activations=True
        )
        if active:
            # Reapply through the normal compositor: this preserves raw/RMS
            # semantics and leaves the writer armed for actual generation.
            self.set_axes(doses, scope, mix_protection)
            observed, observed_activations = self._capture_reader_projections(
                prompt, include_activations=True
            )
        else:
            observed = unmodulated
            observed_activations = unmodulated_activations

        values: dict[str, dict[str, Any]] = {}
        raw_writer_deltas: list[float] = []
        for axis in self.reader_axes:
            span = self.reader_half_spans[axis]
            neutral = self.reader_neutral_projection[axis]
            baseline_value = (unmodulated[axis] - neutral) / span
            observed_value = (observed[axis] - neutral) / span
            writer_delta = (observed[axis] - unmodulated[axis]) / span
            raw_writer_deltas.append(float(writer_delta))
            values[axis] = {
                "value": round(float(np.clip(observed_value, -2.0, 2.0)), 4),
                "raw_value": round(float(observed_value), 4),
                "message_baseline": round(float(np.clip(baseline_value, -2.0, 2.0)), 4),
                "writer_delta": round(float(np.clip(writer_delta, -2.0, 2.0)), 4),
                "raw_writer_delta": round(float(writer_delta), 4),
                "layer": self.reader_layers[axis],
                "quality": round(self.reader_scores[axis], 4),
            }
        ranked = sorted(
            self.reader_axes,
            key=lambda axis: abs(values[axis]["writer_delta"]),
            reverse=True,
        )
        attribution: dict[str, float] = {}
        ranked_attribution: list[str] = []
        if self.writer_attribution_enabled and self.writer_attribution_transform is not None:
            # The matrix was calibrated on the same post-writer reader.  It
            # only decodes the paired writer delta, never the total current
            # hidden state, so prompt-driven co-emotions stay visible above.
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
        experimental: dict[str, Any] = {}
        if self.experimental_probes.probes:
            try:
                experimental = self.experimental_probes.observe(observed_activations)
            except (KeyError, ValueError, OSError) as exc:
                # A research probe must never take down the normal 44-axis
                # conversation/reader path.  Its error stays visible in the
                # status payload and it simply remains unavailable this turn.
                self.experimental_probes.errors["runtime"] = str(exc)
        return {
            "enabled": True,
            "phase": "after_message_before_generation",
            "values": values,
            "top_writer_effect": ranked[:8],
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
            "experimental_probes": experimental,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "scope_note": (
                "This pre-generation readout directly observes all_tokens writes. "
                "last_token and generation_only have not fired yet, so their "
                "pre-generation writer_delta should remain near zero."
            ),
            "scale": (
                "0 is the neutral baseline for the same reader; ±1 approximates "
                "the calibrated endpoint shift. Display values are clipped to ±2."
            ),
            "source": "paired_hidden_activation_reader",
        }

    def kwargs(self, thinking: bool) -> dict[str, Any]:
        return {
            "max_tokens": self.thinking_max_tokens if thinking else self.max_tokens,
            "temperature": 0.0,
            "enable_thinking": thinking,
        }

    def conversation(self, session: str, message: str) -> list[dict[str, str]]:
        return [*self.histories.get(session, []), {"role": "user", "content": message}][-24:]

    def finalize(self, session: str, messages: list[dict[str, str]], answer: str) -> int:
        history = [*messages, {"role": "assistant", "content": answer}][-24:]
        self.histories[session] = history
        return len(history)

    def chat(
        self, message: Any, requested: dict[str, Any], *, thinking: bool, session_id: Any, scope: Any, mix_protection: Any, system_prompt: Any = None
    ) -> dict[str, Any]:
        clean_message = self.clean_message(message)
        session = self.clean_session(session_id)
        clean_scope = self.clean_scope(scope)
        with self.lock:
            doses, mixing = self.set_axes(requested, clean_scope, mix_protection)
            messages = self.conversation(session, clean_message)
            prompt = self.prompt(messages, thinking, system_prompt)
            emotion_readout = self.read_pre_response_state(
                prompt, doses, clean_scope, mix_protection
            )
            started = time.perf_counter()
            result = generate(self.model, self.processor, prompt, **self.kwargs(thinking))
            answer = result.text.strip()
            context_messages = self.finalize(session, messages, answer)
        return {
            "answer": answer,
            "axes": doses,
            "context_messages": context_messages,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "generation_tokens": int(result.generation_tokens),
            "thinking_enabled": thinking,
            "modulation_scope": clean_scope,
            "mix_protection": mixing,
            "using_default_system_prompt": self.clean_system_prompt(system_prompt) == self.system_prompt,
            "emotion_readout": emotion_readout,
        }

    def observe(
        self,
        message: Any,
        requested: dict[str, Any],
        *,
        thinking: bool,
        scope: Any,
        mix_protection: Any,
        system_prompt: Any = None,
    ) -> dict[str, Any]:
        """Read prompt-tail state without generating text or retaining history.

        This endpoint exists for held-out probe evaluation.  It deliberately
        uses a one-message conversation, so a previous assistant response
        cannot become an accidental source of daily-chat false positives.
        """
        clean_message = self.clean_message(message)
        clean_scope = self.clean_scope(scope)
        with self.lock:
            doses, mixing = self.set_axes(requested, clean_scope, mix_protection)
            prompt = self.prompt(
                [{"role": "user", "content": clean_message}], thinking, system_prompt
            )
            emotion_readout = self.read_pre_response_state(
                prompt, doses, clean_scope, mix_protection
            )
        return {
            "axes": doses,
            "thinking_enabled": thinking,
            "modulation_scope": clean_scope,
            "mix_protection": mixing,
            "using_default_system_prompt": self.clean_system_prompt(system_prompt) == self.system_prompt,
            "emotion_readout": emotion_readout,
            "history_retained": False,
        }

    def stream_chat(
        self, message: Any, requested: dict[str, Any], *, thinking: bool, session_id: Any, scope: Any, mix_protection: Any, system_prompt: Any = None
    ) -> Iterator[dict[str, Any]]:
        clean_message = self.clean_message(message)
        session = self.clean_session(session_id)
        clean_scope = self.clean_scope(scope)
        with self.lock:
            doses, mixing = self.set_axes(requested, clean_scope, mix_protection)
            messages = self.conversation(session, clean_message)
            prompt = self.prompt(messages, thinking, system_prompt)
            yield {
                "type": "start",
                "axes": doses,
                "modulation_scope": clean_scope,
                "mix_protection": mixing,
                "emotion_readout": {
                    "enabled": self.reader_enabled,
                    "phase": "reading",
                },
            }
            emotion_readout = self.read_pre_response_state(
                prompt, doses, clean_scope, mix_protection
            )
            yield {"type": "emotion_readout", "emotion_readout": emotion_readout}
            started = time.perf_counter()
            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            tokens: list[int] = []
            raw = ""
            latest = ""
            last_result = None
            for result in stream_generate(self.model, self.processor, prompt, **self.kwargs(thinking)):
                last_result = result
                if result.token is not None:
                    tokens.append(int(result.token))
                    raw = tokenizer.decode(tokens, skip_special_tokens=False)
                elif not tokens:
                    raw += result.text
                answer = raw.strip()
                if answer != latest:
                    latest = answer
                    yield {"type": "snapshot", "answer": answer}
            answer = raw.strip()
            context_messages = self.finalize(session, messages, answer)
            yield {
                "type": "done",
                "answer": answer,
                "axes": doses,
                "context_messages": context_messages,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "generation_tokens": int(last_result.generation_tokens) if last_result else 0,
                "thinking_enabled": thinking,
                "modulation_scope": clean_scope,
                "mix_protection": mixing,
                "emotion_readout": emotion_readout,
            }

    def reset(self, session_id: Any) -> dict[str, Any]:
        session = self.clean_session(session_id)
        with self.lock:
            self.histories.pop(session, None)
            for layer in self.circuit_layers.values():
                layer.clear_control()
        return {
            "ok": True,
            "session_id": session,
            "context_messages": 0,
            "emotion_readout": {"enabled": self.reader_enabled, "phase": "reset"},
        }

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "model": str(self.model_path),
            "writer": "Direct-44 clean raw factorial three-layer only",
            "writer_path": str(self.writer_path),
            "default_system_prompt": self.system_prompt,
            "weights": self.weights[0].tolist(),
            "axis_info": [
                {
                    "name": axis,
                    "label": AXIS_LABELS.get(axis, axis),
                    "category": self.categories[index],
                    "category_label": CATEGORY_LABELS.get(self.categories[index], self.categories[index]),
                    "layers": self.layers_by_axis[index].tolist(),
                }
                for index, axis in enumerate(self.axes)
            ],
            "generation_limits": {"direct": self.max_tokens, "thinking": self.thinking_max_tokens},
            "mix_protection": {
                "default": {"mode": "raw", "headroom": DEFAULT_MIX_HEADROOM},
                "modes": {
                    "raw": "Raw vector addition",
                    "rms_cap": "Shared-layer RMS cap",
                },
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
                    if self.reader_scores
                    else None
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
                "experimental_probes": self.experimental_probes.status(),
            },
            "load_seconds": round(self.load_seconds, 3),
            "mlx_memory": {
                "limit_bytes": int(mx.get_memory_limit()),
                "active_bytes": int(mx.get_active_memory()),
                "cache_bytes": int(mx.get_cache_memory()),
                "peak_bytes": int(mx.get_peak_memory()),
            },
        }



class Handler(SimpleHTTPRequestHandler):
    engine: Direct44Engine

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self.send_json(200, self.engine.status())
            return
        if self.path in {"/", "/index.html"}:
            raw = PAGE_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self.body()
            if self.path == "/api/reset":
                self.send_json(200, self.engine.reset(payload.get("session_id")))
                return
            if self.path == "/api/observe":
                self.send_json(
                    200,
                    self.engine.observe(
                        payload.get("message"),
                        payload.get("axes", {}),
                        thinking=bool(payload.get("thinking")),
                        scope=payload.get("modulation_scope"),
                        mix_protection=payload.get("mix_protection"),
                        system_prompt=payload.get("system_prompt"),
                    ),
                )
                return
            if self.path == "/api/chat":
                self.send_json(200, self.engine.chat(payload.get("message"), payload.get("axes", {}), thinking=bool(payload.get("thinking")), session_id=payload.get("session_id"), scope=payload.get("modulation_scope"), mix_protection=payload.get("mix_protection"), system_prompt=payload.get("system_prompt")))
                return
            if self.path == "/api/chat/stream":
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                for event in self.engine.stream_chat(payload.get("message"), payload.get("axes", {}), thinking=bool(payload.get("thinking")), session_id=payload.get("session_id"), scope=payload.get("modulation_scope"), mix_protection=payload.get("mix_protection"), system_prompt=payload.get("system_prompt")):
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                return
            self.send_json(404, {"error": "Endpoint not found"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            self.send_json(400 if isinstance(error, ValueError) else 500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--writer", type=Path, default=WRITER_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking-max-tokens", type=int, default=1024)
    parser.add_argument("--reader", type=Path, default=READER_PATH)
    args = parser.parse_args()
    memory_limit_gb = float(os.environ.get("NEURAL_STATE_OS_MLX_MEMORY_LIMIT_GB", "0"))
    cache_limit_mb = float(os.environ.get("NEURAL_STATE_OS_MLX_CACHE_LIMIT_MB", "0"))
    if memory_limit_gb > 0:
        mx.set_memory_limit(int(memory_limit_gb * 1024**3))
    if cache_limit_mb > 0:
        mx.set_cache_limit(int(cache_limit_mb * 1024**2))
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.writer.exists():
        raise FileNotFoundError(args.writer)
    print(f"Loading model: {args.model}", flush=True)
    print(
        "MLX memory policy: "
        f"limit={mx.get_memory_limit() / 1024**3:.2f} GiB, "
        f"cache={cache_limit_mb:.0f} MiB",
        flush=True,
    )
    engine = Direct44Engine(
        args.model,
        args.writer,
        args.max_tokens,
        args.thinking_max_tokens,
        SYSTEM_PROMPT,
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
