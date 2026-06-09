from __future__ import annotations

import math
import json
import difflib
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from viitorvoice import paths

from viitorvoice.llm.model import ViiTorVoiceLLMModel
from viitorvoice.llm.runtime import DEFAULT_CHECKPOINT_DIR, clear_proxies
from viitorvoice.llm.text_utils import (
    NO_REF_TEXT_SPECIAL_TOKENS,
    RuleDurationEstimator,
    add_punctuation,
    chunk_text_punctuation,
    combine_text,
    estimate_ref_text_mask_len,
    extract_leading_emotion_tag,
    has_leading_emotion_tag,
    has_nvv_tag,
    insert_text_pause_anchors,
    make_mask_text,
    normalize_nvv_tags,
    prepare_text_for_tokenizer,
    resolve_language,
    strip_leading_emotion_tag,
    strip_nvv_tags,
    tokenize_with_special_tokens,
    wrap_prompt_target_text,
)
from viitorvoice.llm.voice_design import (
    _INSTRUCT_ALL_VALID,
    _INSTRUCT_EN_TO_ZH,
    _INSTRUCT_MUTUALLY_EXCLUSIVE,
    _INSTRUCT_VALID_EN,
    _INSTRUCT_VALID_ZH,
    _INSTRUCT_ZH_TO_EN,
    _ZH_RE,
)

@dataclass
class LLMGenerationConfig:
    num_step: int = 32
    guidance_scale: float = 0.0
    emotion_guidance_scale: float = 0.0
    nvv_guidance_scale: float = 0.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = False
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> "LLMGenerationConfig":
        valid_keys = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in valid_keys})


@dataclass
class GenerationTask:
    batch_size: int
    texts: list[str]
    target_lens: list[int]
    langs: list[Optional[str]]
    instructs: list[Optional[str]]
    ref_texts: list[Optional[str]]
    ref_audio_tokens: list[Optional[torch.Tensor]]
    ref_text_mask_lens: Optional[list[Optional[int]]] = None
    no_ref_silence_prefix_lens: Optional[list[int]] = None
    speed: Optional[list[float]] = None

    def get_indices(self, config: LLMGenerationConfig, frame_rate: int) -> tuple[list[int], list[int]]:
        threshold = int(config.audio_chunk_threshold * frame_rate)
        short_idx = [i for i, length in enumerate(self.target_lens) if length <= threshold]
        long_idx = [i for i, length in enumerate(self.target_lens) if length > threshold]
        return short_idx, long_idx

    def slice_task(self, indices: list[int]) -> Optional["GenerationTask"]:
        if not indices:
            return None
        return GenerationTask(
            batch_size=len(indices),
            texts=[self.texts[i] for i in indices],
            target_lens=[self.target_lens[i] for i in indices],
            langs=[self.langs[i] for i in indices],
            instructs=[self.instructs[i] for i in indices],
            ref_texts=[self.ref_texts[i] for i in indices],
            ref_audio_tokens=[self.ref_audio_tokens[i] for i in indices],
            ref_text_mask_lens=(
                [self.ref_text_mask_lens[i] for i in indices]
                if self.ref_text_mask_lens
                else None
            ),
            no_ref_silence_prefix_lens=(
                [self.no_ref_silence_prefix_lens[i] for i in indices]
                if self.no_ref_silence_prefix_lens
                else None
            ),
            speed=[self.speed[i] for i in indices] if self.speed else None,
        )


GeneratedItem = Union[torch.Tensor, list[torch.Tensor]]


class ViiTorVoiceLLMGenerator:
    """Standalone LLM token generator for ViiTorVoice audio codebooks."""

    def __init__(
        self,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        onnx_path: str | Path | None = None,
        backend: str = "trt",
        precision: str = "fp16",
        device_id: int = 0,
        trt_cache_root: str | Path | None = None,
        batch_min: int = 1,
        batch_opt: int = 2,
        batch_max: int = 4,
        seq_min: int = 16,
        seq_opt: int = 512,
        seq_max: int = 2048,
        strict_trt: bool = False,
    ) -> None:
        clear_proxies()
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.model = ViiTorVoiceLLMModel(
            checkpoint_dir=self.checkpoint_dir,
            onnx_path=onnx_path,
            backend=backend,
            precision=precision,
            device_id=device_id,
            trt_cache_root=trt_cache_root,
            batch_min=batch_min,
            batch_opt=batch_opt,
            batch_max=batch_max,
            seq_min=seq_min,
            seq_opt=seq_opt,
            seq_max=seq_max,
            strict_trt=strict_trt,
        )
        self.config = self.model.config
        self.device = self.model.device
        self.tokenizer = self._load_tokenizer(self.checkpoint_dir)
        self.duration_estimator = RuleDurationEstimator()
        self.frame_rate = int(self.config.get("audio_frame_rate", 25))
        self.num_audio_codebook = int(self.config["num_audio_codebook"])
        self.audio_mask_ids = list(self.config["audio_mask_ids"])
        self.audio_pause_ids = self.config.get("audio_pause_ids")
        self.audio_separator_ids = self.config.get("audio_separator_ids")
        self.audio_valid_vocab_sizes = list(self.config["audio_valid_vocab_sizes"])
        self.active_providers = self.model.active_providers
        self._no_ref_text_silence_codec_cache: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def generate(
        self,
        text: str | list[str],
        language: str | list[str] | None = None,
        ref_text: str | list[str] | None = None,
        ref_audio_tokens: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray] | None = None,
        instruct: str | list[str] | None = None,
        duration: float | list[Optional[float]] | None = None,
        speed: float | list[Optional[float]] | None = None,
        allow_missing_ref_text: bool = False,
        ref_text_mask_len: int | list[Optional[int]] | None = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        **kwargs: Any,
    ) -> list[GeneratedItem]:
        gen_config = generation_config or LLMGenerationConfig.from_dict(kwargs)
        task = self._preprocess_all(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_audio_tokens=ref_audio_tokens,
            instruct=instruct,
            speed=speed,
            duration=duration,
            preprocess_prompt=gen_config.preprocess_prompt,
            allow_missing_ref_text=allow_missing_ref_text,
            ref_text_mask_len=ref_text_mask_len,
        )
        short_idx, long_idx = task.get_indices(gen_config, self.frame_rate)
        results: list[Optional[GeneratedItem]] = [None] * task.batch_size
        if short_idx:
            short_task = task.slice_task(short_idx)
            assert short_task is not None
            short_results = self._generate_iterative(short_task, gen_config)
            for idx, item in zip(short_idx, short_results):
                results[idx] = item
        if long_idx:
            long_task = task.slice_task(long_idx)
            assert long_task is not None
            long_results = self._generate_chunked(long_task, gen_config)
            for idx, item in zip(long_idx, long_results):
                results[idx] = item
        return [self._require_result(item, i) for i, item in enumerate(results)]

    generate_tokens = generate

    @torch.inference_mode()
    def generate_from_semantic_tokens(
        self,
        *,
        text: str,
        semantic_tokens: torch.Tensor | np.ndarray | Sequence[int],
        ref_audio_tokens: torch.Tensor | np.ndarray,
        language: str | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        allow_missing_ref_text: bool = True,
        ref_text_mask_len: int | None = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> torch.Tensor:
        """Generate full DualCodec tokens from fixed target semantic tokens.

        The returned tensor has shape ``[C, T]``. Layer 0 is copied from
        ``semantic_tokens`` and layers 1..C-1 are generated by the LLM.
        """
        if not text or not text.strip():
            raise ValueError("text is required for semantic-token generation.")
        ref_tokens = self._normalize_single_ref_tokens(ref_audio_tokens, 0)
        assert ref_tokens is not None
        if ref_text is None and not allow_missing_ref_text:
            raise ValueError(
                "ref_text is required when allow_missing_ref_text=False."
            )

        clean_language = resolve_language(language)
        clean_instruct = None
        if instruct is not None:
            use_zh = bool(text and _ZH_RE.search(text))
            clean_instruct = _resolve_instruct(instruct, use_zh=use_zh)
        clean_text = self._condition_text_for_model(normalize_nvv_tags(text).strip())
        clean_ref_text = normalize_nvv_tags(ref_text).strip() if ref_text else None
        if clean_ref_text:
            clean_ref_text = add_punctuation(clean_ref_text)

        target_tokens, editable_mask = self._make_semantic_conditioned_target(
            semantic_tokens
        )
        gen_config = generation_config or LLMGenerationConfig()
        generated = self._generate_edit_iterative_single(
            full_text=clean_text,
            target_audio_tokens=target_tokens,
            editable_audio_mask=editable_mask,
            gen_config=gen_config,
            ref_text=clean_ref_text,
            ref_audio_tokens=ref_tokens,
            lang=clean_language,
            instruct=clean_instruct,
            ref_text_mask_len=ref_text_mask_len,
        )

        semantic = self._normalize_semantic_tokens(semantic_tokens)
        if not torch.equal(generated[0].to(semantic.device), semantic):
            raise RuntimeError("Generated semantic layer changed unexpectedly.")
        return generated.detach().cpu()

    @torch.inference_mode()
    def generate_edit(
        self,
        *,
        full_text: str,
        prefix_audio_tokens: torch.Tensor | np.ndarray,
        suffix_audio_tokens: torch.Tensor | np.ndarray,
        replacement_frames: int,
        language: str | None = None,
        ref_text: str | None = None,
        ref_audio_tokens: torch.Tensor | np.ndarray | None = None,
        instruct: str | None = None,
        allow_missing_ref_text: bool = False,
        ref_text_mask_len: int | None = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        edit_context_frames: int | None = None,
    ) -> torch.Tensor:
        """Regenerate only a masked audio-token span and preserve surrounding tokens.

        ``prefix_audio_tokens`` and ``suffix_audio_tokens`` are the unedited tokens
        before and after the selected span. The returned tensor is the complete
        edited token stream shaped ``[C, T_new]``.
        """
        if not full_text or not full_text.strip():
            raise ValueError("full_text is required for local edit generation.")
        replacement_frames = int(replacement_frames)
        if replacement_frames <= 0:
            raise ValueError(f"replacement_frames should be positive, got {replacement_frames}.")

        gen_config = generation_config or LLMGenerationConfig()
        prefix = self._normalize_single_ref_tokens(prefix_audio_tokens, 0)
        suffix = self._normalize_single_ref_tokens(suffix_audio_tokens, 1)
        assert prefix is not None and suffix is not None
        ref_tokens = self._normalize_single_ref_tokens(ref_audio_tokens, 2) if ref_audio_tokens is not None else None
        if ref_tokens is not None and ref_text is None and not allow_missing_ref_text:
            raise ValueError(
                "ref_text is required when ref_audio_tokens is provided unless "
                "allow_missing_ref_text=True."
            )

        context = None if edit_context_frames is None else max(0, int(edit_context_frames))
        left_context_frames = prefix.size(-1) if context is None else min(prefix.size(-1), context)
        right_context_frames = suffix.size(-1) if context is None else min(suffix.size(-1), context)
        left_context = prefix[:, prefix.size(-1) - left_context_frames :] if left_context_frames else prefix[:, :0]
        right_context = suffix[:, :right_context_frames] if right_context_frames else suffix[:, :0]

        replacement_tokens = self._make_audio_token_tensor(
            batch_size=1,
            seq_len=replacement_frames,
            device=self.device,
        ).squeeze(0)
        target_audio_tokens = torch.cat(
            [left_context, replacement_tokens, right_context],
            dim=-1,
        )
        editable_audio_mask = torch.zeros(
            target_audio_tokens.size(-1),
            dtype=torch.bool,
            device=self.device,
        )
        editable_audio_mask[
            left_context_frames : left_context_frames + replacement_frames
        ] = True

        generated_local = self._generate_edit_iterative_single(
            full_text=self._condition_text_for_model(normalize_nvv_tags(full_text).strip()),
            target_audio_tokens=target_audio_tokens,
            editable_audio_mask=editable_audio_mask,
            gen_config=gen_config,
            ref_text=normalize_nvv_tags(ref_text).strip() if ref_text else None,
            ref_audio_tokens=ref_tokens,
            lang=resolve_language(language),
            instruct=instruct.strip() if instruct else None,
            ref_text_mask_len=ref_text_mask_len,
        )
        untouched_prefix = prefix[:, : prefix.size(-1) - left_context_frames]
        untouched_suffix = suffix[:, right_context_frames:]
        return torch.cat([untouched_prefix, generated_local, untouched_suffix], dim=-1).detach().cpu()

    @torch.inference_mode()
    def generate_edit_masked(
        self,
        *,
        full_text: str,
        target_audio_tokens: torch.Tensor | np.ndarray,
        editable_audio_mask: torch.Tensor | np.ndarray,
        language: str | None = None,
        ref_text: str | None = None,
        ref_audio_tokens: torch.Tensor | np.ndarray | None = None,
        instruct: str | None = None,
        allow_missing_ref_text: bool = False,
        ref_text_mask_len: int | None = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        edit_context_frames: int | None = None,
    ) -> torch.Tensor:
        """Regenerate arbitrary masked positions inside a target audio-token draft."""
        if not full_text or not full_text.strip():
            raise ValueError("full_text is required for local edit generation.")

        gen_config = generation_config or LLMGenerationConfig()
        target = self._normalize_single_ref_tokens(target_audio_tokens, 0)
        assert target is not None
        target_len = target.size(-1)
        editable_mask = torch.as_tensor(
            editable_audio_mask,
            dtype=torch.bool,
            device=self.device,
        ).contiguous()
        if editable_mask.ndim != 1 or editable_mask.numel() != target_len:
            raise ValueError(
                "editable_audio_mask should have shape [target_len], got "
                f"{tuple(editable_mask.shape)} for target_len={target_len}."
            )
        if int(editable_mask.sum().item()) <= 0:
            raise ValueError("editable_audio_mask must select at least one frame.")

        ref_tokens = self._normalize_single_ref_tokens(ref_audio_tokens, 1) if ref_audio_tokens is not None else None
        if ref_tokens is not None and ref_text is None and not allow_missing_ref_text:
            raise ValueError(
                "ref_text is required when ref_audio_tokens is provided unless "
                "allow_missing_ref_text=True."
            )

        edit_positions = torch.nonzero(editable_mask, as_tuple=False).flatten()
        context = None if edit_context_frames is None else max(0, int(edit_context_frames))
        if context is None:
            window_start = 0
            window_end = target_len
        else:
            window_start = max(0, int(edit_positions[0].item()) - context)
            window_end = min(target_len, int(edit_positions[-1].item()) + context + 1)

        target_window = target[:, window_start:window_end].clone()
        editable_window = editable_mask[window_start:window_end].contiguous()
        mask_ids = self._audio_mask_ids_tensor(self.device).view(self.num_audio_codebook, 1)
        target_window = torch.where(
            editable_window.view(1, -1),
            mask_ids.expand_as(target_window),
            target_window,
        )
        generated_window = self._generate_edit_iterative_single(
            full_text=self._condition_text_for_model(normalize_nvv_tags(full_text).strip()),
            target_audio_tokens=target_window,
            editable_audio_mask=editable_window,
            gen_config=gen_config,
            ref_text=normalize_nvv_tags(ref_text).strip() if ref_text else None,
            ref_audio_tokens=ref_tokens,
            lang=resolve_language(language),
            instruct=instruct.strip() if instruct else None,
            ref_text_mask_len=ref_text_mask_len,
        )
        result = target.clone()
        result[:, window_start:window_end] = generated_window
        return result.detach().cpu()

    @torch.inference_mode()
    def forward_step(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        return self.model.forward_step(input_ids, audio_mask, attention_mask)

    def _preprocess_all(
        self,
        text: str | list[str],
        language: str | list[str] | None,
        ref_text: str | list[str] | None,
        ref_audio_tokens: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray] | None,
        instruct: str | list[str] | None,
        speed: float | list[Optional[float]] | None,
        duration: float | list[Optional[float]] | None,
        allow_missing_ref_text: bool,
        ref_text_mask_len: int | list[Optional[int]] | None,
        preprocess_prompt: bool = True,
    ) -> GenerationTask:
        text_list = [text] if isinstance(text, str) else list(text)
        text_list = [self._condition_text_for_model(normalize_nvv_tags(item)) for item in text_list]
        batch_size = len(text_list)
        language_list = [resolve_language(x) for x in self._ensure_list(language, batch_size)]
        instruct_list = self._ensure_list(instruct, batch_size)
        for i, item in enumerate(instruct_list):
            if item is None:
                continue
            use_zh = bool(text_list[i] and _ZH_RE.search(text_list[i]))
            instruct_list[i] = _resolve_instruct(item, use_zh=use_zh)
        ref_text_list = [
            normalize_nvv_tags(item) if item is not None else None
            for item in self._ensure_list(ref_text, batch_size)
        ]
        ref_text_mask_lens = self._ensure_list(ref_text_mask_len, batch_size)
        ref_audio_list = self._normalize_ref_audio_tokens(ref_audio_tokens, batch_size)
        for i, tokens in enumerate(ref_audio_list):
            if tokens is not None and ref_text_list[i] is not None and preprocess_prompt:
                ref_text_list[i] = add_punctuation(ref_text_list[i])
        for i, tokens in enumerate(ref_audio_list):
            if tokens is not None and ref_text_list[i] is None and not allow_missing_ref_text:
                raise ValueError(
                    "ref_text is required when ref_audio_tokens is provided unless "
                    "allow_missing_ref_text=True."
                )

        speeds = None
        if speed is not None:
            speeds = [float(x) if x is not None else 1.0 for x in self._ensure_list(speed, batch_size)]
        durations = None
        if duration is not None:
            durations = [
                float(x) if x is not None else None
                for x in self._ensure_list(duration, batch_size)
            ]

        target_lens: list[int] = []
        speed_list: Optional[list[float]] = None
        for i in range(batch_size):
            has_duration = durations is not None and durations[i] is not None
            item_speed = 1.0 if has_duration else (speeds[i] if speeds else 1.0)
            estimated = self._estimate_target_tokens(
                text_list[i],
                ref_text_list[i],
                ref_audio_list[i].size(-1) if ref_audio_list[i] is not None else None,
                speed=item_speed,
            )
            if has_duration:
                target_lens.append(max(1, int(durations[i] * self.frame_rate)))
            else:
                target_lens.append(estimated)

        no_ref_silence_prefix_lens: list[int] = []
        silence_prefix_seconds = float(
            self.config.get("no_ref_text_silence_prefix_seconds", 1.0)
        )
        for i in range(batch_size):
            use_silence_prefix = (
                silence_prefix_seconds > 0.0
                and ref_text_list[i] is None
                and ref_audio_list[i] is not None
            )
            prefix_len = (
                int(round(silence_prefix_seconds * self.frame_rate))
                if use_silence_prefix
                else 0
            )
            no_ref_silence_prefix_lens.append(prefix_len)

        if durations is not None:
            speed_list = []
            for i, duration_value in enumerate(durations):
                if duration_value is None:
                    speed_list.append(speeds[i] if speeds else 1.0)
                else:
                    total_tokens = max(1, int(duration_value * self.frame_rate))
                    target_lens[i] = max(
                        1,
                        total_tokens - no_ref_silence_prefix_lens[i],
                    )
                    raw_estimate = self._estimate_target_tokens(
                        text_list[i],
                        ref_text_list[i],
                        ref_audio_list[i].size(-1) if ref_audio_list[i] is not None else None,
                        speed=1.0,
                    )
                    speed_list.append(raw_estimate / target_lens[i])
        elif speeds is not None:
            speed_list = speeds

        return GenerationTask(
            batch_size=batch_size,
            texts=text_list,
            target_lens=target_lens,
            langs=language_list,
            instructs=instruct_list,
            ref_texts=ref_text_list,
            ref_audio_tokens=ref_audio_list,
            ref_text_mask_lens=ref_text_mask_lens,
            no_ref_silence_prefix_lens=no_ref_silence_prefix_lens,
            speed=speed_list,
        )

    def describe_text_preparation(
        self,
        *,
        text: str,
        language: str | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        full_text_field: str = "text",
    ) -> dict[str, Any]:
        del instruct
        gen_config = generation_config or LLMGenerationConfig()
        lang = resolve_language(language)
        normalized_text = normalize_nvv_tags(text).strip()
        model_text = self._condition_text_for_model(normalized_text)
        emotion_tag = extract_leading_emotion_tag(normalized_text)
        prepared_target = prepare_text_for_tokenizer(model_text, self.tokenizer, lang)
        normalized_ref_text = normalize_nvv_tags(ref_text).strip() if ref_text else ""
        prepared_ref = prepare_text_for_tokenizer(normalized_ref_text, self.tokenizer, lang) if normalized_ref_text else ""
        use_emotion_cfg = gen_config.emotion_guidance_scale != 0.0 and has_leading_emotion_tag(model_text)
        use_nvv_cfg = gen_config.nvv_guidance_scale != 0.0 and has_nvv_tag(model_text)
        debug = {
            full_text_field: text,
            "normalized_text": normalized_text,
            "model_text": model_text,
            "tokenizer_target_text": prepared_target,
            "ref_text": ref_text or "",
            "normalized_ref_text": normalized_ref_text,
            "tokenizer_ref_text": prepared_ref,
            "language": lang,
            "leading_emotion_tag": emotion_tag or "",
            "leading_emotion_tag_in_tokenizer_vocab": self._tokenizer_has_token(emotion_tag),
            "stripped_unsupported_emotion_tag": bool(emotion_tag and normalized_text != model_text),
            "branches": {
                "full": model_text,
                "uncond": gen_config.guidance_scale != 0.0,
                "no_emotion": strip_leading_emotion_tag(model_text) if use_emotion_cfg else "",
                "no_nvv": strip_nvv_tags(model_text) if use_nvv_cfg else "",
            },
            "effective_guidance": {
                "cfg_scale": gen_config.guidance_scale if gen_config.guidance_scale != 0.0 else 0.0,
                "emotion_guidance_scale": gen_config.emotion_guidance_scale if use_emotion_cfg else 0.0,
                "nvv_guidance_scale": gen_config.nvv_guidance_scale if use_nvv_cfg else 0.0,
            },
        }
        return debug

    def _condition_text_for_model(self, text: str) -> str:
        if not text:
            return text
        emotion_tag = extract_leading_emotion_tag(text)
        if emotion_tag and not self._tokenizer_has_token(emotion_tag):
            return strip_leading_emotion_tag(text)
        return text.strip()

    def _tokenizer_has_token(self, token: str | None) -> bool:
        if not token:
            return False
        return token in self.tokenizer.get_vocab()

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        denoise: bool = False,
        ref_text_mask_len: Optional[int] = None,
        no_ref_silence_prefix_len: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        del denoise
        style_text = ""
        style_text += f"<|lang_start|>{lang if lang else 'None'}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct if instruct else 'None'}<|instruct_end|>"
        style_tokens = (
            tokenize_with_special_tokens(style_text, self.tokenizer)
            .repeat(self.num_audio_codebook, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        if ref_audio_tokens is not None and (
            ref_text is None or self.audio_separator_ids is not None
        ):
            self._ensure_no_ref_text_tokens_available()
            if ref_text is None:
                if ref_text_mask_len is None:
                    ref_text_mask_len = estimate_ref_text_mask_len(
                        language_id=lang,
                        prompt_audio_frames=ref_audio_tokens.size(-1),
                        audio_frame_rate=self.frame_rate,
                        tokens_per_second=dict(
                            self.config.get("no_ref_text_tokens_per_second", {})
                        ),
                        min_tokens=int(self.config.get("no_ref_text_min_mask_tokens", 1)),
                        max_tokens=int(self.config.get("no_ref_text_max_mask_tokens", 256)),
                        jitter_ratio=0.0,
                        jitter=False,
                        default_tokens_per_second=float(
                            self.config.get("no_ref_text_default_tokens_per_second", 4.0)
                        ),
                    )
                prompt_text = make_mask_text(ref_text_mask_len)
            else:
                prompt_text = prepare_text_for_tokenizer(ref_text, self.tokenizer, lang)
                if self.audio_pause_ids is not None:
                    prompt_text = insert_text_pause_anchors(prompt_text)
            target_text = prepare_text_for_tokenizer(text, self.tokenizer, lang)
            if self.audio_pause_ids is not None:
                target_text = insert_text_pause_anchors(target_text)
            full_text = wrap_prompt_target_text(prompt_text, target_text)
        else:
            full_text = combine_text(text=text, ref_text=ref_text)
            full_text = prepare_text_for_tokenizer(full_text, self.tokenizer, lang)
            if self.audio_pause_ids is not None:
                full_text = insert_text_pause_anchors(full_text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            tokenize_with_special_tokens(wrapped_text, self.tokenizer)
            .repeat(self.num_audio_codebook, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        target_audio_tokens = self._make_audio_token_tensor(1, num_target_tokens, self.device)
        no_ref_silence_prefix_tokens = self._get_no_ref_text_silence_prefix_tokens(
            no_ref_silence_prefix_len
        )
        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
            if self.audio_separator_ids is not None:
                parts.append(self._make_audio_separator_tensor(self.device))
        if no_ref_silence_prefix_tokens is not None:
            parts.append(no_ref_silence_prefix_tokens.unsqueeze(0).to(self.device))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)
            if self.audio_separator_ids is not None:
                cond_audio_start_idx -= 1
        if no_ref_silence_prefix_tokens is not None:
            cond_audio_start_idx -= no_ref_silence_prefix_tokens.size(-1)
        cond_audio_mask = torch.zeros(
            1,
            cond_total_length,
            dtype=torch.bool,
            device=self.device,
        )
        cond_audio_mask[0, cond_audio_start_idx:] = True
        return {
            "input_ids": cond_input_ids,
            "audio_mask": cond_audio_mask,
            "target_prefix_tokens": no_ref_silence_prefix_tokens,
        }

    def _prepare_edit_inference_inputs(
        self,
        full_text: str,
        target_audio_tokens: torch.Tensor,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_text_mask_len: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        style_text = ""
        style_text += f"<|lang_start|>{lang if lang else 'None'}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct if instruct else 'None'}<|instruct_end|>"
        style_tokens = (
            tokenize_with_special_tokens(style_text, self.tokenizer)
            .repeat(self.num_audio_codebook, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        if ref_audio_tokens is not None and (
            ref_text is None or self.audio_separator_ids is not None
        ):
            self._ensure_no_ref_text_tokens_available()
            if ref_text is None:
                if ref_text_mask_len is None:
                    ref_text_mask_len = estimate_ref_text_mask_len(
                        language_id=lang,
                        prompt_audio_frames=ref_audio_tokens.size(-1),
                        audio_frame_rate=self.frame_rate,
                        tokens_per_second=dict(
                            self.config.get("no_ref_text_tokens_per_second", {})
                        ),
                        min_tokens=int(self.config.get("no_ref_text_min_mask_tokens", 1)),
                        max_tokens=int(self.config.get("no_ref_text_max_mask_tokens", 256)),
                        jitter_ratio=0.0,
                        jitter=False,
                        default_tokens_per_second=float(
                            self.config.get("no_ref_text_default_tokens_per_second", 4.0)
                        ),
                    )
                prompt_text = make_mask_text(ref_text_mask_len)
            else:
                prompt_text = prepare_text_for_tokenizer(ref_text, self.tokenizer, lang)
                if self.audio_pause_ids is not None:
                    prompt_text = insert_text_pause_anchors(prompt_text)
            target_text = prepare_text_for_tokenizer(full_text, self.tokenizer, lang)
            if self.audio_pause_ids is not None:
                target_text = insert_text_pause_anchors(target_text)
            conditioned_text = wrap_prompt_target_text(prompt_text, target_text)
        else:
            conditioned_text = combine_text(text=full_text, ref_text=ref_text)
            conditioned_text = prepare_text_for_tokenizer(
                conditioned_text,
                self.tokenizer,
                lang,
            )
            if self.audio_pause_ids is not None:
                conditioned_text = insert_text_pause_anchors(conditioned_text)

        wrapped_text = f"<|text_start|>{conditioned_text}<|text_end|>"
        text_tokens = (
            tokenize_with_special_tokens(wrapped_text, self.tokenizer)
            .repeat(self.num_audio_codebook, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
            if self.audio_separator_ids is not None:
                parts.append(self._make_audio_separator_tensor(self.device))
        parts.append(target_audio_tokens.unsqueeze(0).to(self.device))
        cond_input_ids = torch.cat(parts, dim=2)

        target_len = target_audio_tokens.size(-1)
        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - target_len
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)
            if self.audio_separator_ids is not None:
                cond_audio_start_idx -= 1
        cond_audio_mask = torch.zeros(
            1,
            cond_total_length,
            dtype=torch.bool,
            device=self.device,
        )
        cond_audio_mask[0, cond_audio_start_idx:] = True
        return {
            "input_ids": cond_input_ids,
            "audio_mask": cond_audio_mask,
        }

    def _generate_chunked(
        self,
        task: GenerationTask,
        gen_config: LLMGenerationConfig,
    ) -> list[list[torch.Tensor]]:
        all_chunks: list[list[str]] = []
        for i in range(task.batch_size):
            avg_tokens_per_char = task.target_lens[i] / max(1, len(task.texts[i]))
            text_chunk_len = max(
                1,
                int(gen_config.audio_chunk_duration * self.frame_rate / avg_tokens_per_char),
            )
            all_chunks.append(
                chunk_text_punctuation(
                    text=task.texts[i],
                    chunk_len=text_chunk_len,
                    min_chunk_len=3,
                )
            )

        has_ref = [tokens is not None for tokens in task.ref_audio_tokens]
        if any(has_ref) and not all(has_ref):
            raise ValueError(
                "Chunked generation requires all items to either have or not have ref_audio_tokens."
            )

        max_num_chunks = max(len(chunks) for chunks in all_chunks)
        chunk_results: list[list[torch.Tensor]] = [[] for _ in range(task.batch_size)]

        def run_batch(
            indices: list[int],
            texts: list[str],
            ref_audios: list[Optional[torch.Tensor]],
            ref_texts: list[Optional[str]],
            prefix_lens: Optional[list[int]] = None,
        ) -> None:
            target_lens = [
                self._estimate_target_tokens(
                    texts[j],
                    ref_texts[j],
                    ref_audios[j].size(-1) if ref_audios[j] is not None else None,
                    speed=task.speed[i] if task.speed else 1.0,
                )
                for j, i in enumerate(indices)
            ]
            sub_task = GenerationTask(
                batch_size=len(indices),
                texts=texts,
                target_lens=target_lens,
                langs=[task.langs[i] for i in indices],
                instructs=[task.instructs[i] for i in indices],
                ref_texts=ref_texts,
                ref_audio_tokens=ref_audios,
                ref_text_mask_lens=(
                    [task.ref_text_mask_lens[i] for i in indices]
                    if task.ref_text_mask_lens
                    else None
                ),
                no_ref_silence_prefix_lens=(
                    prefix_lens if prefix_lens is not None else [0] * len(indices)
                ),
                speed=[task.speed[i] for i in indices] if task.speed else None,
            )
            generated = self._generate_iterative(sub_task, gen_config)
            for j, idx in enumerate(indices):
                chunk_results[idx].append(generated[j])

        if all(has_ref):
            for chunk_idx in range(max_num_chunks):
                indices = [i for i in range(task.batch_size) if chunk_idx < len(all_chunks[i])]
                if indices:
                    run_batch(
                        indices,
                        texts=[all_chunks[i][chunk_idx] for i in indices],
                        ref_audios=[task.ref_audio_tokens[i] for i in indices],
                        ref_texts=[task.ref_texts[i] for i in indices],
                        prefix_lens=(
                            [task.no_ref_silence_prefix_lens[i] for i in indices]
                            if chunk_idx == 0 and task.no_ref_silence_prefix_lens
                            else [0] * len(indices)
                        ),
                    )
        else:
            indices_0 = [i for i in range(task.batch_size) if all_chunks[i]]
            run_batch(
                indices_0,
                texts=[all_chunks[i][0] for i in indices_0],
                ref_audios=[None] * len(indices_0),
                ref_texts=[None] * len(indices_0),
            )
            first_chunk_map = {idx: chunk_results[idx][0] for idx in indices_0}
            for chunk_idx in range(1, max_num_chunks):
                indices = [i for i in range(task.batch_size) if chunk_idx < len(all_chunks[i])]
                if indices:
                    run_batch(
                        indices,
                        texts=[all_chunks[i][chunk_idx] for i in indices],
                        ref_audios=[first_chunk_map[i] for i in indices],
                        ref_texts=[all_chunks[i][0] for i in indices],
                    )
        return chunk_results

    def _generate_iterative(
        self,
        task: GenerationTask,
        gen_config: LLMGenerationConfig,
    ) -> list[torch.Tensor]:
        batch_size = task.batch_size

        def prepare_item(i: int, text: str) -> dict[str, torch.Tensor]:
            return self._prepare_inference_inputs(
                text=text,
                num_target_tokens=task.target_lens[i],
                ref_text=task.ref_texts[i],
                ref_audio_tokens=task.ref_audio_tokens[i],
                lang=task.langs[i],
                instruct=task.instructs[i],
                denoise=gen_config.denoise,
                ref_text_mask_len=task.ref_text_mask_lens[i] if task.ref_text_mask_lens else None,
                no_ref_silence_prefix_len=(
                    task.no_ref_silence_prefix_lens[i]
                    if task.no_ref_silence_prefix_lens
                    else None
                ),
            )

        inputs_list = [prepare_item(i, task.texts[i]) for i in range(batch_size)]
        target_prefixes = [item.get("target_prefix_tokens") for item in inputs_list]
        use_cfg = gen_config.guidance_scale != 0.0
        use_emotion_cfg = gen_config.emotion_guidance_scale != 0.0
        use_nvv_cfg = gen_config.nvv_guidance_scale != 0.0

        branch_rows: list[dict[str, Any]] = []
        branch_lookup: list[dict[str, int]] = [{} for _ in range(batch_size)]

        def add_branch(
            sample_index: int,
            kind: str,
            item: dict[str, torch.Tensor] | None,
            *,
            cond_len: int,
            effective_scale: float = 0.0,
        ) -> None:
            row_index = len(branch_rows)
            branch_rows.append(
                {
                    "sample_index": sample_index,
                    "kind": kind,
                    "item": item,
                    "cond_len": cond_len,
                    "effective_scale": effective_scale,
                }
            )
            branch_lookup[sample_index][kind] = row_index

        for i, item in enumerate(inputs_list):
            cond_len = item["input_ids"].size(2)
            add_branch(i, "full", item, cond_len=cond_len)
            if use_cfg:
                add_branch(i, "uncond", None, cond_len=task.target_lens[i])
            if use_emotion_cfg and has_leading_emotion_tag(task.texts[i]):
                no_emotion_item = prepare_item(i, strip_leading_emotion_tag(task.texts[i]))
                add_branch(
                    i,
                    "no_emotion",
                    no_emotion_item,
                    cond_len=no_emotion_item["input_ids"].size(2),
                    effective_scale=gen_config.emotion_guidance_scale,
                )
            if use_nvv_cfg and has_nvv_tag(task.texts[i]):
                no_nvv_item = prepare_item(i, strip_nvv_tags(task.texts[i]))
                add_branch(
                    i,
                    "no_nvv",
                    no_nvv_item,
                    cond_len=no_nvv_item["input_ids"].size(2),
                    effective_scale=gen_config.nvv_guidance_scale,
                )

        max_cond_len = max(row["cond_len"] for row in branch_rows)
        batch_input_ids = self._make_audio_token_tensor(
            batch_size=len(branch_rows),
            seq_len=max_cond_len,
            device=self.device,
        )
        batch_audio_mask = torch.zeros(
            (len(branch_rows), max_cond_len),
            dtype=torch.bool,
            device=self.device,
        )
        batch_attention_mask = torch.zeros(
            (len(branch_rows), 1, max_cond_len, max_cond_len),
            dtype=torch.bool,
            device=self.device,
        )

        for row_index, row in enumerate(branch_rows):
            i = int(row["sample_index"])
            target_len = task.target_lens[i]
            if row["kind"] == "uncond":
                item = inputs_list[i]
                batch_input_ids[row_index, :, :target_len] = item["input_ids"][..., -target_len:]
                batch_audio_mask[row_index, :target_len] = item["audio_mask"][..., -target_len:]
                batch_attention_mask[row_index, :, :target_len, :target_len] = True
                if max_cond_len > target_len:
                    pad_diag = torch.arange(target_len, max_cond_len, device=self.device)
                    batch_attention_mask[row_index, :, pad_diag, pad_diag] = True
                continue

            item = row["item"]
            if item is None:
                raise RuntimeError(f"Missing conditional inputs for branch {row['kind']!r}.")
            cond_len = int(row["cond_len"])
            batch_input_ids[row_index, :, :cond_len] = item["input_ids"]
            batch_audio_mask[row_index, :cond_len] = item["audio_mask"]
            batch_attention_mask[row_index, :, :cond_len, :cond_len] = True

        tokens = self._make_audio_token_tensor(
            batch_size=batch_size,
            seq_len=max(task.target_lens),
            device=self.device,
        )
        schedules = self._build_schedules(task.target_lens, gen_config)
        layer_ids = torch.arange(self.num_audio_codebook, device=self.device).view(1, -1, 1)

        def get_branch_logits(
            semantic_logits: torch.Tensor,
            acoustic_logits: torch.Tensor,
            sample_index: int,
            kind: str,
            target_len: int,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            row_index = branch_lookup[sample_index].get(kind)
            if row_index is None:
                return None, None
            row = branch_rows[row_index]
            if kind == "uncond":
                return (
                    semantic_logits[row_index : row_index + 1, :target_len, :],
                    acoustic_logits[row_index : row_index + 1, :, :target_len, :],
                )
            cond_len = int(row["cond_len"])
            return (
                semantic_logits[row_index : row_index + 1, cond_len - target_len : cond_len, :],
                acoustic_logits[row_index : row_index + 1, :, cond_len - target_len : cond_len, :],
            )

        for step in range(gen_config.num_step):
            outputs = self.model.forward_step(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            )
            semantic_logits = outputs.semantic_logits.to(torch.float32)
            acoustic_logits = outputs.acoustic_logits.to(torch.float32)

            for i in range(batch_size):
                k = schedules[i][step]
                if k <= 0:
                    continue
                target_len = task.target_lens[i]
                c_semantic_logits, c_acoustic_logits = get_branch_logits(
                    semantic_logits,
                    acoustic_logits,
                    i,
                    "full",
                    target_len,
                )
                u_semantic_logits, u_acoustic_logits = get_branch_logits(
                    semantic_logits,
                    acoustic_logits,
                    i,
                    "uncond",
                    target_len,
                )
                ne_semantic_logits, ne_acoustic_logits = get_branch_logits(
                    semantic_logits,
                    acoustic_logits,
                    i,
                    "no_emotion",
                    target_len,
                )
                nn_semantic_logits, nn_acoustic_logits = get_branch_logits(
                    semantic_logits,
                    acoustic_logits,
                    i,
                    "no_nvv",
                    target_len,
                )
                if c_semantic_logits is None or c_acoustic_logits is None:
                    raise RuntimeError("Missing full conditional logits.")
                emotion_scale = (
                    branch_rows[branch_lookup[i]["no_emotion"]]["effective_scale"]
                    if "no_emotion" in branch_lookup[i]
                    else 0.0
                )
                nvv_scale = (
                    branch_rows[branch_lookup[i]["no_nvv"]]["effective_scale"]
                    if "no_nvv" in branch_lookup[i]
                    else 0.0
                )
                pred_tokens, scores = self._predict_tokens_with_scoring(
                    c_semantic_logits,
                    u_semantic_logits,
                    c_acoustic_logits,
                    u_acoustic_logits,
                    gen_config,
                    ne_semantic_logits=ne_semantic_logits,
                    ne_acoustic_logits=ne_acoustic_logits,
                    nn_semantic_logits=nn_semantic_logits,
                    nn_acoustic_logits=nn_acoustic_logits,
                    emotion_guidance_scale=float(emotion_scale),
                    nvv_guidance_scale=float(nvv_scale),
                )
                scores = scores - (layer_ids * gen_config.layer_penalty_factor)
                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, gen_config.position_temperature)

                sample_tokens = tokens[i : i + 1, :, :target_len]
                mask_ids = self._audio_mask_ids_tensor(sample_tokens.device).view(1, -1, 1)
                scores.masked_fill_(sample_tokens != mask_ids, -float("inf"))
                _, topk_idx = torch.topk(scores.flatten(), k)
                sample_tokens.copy_(
                    self._apply_selected_token_updates(sample_tokens, pred_tokens, topk_idx)
                )
                tokens[i : i + 1, :, :target_len] = sample_tokens

                for row_index in branch_lookup[i].values():
                    row = branch_rows[row_index]
                    if row["kind"] == "uncond":
                        batch_input_ids[row_index : row_index + 1, :, :target_len] = sample_tokens
                    else:
                        cond_len = int(row["cond_len"])
                        batch_input_ids[
                            row_index : row_index + 1,
                            :,
                            cond_len - target_len : cond_len,
                        ] = sample_tokens

        results = []
        for i in range(batch_size):
            generated = tokens[i, :, : task.target_lens[i]]
            prefix = target_prefixes[i]
            if prefix is not None and prefix.numel() > 0:
                generated = torch.cat([prefix.to(generated.device), generated], dim=-1)
            results.append(generated.detach().cpu())
        return results

    def _generate_edit_iterative_single(
        self,
        *,
        full_text: str,
        target_audio_tokens: torch.Tensor,
        editable_audio_mask: torch.Tensor,
        gen_config: LLMGenerationConfig,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_text_mask_len: Optional[int] = None,
    ) -> torch.Tensor:
        target_audio_tokens = target_audio_tokens.to(self.device, dtype=torch.long).contiguous()
        editable_audio_mask = editable_audio_mask.to(self.device, dtype=torch.bool).contiguous()
        target_len = target_audio_tokens.size(-1)
        if editable_audio_mask.ndim != 1 or editable_audio_mask.numel() != target_len:
            raise ValueError(
                "editable_audio_mask should have shape [target_len], got "
                f"{tuple(editable_audio_mask.shape)} for target_len={target_len}."
            )
        editable_frames = int(editable_audio_mask.sum().item())
        if editable_frames <= 0:
            raise ValueError("editable_audio_mask must select at least one frame.")

        def prepare_item(text: str) -> dict[str, torch.Tensor]:
            return self._prepare_edit_inference_inputs(
                full_text=text,
                target_audio_tokens=target_audio_tokens,
                ref_text=ref_text,
                ref_audio_tokens=ref_audio_tokens,
                lang=lang,
                instruct=instruct,
                ref_text_mask_len=ref_text_mask_len,
            )

        item = prepare_item(full_text)
        use_cfg = gen_config.guidance_scale != 0.0
        use_emotion_cfg = gen_config.emotion_guidance_scale != 0.0 and has_leading_emotion_tag(full_text)
        use_nvv_cfg = gen_config.nvv_guidance_scale != 0.0 and has_nvv_tag(full_text)

        branch_rows: list[dict[str, Any]] = []
        branch_lookup: dict[str, int] = {}

        def add_branch(
            kind: str,
            item_value: dict[str, torch.Tensor] | None,
            *,
            cond_len: int,
            effective_scale: float = 0.0,
        ) -> None:
            row_index = len(branch_rows)
            branch_rows.append(
                {
                    "kind": kind,
                    "item": item_value,
                    "cond_len": cond_len,
                    "effective_scale": effective_scale,
                }
            )
            branch_lookup[kind] = row_index

        add_branch("full", item, cond_len=item["input_ids"].size(2))
        if use_cfg:
            add_branch("uncond", None, cond_len=target_len)
        if use_emotion_cfg:
            no_emotion_item = prepare_item(strip_leading_emotion_tag(full_text))
            add_branch(
                "no_emotion",
                no_emotion_item,
                cond_len=no_emotion_item["input_ids"].size(2),
                effective_scale=gen_config.emotion_guidance_scale,
            )
        if use_nvv_cfg:
            no_nvv_item = prepare_item(strip_nvv_tags(full_text))
            add_branch(
                "no_nvv",
                no_nvv_item,
                cond_len=no_nvv_item["input_ids"].size(2),
                effective_scale=gen_config.nvv_guidance_scale,
            )

        max_cond_len = max(row["cond_len"] for row in branch_rows)
        batch_input_ids = self._make_audio_token_tensor(
            batch_size=len(branch_rows),
            seq_len=max_cond_len,
            device=self.device,
        )
        batch_audio_mask = torch.zeros(
            (len(branch_rows), max_cond_len),
            dtype=torch.bool,
            device=self.device,
        )
        batch_attention_mask = torch.zeros(
            (len(branch_rows), 1, max_cond_len, max_cond_len),
            dtype=torch.bool,
            device=self.device,
        )

        for row_index, row in enumerate(branch_rows):
            if row["kind"] == "uncond":
                batch_input_ids[row_index, :, :target_len] = target_audio_tokens
                batch_audio_mask[row_index, :target_len] = True
                batch_attention_mask[row_index, :, :target_len, :target_len] = True
                if max_cond_len > target_len:
                    pad_diag = torch.arange(target_len, max_cond_len, device=self.device)
                    batch_attention_mask[row_index, :, pad_diag, pad_diag] = True
                continue

            item_value = row["item"]
            if item_value is None:
                raise RuntimeError(f"Missing conditional inputs for branch {row['kind']!r}.")
            cond_len = int(row["cond_len"])
            batch_input_ids[row_index, :, :cond_len] = item_value["input_ids"]
            batch_audio_mask[row_index, :cond_len] = item_value["audio_mask"]
            batch_attention_mask[row_index, :, :cond_len, :cond_len] = True

        sample_tokens = target_audio_tokens.unsqueeze(0).clone()
        layer_ids = torch.arange(self.num_audio_codebook, device=self.device).view(1, -1, 1)
        mask_ids = self._audio_mask_ids_tensor(self.device).view(1, -1, 1)
        editable_scores_mask = editable_audio_mask.view(1, 1, target_len)
        editable_token_count = int(
            (
                (sample_tokens == mask_ids)
                & editable_scores_mask.expand_as(sample_tokens)
            )
            .sum()
            .item()
        )
        if editable_token_count <= 0:
            raise ValueError("target_audio_tokens has no editable mask tokens.")
        schedule = self._build_mask_token_schedules(
            [editable_token_count],
            gen_config,
        )[0]

        def get_branch_logits(
            semantic_logits: torch.Tensor,
            acoustic_logits: torch.Tensor,
            kind: str,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            row_index = branch_lookup.get(kind)
            if row_index is None:
                return None, None
            row = branch_rows[row_index]
            if kind == "uncond":
                return (
                    semantic_logits[row_index : row_index + 1, :target_len, :],
                    acoustic_logits[row_index : row_index + 1, :, :target_len, :],
                )
            cond_len = int(row["cond_len"])
            return (
                semantic_logits[row_index : row_index + 1, cond_len - target_len : cond_len, :],
                acoustic_logits[row_index : row_index + 1, :, cond_len - target_len : cond_len, :],
            )

        for step in range(gen_config.num_step):
            outputs = self.model.forward_step(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            )
            semantic_logits = outputs.semantic_logits.to(torch.float32)
            acoustic_logits = outputs.acoustic_logits.to(torch.float32)

            c_semantic_logits, c_acoustic_logits = get_branch_logits(
                semantic_logits,
                acoustic_logits,
                "full",
            )
            u_semantic_logits, u_acoustic_logits = get_branch_logits(
                semantic_logits,
                acoustic_logits,
                "uncond",
            )
            ne_semantic_logits, ne_acoustic_logits = get_branch_logits(
                semantic_logits,
                acoustic_logits,
                "no_emotion",
            )
            nn_semantic_logits, nn_acoustic_logits = get_branch_logits(
                semantic_logits,
                acoustic_logits,
                "no_nvv",
            )
            if c_semantic_logits is None or c_acoustic_logits is None:
                raise RuntimeError("Missing full conditional logits.")
            emotion_scale = (
                branch_rows[branch_lookup["no_emotion"]]["effective_scale"]
                if "no_emotion" in branch_lookup
                else 0.0
            )
            nvv_scale = (
                branch_rows[branch_lookup["no_nvv"]]["effective_scale"]
                if "no_nvv" in branch_lookup
                else 0.0
            )
            pred_tokens, scores = self._predict_tokens_with_scoring(
                c_semantic_logits,
                u_semantic_logits,
                c_acoustic_logits,
                u_acoustic_logits,
                gen_config,
                ne_semantic_logits=ne_semantic_logits,
                ne_acoustic_logits=ne_acoustic_logits,
                nn_semantic_logits=nn_semantic_logits,
                nn_acoustic_logits=nn_acoustic_logits,
                emotion_guidance_scale=float(emotion_scale),
                nvv_guidance_scale=float(nvv_scale),
            )
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)

            scores.masked_fill_(~editable_scores_mask, -float("inf"))
            scores.masked_fill_(sample_tokens != mask_ids, -float("inf"))
            available = int(torch.isfinite(scores).sum().item())
            if available <= 0:
                break
            k = min(int(schedule[step]), available)
            if k <= 0:
                continue
            _, topk_idx = torch.topk(scores.flatten(), k)
            sample_tokens.copy_(
                self._apply_selected_token_updates(sample_tokens, pred_tokens, topk_idx)
            )
            sample_tokens = torch.where(
                editable_scores_mask.expand_as(sample_tokens),
                sample_tokens,
                target_audio_tokens.unsqueeze(0),
            )
            for row_index, row in enumerate(branch_rows):
                if row["kind"] == "uncond":
                    batch_input_ids[row_index : row_index + 1, :, :target_len] = sample_tokens
                else:
                    cond_len = int(row["cond_len"])
                    batch_input_ids[
                        row_index : row_index + 1,
                        :,
                        cond_len - target_len : cond_len,
                    ] = sample_tokens

        remaining = sample_tokens == mask_ids
        if bool((remaining & editable_scores_mask.expand_as(sample_tokens)).any()):
            raise RuntimeError("Local edit generation ended with unfilled mask tokens.")
        return sample_tokens.squeeze(0).detach()

    def _predict_tokens_with_scoring(
        self,
        c_semantic_logits: torch.Tensor,
        u_semantic_logits: torch.Tensor | None,
        c_acoustic_logits: torch.Tensor,
        u_acoustic_logits: torch.Tensor | None,
        gen_config: LLMGenerationConfig,
        *,
        ne_semantic_logits: torch.Tensor | None = None,
        ne_acoustic_logits: torch.Tensor | None = None,
        nn_semantic_logits: torch.Tensor | None = None,
        nn_acoustic_logits: torch.Tensor | None = None,
        emotion_guidance_scale: float = 0.0,
        nvv_guidance_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def combine_log_probs(
            c_logits: torch.Tensor,
            u_logits: torch.Tensor | None,
            ne_logits: torch.Tensor | None,
            nn_logits: torch.Tensor | None,
        ) -> torch.Tensor:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            guided = c_log_probs
            if gen_config.guidance_scale != 0.0:
                if u_logits is None:
                    raise ValueError(
                        "Unconditional logits are required when guidance_scale is non-zero."
                    )
                u_log_probs = F.log_softmax(u_logits, dim=-1)
                guided = guided + gen_config.guidance_scale * (c_log_probs - u_log_probs)
            if emotion_guidance_scale != 0.0:
                if ne_logits is None:
                    raise ValueError(
                        "No-emotion logits are required when emotion_guidance_scale is non-zero."
                    )
                ne_log_probs = F.log_softmax(ne_logits, dim=-1)
                guided = guided + emotion_guidance_scale * (c_log_probs - ne_log_probs)
            if nvv_guidance_scale != 0.0:
                if nn_logits is None:
                    raise ValueError(
                        "No-NVV logits are required when nvv_guidance_scale is non-zero."
                    )
                nn_log_probs = F.log_softmax(nn_logits, dim=-1)
                guided = guided + nvv_guidance_scale * (c_log_probs - nn_log_probs)
            return torch.log_softmax(guided, dim=-1)

        semantic_log_probs = combine_log_probs(
            c_semantic_logits,
            u_semantic_logits,
            ne_semantic_logits,
            nn_semantic_logits,
        )
        acoustic_log_probs = combine_log_probs(
            c_acoustic_logits,
            u_acoustic_logits,
            ne_acoustic_logits,
            nn_acoustic_logits,
        )

        semantic_log_probs[..., self.audio_mask_ids[0]] = -float("inf")
        acoustic_log_probs[..., self.audio_mask_ids[1]] = -float("inf")
        if self.audio_separator_ids is not None:
            semantic_log_probs[..., self.audio_separator_ids[0]] = -float("inf")
            acoustic_log_probs[..., self.audio_separator_ids[1]] = -float("inf")

        if gen_config.class_temperature > 0.0:
            semantic_pred_tokens = _gumbel_sample(
                _filter_top_k(semantic_log_probs, ratio=0.1),
                gen_config.class_temperature,
            ).argmax(dim=-1)
            acoustic_pred_tokens = _gumbel_sample(
                _filter_top_k(acoustic_log_probs, ratio=0.1),
                gen_config.class_temperature,
            ).argmax(dim=-1)
        else:
            semantic_pred_tokens = semantic_log_probs.argmax(dim=-1)
            acoustic_pred_tokens = acoustic_log_probs.argmax(dim=-1)

        semantic_scores = semantic_log_probs.max(dim=-1)[0].unsqueeze(1)
        acoustic_scores = acoustic_log_probs.max(dim=-1)[0]
        confidence_scores = torch.cat([semantic_scores, acoustic_scores], dim=1)
        pred_tokens = torch.cat([semantic_pred_tokens.unsqueeze(1), acoustic_pred_tokens], dim=1)
        return pred_tokens, confidence_scores

    def _apply_selected_token_updates(
        self,
        sample_tokens: torch.Tensor,
        pred_tokens: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> torch.Tensor:
        flat_tokens = sample_tokens.flatten()
        flat_pred_tokens = pred_tokens.flatten()
        flat_tokens[topk_idx] = flat_pred_tokens[topk_idx]
        updated_tokens = flat_tokens.view_as(sample_tokens)

        pause_ids = self._audio_pause_ids_tensor(sample_tokens.device)
        if pause_ids is None:
            return updated_tokens
        num_layers = updated_tokens.shape[1]
        seq_len = updated_tokens.shape[2]
        pause_ids = pause_ids[:num_layers]
        semantic_flat = topk_idx[topk_idx < seq_len]
        if semantic_flat.numel() == 0:
            return updated_tokens
        semantic_pred = pred_tokens[0, 0, semantic_flat]
        semantic_frame_idx = semantic_flat[semantic_pred == pause_ids[0]]
        if semantic_frame_idx.numel() == 0:
            return updated_tokens
        updated_tokens[:, :, semantic_frame_idx] = pause_ids.view(1, num_layers, 1)
        return updated_tokens

    def _build_schedules(
        self,
        target_lens: list[int],
        gen_config: LLMGenerationConfig,
    ) -> list[list[int]]:
        total_masks = [
            int(target_len) * self.num_audio_codebook for target_len in target_lens
        ]
        return self._build_mask_token_schedules(total_masks, gen_config)

    def _build_mask_token_schedules(
        self,
        total_mask_tokens: list[int],
        gen_config: LLMGenerationConfig,
    ) -> list[list[int]]:
        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step + 1,
            t_shift=gen_config.t_shift,
            device=self.device,
        ).tolist()
        schedules: list[list[int]] = []
        for total_mask in total_mask_tokens:
            remaining = total_mask
            schedule = []
            for step in range(gen_config.num_step):
                if step == gen_config.num_step - 1:
                    num = remaining
                else:
                    num = min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        remaining,
                    )
                schedule.append(int(num))
                remaining -= int(num)
            schedules.append(schedule)
        return schedules

    def _estimate_target_tokens(
        self,
        text: str,
        ref_text: Optional[str],
        num_ref_audio_tokens: Optional[int],
        speed: float = 1.0,
    ) -> int:
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = self.frame_rate
        estimated = self.duration_estimator.estimate_duration(
            text,
            ref_text,
            num_ref_audio_tokens,
        )
        if speed > 0 and speed != 1.0:
            estimated = estimated / speed
        return max(1, int(estimated))

    def _make_audio_token_tensor(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask_ids = self._audio_mask_ids_tensor(device).view(1, self.num_audio_codebook, 1)
        return mask_ids.expand(batch_size, self.num_audio_codebook, seq_len).clone()

    def _make_audio_separator_tensor(self, device: torch.device) -> torch.Tensor:
        if self.audio_separator_ids is None:
            raise RuntimeError("audio_separator_ids is not configured.")
        sep = torch.tensor(self.audio_separator_ids, dtype=torch.long, device=device)
        return sep.view(1, self.num_audio_codebook, 1)

    def _resolve_no_ref_text_silence_codec_path(self) -> Optional[Path]:
        codec_path = self.config.get(
            "no_ref_text_silence_codec_path",
            "viitorvoice/assets/dualcodec_silence_2s.pt",
        )
        if not codec_path:
            return None
        path = Path(str(codec_path)).expanduser()
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        return paths.resolve_model_asset_path(path, fallback=paths.silence_codec_path())

    def _load_no_ref_text_silence_codec(self) -> Optional[torch.Tensor]:
        if self._no_ref_text_silence_codec_cache is not None:
            return self._no_ref_text_silence_codec_cache

        path = self._resolve_no_ref_text_silence_codec_path()
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"No-ref-text silence codec not found: {path}")

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        tokens = payload["tokens"] if isinstance(payload, dict) else payload
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        if tokens.ndim == 3:
            if tokens.size(0) != 1:
                raise ValueError(
                    "No-ref-text silence codec batch size should be 1, "
                    f"got {tokens.size(0)}."
                )
            tokens = tokens.squeeze(0)
        if tokens.ndim != 2:
            raise ValueError(
                "No-ref-text silence codec should have shape [C, T] or [1, C, T], "
                f"got {tuple(tokens.shape)}."
            )
        if tokens.size(0) != self.num_audio_codebook:
            raise ValueError(
                "No-ref-text silence codec codebook count mismatch: "
                f"expected {self.num_audio_codebook}, got {tokens.size(0)}."
            )
        self._no_ref_text_silence_codec_cache = tokens.contiguous()
        return self._no_ref_text_silence_codec_cache

    def _get_no_ref_text_silence_prefix_tokens(
        self,
        prefix_len: Optional[int],
    ) -> Optional[torch.Tensor]:
        if prefix_len is None or prefix_len <= 0:
            return None
        silence_tokens = self._load_no_ref_text_silence_codec()
        if silence_tokens is None:
            return None
        if prefix_len > silence_tokens.size(-1):
            raise ValueError(
                "Requested no-ref-text silence prefix is longer than the saved "
                f"silence codec: requested={prefix_len}, available={silence_tokens.size(-1)}."
            )
        return silence_tokens[:, :prefix_len].to(self.device)

    def _audio_mask_ids_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.audio_mask_ids, dtype=torch.long, device=device)

    def _audio_pause_ids_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.audio_pause_ids is None:
            return None
        return torch.tensor(self.audio_pause_ids, dtype=torch.long, device=device)

    def _normalize_semantic_tokens(
        self,
        value: torch.Tensor | np.ndarray | Sequence[int],
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.long)
        if tensor.ndim == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 2:
            if tensor.size(0) == self.num_audio_codebook:
                tensor = tensor[0]
            elif tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            elif tensor.size(1) == 1:
                tensor = tensor.squeeze(1)
            else:
                raise ValueError(
                    "semantic_tokens should have shape [T], [1, T], or [C, T], "
                    f"got {tuple(tensor.shape)}."
                )
        if tensor.ndim != 1:
            raise ValueError(
                "semantic_tokens should have shape [T], got "
                f"{tuple(tensor.shape)}."
            )
        if tensor.numel() <= 0:
            raise ValueError("semantic_tokens must contain at least one token.")

        mask_id = int(self.audio_mask_ids[0])
        invalid = tensor == mask_id
        if self.audio_separator_ids is not None:
            invalid |= tensor == int(self.audio_separator_ids[0])
        if bool(invalid.any()):
            bad = torch.nonzero(invalid, as_tuple=False).flatten()[:8].tolist()
            raise ValueError(
                "semantic_tokens contains mask/separator structural ids at "
                f"positions {bad}."
            )

        valid_size = int(self.audio_valid_vocab_sizes[0])
        if bool(((tensor < 0) | (tensor >= valid_size)).any()):
            bad = torch.nonzero((tensor < 0) | (tensor >= valid_size), as_tuple=False)
            sample = bad.flatten()[:8].tolist()
            raise ValueError(
                "semantic_tokens contains ids outside the valid semantic vocab "
                f"[0, {valid_size}); positions {sample}."
            )
        return tensor.to(self.device, non_blocking=True).contiguous()

    def _make_semantic_conditioned_target(
        self,
        semantic_tokens: torch.Tensor | np.ndarray | Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = self._normalize_semantic_tokens(semantic_tokens)
        target = self._make_audio_token_tensor(
            batch_size=1,
            seq_len=int(semantic.numel()),
            device=self.device,
        ).squeeze(0)
        target[0] = semantic

        editable_mask = torch.ones(
            semantic.numel(),
            dtype=torch.bool,
            device=self.device,
        )
        pause_ids = self._audio_pause_ids_tensor(self.device)
        if pause_ids is not None:
            pause_frames = semantic == pause_ids[0]
            if bool(pause_frames.any()):
                target[:, pause_frames] = pause_ids.view(-1, 1)
                editable_mask[pause_frames] = False
        return target.contiguous(), editable_mask

    def _ensure_no_ref_text_tokens_available(self) -> None:
        vocab = self.tokenizer.get_vocab()
        missing = [token for token in NO_REF_TEXT_SPECIAL_TOKENS if token not in vocab]
        if missing:
            raise ValueError(
                "No-ref-text inference requires tokenizer special tokens: "
                + ", ".join(missing)
            )

    def _normalize_ref_audio_tokens(
        self,
        value: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray] | None,
        batch_size: int,
    ) -> list[Optional[torch.Tensor]]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, (torch.Tensor, np.ndarray)):
            values: list[Any] = [value]
        else:
            values = list(value)
        if len(values) not in {1, batch_size}:
            raise ValueError(
                f"ref_audio_tokens should have length 1 or batch size {batch_size}, got {len(values)}."
            )
        if len(values) == 1 and batch_size > 1:
            values = values * batch_size
        return [self._normalize_single_ref_tokens(item, i) for i, item in enumerate(values)]

    def _normalize_single_ref_tokens(
        self,
        value: torch.Tensor | np.ndarray | None,
        index: int,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.long)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"ref_audio_tokens[{index}] should have shape [C, T], got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] != self.num_audio_codebook:
            raise ValueError(
                f"ref_audio_tokens[{index}] should have {self.num_audio_codebook} layers, "
                f"got {tensor.shape[0]}."
            )
        return tensor.to(self.device, non_blocking=True).contiguous()

    @staticmethod
    def _ensure_list(x: Any, batch_size: int, auto_repeat: bool = True) -> list[Any]:
        x_list = x if isinstance(x, list) else [x]
        if len(x_list) not in {1, batch_size}:
            raise ValueError(f"Expected length 1 or {batch_size}, got {len(x_list)}.")
        if auto_repeat and len(x_list) == 1:
            x_list = x_list * batch_size
        return x_list

    @staticmethod
    def _require_result(item: Optional[GeneratedItem], index: int) -> GeneratedItem:
        if item is None:
            raise RuntimeError(f"Missing generated result for item {index}.")
        return item

    @staticmethod
    def _load_tokenizer(checkpoint_dir: Path) -> PreTrainedTokenizerFast:
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        tokenizer_config: dict[str, Any] = {}
        if tokenizer_config_path.is_file():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            bos_token=tokenizer_config.get("bos_token"),
            eos_token=tokenizer_config.get("eos_token"),
            unk_token=tokenizer_config.get("unk_token"),
            pad_token=tokenizer_config.get("pad_token"),
            mask_token=tokenizer_config.get("mask_token"),
            additional_special_tokens=tokenizer_config.get("extra_special_tokens", []),
            clean_up_tokenization_spaces=tokenizer_config.get(
                "clean_up_tokenization_spaces",
                False,
            ),
        )


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    value, index = logits.topk(k, dim=-1)
    filtered = torch.full_like(logits, -float("inf"))
    filtered.scatter_(-1, index, value)
    return filtered


def _resolve_instruct(instruct: Optional[str], use_zh: bool = False) -> Optional[str]:
    if instruct is None:
        return None

    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    raw_items = re.split(r"\s*[,，]\s*", instruct_str)
    raw_items = [item for item in raw_items if item]

    unknown = []
    normalised = []
    for raw in raw_items:
        item = raw.strip().lower()
        if item in _INSTRUCT_ALL_VALID:
            normalised.append(item)
        else:
            suggestion = difflib.get_close_matches(
                item,
                _INSTRUCT_ALL_VALID,
                n=1,
                cutoff=0.6,
            )
            unknown.append((raw, item, suggestion[0] if suggestion else None))

    if unknown:
        lines = []
        for raw, item, suggestion in unknown:
            if suggestion:
                lines.append(
                    f"  '{raw}' -> '{item}' (unsupported; did you mean '{suggestion}'?)"
                )
            else:
                lines.append(f"  '{raw}' -> '{item}' (unsupported)")
        raise ValueError(
            f"Unsupported instruct items found in {instruct_str}:\n"
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\n\nTip: Use only English or only Chinese instructs. "
            "English instructs should use comma + space (e.g. "
            "'male, indian accent'),\nChinese instructs should use full-width "
            "comma (e.g. '男，河南话')."
        )

    has_dialect = any(item.endswith("话") for item in normalised)
    has_accent = any(" accent" in item for item in normalised)
    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix Chinese dialect and English accent in a single instruct. "
            "Dialects are for Chinese speech, accents for English speech."
        )

    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    if use_zh:
        normalised = [_INSTRUCT_EN_TO_ZH.get(item, item) for item in normalised]
    else:
        normalised = [_INSTRUCT_ZH_TO_EN.get(item, item) for item in normalised]

    conflicts = []
    for category in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [item for item in normalised if item in category]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        parts = []
        for group in conflicts:
            parts.append(" vs ".join(f"'{item}'" for item in group))
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(parts)
            + ". Each category (gender, age, pitch, style, accent, dialect) "
            "allows at most one item."
        )

    has_zh = any(any("\u4e00" <= char <= "\u9fff" for char in item) for item in normalised)
    separator = "，" if has_zh else ", "
    return separator.join(normalised)


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float,
    t_end: float,
    num_step: int,
    t_shift: float,
    device: torch.device,
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1, device=device)
    return t_shift * timesteps / (1 + (t_shift - 1) * timesteps)


