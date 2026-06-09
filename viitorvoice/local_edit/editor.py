from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch

from viitorvoice.local_edit.alignment import AlignmentItem
from viitorvoice.local_edit.duration import estimate_replacement_frames
from viitorvoice.local_edit.span import EditSpan, MultiEditSpan, resolve_edit_span, resolve_multi_edit_span


@dataclass
class LocalEditTokenPlan:
    span: EditSpan
    source_tokens: torch.Tensor
    prefix_tokens: torch.Tensor
    suffix_tokens: torch.Tensor
    reference_tokens: torch.Tensor
    replacement_frames: int
    audio_duration: float

    @property
    def old_total_frames(self) -> int:
        return int(self.source_tokens.shape[-1])

    @property
    def new_total_frames(self) -> int:
        return int(self.prefix_tokens.shape[-1] + self.replacement_frames + self.suffix_tokens.shape[-1])


@dataclass
class LocalMultiEditTokenPlan:
    span: MultiEditSpan
    source_tokens: torch.Tensor
    target_tokens: torch.Tensor
    editable_audio_mask: torch.Tensor
    reference_tokens: torch.Tensor
    replacement_frames: list[int]
    audio_duration: float

    @property
    def old_total_frames(self) -> int:
        return int(self.source_tokens.shape[-1])

    @property
    def new_total_frames(self) -> int:
        return int(self.target_tokens.shape[-1])

    @property
    def total_replacement_frames(self) -> int:
        return sum(int(frames) for frames in self.replacement_frames)


def build_local_edit_token_plan(
    *,
    source_tokens: torch.Tensor | np.ndarray,
    alignment_items: list[AlignmentItem],
    original_text: str,
    selection: str,
    replacement_text: str | Sequence[str],
    audio_duration: float,
    padding_ms: float,
    frame_rate: int = 25,
    length_mode: str = "auto",
    manual_seconds: float | None = None,
    manual_frames: int | None = None,
    length_scale: float = 1.0,
    min_edit_frames: int = 6,
    reference_context_frames: int = 300,
) -> LocalEditTokenPlan:
    tokens = torch.as_tensor(source_tokens, dtype=torch.long)
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    if tokens.ndim != 2:
        raise ValueError(f"source_tokens should have shape [C, T], got {tuple(tokens.shape)}.")
    token_count = int(tokens.shape[-1])
    if token_count <= 0:
        raise ValueError("source_tokens is empty.")
    span = resolve_edit_span(
        items=alignment_items,
        original_text=original_text,
        selection=selection,
        replacement_text=replacement_text,
        padding_ms=padding_ms,
        audio_duration=audio_duration,
        token_count=token_count,
    )
    prefix = tokens[:, : span.start_frame].contiguous()
    suffix = tokens[:, span.end_frame :].contiguous()
    replacement_frames = estimate_replacement_frames(
        span.old_frames,
        span.selected_text,
        replacement_text,
        frame_rate=frame_rate,
        length_mode=length_mode,
        manual_seconds=manual_seconds,
        manual_frames=manual_frames,
        length_scale=length_scale,
        min_frames=min_edit_frames,
    )
    reference = build_reference_tokens(
        tokens,
        span.start_frame,
        span.end_frame,
        max_frames=reference_context_frames,
    )
    return LocalEditTokenPlan(
        span=span,
        source_tokens=tokens.contiguous(),
        prefix_tokens=prefix,
        suffix_tokens=suffix,
        reference_tokens=reference,
        replacement_frames=replacement_frames,
        audio_duration=float(audio_duration),
    )


def build_multi_local_edit_token_plan(
    *,
    source_tokens: torch.Tensor | np.ndarray,
    alignment_items: list[AlignmentItem],
    original_text: str,
    selection: str,
    replacement_text: str,
    audio_duration: float,
    padding_ms: float,
    expand_mask_ratio: float = 1.0,
    frame_rate: int = 25,
    length_mode: str = "auto",
    manual_seconds: float | None = None,
    manual_frames: int | None = None,
    length_scale: float = 1.0,
    min_edit_frames: int = 6,
    reference_context_frames: int = 300,
) -> LocalMultiEditTokenPlan:
    tokens = torch.as_tensor(source_tokens, dtype=torch.long)
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    if tokens.ndim != 2:
        raise ValueError(f"source_tokens should have shape [C, T], got {tuple(tokens.shape)}.")
    token_count = int(tokens.shape[-1])
    if token_count <= 0:
        raise ValueError("source_tokens is empty.")

    span = resolve_multi_edit_span(
        items=alignment_items,
        original_text=original_text,
        selection=selection,
        replacement_text=replacement_text,
        padding_ms=padding_ms,
        audio_duration=audio_duration,
        token_count=token_count,
    )
    replacement_frames = [
        estimate_replacement_frames(
            segment.old_frames,
            segment.selected_text,
            segment.replacement_text,
            frame_rate=frame_rate,
            length_mode=length_mode,
            manual_seconds=manual_seconds,
            manual_frames=manual_frames,
            length_scale=length_scale,
            min_frames=min_edit_frames,
        )
        for segment in span.segments
    ]

    parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    editable_ranges: list[tuple[int, int]] = []
    cursor = 0
    output_cursor = 0
    codebooks = int(tokens.shape[0])
    for segment, frames in zip(span.segments, replacement_frames):
        if segment.start_frame < cursor:
            raise ValueError("Edit segments overlap in audio-token frames.")
        unchanged = tokens[:, cursor : segment.start_frame]
        if unchanged.shape[-1]:
            parts.append(unchanged)
            mask_parts.append(torch.zeros(unchanged.shape[-1], dtype=torch.bool))
            output_cursor += int(unchanged.shape[-1])
        replacement_placeholder = torch.zeros((codebooks, int(frames)), dtype=tokens.dtype)
        parts.append(replacement_placeholder)
        mask_parts.append(torch.ones(int(frames), dtype=torch.bool))
        editable_ranges.append((output_cursor, output_cursor + int(frames)))
        output_cursor += int(frames)
        cursor = segment.end_frame
    tail = tokens[:, cursor:]
    if tail.shape[-1]:
        parts.append(tail)
        mask_parts.append(torch.zeros(tail.shape[-1], dtype=torch.bool))

    target_tokens = torch.cat(parts, dim=-1).contiguous()
    editable_audio_mask = _expand_editable_mask(
        torch.cat(mask_parts, dim=0).contiguous(),
        editable_ranges,
        expand_mask_ratio,
    )
    reference = build_multi_reference_tokens(
        tokens,
        [(segment.start_frame, segment.end_frame) for segment in span.segments],
        max_frames=reference_context_frames,
    )
    return LocalMultiEditTokenPlan(
        span=span,
        source_tokens=tokens.contiguous(),
        target_tokens=target_tokens,
        editable_audio_mask=editable_audio_mask,
        reference_tokens=reference,
        replacement_frames=replacement_frames,
        audio_duration=float(audio_duration),
    )


def _expand_editable_mask(
    mask: torch.Tensor,
    ranges: list[tuple[int, int]],
    expand_mask_ratio: float,
) -> torch.Tensor:
    ratio = float(expand_mask_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"expand_mask_ratio should be a positive finite number, got {expand_mask_ratio!r}.")
    ratio = max(1.0, ratio)
    if ratio == 1.0 or not ranges:
        return mask

    expanded = mask.clone()
    token_count = int(expanded.shape[0])
    for start, end in ranges:
        start = max(0, min(token_count, int(start)))
        end = max(start, min(token_count, int(end)))
        if end <= start:
            continue
        center = (start + end) / 2.0
        radius = max(0.5, (end - start) / 2.0) * ratio
        expanded_start = max(0, int(math.floor(center - radius)))
        expanded_end = min(token_count, int(math.ceil(center + radius)))
        if expanded_end > expanded_start:
            expanded[expanded_start:expanded_end] = True
    return expanded.contiguous()


def build_reference_tokens(
    source_tokens: torch.Tensor,
    start_frame: int,
    end_frame: int,
    *,
    max_frames: int = 300,
) -> torch.Tensor:
    """Pick unedited nearby context tokens for no-ref-text voice conditioning."""
    if max_frames <= 0 or source_tokens.shape[-1] <= max_frames:
        return source_tokens.contiguous()
    half = max(1, int(max_frames) // 2)
    left = source_tokens[:, max(0, start_frame - half) : start_frame]
    right = source_tokens[:, end_frame : min(source_tokens.shape[-1], end_frame + half)]
    ref = torch.cat([left, right], dim=-1)
    if ref.shape[-1] == 0:
        ref = source_tokens[:, :max_frames]
    if ref.shape[-1] > max_frames:
        ref = ref[:, :max_frames]
    return ref.contiguous()


def build_multi_reference_tokens(
    source_tokens: torch.Tensor,
    spans: list[tuple[int, int]],
    *,
    max_frames: int = 300,
) -> torch.Tensor:
    """Pick unedited source tokens around one or more edited regions."""
    if max_frames <= 0 or source_tokens.shape[-1] <= max_frames:
        return source_tokens.contiguous()
    if not spans:
        return source_tokens[:, :max_frames].contiguous()

    token_count = int(source_tokens.shape[-1])
    clean_spans = [
        (max(0, min(token_count, int(start))), max(0, min(token_count, int(end))))
        for start, end in spans
    ]
    clean_spans = [(start, end) for start, end in clean_spans if end > start]
    if not clean_spans:
        return source_tokens[:, :max_frames].contiguous()

    per_side = max(1, int(max_frames) // max(1, 2 * len(clean_spans)))
    pieces: list[torch.Tensor] = []
    for start, end in clean_spans:
        left = source_tokens[:, max(0, start - per_side) : start]
        right = source_tokens[:, end : min(token_count, end + per_side)]
        if left.shape[-1]:
            pieces.append(left)
        if right.shape[-1]:
            pieces.append(right)

    if pieces:
        ref = torch.cat(pieces, dim=-1)
    else:
        keep = torch.ones(token_count, dtype=torch.bool)
        for start, end in clean_spans:
            keep[start:end] = False
        ref = source_tokens[:, keep]
    if ref.shape[-1] == 0:
        ref = source_tokens[:, :max_frames]
    if ref.shape[-1] > max_frames:
        ref = ref[:, :max_frames]
    return ref.contiguous()
