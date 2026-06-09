from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from viitorvoice.local_edit.alignment import AlignmentItem


@dataclass
class EditSpan:
    selected_indices: list[int]
    selected_text: str
    replacement_text: str
    start_time: float
    end_time: float
    padded_start_time: float
    padded_end_time: float
    start_frame: int
    end_frame: int
    edited_full_text: str

    @property
    def old_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class EditSegment:
    selected_indices: list[int]
    selected_text: str
    replacement_text: str
    start_time: float
    end_time: float
    padded_start_time: float
    padded_end_time: float
    start_frame: int
    end_frame: int

    @property
    def old_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class MultiEditSpan:
    segments: list[EditSegment]
    edited_full_text: str

    @property
    def selected_indices(self) -> list[int]:
        return [idx for segment in self.segments for idx in segment.selected_indices]

    @property
    def selected_text(self) -> str:
        return " | ".join(segment.selected_text for segment in self.segments)

    @property
    def replacement_text(self) -> str:
        return " | ".join(segment.replacement_text for segment in self.segments)

    @property
    def start_time(self) -> float:
        return min(segment.start_time for segment in self.segments)

    @property
    def end_time(self) -> float:
        return max(segment.end_time for segment in self.segments)

    @property
    def padded_start_time(self) -> float:
        return min(segment.padded_start_time for segment in self.segments)

    @property
    def padded_end_time(self) -> float:
        return max(segment.padded_end_time for segment in self.segments)

    @property
    def start_frame(self) -> int:
        return min(segment.start_frame for segment in self.segments)

    @property
    def end_frame(self) -> int:
        return max(segment.end_frame for segment in self.segments)

    @property
    def old_frames(self) -> int:
        return sum(segment.old_frames for segment in self.segments)


def parse_selection_indices(selection: str | int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(selection, int):
        return [selection]
    if isinstance(selection, (list, tuple)):
        values = sorted({int(item) for item in selection})
        if not values:
            raise ValueError("Selection is empty.")
        return values
    text = str(selection or "").strip()
    if not text:
        raise ValueError("Selection indices are required, for example '3' or '3-5'.")
    values: set[int] = set()
    for part in re.split(r"[,，\s]+", text):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    if not values:
        raise ValueError("Selection is empty.")
    return sorted(values)


def parse_selection_groups(selection: str | int | list[int] | tuple[int, ...]) -> list[list[int]]:
    if isinstance(selection, (int, list, tuple)):
        return [parse_selection_indices(selection)]
    text = str(selection or "").strip()
    if not text:
        raise ValueError("Selection indices are required, for example '3' or '3-5'.")
    groups = [
        parse_selection_indices(part.strip())
        for part in re.split(r"[;\n]+", text)
        if part.strip()
    ]
    if not groups:
        raise ValueError("Selection is empty.")
    seen: set[int] = set()
    duplicated: set[int] = set()
    for group in groups:
        for idx in group:
            if idx in seen:
                duplicated.add(idx)
            seen.add(idx)
    if duplicated:
        raise ValueError(f"Selection indices appear in multiple edit segments: {sorted(duplicated)}")
    return groups


def parse_replacement_texts(replacement_text: str | Sequence[str], segment_count: int) -> list[str]:
    if isinstance(replacement_text, Sequence) and not isinstance(replacement_text, str):
        parts = [str(part) for part in replacement_text]
        if len(parts) != segment_count:
            raise ValueError(
                "Replacement text segment count should match selection segment count. "
                f"Got {len(parts)} replacement segments for {segment_count} selections."
            )
        return parts

    text = str(replacement_text or "").strip()
    if segment_count <= 1:
        return [text]
    parts = [part.strip() for part in re.split(r"[;\n]+", text) if part.strip()]
    if len(parts) != segment_count:
        raise ValueError(
            "Replacement text segment count should match selection segment count. "
            f"Got {len(parts)} replacement segments for {segment_count} selections. "
            "Use ';' or new lines to separate multiple replacements."
        )
    return parts


def resolve_edit_span(
    *,
    items: list[AlignmentItem],
    original_text: str,
    selection: str | int | list[int] | tuple[int, ...],
    replacement_text: str | Sequence[str],
    padding_ms: float,
    audio_duration: float,
    token_count: int,
) -> EditSpan:
    if not items:
        raise ValueError("Alignment items are required.")
    if replacement_text is None:
        raise ValueError("Replacement text is required.")
    indices = parse_selection_indices(selection)
    by_index = {item.index: item for item in items}
    missing = [idx for idx in indices if idx not in by_index]
    if missing:
        raise ValueError(f"Selection index not found in alignment: {missing}")
    selected = [by_index[idx] for idx in indices]
    selected.sort(key=lambda item: item.start_time)
    selected_text = "".join(item.text for item in selected)
    start_time = min(item.start_time for item in selected)
    end_time = max(item.end_time for item in selected)
    if end_time <= start_time:
        raise ValueError("Selected alignment span has non-positive duration.")

    pad = max(0.0, float(padding_ms)) / 1000.0
    padded_start = max(0.0, start_time - pad)
    padded_end = min(max(audio_duration, end_time), end_time + pad)
    start_frame = _time_to_frame(padded_start, audio_duration, token_count, floor=True)
    end_frame = _time_to_frame(padded_end, audio_duration, token_count, floor=False)
    end_frame = max(start_frame + 1, min(token_count, end_frame))

    edited_full_text = _replace_text_span(original_text, selected, selected_text, replacement_text)
    return EditSpan(
        selected_indices=indices,
        selected_text=selected_text,
        replacement_text=replacement_text,
        start_time=start_time,
        end_time=end_time,
        padded_start_time=padded_start,
        padded_end_time=padded_end,
        start_frame=start_frame,
        end_frame=end_frame,
        edited_full_text=edited_full_text,
    )


def resolve_multi_edit_span(
    *,
    items: list[AlignmentItem],
    original_text: str,
    selection: str | int | list[int] | tuple[int, ...],
    replacement_text: str,
    padding_ms: float,
    audio_duration: float,
    token_count: int,
) -> MultiEditSpan:
    if not items:
        raise ValueError("Alignment items are required.")
    groups = parse_selection_groups(selection)
    replacements = parse_replacement_texts(replacement_text, len(groups))
    by_index = {item.index: item for item in items}
    pad = max(0.0, float(padding_ms)) / 1000.0

    segments: list[EditSegment] = []
    replacements_by_text_span: list[tuple[int, int, str]] = []
    fallback_cursor = 0
    for indices, replacement in zip(groups, replacements):
        missing = [idx for idx in indices if idx not in by_index]
        if missing:
            raise ValueError(f"Selection index not found in alignment: {missing}")
        selected = [by_index[idx] for idx in indices]
        selected.sort(key=lambda item: item.start_time)
        selected_text = "".join(item.text for item in selected)
        start_time = min(item.start_time for item in selected)
        end_time = max(item.end_time for item in selected)
        if end_time <= start_time:
            raise ValueError("Selected alignment span has non-positive duration.")

        padded_start = max(0.0, start_time - pad)
        padded_end = min(max(audio_duration, end_time), end_time + pad)
        start_frame = _time_to_frame(padded_start, audio_duration, token_count, floor=True)
        end_frame = _time_to_frame(padded_end, audio_duration, token_count, floor=False)
        end_frame = max(start_frame + 1, min(token_count, end_frame))

        text_start, text_end = _locate_text_span(
            original_text,
            selected,
            selected_text,
            fallback_cursor,
        )
        text_start, text_end = _adjust_text_span_for_replacement(
            original_text,
            text_start,
            text_end,
            replacement,
        )
        fallback_cursor = text_end
        replacements_by_text_span.append((text_start, text_end, replacement))
        segments.append(
            EditSegment(
                selected_indices=indices,
                selected_text=selected_text,
                replacement_text=replacement,
                start_time=start_time,
                end_time=end_time,
                padded_start_time=padded_start,
                padded_end_time=padded_end,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    ordered = sorted(zip(segments, replacements_by_text_span), key=lambda item: item[0].start_frame)
    for (left, _), (right, _) in zip(ordered, ordered[1:]):
        if left.end_frame > right.start_frame:
            raise ValueError(
                "Padded edit segments overlap in audio-token frames. "
                "Reduce padding_ms or merge those selections into one segment."
            )
    text_ordered = sorted(replacements_by_text_span, key=lambda item: item[0])
    for left, right in zip(text_ordered, text_ordered[1:]):
        if left[1] > right[0]:
            raise ValueError("Edit text spans overlap. Please merge overlapping selections.")

    edited_full_text = original_text
    for start, end, replacement in reversed(text_ordered):
        edited_full_text = edited_full_text[:start] + replacement + edited_full_text[end:]

    return MultiEditSpan(
        segments=[segment for segment, _ in ordered],
        edited_full_text=edited_full_text,
    )


def alignment_items_from_dicts(rows: list[dict[str, Any]]) -> list[AlignmentItem]:
    items: list[AlignmentItem] = []
    for idx, row in enumerate(rows):
        items.append(
            AlignmentItem(
                index=int(row.get("index", idx)),
                text=str(row.get("text", "")),
                start_time=float(row.get("start_time", 0.0)),
                end_time=float(row.get("end_time", 0.0)),
                start_char=_optional_int(row.get("start_char")),
                end_char=_optional_int(row.get("end_char")),
                kind=row.get("kind", "word"),
            )
        )
    return items


def _time_to_frame(
    time_value: float,
    audio_duration: float,
    token_count: int,
    *,
    floor: bool,
) -> int:
    if token_count <= 0:
        raise ValueError(f"token_count should be positive, got {token_count}.")
    if audio_duration <= 0:
        frame = time_value * 25.0
    else:
        frame = time_value / audio_duration * token_count
    value = math.floor(frame) if floor else math.ceil(frame)
    return max(0, min(int(token_count), int(value)))


def _replace_text_span(
    original_text: str,
    selected: list[AlignmentItem],
    selected_text: str,
    replacement_text: str,
) -> str:
    start_chars = [item.start_char for item in selected if item.start_char is not None]
    end_chars = [item.end_char for item in selected if item.end_char is not None]
    if start_chars and end_chars:
        start = min(start_chars)
        end = max(end_chars)
        start, end = _adjust_text_span_for_replacement(original_text, start, end, replacement_text)
        return original_text[:start] + replacement_text + original_text[end:]
    pos = original_text.find(selected_text)
    if pos >= 0:
        end = pos + len(selected_text)
        pos, end = _adjust_text_span_for_replacement(original_text, pos, end, replacement_text)
        return original_text[:pos] + replacement_text + original_text[end:]
    raise ValueError(
        "Could not map selected alignment text back to original_text. "
        "Please use char-level alignment or adjust the original text."
    )


def _locate_text_span(
    original_text: str,
    selected: list[AlignmentItem],
    selected_text: str,
    search_start: int,
) -> tuple[int, int]:
    start_chars = [item.start_char for item in selected if item.start_char is not None]
    end_chars = [item.end_char for item in selected if item.end_char is not None]
    if start_chars and end_chars:
        return min(start_chars), max(end_chars)
    pos = original_text.find(selected_text, max(0, int(search_start)))
    if pos < 0 and search_start:
        pos = original_text.find(selected_text)
    if pos >= 0:
        return pos, pos + len(selected_text)
    raise ValueError(
        "Could not map selected alignment text back to original_text. "
        "Please use char-level alignment or adjust the original text."
    )


def _adjust_text_span_for_replacement(
    original_text: str,
    start: int,
    end: int,
    replacement_text: str,
) -> tuple[int, int]:
    if replacement_text != "":
        return start, end
    if (
        start > 0
        and end < len(original_text)
        and original_text[start - 1].isspace()
        and original_text[end].isspace()
    ):
        while end < len(original_text) and original_text[end].isspace():
            end += 1
    elif start <= 0:
        while end < len(original_text) and original_text[end].isspace():
            end += 1
    elif end >= len(original_text):
        while start > 0 and original_text[start - 1].isspace():
            start -= 1
    return start, end


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
