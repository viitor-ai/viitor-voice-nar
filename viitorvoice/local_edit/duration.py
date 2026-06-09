from __future__ import annotations

import math
import re


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_NONVERBAL_DURATION_TAGS = frozenset(
    {
        "<|nv-Cough|>",
        "<|nv-Sigh|>",
        "<|nv-Breathing|>",
        "<|nv-Surprise-oh|>",
        "<|nv-Surprise-ah|>",
        "<|nv-Dissatisfaction-hnn|>",
        "<|nv-Laughter|>",
        "<|nv-Uhm|>",
        "<|nv-Question-en|>",
        "<|nv-Question-oh|>",
        "<|nv-Question-ah|>",
        "<|nv-Question-ei|>",
        "<|nv-Question-huh|>",
        "<|nv-Confirmation-en|>",
        "<|nv-Throat clearing|>",
        "<|nv-Gasp|>",
        "<|nv-Groaning|>",
        "<|nv-Surprise-wa|>",
        "<|nv-Surprise-yo|>",
        "<|nv-Question-yi|>",
        "(burps)",
        "(chuckle)",
        "(clear-throat)",
        "(coughs)",
        "(emm)",
        "(exhale)",
        "(gasps)",
        "(groans)",
        "(hissing)",
        "(humming)",
        "(inhale)",
        "(laughs)",
        "(lip-smacking)",
        "(pant)",
        "(sighs)",
        "(sneezes)",
        "(sniffs)",
        "(snorts)",
    }
)
_NONVERBAL_DURATION_TAG_RE = re.compile(
    "|".join(re.escape(tag) for tag in sorted(_NONVERBAL_DURATION_TAGS, key=len, reverse=True))
)


def text_duration_weight(text: str) -> float:
    if not text:
        return 0.0
    weight = 0.0
    consumed = [False] * len(text)
    for match in _NONVERBAL_DURATION_TAG_RE.finditer(text):
        weight += 1.0
        for idx in range(match.start(), match.end()):
            consumed[idx] = True
    for match in _WORD_RE.finditer(text):
        if any(consumed[match.start() : match.end()]):
            continue
        token = match.group(0)
        # A compact approximation that avoids making long English words too slow.
        weight += max(1.0, len(token) / 4.0)
        for idx in range(match.start(), match.end()):
            consumed[idx] = True
    for idx, char in enumerate(text):
        if consumed[idx] or char.isspace():
            continue
        if _CJK_RE.fullmatch(char) or _KANA_RE.fullmatch(char):
            weight += 1.0
        elif char.isalnum():
            weight += 0.35
        elif char in ",.;:!?，。；：！？、":
            weight += 0.15
        else:
            weight += 0.25
    return max(weight, 0.0)


def estimate_replacement_frames(
    old_frames: int,
    old_text: str,
    replacement_text: str,
    *,
    frame_rate: int = 25,
    length_mode: str = "auto",
    manual_seconds: float | None = None,
    manual_frames: int | None = None,
    length_scale: float = 1.0,
    min_frames: int = 6,
) -> int:
    mode = (length_mode or "auto").strip().lower()
    if mode == "manual_frames" and manual_frames is not None and int(manual_frames) > 0:
        return max(int(min_frames), int(manual_frames))
    if mode == "manual_seconds" and manual_seconds is not None and float(manual_seconds) > 0:
        return max(int(min_frames), int(round(float(manual_seconds) * int(frame_rate))))

    old_weight = text_duration_weight(old_text)
    new_weight = text_duration_weight(replacement_text)
    if old_weight <= 0:
        old_weight = max(1.0, new_weight)
    ratio = new_weight / old_weight if new_weight > 0 else 1.0
    estimated = int(round(max(1, int(old_frames)) * ratio * float(length_scale)))
    return max(int(min_frames), estimated)
