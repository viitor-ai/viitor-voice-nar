from __future__ import annotations

import bisect
import math
import random
import re
import unicodedata
from functools import lru_cache
from typing import Optional

import torch


PROMPT_TEXT_START_TOKEN = "<|prompt_text_start|>"
PROMPT_TEXT_END_TOKEN = "<|prompt_text_end|>"
TARGET_TEXT_START_TOKEN = "<|target_text_start|>"
TARGET_TEXT_END_TOKEN = "<|target_text_end|>"
TEXT_MASK_TOKEN = "<|text_mask|>"

NO_REF_TEXT_SPECIAL_TOKENS = [
    PROMPT_TEXT_START_TOKEN,
    PROMPT_TEXT_END_TOKEN,
    TARGET_TEXT_START_TOKEN,
    TARGET_TEXT_END_TOKEN,
    TEXT_MASK_TOKEN,
]

PAUSE_ANCHOR_TOKEN = "<|pause_anchor|>"
PAUSE_ANCHOR_ALIASES = ("<|pause-anchor|>",)
AUTO_PAUSE_PUNCTUATION = frozenset(".!?。！？;；:：")
SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = {'"', "'", "）", "]", "》", ">", "」", "】"}
ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Rev.",
    "Fr.",
    "Hon.",
    "Pres.",
    "Gov.",
    "Capt.",
    "Gen.",
    "Sen.",
    "Rep.",
    "Col.",
    "Maj.",
    "Lt.",
    "Cmdr.",
    "Sgt.",
    "Cpl.",
    "Co.",
    "Corp.",
    "Inc.",
    "Ltd.",
    "Est.",
    "Dept.",
    "St.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "Mt.",
    "Ft.",
    "No.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "i.e.",
    "e.g.",
    "vs.",
    "Vs.",
    "Etc.",
    "approx.",
    "fig.",
    "def.",
}
END_PUNCTUATION = {
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "…",
    ")",
    "]",
    "}",
    '"',
    "'",
    "；",
    "：",
    "，",
    "。",
    "！",
    "？",
    "、",
    "）",
    "】",
}

_JA_KANJI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_JA_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_NONVERBAL_TAG_PATTERN = (
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)
_VIITORVOICE_TAG_PATTERN = r"<\|[^|]+?\|>"
_SPECIAL_TOKEN_PATTERN = re.compile(f"{_NONVERBAL_TAG_PATTERN}|{_VIITORVOICE_TAG_PATTERN}")
NVV_TAG_MAP = {
    "<|nv-Cough|>": "(coughs)",
    "<|nv-Sigh|>": "(sighs)",
    "<|nv-Breathing|>": "(inhale)",
    "<|nv-Surprise-oh|>": "(gasps)",
    "<|nv-Surprise-ah|>": "(gasps)",
    "<|nv-Dissatisfaction-hnn|>": "(groans)",
    "<|nv-Laughter|>": "(laughs)",
    "<|nv-Uhm|>": "(emm)",
    "<|nv-Question-en|>": "(humming)",
    "<|nv-Question-oh|>": "(humming)",
    "<|nv-Question-ah|>": "(humming)",
    "<|nv-Question-ei|>": "(humming)",
    "<|nv-Question-huh|>": "(emm)",
    "<|nv-Confirmation-en|>": "(humming)",
    "<|nv-Throat clearing|>": "(clear-throat)",
    "<|nv-Gasp|>": "(gasps)",
    "<|nv-Groaning|>": "(groans)",
    "<|nv-Surprise-wa|>": "(gasps)",
    "<|nv-Surprise-yo|>": "(gasps)",
    "<|nv-Question-yi|>": "(humming)",
}
_NVV_TAG_RE = re.compile("|".join(re.escape(tag) for tag in sorted(NVV_TAG_MAP, key=len, reverse=True)))
_PAUSE_ANCHOR_ALIAS_RE = re.compile("|".join(re.escape(tag) for tag in PAUSE_ANCHOR_ALIASES))
_EMOTION_TAG_RE = re.compile(r"^\s*<\|emotion-[^|>]+?\|>\s*")
_NVV_PAREN_VALUES = frozenset(NVV_TAG_MAP.values())
_NVV_PAREN_TAG_RE = re.compile(
    "|".join(re.escape(tag) for tag in sorted(_NVV_PAREN_VALUES, key=len, reverse=True)),
    flags=re.IGNORECASE,
)


def _normalize_stripped_tag_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?。，；：！？、])", r"\1", text)
    text = re.sub(r"([（(])\s+", r"\1", text)
    text = re.sub(r"\s+([）)])", r"\1", text)
    return text.strip()


def has_leading_emotion_tag(text: str) -> bool:
    return bool(text and _EMOTION_TAG_RE.match(text))


def extract_leading_emotion_tag(text: str) -> str | None:
    if not text:
        return None
    match = _EMOTION_TAG_RE.match(text)
    return match.group(0).strip() if match else None


def strip_leading_emotion_tag(text: str) -> str:
    if not text:
        return text
    return _normalize_stripped_tag_text(_EMOTION_TAG_RE.sub("", text, count=1))


def has_nvv_tag(text: str) -> bool:
    if not text:
        return False
    return bool(_NVV_TAG_RE.search(text) or _NVV_PAREN_TAG_RE.search(text))


def strip_nvv_tags(text: str) -> str:
    if not text:
        return text
    text = _NVV_TAG_RE.sub("", text)
    text = _NVV_PAREN_TAG_RE.sub("", text)
    return _normalize_stripped_tag_text(text)


def strip_cfg_tag_branch(text: str, *, remove_emotion: bool, remove_nvv: bool) -> str:
    if remove_emotion:
        text = strip_leading_emotion_tag(text)
    if remove_nvv:
        text = strip_nvv_tags(text)
    return _normalize_stripped_tag_text(text)

_LANG_NAME_TO_ID = {
    "english": "en",
    "chinese": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "portuguese": "pt",
    "italian": "it",
}
_LANG_IDS = frozenset(
    {
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "hi",
        "it",
        "ja",
        "ko",
        "pt",
        "ru",
        "zh",
        *_LANG_NAME_TO_ID.values(),
    }
)


def resolve_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip().lower().replace("_", "-")
    if not normalized or normalized == "none":
        return None
    aliases = {
        "en-us": "en",
        "en-gb": "en",
        "eng": "en",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "zho": "zh",
        "cmn": "zh",
        "cn": "zh",
        "ja-jp": "ja",
        "jpn": "ja",
        "jp": "ja",
        "ko-kr": "ko",
        "kor": "ko",
        "kr": "ko",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in _LANG_IDS:
        return normalized
    return _LANG_NAME_TO_ID.get(normalized)


def add_punctuation(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in END_PUNCTUATION:
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)
        text += "。" if is_chinese else "."
    return text


def normalize_nvv_tags(text: str) -> str:
    if not text:
        return text
    text = _NVV_TAG_RE.sub(lambda match: NVV_TAG_MAP[match.group(0)], text)
    return _PAUSE_ANCHOR_ALIAS_RE.sub(PAUSE_ANCHOR_TOKEN, text)


def combine_text(text: str, ref_text: Optional[str] = None) -> str:
    text = normalize_nvv_tags(text)
    ref_text = normalize_nvv_tags(ref_text) if ref_text else ref_text
    full_text = ref_text.strip() + " " + text.strip() if ref_text else text.strip()
    full_text = re.sub(r"[\r\n]+", "", full_text)
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")
    full_text = re.sub(r"[ \t]+", " ", full_text)
    chinese_range = r"[\u4e00-\u9fff]"
    return re.sub(rf"(?<={chinese_range})\s+|\s+(?={chinese_range})", "", full_text)


def prepare_text_for_tokenizer(text: str, tokenizer, language: Optional[str] = None) -> str:
    if not text:
        return text
    text = normalize_nvv_tags(text)
    is_japanese = language == "ja" or bool(_JA_KANA_RE.search(text))
    if not is_japanese:
        return text
    vocab_set = set(tokenizer.get_vocab())
    formatted = []
    for char in text:
        if _JA_KANJI_RE.fullmatch(char):
            ja_tag = f"<|ja-{char}|>"
            formatted.append(ja_tag if ja_tag in vocab_set else char)
        else:
            formatted.append(char)
    return "".join(formatted)


def insert_text_pause_anchors(text: str) -> str:
    if not text or PAUSE_ANCHOR_TOKEN in text:
        return text
    chars = list(text)
    anchored: list[str] = []
    for i, char in enumerate(chars):
        anchored.append(char)
        if char not in AUTO_PAUSE_PUNCTUATION:
            continue
        if any(not c.isspace() for c in chars[i + 1 :]):
            anchored.append(PAUSE_ANCHOR_TOKEN)
    return "".join(anchored)


def tokenize_with_special_tokens(text: str, tokenizer) -> torch.Tensor:
    parts: list[list[int]] = []
    last_end = 0
    for match in _SPECIAL_TOKEN_PATTERN.finditer(text):
        if match.start() > last_end:
            segment = text[last_end : match.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(match.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = match.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)
    if not parts:
        return tokenizer(text, return_tensors="pt").input_ids
    combined = []
    for token_ids in parts:
        combined.extend(token_ids)
    return torch.tensor([combined], dtype=torch.long)


def make_mask_text(mask_len: int) -> str:
    return TEXT_MASK_TOKEN * max(1, int(mask_len))


def wrap_prompt_target_text(prompt_text: str, target_text: str) -> str:
    return (
        f"{PROMPT_TEXT_START_TOKEN}{prompt_text}{PROMPT_TEXT_END_TOKEN}"
        f"{TARGET_TEXT_START_TOKEN}{target_text}{TARGET_TEXT_END_TOKEN}"
    )


def estimate_ref_text_mask_len(
    language_id: str | None,
    prompt_audio_frames: int,
    audio_frame_rate: int,
    tokens_per_second: dict[str, float],
    min_tokens: int,
    max_tokens: int,
    default_tokens_per_second: float,
    jitter_ratio: float = 0.0,
    jitter: bool = False,
    rng: random.Random | None = None,
) -> int:
    if audio_frame_rate <= 0:
        raise ValueError(f"audio_frame_rate should be positive, got {audio_frame_rate}.")
    if min_tokens <= 0:
        raise ValueError(f"min_tokens should be positive, got {min_tokens}.")
    if max_tokens < min_tokens:
        raise ValueError(
            f"max_tokens should be >= min_tokens, got {max_tokens} < {min_tokens}."
        )
    seconds = max(0.0, prompt_audio_frames / audio_frame_rate)
    rate = tokens_per_second.get(language_id or "", default_tokens_per_second)
    if rate <= 0:
        rate = default_tokens_per_second
    estimated = seconds * rate
    if jitter and jitter_ratio > 0.0:
        chooser = rng if rng is not None else random
        low = max(0.0, 1.0 - jitter_ratio)
        high = 1.0 + jitter_ratio
        estimated *= chooser.uniform(low, high)
    return max(min_tokens, min(max_tokens, int(round(estimated))))


def chunk_text_punctuation(
    text: str,
    chunk_len: int,
    min_chunk_len: Optional[int] = None,
) -> list[str]:
    sentences: list[list[str]] = []
    current_sentence: list[str] = []
    for token in list(text):
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in SPLIT_PUNCTUATION or token in CLOSING_MARKS)
        ):
            sentences[-1].append(token)
        else:
            current_sentence.append(token)
            if token in SPLIT_PUNCTUATION:
                is_abbreviation = False
                if token == ".":
                    temp_str = "".join(current_sentence).strip()
                    if temp_str and temp_str.split()[-1] in ABBREVIATIONS:
                        is_abbreviation = True
                if not is_abbreviation:
                    sentences.append(current_sentence)
                    current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)

    merged_chunks: list[list[str]] = []
    current_chunk: list[str] = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if current_chunk:
                merged_chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        merged_chunks.append(current_chunk)

    if min_chunk_len is None:
        final_chunks = merged_chunks
    else:
        first_short = bool(merged_chunks and len(merged_chunks[0]) < min_chunk_len)
        final_chunks: list[list[str]] = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_short:
                final_chunks[-1].extend(chunk)
            elif len(chunk) >= min_chunk_len or not final_chunks:
                final_chunks.append(chunk)
            else:
                final_chunks[-1].extend(chunk)
    return ["".join(chunk).strip() for chunk in final_chunks if "".join(chunk).strip()]


class RuleDurationEstimator:
    def __init__(self) -> None:
        self.weights = {
            "cjk": 3.0,
            "hangul": 2.5,
            "kana": 2.2,
            "indic": 1.8,
            "thai_lao": 1.5,
            "khmer_myanmar": 1.8,
            "arabic": 1.5,
            "hebrew": 1.5,
            "latin": 1.0,
            "cyrillic": 1.0,
            "greek": 1.0,
            "space": 0.2,
            "digit": 3.5,
            "punctuation": 0.5,
            "mark": 0.0,
            "default": 1.0,
        }
        self.ranges = [
            (0x02AF, "latin"),
            (0x03FF, "greek"),
            (0x052F, "cyrillic"),
            (0x05FF, "hebrew"),
            (0x08FF, "arabic"),
            (0x0DFF, "indic"),
            (0x0EFF, "thai_lao"),
            (0x109F, "khmer_myanmar"),
            (0x11FF, "hangul"),
            (0x309F, "kana"),
            (0x30FF, "kana"),
            (0x9FFF, "cjk"),
            (0xD7AF, "hangul"),
            (0xFAFF, "cjk"),
            (0xFFEF, "latin"),
        ]
        self.breakpoints = [item[0] for item in self.ranges]

    @lru_cache(maxsize=4096)
    def _get_char_weight(self, char: str) -> float:
        code = ord(char)
        if (65 <= code <= 90) or (97 <= code <= 122):
            return self.weights["latin"]
        if code == 32:
            return self.weights["space"]
        category = unicodedata.category(char)
        if category.startswith("M"):
            return self.weights["mark"]
        if category.startswith("P") or category.startswith("S"):
            return self.weights["punctuation"]
        if category.startswith("Z"):
            return self.weights["space"]
        if category.startswith("N"):
            return self.weights["digit"]
        idx = bisect.bisect_left(self.breakpoints, code)
        if idx < len(self.ranges):
            return self.weights.get(self.ranges[idx][1], self.weights["default"])
        return self.weights["cjk"] if code > 0x20000 else self.weights["default"]

    def calculate_total_weight(self, text: str) -> float:
        return sum(self._get_char_weight(c) for c in text)

    def estimate_duration(
        self,
        target_text: str,
        ref_text: str,
        ref_duration: float,
        low_threshold: Optional[float] = 50,
        boost_strength: float = 3,
    ) -> float:
        if ref_duration <= 0 or not ref_text:
            return 0.0
        ref_weight = self.calculate_total_weight(ref_text)
        if ref_weight == 0:
            return 0.0
        estimated_duration = self.calculate_total_weight(target_text) / (ref_weight / ref_duration)
        if low_threshold is not None and estimated_duration < low_threshold:
            alpha = 1.0 / boost_strength
            return low_threshold * math.pow(estimated_duration / low_threshold, alpha)
        return estimated_duration
