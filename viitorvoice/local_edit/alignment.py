from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


@dataclass
class AlignmentItem:
    index: int
    text: str
    start_time: float
    end_time: float
    start_char: int | None = None
    end_char: int | None = None
    kind: Literal["char", "word"] = "word"

    def to_row(self) -> list[Any]:
        return [
            self.index,
            self.text,
            round(self.start_time, 3),
            round(self.end_time, 3),
            self.start_char if self.start_char is not None else "",
            self.end_char if self.end_char is not None else "",
            self.kind,
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Qwen3ForcedAlignerWrapper:
    """Small local-model wrapper around Qwen3-ASR forced alignment."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        default_language: str = "zh",
        dtype: str | None = "auto",
        attn_implementation: str | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Qwen3-ASR aligner model path does not exist: {path}. "
                "Download the model locally first and pass the local directory."
            )
        try:
            from qwen_asr import Qwen3ForcedAligner
        except Exception:
            try:
                from qwen3_asr import Qwen3ForcedAligner
            except Exception as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError(
                    "qwen-asr is required for local edit alignment. Install a version "
                    "compatible with this environment, then retry. Original error: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        if Qwen3ForcedAligner is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "qwen-asr did not expose Qwen3ForcedAligner in this environment."
            )

        self.model_path = path.resolve()
        self.device = device
        self.default_language = default_language
        self.dtype = dtype or "auto"
        self.attn_implementation = (attn_implementation or "").strip() or None
        load_kwargs: dict[str, Any] = {
            "device_map": self._normalize_device_map(device),
        }
        normalized_dtype = _torch_dtype_from_name(self.dtype)
        if normalized_dtype is not None:
            load_kwargs["dtype"] = normalized_dtype
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            str(self.model_path),
            **load_kwargs,
        )
        model = getattr(self.aligner, "model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    @staticmethod
    def _normalize_device_map(device: str) -> str:
        value = (device or "cuda").strip()
        if value == "cuda":
            return "cuda:0"
        return value

    def align(
        self,
        audio: str | Path,
        text: str,
        language: str | None = None,
        granularity: Literal["auto", "word", "char"] = "auto",
    ) -> list[AlignmentItem]:
        audio_path = Path(audio).expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not text or not text.strip():
            raise ValueError("Original text is required for forced alignment.")
        result = self.aligner.align(
            audio=str(audio_path),
            text=text,
            language=language or self.default_language,
        )
        items = normalize_alignment_result(result, text)
        if granularity == "char":
            items = expand_items_to_chars(items, text)
        elif granularity == "word":
            items = [item for item in items if item.text.strip()]
            for idx, item in enumerate(items):
                item.index = idx
                item.kind = "word"
        return items


def normalize_alignment_result(result: Any, original_text: str) -> list[AlignmentItem]:
    raw_items = list(_iter_alignment_dicts(result))
    items: list[AlignmentItem] = []
    cursor = 0
    for raw in raw_items:
        token_text = str(_pick(raw, "text", "word", "char", "token", "label", default="")).strip()
        start = _pick(raw, "start_time", "start", "begin", "ts_begin", default=None)
        end = _pick(raw, "end_time", "end", "finish", "ts_end", default=None)
        if token_text == "" or start is None or end is None:
            continue
        start_time = float(start)
        end_time = float(end)
        if end_time <= start_time:
            continue
        start_char, end_char = _find_text_span(original_text, token_text, cursor)
        if end_char is not None:
            cursor = end_char
        items.append(
            AlignmentItem(
                index=len(items),
                text=token_text,
                start_time=start_time,
                end_time=end_time,
                start_char=start_char,
                end_char=end_char,
                kind="char" if len(token_text) == 1 else "word",
            )
        )
    if not items:
        raise ValueError("Forced aligner returned no usable word/char timestamps.")
    return items


def expand_items_to_chars(
    items: list[AlignmentItem],
    original_text: str,
) -> list[AlignmentItem]:
    expanded: list[AlignmentItem] = []
    for item in items:
        chars = [char for char in item.text if not char.isspace()]
        if len(chars) <= 1:
            item.index = len(expanded)
            item.kind = "char"
            expanded.append(item)
            continue
        duration = item.end_time - item.start_time
        char_start = item.start_char
        for offset, char in enumerate(chars):
            start_time = item.start_time + duration * offset / len(chars)
            end_time = item.start_time + duration * (offset + 1) / len(chars)
            start_char = None if char_start is None else char_start + offset
            expanded.append(
                AlignmentItem(
                    index=len(expanded),
                    text=char,
                    start_time=start_time,
                    end_time=end_time,
                    start_char=start_char,
                    end_char=None if start_char is None else start_char + 1,
                    kind="char",
                )
            )
    return expanded


def _iter_alignment_dicts(result: Any) -> Iterable[Any]:
    if result is None:
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            yield from _iter_alignment_dicts(item)
        return
    if isinstance(result, dict):
        if any(key in result for key in ("text", "word", "char", "token", "label")) and any(
            key in result for key in ("start_time", "start", "begin", "ts_begin")
        ):
            yield result
            return
        for key in ("segments", "words", "chars", "characters", "timestamps", "items", "result"):
            if key in result:
                yield from _iter_alignment_dicts(result[key])
        return
    if hasattr(result, "__dict__"):
        yield from _iter_alignment_dicts(vars(result))


def _pick(raw: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(raw, dict) and key in raw:
            return raw[key]
        if hasattr(raw, key):
            return getattr(raw, key)
    return default


def _torch_dtype_from_name(value: str | None) -> Any:
    text = (value or "auto").strip().lower().replace("-", "").replace("_", "")
    if text in {"", "none", "default"}:
        return None
    if text == "auto":
        return "auto"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - qwen-asr runtime dependency path
        raise RuntimeError(f"Aligner dtype {value!r} requires torch to be installed.") from exc
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported aligner dtype {value!r}; use auto, bf16, fp16, fp32, or default.")


def _find_text_span(
    original_text: str,
    token_text: str,
    cursor: int,
) -> tuple[int | None, int | None]:
    if not token_text:
        return None, None
    pos = original_text.find(token_text, max(0, cursor))
    if pos < 0:
        pos = original_text.find(token_text)
    if pos < 0:
        return None, None
    return pos, pos + len(token_text)
