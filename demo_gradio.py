from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def _clear_proxy_env() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        os.environ.pop(key, None)


_clear_proxy_env()

import gradio as gr  # noqa: E402
import requests  # noqa: E402


DEFAULT_BASE_URL = os.environ.get("VIITORVOICE_HTTP_BASE_URL", "http://127.0.0.1:7861")
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
LANGUAGE_CHOICES = ["en", "zh", "ja", "ko", "yue"]
OUTPUT_FORMAT = "wav"
REQUEST_TIMEOUT_SEC = 900
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_JA_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_KO_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_EMOTION_TAG_RE = re.compile(r"\s*<\|emotion-[^|>]+?\|>\s*")
_NVV_TAG_RE = re.compile(r"\s*<\|nv-[^|>]+?\|>\s*")

CLONE_EXAMPLES = [
    [
        "assets/emotion_prompt_en.wav",
        "<|emotion-angry|> How dare you do this to me. I trusted you again and again, and you shattered everything!",
        "en",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/emotion_prompt_en.wav",
        "<|emotion-happy|> This is incredible. I have waited so long for this moment, and I am bursting with joy!",
        "en",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/emotion_prompt_en.wav",
        "<|emotion-sad|> I cannot hold myself together anymore. every bit of hope is slipping away from me.",
        "en",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/emotion_prompt_en.wav",
        "<|emotion-fearful|> Do not come any closer. I am terrified, please, just stay away from me!",
        "en",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/emotion_prompt_zh.wav",
        "<|emotion-angry|> 你怎么可以这样对我？我一次又一次相信你，可你却把所有承诺都打碎了！",
        "zh",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/emotion_prompt_zh.wav",
        "<|emotion-happy|> 太好了，我等这一刻等了很久，现在心里充满了喜悦和期待！",
        "zh",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/emotion_prompt_zh.wav",
        "<|emotion-sad|> 我真的快撑不住了，最后一点希望也在慢慢离我远去。",
        "zh",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/emotion_prompt_zh.wav",
        "<|emotion-fearful|> 别再靠近了，我真的很害怕，求你就站在那里不要过来。",
        "zh",
        32,
        2.0,
        1.0,
        0.0,
        None,
    ],
    [
        "assets/nvv_prompt_en.wav",
        "I thought everything was finally ready <|nv-Laughter|> but then the lights went out again.",
        "en",
        32,
        2.0,
        0.0,
        0.5,
        None,
    ],
    [
        "assets/nvv_prompt_en.wav",
        "Well, I suppose we can try one more time <|nv-Sigh|> but please do not make me explain it again.",
        "en",
        32,
        2.0,
        0.0,
        0.5,
        None,
    ],
    [
        "assets/nvv_prompt_zh.wav",
        "我本来以为这次终于能顺利结束了<|nv-Laughter|>结果还是出现了新的问题。",
        "zh",
        32,
        2.0,
        0.0,
        0.5,
        None,
    ],
    [
        "assets/nvv_prompt_zh.wav",
        "好吧，我们就再试一次<|nv-Sigh|>但这次请你一定认真听我说完。",
        "zh",
        32,
        2.0,
        0.0,
        0.5,
        None,
    ],
]
CLONE_EXAMPLE_HEADERS = [
    "Example",
    "Audio",
    "Text Preview",
    "Language",
    "Key Params",
]
CLONE_EXAMPLE_NAMES = [
    "EN Angry",
    "EN Happy",
    "EN Sad",
    "EN Fearful",
    "ZH Angry",
    "ZH Happy",
    "ZH Sad",
    "ZH Fearful",
    "EN NVV Laughter",
    "EN NVV Sigh",
    "ZH NVV Laughter",
    "ZH NVV Sigh",
]

EDIT_EXAMPLES = [
    [
        "assets/edit_prompt_en.wav",
        "In short, we embark on a mission to make America great again for all Americans.",
        "In short, we embark on a mission to break language barriers for everyone.",
        "en",
        1.5,
        32,
        2.0,
        0.0,
        0.0,
    ],
    [
        "assets/edit_prompt_zh.wav",
        "当你明白如何为他人考虑,如何不再被冲动控制头脑,你的父亲应该就已经满足了。",
        "当你明白如何为自己考虑,如何不再被他人控制头脑,你的灵魂应该就已经满足了。",
        "zh",
        1.5,
        32,
        2.0,
        0.0,
        0.0,
    ],
]
EDIT_EXAMPLE_HEADERS = [
    "Example",
    "Audio",
    "Original Preview",
    "Edited Preview",
    "Language",
    "Key Params",
]
EDIT_EXAMPLE_NAMES = ["English Edit", "Chinese Edit"]


def _base_url(value: str) -> str:
    url = (value or DEFAULT_BASE_URL).strip().rstrip("/")
    if not url:
        raise gr.Error("HTTP Base URL is required.")
    return url


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _duration_count_text(text: str, language: str) -> str:
    lang = _normalize_language(language)
    nvv_placeholder = "字字" if lang in {"zh", "ja", "ko", "yue"} else " word "
    text = _EMOTION_TAG_RE.sub("", text)
    text = _NVV_TAG_RE.sub(nvv_placeholder, text)
    return text


def _estimated_duration(text: str, language: str) -> float:
    count_text = _duration_count_text(text, language)
    stripped = "".join(count_text.split())
    if not stripped:
        return 1.0
    if _normalize_language(language) in {"zh", "ja", "ko", "yue"}:
        duration = len(stripped) / 4.8
    else:
        words = len(count_text.split())
        duration = max(words / 2.6, len(stripped) / 16.0)
    return round(min(max(duration, 1.0), 30.0), 2)


def _normalize_language(language: str | None) -> str:
    text = (language or "").strip().lower().replace("_", "-")
    aliases = {
        "english": "en",
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "zho": "zh",
        "cmn": "zh",
        "cn": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "japanese": "ja",
        "jpn": "ja",
        "jp": "ja",
        "ja-jp": "ja",
        "korean": "ko",
        "kor": "ko",
        "kr": "ko",
        "ko-kr": "ko",
    }
    return aliases.get(text, text)


def _edit_language_and_granularity(language: str, original_text: str, edited_text: str) -> tuple[str, str]:
    lang = _normalize_language(language)
    combined_text = f"{original_text}\n{edited_text}"
    has_cjk = bool(_CJK_RE.search(combined_text))
    has_kana = bool(_JA_KANA_RE.search(combined_text))
    has_hangul = bool(_KO_HANGUL_RE.search(combined_text))

    if lang in {"zh", "yue", "ja", "ko"}:
        return lang, "char"
    if has_hangul:
        return "ko", "char"
    if has_kana:
        return "ja", "char"
    if has_cjk:
        return "zh", "char"
    if lang == "en" or _ASCII_WORD_RE.search(combined_text):
        return lang or "en", "word"
    return lang or "zh", "char"


def _audio_suffix(output_format: str) -> str:
    return ".flac" if output_format == "flac" else ".wav"


def _write_audio_response(response: requests.Response, output_format: str, prefix: str) -> tuple[str, dict[str, Any]]:
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=_audio_suffix(output_format), delete=False) as handle:
        handle.write(response.content)
        output_path = handle.name
    meta = {
        "http_status": response.status_code,
        "trace_id": response.headers.get("X-ViiTorVoice-Trace-Id", ""),
        "sample_rate": response.headers.get("X-ViiTorVoice-Sample-Rate", ""),
        "duration_sec": response.headers.get("X-ViiTorVoice-Duration-Sec", ""),
        "content_type": response.headers.get("Content-Type", ""),
        "bytes": len(response.content),
        "output_path": output_path,
    }
    return output_path, meta


def _selected_row_index(evt: gr.SelectData) -> int:
    index = evt.index
    if isinstance(index, (list, tuple)):
        if not index:
            raise gr.Error("Please select an example row.")
        return int(index[0])
    return int(index)


def _clone_example_display_rows() -> list[list[Any]]:
    rows = []
    for name, row in zip(CLONE_EXAMPLE_NAMES, CLONE_EXAMPLES, strict=True):
        target_duration = "auto" if row[7] is None else f"{row[7]}s"
        rows.append(
            [
                name,
                row[0],
                _preview(row[1]),
                row[2],
                (
                    f"steps={row[3]}, cfg={row[4]}, emotion={row[5]}, "
                    f"nvv={row[6]}, duration={target_duration}"
                ),
            ]
        )
    return rows


def _edit_example_display_rows() -> list[list[Any]]:
    rows = []
    for name, row in zip(EDIT_EXAMPLE_NAMES, EDIT_EXAMPLES, strict=True):
        rows.append(
            [
                name,
                row[0],
                _preview(row[1]),
                _preview(row[2]),
                row[3],
                f"expand={row[4]}, steps={row[5]}, cfg={row[6]}",
            ]
        )
    return rows


def _preview(text: str, max_chars: int = 58) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _load_clone_example(evt: gr.SelectData) -> list[Any]:
    row = CLONE_EXAMPLES[_selected_row_index(evt)]
    return list(row)


def _load_edit_example(evt: gr.SelectData) -> list[Any]:
    row = EDIT_EXAMPLES[_selected_row_index(evt)]
    return list(row)


def _post_audio(
    base_url: str,
    endpoint: str,
    data: dict[str, str],
    audio_field: str,
    audio_path: str | None,
    output_format: str = OUTPUT_FORMAT,
) -> tuple[str, dict[str, Any]]:
    if not audio_path:
        raise gr.Error("Please upload an audio file.")
    path = Path(audio_path)
    if not path.is_file():
        raise gr.Error(f"Audio file does not exist: {audio_path}")

    url = f"{_base_url(base_url)}{endpoint}"
    with path.open("rb") as audio_file:
        files = {audio_field: (path.name, audio_file, "application/octet-stream")}
        try:
            response = _session().post(url, data=data, files=files, timeout=REQUEST_TIMEOUT_SEC)
        except requests.RequestException as exc:
            raise gr.Error(f"Request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise gr.Error(f"HTTP {response.status_code}: {detail}")

    return _write_audio_response(response, output_format, "viitorvoice_")


def voice_clone(
    base_url: str,
    ref_audio: str | None,
    text: str,
    language: str,
    num_steps: int,
    cfg_scale: float,
    emotion_guidance_scale: float,
    nvv_guidance_scale: float,
    target_duration: float | None,
) -> tuple[str, dict[str, Any]]:
    if not text.strip():
        raise gr.Error("Synthesis text is required.")
    if not language.strip():
        raise gr.Error("Language is required.")

    duration = float(target_duration) if target_duration else _estimated_duration(text, language)
    data = {
        "text": text,
        "language": language,
        "num_steps": str(int(num_steps)),
        "cfg_scale": str(float(cfg_scale)),
        "emotion_guidance_scale": str(float(emotion_guidance_scale)),
        "nvv_guidance_scale": str(float(nvv_guidance_scale)),
        "duration": str(duration),
    }
    audio_path, meta = _post_audio(
        base_url=base_url,
        endpoint="/v1/voice-clone",
        data=data,
        audio_field="ref_audio",
        audio_path=ref_audio,
    )
    meta["estimated_or_selected_target_duration_sec"] = duration
    return audio_path, meta


def text_local_edit(
    base_url: str,
    source_audio: str | None,
    original_text: str,
    edited_text: str,
    language: str,
    expand_mask_ratio: float,
    num_steps: int,
    cfg_scale: float,
    emotion_guidance_scale: float,
    nvv_guidance_scale: float,
) -> tuple[str, dict[str, Any]]:
    if not original_text.strip():
        raise gr.Error("Original text is required.")
    if not edited_text.strip():
        raise gr.Error("Edited text is required.")
    if not language.strip():
        raise gr.Error("Language is required.")

    effective_language, align_granularity = _edit_language_and_granularity(language, original_text, edited_text)
    data = {
        "original_text": original_text,
        "edited_text": edited_text,
        "language": effective_language,
        "align_granularity": align_granularity,
        "expand_mask_ratio": str(float(expand_mask_ratio)),
        "num_steps": str(int(num_steps)),
        "cfg_scale": str(float(cfg_scale)),
        "emotion_guidance_scale": str(float(emotion_guidance_scale)),
        "nvv_guidance_scale": str(float(nvv_guidance_scale)),
    }
    return _post_audio(
        base_url=base_url,
        endpoint="/v1/text-local-edit",
        data=data,
        audio_field="source_audio",
        audio_path=source_audio,
    )


def build_demo(default_base_url: str) -> gr.Blocks:
    def run_voice_clone(
        ref_audio: str | None,
        text: str,
        language: str,
        num_steps: int,
        cfg_scale: float,
        emotion_guidance_scale: float,
        nvv_guidance_scale: float,
        target_duration: float | None,
    ) -> tuple[str, dict[str, Any]]:
        return voice_clone(
            default_base_url,
            ref_audio,
            text,
            language,
            num_steps,
            cfg_scale,
            emotion_guidance_scale,
            nvv_guidance_scale,
            target_duration,
        )

    def run_text_local_edit(
        source_audio: str | None,
        original_text: str,
        edited_text: str,
        language: str,
        expand_mask_ratio: float,
        num_steps: int,
        cfg_scale: float,
        emotion_guidance_scale: float,
        nvv_guidance_scale: float,
    ) -> tuple[str, dict[str, Any]]:
        return text_local_edit(
            default_base_url,
            source_audio,
            original_text,
            edited_text,
            language,
            expand_mask_ratio,
            num_steps,
            cfg_scale,
            emotion_guidance_scale,
            nvv_guidance_scale,
        )

    with gr.Blocks(title="ViiTorVoice Demo") as demo:
        gr.Markdown(
            """
# **ViiTorVoice**: Clone, Edit, and Generate Human-Like Speech

[![GitHub](https://img.shields.io/badge/GitHub-viitor--voice--nar-181717?logo=github)](https://github.com/viitor-ai/viitor-voice-nar)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ViiTorVoice--NAR-FFD21E?logo=huggingface)](https://huggingface.co/ZzWater/ViiTorVoice-NAR)
"""
        )

        with gr.Tabs():
            with gr.Tab("Voice Edit"):
                with gr.Row():
                    with gr.Column():
                        edit_source_audio = gr.Audio(
                            label="Source Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                            value=EDIT_EXAMPLES[0][0],
                        )
                        edit_original_text = gr.Textbox(
                            label="Original Text",
                            value=EDIT_EXAMPLES[0][1],
                            lines=3,
                        )
                        edit_edited_text = gr.Textbox(
                            label="Edited Text",
                            value=EDIT_EXAMPLES[0][2],
                            lines=3,
                        )
                        edit_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGE_CHOICES,
                            value=EDIT_EXAMPLES[0][3],
                            allow_custom_value=True,
                        )
                        with gr.Accordion("Optional Parameters", open=True):
                            edit_expand_mask_ratio = gr.Number(
                                label="expand_mask_ratio",
                                value=EDIT_EXAMPLES[0][4],
                                precision=2,
                            )
                            edit_num_steps = gr.Slider(
                                label="num_steps",
                                minimum=1,
                                maximum=64,
                                value=EDIT_EXAMPLES[0][5],
                                step=1,
                            )
                            edit_cfg_scale = gr.Slider(
                                label="cfg_scale",
                                minimum=0.0,
                                maximum=10.0,
                                value=EDIT_EXAMPLES[0][6],
                                step=0.1,
                            )
                            edit_emotion_guidance_scale = gr.Slider(
                                label="emotion_guidance_scale",
                                minimum=0.0,
                                maximum=12.0,
                                value=EDIT_EXAMPLES[0][7],
                                step=0.1,
                            )
                            edit_nvv_guidance_scale = gr.Slider(
                                label="nvv_guidance_scale",
                                minimum=0.0,
                                maximum=12.0,
                                value=EDIT_EXAMPLES[0][8],
                                step=0.1,
                            )
                        edit_button = gr.Button("Edit", variant="primary")
                    with gr.Column():
                        edit_output = gr.Audio(label="Edited Audio", type="filepath")
                        edit_status = gr.JSON(label="Result")

                edit_button.click(
                    run_text_local_edit,
                    inputs=[
                        edit_source_audio,
                        edit_original_text,
                        edit_edited_text,
                        edit_language,
                        edit_expand_mask_ratio,
                        edit_num_steps,
                        edit_cfg_scale,
                        edit_emotion_guidance_scale,
                        edit_nvv_guidance_scale,
                    ],
                    outputs=[edit_output, edit_status],
                )
                edit_examples = gr.Dataframe(
                    value=_edit_example_display_rows(),
                    headers=EDIT_EXAMPLE_HEADERS,
                    label="Examples",
                    interactive=False,
                    wrap=True,
                    type="array",
                )
                edit_examples.select(
                    _load_edit_example,
                    outputs=[
                        edit_source_audio,
                        edit_original_text,
                        edit_edited_text,
                        edit_language,
                        edit_expand_mask_ratio,
                        edit_num_steps,
                        edit_cfg_scale,
                        edit_emotion_guidance_scale,
                        edit_nvv_guidance_scale,
                    ],
                )

            with gr.Tab("Voice Clone"):
                with gr.Row():
                    with gr.Column():
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                            value=CLONE_EXAMPLES[0][0],
                        )
                        clone_text = gr.Textbox(
                            label="Synthesis Text",
                            value=CLONE_EXAMPLES[0][1],
                            lines=4,
                        )
                        clone_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGE_CHOICES,
                            value=CLONE_EXAMPLES[0][2],
                            allow_custom_value=True,
                        )
                        with gr.Accordion("Optional Parameters", open=True):
                            clone_num_steps = gr.Slider(
                                label="num_steps",
                                minimum=1,
                                maximum=64,
                                value=CLONE_EXAMPLES[0][3],
                                step=1,
                            )
                            clone_cfg_scale = gr.Slider(
                                label="cfg_scale",
                                minimum=0.0,
                                maximum=10.0,
                                value=CLONE_EXAMPLES[0][4],
                                step=0.1,
                            )
                            clone_emotion_guidance_scale = gr.Slider(
                                label="emotion_guidance_scale",
                                minimum=0.0,
                                maximum=12.0,
                                value=CLONE_EXAMPLES[0][5],
                                step=0.1,
                            )
                            clone_nvv_guidance_scale = gr.Slider(
                                label="nvv_guidance_scale",
                                minimum=0.0,
                                maximum=12.0,
                                value=CLONE_EXAMPLES[0][6],
                                step=0.1,
                            )
                            clone_target_duration = gr.Number(
                                label="Target Duration (seconds)",
                                value=CLONE_EXAMPLES[0][7],
                                precision=2,
                            )
                        clone_button = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        clone_output = gr.Audio(label="Generated Audio", type="filepath")
                        clone_status = gr.JSON(label="Result")

                clone_button.click(
                    run_voice_clone,
                    inputs=[
                        clone_ref_audio,
                        clone_text,
                        clone_language,
                        clone_num_steps,
                        clone_cfg_scale,
                        clone_emotion_guidance_scale,
                        clone_nvv_guidance_scale,
                        clone_target_duration,
                    ],
                    outputs=[clone_output, clone_status],
                )
                clone_examples = gr.Dataframe(
                    value=_clone_example_display_rows(),
                    headers=CLONE_EXAMPLE_HEADERS,
                    label="Examples",
                    interactive=False,
                    wrap=True,
                    type="array",
                )
                clone_examples.select(
                    _load_clone_example,
                    outputs=[
                        clone_ref_audio,
                        clone_text,
                        clone_language,
                        clone_num_steps,
                        clone_cfg_scale,
                        clone_emotion_guidance_scale,
                        clone_nvv_guidance_scale,
                        clone_target_duration,
                    ],
                )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for ViiTorVoice HTTP endpoints.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7862")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(args.base_url)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
