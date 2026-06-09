from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from viitorvoice.grpc_server.config import DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE
from viitorvoice.grpc_server.audio_utils import (
    fade_and_pad_audio,
    load_audio,
    load_audio_bytes,
    remove_silence,
    trim_long_audio,
)
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2


@dataclass
class LoadedAudio:
    encoder_audio: np.ndarray
    wave: np.ndarray
    sample_rate: int
    duration_sec: float
    temp_path: str | None = None


def load_audio_for_encoder(
    audio: common_pb2.AudioInput,
    *,
    preprocess_prompt: bool,
    can_trim_long_audio: bool = False,
) -> LoadedAudio:
    wave = _load_audio_payload(audio)
    if preprocess_prompt:
        if can_trim_long_audio:
            wave = trim_long_audio(wave, DEFAULT_SAMPLE_RATE, trim_threshold=20.0)
        wave = remove_silence(
            wave,
            DEFAULT_SAMPLE_RATE,
            mid_sil=200,
            lead_sil=100,
            trail_sil=200,
        )
        if wave.shape[-1] == 0:
            raise ValueError("Audio is empty after silence removal.")
    clip_size = int(wave.shape[-1] % DEFAULT_HOP_LENGTH)
    if clip_size > 0:
        wave = wave[:, :-clip_size]
    if wave.shape[-1] == 0:
        raise ValueError("Audio is empty after hop-length alignment.")
    wave = np.ascontiguousarray(wave, dtype=np.float32)
    encoder_audio = np.ascontiguousarray(wave[np.newaxis, :, :], dtype=np.float32)
    return LoadedAudio(
        encoder_audio=encoder_audio,
        wave=wave.reshape(-1).astype(np.float32, copy=False),
        sample_rate=DEFAULT_SAMPLE_RATE,
        duration_sec=float(wave.shape[-1]) / DEFAULT_SAMPLE_RATE,
    )


def audio_input_to_temp_wav(audio: common_pb2.AudioInput) -> str:
    if audio.audio_path:
        path = Path(audio.audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return str(path)
    if not audio.audio_bytes:
        raise ValueError("audio_bytes or audio_path is required.")
    wave = _load_audio_payload(audio)
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    sf.write(handle.name, wave.reshape(-1), DEFAULT_SAMPLE_RATE)
    return handle.name


def encode_wav_bytes(wave: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    arr = np.asarray(wave, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim == 2:
        arr = arr.T
    elif arr.ndim != 1:
        raise ValueError(f"Expected wave shape [T] or [C, T], got {arr.shape}.")
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pad_audio_input_silence(
    audio: common_pb2.AudioInput,
    *,
    padding_sec: float,
) -> common_pb2.AudioInput:
    wave = _load_audio_payload(audio)
    pad_samples = max(0, int(round(float(padding_sec) * DEFAULT_SAMPLE_RATE)))
    if pad_samples <= 0:
        result = common_pb2.AudioInput()
        result.CopyFrom(audio)
        return result
    silence = np.zeros((wave.shape[0], pad_samples), dtype=np.float32)
    padded = np.concatenate([silence, wave.astype(np.float32, copy=False), silence], axis=-1)
    return common_pb2.AudioInput(
        audio_bytes=encode_wav_bytes(padded, DEFAULT_SAMPLE_RATE),
        sample_rate=DEFAULT_SAMPLE_RATE,
        format=common_pb2.AUDIO_FORMAT_WAV,
    )


def trim_audio_result_edges(
    audio: common_pb2.AudioResult,
    *,
    trim_sec: float,
) -> common_pb2.AudioResult:
    trim_samples = max(0, int(round(float(trim_sec) * DEFAULT_SAMPLE_RATE)))
    if trim_samples <= 0:
        result = common_pb2.AudioResult()
        result.CopyFrom(audio)
        return result
    if not audio.audio_bytes:
        raise ValueError("audio.audio_bytes is required.")
    wave = load_audio_bytes(bytes(audio.audio_bytes), DEFAULT_SAMPLE_RATE)
    total_samples = int(wave.shape[-1])
    if total_samples > trim_samples * 2:
        wave = wave[:, trim_samples:-trim_samples]
    else:
        wave = wave[:, 0:0]
    return common_pb2.AudioResult(
        audio_bytes=encode_wav_bytes(wave, DEFAULT_SAMPLE_RATE),
        sample_rate=DEFAULT_SAMPLE_RATE,
        format=common_pb2.AUDIO_FORMAT_WAV,
        channels=1,
        duration_sec=float(wave.shape[-1]) / DEFAULT_SAMPLE_RATE,
    )


def post_process_wave(wave: np.ndarray, *, postprocess_output: bool) -> np.ndarray:
    arr = np.asarray(wave, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {arr.shape}.")
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected wave shape [C, T], got {arr.shape}.")
    if postprocess_output:
        arr = remove_silence(
            arr,
            DEFAULT_SAMPLE_RATE,
            mid_sil=500,
            lead_sil=100,
            trail_sil=100,
        )
    return fade_and_pad_audio(arr, sample_rate=DEFAULT_SAMPLE_RATE).reshape(-1).astype(np.float32)


def _load_audio_payload(audio: common_pb2.AudioInput) -> np.ndarray:
    if audio.audio_bytes:
        return load_audio_bytes(bytes(audio.audio_bytes), DEFAULT_SAMPLE_RATE)
    if audio.audio_path:
        path = Path(audio.audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return load_audio(str(path), DEFAULT_SAMPLE_RATE)
    raise ValueError("audio_bytes or audio_path is required.")
