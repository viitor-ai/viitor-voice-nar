from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent, split_on_silence


def load_waveform(audio_path: str) -> tuple[np.ndarray, int]:
    try:
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        return data.T, sample_rate
    except Exception:
        import librosa

        data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data, sample_rate


def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    data, sample_rate = load_waveform(audio_path)
    return _to_mono_resampled(data, sample_rate, sampling_rate)


def load_audio_bytes(raw: bytes, sampling_rate: int) -> np.ndarray:
    buffer = io.BytesIO(raw)
    try:
        data, sample_rate = sf.read(buffer, dtype="float32", always_2d=True)
        data = data.T
    except Exception:
        import librosa

        buffer.seek(0)
        data, sample_rate = librosa.load(buffer, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
    return _to_mono_resampled(data, sample_rate, sampling_rate)


def numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def audiosegment_to_numpy(audio: AudioSegment) -> np.ndarray:
    data = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    if audio.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, audio.channels).T


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    wave = numpy_to_audiosegment(audio, sampling_rate)
    if mid_sil > 0:
        non_silent_segments = split_on_silence(
            wave,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        wave = AudioSegment.silent(duration=0)
        for segment in non_silent_segments:
            wave += segment
    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50)
    return audiosegment_to_numpy(wave)


def remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int = 100,
    trail_sil: int = 300,
    silence_threshold: float = -50,
) -> AudioSegment:
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[max(0, start_idx - lead_sil) :]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[max(0, start_idx - trail_sil) :]
    return audio.reverse()


def fade_and_pad_audio(
    audio: np.ndarray,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)
    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            processed[..., :k] *= np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]

    if pad_samples > 0:
        silence = np.zeros((processed.shape[0], pad_samples), dtype=processed.dtype)
        processed = np.concatenate([silence, processed, silence], axis=-1)
    return processed


def trim_long_audio(
    audio: np.ndarray,
    sampling_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 3.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    duration = audio.shape[-1] / sampling_rate
    if duration <= trim_threshold:
        return audio

    segment = numpy_to_audiosegment(audio, sampling_rate)
    nonsilent = detect_nonsilent(segment, min_silence_len=100, silence_thresh=-40, seek_step=10)
    if not nonsilent:
        return audio

    max_ms = int(max_duration * 1000)
    min_ms = int(min_duration * 1000)
    best_split = 0
    for start, end in nonsilent:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break
    if best_split < min_ms:
        best_split = min(max_ms, len(segment))
    return audiosegment_to_numpy(segment[:best_split])


def cross_fade_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]

    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n
    merged = chunks[0].copy()

    for chunk in chunks[1:]:
        parts = [merged]
        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            parts[-1][..., -fout_n:] *= np.linspace(1, 0, fout_n, dtype=np.float32)[np.newaxis, :]

        parts.append(np.zeros((chunks[0].shape[0], silence_n), dtype=np.float32))

        fade_in = chunk.copy()
        fin_n = min(fade_n, fade_in.shape[-1])
        if fin_n > 0:
            fade_in[..., :fin_n] *= np.linspace(0, 1, fin_n, dtype=np.float32)[np.newaxis, :]
        parts.append(fade_in)
        merged = np.concatenate(parts, axis=-1)
    return merged


def _to_mono_resampled(data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if orig_rate != target_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data),
            orig_freq=orig_rate,
            new_freq=target_rate,
        ).numpy()
    return data.astype(np.float32, copy=False)
