# ViiTorVoice HTTP API Usage

This document describes the end-to-end HTTP APIs exposed by `viitorvoice/grpc_server/http/server.py`.

Local deployment URL:

```text
http://127.0.0.1:7861
```

Start the local service from the root of this standalone repository:

```bash
./run_grpc_v2.sh start all
```

## API List

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/v1/voice-clone` | Voice cloning: input a prompt codebook or raw prompt audio plus target text, and return synthesized audio |
| `POST` | `/v1/text-local-edit` | Speech editing: input source audio, original text, and edited text, and return edited audio |

All `POST` APIs use `multipart/form-data`.

## Common Responses

On success, audio APIs return audio bytes directly. The response headers include:

| Header | Meaning | Example |
| --- | --- | --- |
| `Content-Type` | Output audio format | `audio/wav` |
| `X-ViiTorVoice-Trace-Id` | Server trace id for log debugging | `c3c145a84a684e4f87856244503dcf6f` |
| `X-ViiTorVoice-Sample-Rate` | Output sample rate | `24000` |
| `X-ViiTorVoice-Duration-Sec` | Output audio duration in seconds | `3.840000` |

On failure, the API returns JSON:

```json
{
  "detail": "INVALID_ARGUMENT: error message"
}
```

Common HTTP status codes:

| Status Code | Meaning |
| --- | --- |
| `400` | Invalid parameters, missing audio input, or unsupported format |
| `503` | Backend gRPC service unavailable |
| `504` | Request timeout |

## Audio Input Methods

There are three audio input methods. For the same audio field, choose exactly one.

### 1. Upload an audio file

Voice cloning uses `ref_audio`:

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

Speech editing uses `source_audio`:

```bash
curl -X POST "$BASE_URL/v1/text-local-edit" \
  -F 'source_audio=@source.wav' \
  -F 'original_text=I like all americans.' \
  -F 'edited_text=I like all chinese.' \
  --output edited.wav
```

### 2. Pass a server-local path

The path must be accessible on the machine running the HTTP service, not on the caller machine.

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio_path=/data/audio/prompt.wav' \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

```bash
curl -X POST "$BASE_URL/v1/text-local-edit" \
  -F 'source_audio_path=/data/audio/source.wav' \
  -F 'original_text=I like all americans.' \
  -F 'edited_text=I like all chinese.' \
  --output edited.wav
```

### 3. Pass base64 audio

```bash
AUDIO_B64="$(base64 -w 0 prompt.wav)"

curl -X POST "$BASE_URL/v1/voice-clone" \
  -F "ref_audio_base64=$AUDIO_B64" \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

```bash
AUDIO_B64="$(base64 -w 0 source.wav)"

curl -X POST "$BASE_URL/v1/text-local-edit" \
  -F "source_audio_base64=$AUDIO_B64" \
  -F 'original_text=I like all americans.' \
  -F 'edited_text=I like all chinese.' \
  --output edited.wav
```

## Codebook Input Methods

Currently, only `/v1/voice-clone` supports passing a prompt audio codebook directly.

The codebook JSON supports two structures:

```json
{
  "values": [1, 2, 3],
  "shape": [12, 305]
}
```

or:

```json
{
  "audio_codebook": {
    "values": [1, 2, 3],
    "shape": [12, 305]
  }
}
```

Fields:

| Field | Meaning | Example |
| --- | --- | --- |
| `values` | Flattened int64 token array | `[100, 23, 54, ...]` |
| `shape` | Tensor shape, usually `[12, T]` | `[12, 305]` |

### 1. Form string

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio_codebook={"values":[1,2,3],"shape":[12,305]}' \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

### 2. Upload a JSON file

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio_codebook_file=@prompt_codebook.json;type=application/json' \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

`ref_audio_codebook` and `ref_audio_codebook_file` are mutually exclusive. When using a codebook, avoid also passing `ref_audio`, `ref_audio_path`, or `ref_audio_base64`.

## Parameters: `GET /health`

No request parameters.

```bash
BASE_URL="http://127.0.0.1:7861"
curl "$BASE_URL/health"
```

Example response:

```json
{
  "state": 2,
  "message": "ready",
  "version": "viitorvoice.grpc_server",
  "active_backends": ["encoder", "llm", "decoder"],
  "trace_id": "..."
}
```

## Parameters: `POST /v1/voice-clone`

### Required Parameters

| Parameter | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
| `text` | string | None | Text to synthesize | `-F 'text=hello from ViiTorVoice'` |

The prompt input must provide one of the following:

| Input Type | Parameter | Example |
| --- | --- | --- |
| Uploaded audio | `ref_audio` | `-F 'ref_audio=@prompt.wav'` |
| Server audio path | `ref_audio_path` | `-F 'ref_audio_path=/data/audio/prompt.wav'` |
| Base64 audio | `ref_audio_base64` | `-F "ref_audio_base64=$AUDIO_B64"` |
| Codebook JSON string | `ref_audio_codebook` | `-F 'ref_audio_codebook={"values":[...],"shape":[12,305]}'` |
| Codebook JSON file | `ref_audio_codebook_file` | `-F 'ref_audio_codebook_file=@prompt_codebook.json;type=application/json'` |

### Optional Parameters

| Parameter | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
| `language` | string | `en` | Text language. Common values: `en`, `zh`, `ja`, `ko`, `yue` | `-F 'language=ja'` |
| `ref_text` | string | Empty string | Transcript for the prompt audio; when empty, the model uses the no-ref-text logic | `-F 'ref_text=this is the prompt transcript'` |
| `instruct` | string | Empty string | Additional style or instruction text | `-F 'instruct=speak calmly'` |
| `allow_missing_ref_text` | bool | `true` | Whether `ref_text` may be omitted | `-F 'allow_missing_ref_text=true'` |
| `ref_text_mask_len` | int | `10` | Reference text mask length for no-ref-text mode | `-F 'ref_text_mask_len=10'` |
| `sample_rate` | int | `0` | Input audio sample rate; `0` means parse it from the file | `-F 'sample_rate=24000'` |
| `input_format` | string | `wav` | Input audio format: `wav`, `flac`, `pcm_s16le` | `-F 'input_format=wav'` |
| `output_format` | string | `wav` | Output audio format: `wav`, `flac`, `pcm_s16le` | `-F 'output_format=wav'` |
| `num_steps` | int | `32` | LLM generation steps | `-F 'num_steps=32'` |
| `cfg_scale` | float | `2.0` | classifier-free guidance scale | `-F 'cfg_scale=2.0'` |
| `emotion_guidance_scale` | float | `0.0` | leading emotion tag CFG scale; ignored when no `<|emotion-xxx|>` tag is present | `-F 'emotion_guidance_scale=6.0'` |
| `nvv_guidance_scale` | float | `0.0` | NVV tag CFG scale; ignored when no NVV tag is present | `-F 'nvv_guidance_scale=2.0'` |
| `position_temperature` | float | `1.0` | Position sampling temperature | `-F 'position_temperature=1.0'` |
| `class_temperature` | float | `0.0` | Class sampling temperature | `-F 'class_temperature=0.0'` |
| `t_shift` | float | `0.1` | diffusion/flow time-shift parameter | `-F 't_shift=0.1'` |
| `layer_penalty_factor` | float | `5.0` | Multi-codebook-layer penalty factor | `-F 'layer_penalty_factor=5.0'` |
| `duration` | float | Not set | Target output duration in seconds | `-F 'duration=3.5'` |
| `speed` | float | Not set | Speech speed control; larger values are usually faster | `-F 'speed=1.05'` |
| `preprocess_prompt` | bool | `true` | Whether to preprocess the prompt audio | `-F 'preprocess_prompt=true'` |
| `postprocess_output` | bool | `true` | Whether to postprocess the output audio | `-F 'postprocess_output=true'` |
| `timeout_sec` | int | `600` | HTTP-to-gRPC call timeout in seconds | `-F 'timeout_sec=600'` |

### Full curl example: prompt audio

```bash
BASE_URL="http://127.0.0.1:7861"

curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=hello from ViiTorVoice' \
  -F 'language=en' \
  -F 'ref_text=this is my prompt voice' \
  -F 'allow_missing_ref_text=true' \
  -F 'ref_text_mask_len=10' \
  -F 'input_format=wav' \
  -F 'output_format=wav' \
  -F 'num_steps=32' \
  -F 'cfg_scale=2.0' \
  -F 'position_temperature=1.0' \
  -F 'class_temperature=0.0' \
  -F 't_shift=0.1' \
  -F 'layer_penalty_factor=5.0' \
  -F 'preprocess_prompt=true' \
  -F 'postprocess_output=true' \
  -F 'timeout_sec=600' \
  --output clone.wav
```

### Full curl example: prompt codebook

```bash
BASE_URL="http://127.0.0.1:7861"

curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio_codebook_file=@prompt_codebook.json;type=application/json' \
  -F 'text=hello from ViiTorVoice' \
  -F 'language=en' \
  -F 'output_format=wav' \
  -F 'num_steps=32' \
  --output clone.wav
```

## Parameters: `POST /v1/text-local-edit`

### Required Parameters

| Parameter | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
| `original_text` | string | None | Text corresponding to the source audio. Alignment and diff use this original text | `-F 'original_text=I like all americans.'` |
| `edited_text` | string | None | Complete edited text | `-F 'edited_text=I like all chinese.'` |

The source audio must provide one of the following:

| Input Type | Parameter | Example |
| --- | --- | --- |
| Uploaded audio | `source_audio` | `-F 'source_audio=@source.wav'` |
| Server audio path | `source_audio_path` | `-F 'source_audio_path=/data/audio/source.wav'` |
| Base64 audio | `source_audio_base64` | `-F "source_audio_base64=$AUDIO_B64"` |

### Optional Parameters

| Parameter | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
| `language` | string | `en` | Text language. `zh`, `ja`, and `ko` default to character-level alignment; English defaults to word-level alignment | `-F 'language=en'` |
| `sample_rate` | int | `0` | Input audio sample rate; `0` means parse it from the file | `-F 'sample_rate=24000'` |
| `input_format` | string | `wav` | Input audio format: `wav`, `flac`, `pcm_s16le` | `-F 'input_format=wav'` |
| `output_format` | string | `wav` | Output audio format: `wav`, `flac`, `pcm_s16le` | `-F 'output_format=wav'` |
| `align_granularity` | string | Empty string | Force alignment granularity: empty for auto, or `word` / `char` / `character` | `-F 'align_granularity=word'` |
| `padding_ms` | float | Not set | Add padding to both sides of the audio span found by text diff, in milliseconds | `-F 'padding_ms=250'` |
| `expand_mask_ratio` | float | Not set | Expand the final edit mask around the original mask center; `1.0` means no expansion | `-F 'expand_mask_ratio=1.5'` |
| `length_mode` | string | Not set | Replacement segment length strategy: `auto`, `manual_seconds`, `manual_frames` | `-F 'length_mode=auto'` |
| `manual_duration` | float | Not set | Target replacement segment duration in seconds when `length_mode=manual_seconds` | `-F 'length_mode=manual_seconds' -F 'manual_duration=0.8'` |
| `manual_frames` | int | Not set | Target replacement segment codec frame count when `length_mode=manual_frames` | `-F 'length_mode=manual_frames' -F 'manual_frames=20'` |
| `length_scale` | float | Not set | Scale the automatically estimated length when `length_mode=auto` | `-F 'length_mode=auto' -F 'length_scale=1.2'` |
| `min_mask_frames` | int | `6` | Minimum edit mask frame count | `-F 'min_mask_frames=6'` |
| `edit_context_frames` | int | `40` | Model context frames kept on both sides of the edit region | `-F 'edit_context_frames=40'` |
| `edit_ref_context_frames` | int | `120` | Context frames used as voice reference | `-F 'edit_ref_context_frames=120'` |
| `preprocess_source_audio` | bool | Not set | Whether to preprocess source audio; when unset, the service default is used | `-F 'preprocess_source_audio=true'` |
| `postprocess_output` | bool | `true` | Whether to postprocess output audio | `-F 'postprocess_output=true'` |
| `num_steps` | int | `32` | LLM generation steps | `-F 'num_steps=32'` |
| `cfg_scale` | float | `2.0` | classifier-free guidance scale | `-F 'cfg_scale=2.0'` |
| `emotion_guidance_scale` | float | `0.0` | leading emotion tag CFG scale; ignored when no `<|emotion-xxx|>` tag is present | `-F 'emotion_guidance_scale=6.0'` |
| `nvv_guidance_scale` | float | `0.0` | NVV tag CFG scale; ignored when no NVV tag is present | `-F 'nvv_guidance_scale=2.0'` |
| `position_temperature` | float | `1.0` | Position sampling temperature | `-F 'position_temperature=1.0'` |
| `class_temperature` | float | `0.0` | Class sampling temperature | `-F 'class_temperature=0.0'` |
| `t_shift` | float | `0.1` | diffusion/flow time-shift parameter | `-F 't_shift=0.1'` |
| `layer_penalty_factor` | float | `5.0` | Multi-codebook-layer penalty factor | `-F 'layer_penalty_factor=5.0'` |
| `timeout_sec` | int | `900` | HTTP-to-gRPC call timeout in seconds | `-F 'timeout_sec=900'` |

### Full curl example

```bash
BASE_URL="http://127.0.0.1:7861"

curl -X POST "$BASE_URL/v1/text-local-edit" \
  -F 'source_audio=@source.wav' \
  -F 'original_text=I like all americans.' \
  -F 'edited_text=I like all chinese.' \
  -F 'language=en' \
  -F 'align_granularity=word' \
  -F 'padding_ms=250' \
  -F 'expand_mask_ratio=1.5' \
  -F 'length_mode=auto' \
  -F 'length_scale=1.1' \
  -F 'min_mask_frames=6' \
  -F 'edit_context_frames=40' \
  -F 'edit_ref_context_frames=120' \
  -F 'postprocess_output=true' \
  -F 'output_format=wav' \
  -F 'num_steps=32' \
  -F 'cfg_scale=2.0' \
  -F 'position_temperature=1.0' \
  -F 'class_temperature=0.0' \
  -F 't_shift=0.1' \
  -F 'layer_penalty_factor=5.0' \
  -F 'timeout_sec=900' \
  --output edited.wav
```

## Python Examples

If the client environment does not have `requests`, install it in the current virtual environment:

```bash
uv pip install requests
```

### Python: voice cloning with uploaded prompt audio

```python
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:7861"

with open("prompt.wav", "rb") as audio_file:
    response = requests.post(
        f"{BASE_URL}/v1/voice-clone",
        files={"ref_audio": ("prompt.wav", audio_file, "audio/wav")},
        data={
            "text": "hello from ViiTorVoice",
            "language": "en",
            "ref_text": "this is my prompt voice",
            "allow_missing_ref_text": "true",
            "ref_text_mask_len": "10",
            "input_format": "wav",
            "output_format": "wav",
            "num_steps": "32",
            "cfg_scale": "2.0",
            "position_temperature": "1.0",
            "class_temperature": "0.0",
            "t_shift": "0.1",
            "layer_penalty_factor": "5.0",
            "preprocess_prompt": "true",
            "postprocess_output": "true",
            "timeout_sec": "600",
        },
        timeout=620,
    )

response.raise_for_status()
Path("clone.wav").write_bytes(response.content)
print("trace_id:", response.headers.get("X-ViiTorVoice-Trace-Id"))
print("duration:", response.headers.get("X-ViiTorVoice-Duration-Sec"))
```

### Python: voice cloning with prompt codebook

```python
import json
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:7861"

codebook = {
    "values": [1, 2, 3],
    "shape": [12, 305],
}

response = requests.post(
    f"{BASE_URL}/v1/voice-clone",
    data={
        "ref_audio_codebook": json.dumps(codebook),
        "text": "hello from ViiTorVoice",
        "language": "en",
        "output_format": "wav",
        "num_steps": "32",
        "timeout_sec": "600",
    },
    timeout=620,
)

response.raise_for_status()
Path("clone_from_codebook.wav").write_bytes(response.content)
print("trace_id:", response.headers.get("X-ViiTorVoice-Trace-Id"))
```

### Python: voice cloning with base64 audio

```python
import base64
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:7861"

audio_base64 = base64.b64encode(Path("prompt.wav").read_bytes()).decode("ascii")

response = requests.post(
    f"{BASE_URL}/v1/voice-clone",
    data={
        "ref_audio_base64": audio_base64,
        "text": "hello from ViiTorVoice",
        "language": "en",
        "input_format": "wav",
        "output_format": "wav",
    },
    timeout=620,
)

response.raise_for_status()
Path("clone_from_base64.wav").write_bytes(response.content)
```

### Python: speech editing

```python
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:7861"

with open("source.wav", "rb") as audio_file:
    response = requests.post(
        f"{BASE_URL}/v1/text-local-edit",
        files={"source_audio": ("source.wav", audio_file, "audio/wav")},
        data={
            "original_text": "I like all americans.",
            "edited_text": "I like all chinese.",
            "language": "en",
            "align_granularity": "word",
            "padding_ms": "250",
            "expand_mask_ratio": "1.5",
            "length_mode": "auto",
            "length_scale": "1.1",
            "min_mask_frames": "6",
            "edit_context_frames": "40",
            "edit_ref_context_frames": "120",
            "postprocess_output": "true",
            "input_format": "wav",
            "output_format": "wav",
            "num_steps": "32",
            "cfg_scale": "2.0",
            "position_temperature": "1.0",
            "class_temperature": "0.0",
            "t_shift": "0.1",
            "layer_penalty_factor": "5.0",
            "timeout_sec": "900",
        },
        timeout=920,
    )

response.raise_for_status()
Path("edited.wav").write_bytes(response.content)
print("trace_id:", response.headers.get("X-ViiTorVoice-Trace-Id"))
```

## Direct gRPC Documentation

To call the orchestrator gRPC service directly, see [grpc_usage.md](grpc_usage.md).

## Calling Notes

1. `input_format` / `output_format` currently support `wav`, `flac`, and `pcm_s16le`. Regular calls should use `wav`.
2. When `sample_rate=0`, the service tries to parse the sample rate from the audio file. For raw `pcm_s16le`, provide `sample_rate` explicitly.
3. `/v1/text-local-edit` currently generates the edit span automatically from a diff between the original text and edited text. If the literal text is unchanged but you still want resynthesis, a future explicit edit span API is needed.
4. Japanese, Chinese, and Korean default to character-level alignment; English defaults to word-level alignment. `align_granularity` can override the automatic choice.
5. Japanese text uses the original text during alignment and diff. `<|ja-char|>` preprocessing only happens before model tokenization.
6. `expand_mask_ratio` only expands the edit audio mask; it does not change replacement text or diff results.
