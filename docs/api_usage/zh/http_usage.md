# ViiTorVoice HTTP 接口使用说明

本文档说明 `viitorvoice/grpc_server/http/server.py` 暴露的端到端 HTTP 接口。

默认示例使用公网地址：

```text
http://mrwaterzhou.uicp.io:38179
```

本地部署时可替换为：

```text
http://127.0.0.1:7861
```

本地服务从当前独立仓库根目录启动：

```bash
./run_grpc_v2.sh start all
```

## 接口列表

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/v1/voice-clone` | 语音克隆：输入提示音的 codebook 或原始音频 + 待合成文本，输出合成音频 |
| `POST` | `/v1/text-local-edit` | 语音编辑：输入原始音频、原始文本、修改后文本，输出修改后的音频 |

所有 `POST` 接口均使用 `multipart/form-data`。

## 公共返回

成功时，音频接口直接返回音频 bytes，响应头包含：

| Header | 含义 | 示例 |
| --- | --- | --- |
| `Content-Type` | 输出音频格式 | `audio/wav` |
| `X-ViiTorVoice-Trace-Id` | 服务端 trace id，排查日志时使用 | `c3c145a84a684e4f87856244503dcf6f` |
| `X-ViiTorVoice-Sample-Rate` | 输出采样率 | `24000` |
| `X-ViiTorVoice-Duration-Sec` | 输出音频时长，单位秒 | `3.840000` |

失败时返回 JSON：

```json
{
  "detail": "INVALID_ARGUMENT: error message"
}
```

常见 HTTP 状态码：

| 状态码 | 含义 |
| --- | --- |
| `400` | 参数错误、音频输入缺失、格式不支持 |
| `503` | 后端 gRPC 服务不可用 |
| `504` | 请求超时 |

## 音频输入方式

音频输入有三种方式。对同一个音频字段必须三选一。

### 1. 上传音频文件

语音克隆使用 `ref_audio`：

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

语音编辑使用 `source_audio`：

```bash
curl -X POST "$BASE_URL/v1/text-local-edit" \
  -F 'source_audio=@source.wav' \
  -F 'original_text=I like all americans.' \
  -F 'edited_text=I like all chinese.' \
  --output edited.wav
```

### 2. 传服务端本地路径

路径是 HTTP 服务所在机器可访问的路径，不是调用方机器路径。

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

### 3. 传 base64 音频

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

## Codebook 输入方式

目前只有 `/v1/voice-clone` 支持直接传提示音 codebook。

codebook JSON 支持两种结构：

```json
{
  "values": [1, 2, 3],
  "shape": [12, 305]
}
```

或：

```json
{
  "audio_codebook": {
    "values": [1, 2, 3],
    "shape": [12, 305]
  }
}
```

其中：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `values` | int64 token 扁平数组 | `[100, 23, 54, ...]` |
| `shape` | tensor 形状，通常是 `[12, T]` | `[12, 305]` |

### 1. 表单字符串

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio_codebook={"values":[1,2,3],"shape":[12,305]}' \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

### 2. 上传 JSON 文件

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio_codebook_file=@prompt_codebook.json;type=application/json' \
  -F 'text=hello from ViiTorVoice' \
  --output clone.wav
```

`ref_audio_codebook` 与 `ref_audio_codebook_file` 只能二选一。使用 codebook 时建议不要再同时传 `ref_audio`、`ref_audio_path` 或 `ref_audio_base64`。

## 参数：`GET /health`

无请求参数。

```bash
BASE_URL="http://mrwaterzhou.uicp.io:38179"
curl "$BASE_URL/health"
```

返回示例：

```json
{
  "state": 2,
  "message": "ready",
  "version": "viitorvoice.grpc_server",
  "active_backends": ["encoder", "llm", "decoder"],
  "trace_id": "..."
}
```

## 参数：`POST /v1/voice-clone`

### 必填参数

| 参数 | 类型 | 默认值 | 说明 | 调用示例 |
| --- | --- | --- | --- | --- |
| `text` | string | 无 | 待合成文本 | `-F 'text=hello from ViiTorVoice'` |

提示音输入必须满足以下之一：

| 输入类型 | 参数 | 调用示例 |
| --- | --- | --- |
| 上传音频 | `ref_audio` | `-F 'ref_audio=@prompt.wav'` |
| 服务端音频路径 | `ref_audio_path` | `-F 'ref_audio_path=/data/audio/prompt.wav'` |
| base64 音频 | `ref_audio_base64` | `-F "ref_audio_base64=$AUDIO_B64"` |
| codebook JSON 字符串 | `ref_audio_codebook` | `-F 'ref_audio_codebook={"values":[...],"shape":[12,305]}'` |
| codebook JSON 文件 | `ref_audio_codebook_file` | `-F 'ref_audio_codebook_file=@prompt_codebook.json;type=application/json'` |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 | 调用示例 |
| --- | --- | --- | --- | --- |
| `language` | string | `en` | 文本语言。常用：`en`、`zh`、`ja`、`ko`、`yue` | `-F 'language=ja'` |
| `ref_text` | string | 空字符串 | 提示音对应文本；空时由模型按 no-ref-text 逻辑处理 | `-F 'ref_text=this is the prompt transcript'` |
| `instruct` | string | 空字符串 | 额外风格/指令文本 | `-F 'instruct=speak calmly'` |
| `allow_missing_ref_text` | bool | `true` | 是否允许不传 `ref_text` | `-F 'allow_missing_ref_text=true'` |
| `ref_text_mask_len` | int | `10` | no-ref-text 时参考文本 mask 长度 | `-F 'ref_text_mask_len=10'` |
| `sample_rate` | int | `0` | 输入音频采样率；`0` 表示从文件解析 | `-F 'sample_rate=24000'` |
| `input_format` | string | `wav` | 输入音频格式：`wav`、`flac`、`pcm_s16le` | `-F 'input_format=wav'` |
| `output_format` | string | `wav` | 输出音频格式：`wav`、`flac`、`pcm_s16le` | `-F 'output_format=wav'` |
| `num_steps` | int | `8` | LLM 生成步数 | `-F 'num_steps=8'` |
| `cfg_scale` | float | `0.0` | classifier-free guidance scale | `-F 'cfg_scale=0.0'` |
| `emotion_guidance_scale` | float | `0.0` | leading emotion tag CFG scale; ignored when no `<|emotion-xxx|>` tag is present | `-F 'emotion_guidance_scale=6.0'` |
| `nvv_guidance_scale` | float | `0.0` | NVV tag CFG scale; ignored when no NVV tag is present | `-F 'nvv_guidance_scale=2.0'` |
| `position_temperature` | float | `1.0` | 位置采样温度 | `-F 'position_temperature=1.0'` |
| `class_temperature` | float | `0.0` | 类别采样温度 | `-F 'class_temperature=0.0'` |
| `t_shift` | float | `0.1` | diffusion/flow 时间偏移参数 | `-F 't_shift=0.1'` |
| `layer_penalty_factor` | float | `5.0` | 多 codebook 层惩罚因子 | `-F 'layer_penalty_factor=5.0'` |
| `duration` | float | 不设置 | 指定输出目标时长，单位秒 | `-F 'duration=3.5'` |
| `speed` | float | 不设置 | 语速控制；值越大通常越快 | `-F 'speed=1.05'` |
| `preprocess_prompt` | bool | `true` | 是否对提示音做预处理 | `-F 'preprocess_prompt=true'` |
| `postprocess_output` | bool | `true` | 是否对输出音频做后处理 | `-F 'postprocess_output=true'` |
| `timeout_sec` | int | `600` | HTTP 到 gRPC 调用超时，单位秒 | `-F 'timeout_sec=600'` |

### Curl 完整示例：提示音频

```bash
BASE_URL="http://mrwaterzhou.uicp.io:38179"

curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=hello from ViiTorVoice' \
  -F 'language=en' \
  -F 'ref_text=this is my prompt voice' \
  -F 'allow_missing_ref_text=true' \
  -F 'ref_text_mask_len=10' \
  -F 'input_format=wav' \
  -F 'output_format=wav' \
  -F 'num_steps=8' \
  -F 'cfg_scale=0.0' \
  -F 'position_temperature=1.0' \
  -F 'class_temperature=0.0' \
  -F 't_shift=0.1' \
  -F 'layer_penalty_factor=5.0' \
  -F 'preprocess_prompt=true' \
  -F 'postprocess_output=true' \
  -F 'timeout_sec=600' \
  --output clone.wav
```

### Curl 完整示例：提示 codebook

```bash
BASE_URL="http://mrwaterzhou.uicp.io:38179"

curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio_codebook_file=@prompt_codebook.json;type=application/json' \
  -F 'text=hello from ViiTorVoice' \
  -F 'language=en' \
  -F 'output_format=wav' \
  -F 'num_steps=8' \
  --output clone.wav
```

## 参数：`POST /v1/text-local-edit`

### 必填参数

| 参数 | 类型 | 默认值 | 说明 | 调用示例 |
| --- | --- | --- | --- | --- |
| `original_text` | string | 无 | 原始音频对应文本。align 和 diff 使用原始文本 | `-F 'original_text=I like all americans.'` |
| `edited_text` | string | 无 | 修改后的完整文本 | `-F 'edited_text=I like all chinese.'` |

源音频必须三选一：

| 输入类型 | 参数 | 调用示例 |
| --- | --- | --- |
| 上传音频 | `source_audio` | `-F 'source_audio=@source.wav'` |
| 服务端音频路径 | `source_audio_path` | `-F 'source_audio_path=/data/audio/source.wav'` |
| base64 音频 | `source_audio_base64` | `-F "source_audio_base64=$AUDIO_B64"` |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 | 调用示例 |
| --- | --- | --- | --- | --- |
| `language` | string | `en` | 文本语言。`zh`、`ja`、`ko` 默认字符级对齐；英语默认词级对齐 | `-F 'language=en'` |
| `sample_rate` | int | `0` | 输入音频采样率；`0` 表示从文件解析 | `-F 'sample_rate=24000'` |
| `input_format` | string | `wav` | 输入音频格式：`wav`、`flac`、`pcm_s16le` | `-F 'input_format=wav'` |
| `output_format` | string | `wav` | 输出音频格式：`wav`、`flac`、`pcm_s16le` | `-F 'output_format=wav'` |
| `align_granularity` | string | 空字符串 | 强制对齐粒度：空值自动判断，或传 `word` / `char` / `character` | `-F 'align_granularity=word'` |
| `padding_ms` | float | 不设置 | 在文本 diff 得到的音频区间两侧增加 padding，单位毫秒 | `-F 'padding_ms=250'` |
| `expand_mask_ratio` | float | 不设置 | 以原 mask 中心为中心按比例扩张最终编辑 mask；`1.0` 表示不扩张 | `-F 'expand_mask_ratio=1.5'` |
| `length_mode` | string | 不设置 | 替换片段长度策略：`auto`、`manual_seconds`、`manual_frames` | `-F 'length_mode=auto'` |
| `manual_duration` | float | 不设置 | `length_mode=manual_seconds` 时手动指定替换片段目标时长，单位秒 | `-F 'length_mode=manual_seconds' -F 'manual_duration=0.8'` |
| `manual_frames` | int | 不设置 | `length_mode=manual_frames` 时手动指定替换片段目标 codec 帧数 | `-F 'length_mode=manual_frames' -F 'manual_frames=20'` |
| `length_scale` | float | 不设置 | `length_mode=auto` 时，在自动估计长度基础上缩放 | `-F 'length_mode=auto' -F 'length_scale=1.2'` |
| `min_mask_frames` | int | `6` | 最小编辑 mask 帧数 | `-F 'min_mask_frames=6'` |
| `edit_context_frames` | int | `40` | 编辑区域左右保留的模型上下文帧数 | `-F 'edit_context_frames=40'` |
| `edit_ref_context_frames` | int | `120` | 作为参考音色使用的上下文帧数 | `-F 'edit_ref_context_frames=120'` |
| `preprocess_source_audio` | bool | 不设置 | 是否对源音频做预处理；不设置时使用服务默认值 | `-F 'preprocess_source_audio=true'` |
| `postprocess_output` | bool | `true` | 是否对输出音频做后处理 | `-F 'postprocess_output=true'` |
| `num_steps` | int | `8` | LLM 生成步数 | `-F 'num_steps=8'` |
| `cfg_scale` | float | `0.0` | classifier-free guidance scale | `-F 'cfg_scale=0.0'` |
| `emotion_guidance_scale` | float | `0.0` | leading emotion tag CFG scale; ignored when no `<|emotion-xxx|>` tag is present | `-F 'emotion_guidance_scale=6.0'` |
| `nvv_guidance_scale` | float | `0.0` | NVV tag CFG scale; ignored when no NVV tag is present | `-F 'nvv_guidance_scale=2.0'` |
| `position_temperature` | float | `1.0` | 位置采样温度 | `-F 'position_temperature=1.0'` |
| `class_temperature` | float | `0.0` | 类别采样温度 | `-F 'class_temperature=0.0'` |
| `t_shift` | float | `0.1` | diffusion/flow 时间偏移参数 | `-F 't_shift=0.1'` |
| `layer_penalty_factor` | float | `5.0` | 多 codebook 层惩罚因子 | `-F 'layer_penalty_factor=5.0'` |
| `timeout_sec` | int | `900` | HTTP 到 gRPC 调用超时，单位秒 | `-F 'timeout_sec=900'` |

### Curl 完整示例

```bash
BASE_URL="http://mrwaterzhou.uicp.io:38179"

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
  -F 'num_steps=8' \
  -F 'cfg_scale=0.0' \
  -F 'position_temperature=1.0' \
  -F 'class_temperature=0.0' \
  -F 't_shift=0.1' \
  -F 'layer_penalty_factor=5.0' \
  -F 'timeout_sec=900' \
  --output edited.wav
```

## Python 示例

客户端环境如果没有 `requests`，可以在当前虚拟环境中安装：

```bash
uv pip install requests
```

### Python：语音克隆，上传提示音频

```python
from pathlib import Path

import requests

BASE_URL = "http://mrwaterzhou.uicp.io:38179"

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
            "num_steps": "8",
            "cfg_scale": "0.0",
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

### Python：语音克隆，传提示 codebook

```python
import json
from pathlib import Path

import requests

BASE_URL = "http://mrwaterzhou.uicp.io:38179"

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
        "num_steps": "8",
        "timeout_sec": "600",
    },
    timeout=620,
)

response.raise_for_status()
Path("clone_from_codebook.wav").write_bytes(response.content)
print("trace_id:", response.headers.get("X-ViiTorVoice-Trace-Id"))
```

### Python：语音克隆，传 base64 音频

```python
import base64
from pathlib import Path

import requests

BASE_URL = "http://mrwaterzhou.uicp.io:38179"

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

### Python：语音编辑

```python
from pathlib import Path

import requests

BASE_URL = "http://mrwaterzhou.uicp.io:38179"

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
            "num_steps": "8",
            "cfg_scale": "0.0",
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

## gRPC 直连文档

需要直接调用 orchestrator gRPC 服务时，请参考 [grpc_usage.md](grpc_usage.md)。

## 调用注意事项

1. `input_format` / `output_format` 目前支持 `wav`、`flac`、`pcm_s16le`。常规调用建议使用 `wav`。
2. `sample_rate=0` 时服务会尝试从音频文件中解析采样率；传裸 `pcm_s16le` 时应显式提供 `sample_rate`。
3. `/v1/text-local-edit` 当前通过原始文本和修改后文本自动 diff 生成编辑区间；如果字面未变化但仍希望重合成，需要后续新增显式 edit span 接口。
4. 日语、中文、韩语默认使用字符级 align；英语默认使用词级 align。`align_granularity` 可覆盖自动选择。
5. 日语文本在 align/diff 时使用原始文本；只有进入模型 tokenizer 前才会做 `<|ja-字|>` 预处理。
6. `expand_mask_ratio` 只扩大编辑音频 mask，不改变替换文本和 diff 结果。
