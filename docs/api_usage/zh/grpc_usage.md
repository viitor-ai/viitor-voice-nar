# ViiTorVoice gRPC 接口使用说明

HTTP 网关内部调用的是 orchestrator gRPC 服务。以下示例均假设在当前独立仓库根目录运行。默认本地 gRPC target：

```text
127.0.0.1:50051
```

对应 proto 文件：

```text
viitorvoice/grpc_server/proto/viitorvoice_common.proto
viitorvoice/grpc_server/proto/viitorvoice_orchestrator.proto
```

HTTP 与 gRPC 的对应关系：

| HTTP 接口 | gRPC RPC | 请求 message | 返回 message |
| --- | --- | --- | --- |
| `GET /health` | `Health` | `HealthRequest` | `HealthResponse` |
| `POST /v1/voice-clone` | `Synthesize` | `SynthesizeRequest` | `SynthesizeResponse` |
| `POST /v1/text-local-edit` | `TextLocalEdit` | `TextLocalEditRequest` | `TextLocalEditResponse` |

orchestrator 还暴露 lower-level RPC：`EncodeAudio`、`SemanticToWav`、`AlignForEdit`、`LocalEdit`。HTTP 网关目前只封装端到端的 `Synthesize` 和 `TextLocalEdit`。

### gRPC service 定义

```proto
service ViiTorVoiceOrchestratorService {
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc EncodeAudio(EncodeAudioRequest) returns (EncodeAudioResponse);
  rpc Synthesize(SynthesizeRequest) returns (SynthesizeResponse);
  rpc SemanticToWav(SemanticToWavRequest) returns (SemanticToWavResponse);
  rpc AlignForEdit(AlignForEditRequest) returns (AlignForEditResponse);
  rpc LocalEdit(LocalEditRequest) returns (LocalEditResponse);
  rpc TextLocalEdit(TextLocalEditRequest) returns (TextLocalEditResponse);
}
```

### 共用 proto 类型

```proto
enum AudioFormat {
  AUDIO_FORMAT_UNSPECIFIED = 0;
  AUDIO_FORMAT_WAV = 1;
  AUDIO_FORMAT_PCM_S16LE = 2;
  AUDIO_FORMAT_FLAC = 3;
}

enum AlignmentGranularity {
  ALIGNMENT_GRANULARITY_UNSPECIFIED = 0;
  ALIGNMENT_GRANULARITY_WORD = 1;
  ALIGNMENT_GRANULARITY_CHARACTER = 2;
}

message RequestContext {
  string trace_id = 1;
  string request_id = 2;
  string parent_span_id = 3;
  string span_id = 4;
  string caller = 5;
  int64 deadline_ms = 6;
  map<string, string> tags = 7;
}

message ResponseContext {
  string trace_id = 1;
  string request_id = 2;
  string span_id = 3;
  string service = 4;
  string status = 5;
  repeated StageMetric metrics = 6;
}

message AudioInput {
  oneof source {
    bytes audio_bytes = 1;
    string audio_path = 2;
  }
  uint32 sample_rate = 3;
  AudioFormat format = 4;
}

message AudioResult {
  bytes audio_bytes = 1;
  uint32 sample_rate = 2;
  AudioFormat format = 3;
  uint32 channels = 4;
  double duration_sec = 5;
}

message Int64Tensor {
  repeated int64 values = 1;
  repeated int64 shape = 2;
}

message GenerationConfig {
  optional uint32 max_new_tokens = 1;
  optional uint32 num_steps = 2;
  optional float temperature = 3;
  optional float top_p = 4;
  optional uint32 top_k = 5;
  optional float cfg_scale = 6;
  optional uint64 seed = 7;
  optional bool debug = 8;
  optional string debug_request_id = 9;
  optional uint32 request_timeout_sec = 10;
  optional float t_shift = 11;
  optional float layer_penalty_factor = 12;
  optional float position_temperature = 13;
  optional float class_temperature = 14;
  optional bool denoise = 15;
  optional bool preprocess_prompt = 16;
  optional bool postprocess_output = 17;
  optional float audio_chunk_duration = 18;
  optional float audio_chunk_threshold = 19;
  optional float duration = 20;
  optional float speed = 21;
  optional float emotion_guidance_scale = 22;
  optional float nvv_guidance_scale = 23;
}

message TextCondition {
  string text = 1;
  string language = 2;
  string ref_text = 3;
  bool allow_missing_ref_text = 4;
  uint32 ref_text_mask_len = 5;
  string instruct = 6;
}

message AlignmentItem {
  uint32 index = 1;
  string text = 2;
  double start_time = 3;
  double end_time = 4;
  int64 start_char = 5;
  int64 end_char = 6;
  bool has_start_char = 7;
  bool has_end_char = 8;
  string kind = 9;
  int64 start_frame = 10;
  int64 end_frame = 11;
  int64 token_start = 12;
  int64 token_end = 13;
  float confidence = 14;
}

message EditSelection {
  repeated uint32 alignment_indices = 1;
  optional double start_sec = 2;
  optional double end_sec = 3;
  optional int64 start_frame = 4;
  optional int64 end_frame = 5;
}

message EditSegment {
  EditSelection selection = 1;
  string replacement_text = 2;
}
```

### 语音克隆 proto：`Synthesize`

`Synthesize` 对应 HTTP `/v1/voice-clone`。

```proto
message SynthesizeRequest {
  RequestContext context = 1;
  TextCondition condition = 2;
  AudioInput ref_audio = 3;
  Int64Tensor ref_audio_codebook = 4;
  GenerationConfig generation = 5;
  bool return_tokens = 6;
  AudioFormat output_format = 7;
}

message SynthesizeResponse {
  ResponseContext context = 1;
  AudioResult audio = 2;
  Int64Tensor generated_audio_codebook = 3;
  Int64Tensor ref_audio_codebook = 4;
  repeated string warnings = 5;
}
```

请求字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `context` | `RequestContext` | 否 | trace/caller 信息；可只设置 `caller` |
| `condition.text` | string | 是 | 待合成文本 |
| `condition.language` | string | 否 | 语言，如 `en`、`zh`、`ja`、`ko`、`yue` |
| `condition.ref_text` | string | 否 | 提示音对应文本 |
| `condition.allow_missing_ref_text` | bool | 否 | 是否允许缺失 `ref_text` |
| `condition.ref_text_mask_len` | uint32 | 否 | no-ref-text mask 长度 |
| `condition.instruct` | string | 否 | 风格/指令文本 |
| `ref_audio` | `AudioInput` | 二选一 | 提示音音频 bytes 或服务端路径 |
| `ref_audio_codebook` | `Int64Tensor` | 二选一 | 提示音 codebook，通常 shape 为 `[12, T]` |
| `generation` | `GenerationConfig` | 否 | 生成参数 |
| `return_tokens` | bool | 否 | 是否在响应中返回 `generated_audio_codebook` |
| `output_format` | `AudioFormat` | 否 | 输出格式，常用 `AUDIO_FORMAT_WAV` |

`ref_audio` 与 `ref_audio_codebook` 至少提供一个。若两者都提供，服务端优先使用已有 codebook 路径时应以实际 runtime 行为为准；推荐调用方只传一种。

### 语音编辑 proto：`TextLocalEdit`

`TextLocalEdit` 对应 HTTP `/v1/text-local-edit`。

```proto
message TextLocalEditRequest {
  RequestContext context = 1;
  AudioInput source_audio = 2;
  string original_text = 3;
  string edited_text = 4;
  string language = 5;
  GenerationConfig generation = 6;
  AudioFormat output_format = 7;
  optional float padding_ms = 8;
  optional string length_mode = 9;
  optional float manual_duration = 10;
  optional uint32 manual_frames = 11;
  optional float length_scale = 12;
  optional uint32 min_mask_frames = 13;
  optional uint32 edit_context_frames = 14;
  optional uint32 edit_ref_context_frames = 15;
  optional bool preprocess_source_audio = 16;
  optional bool postprocess_output = 17;
  string align_granularity = 18;
  bool return_debug = 19;
  optional float expand_mask_ratio = 20;
}

message TextLocalEditResponse {
  ResponseContext context = 1;
  AudioResult audio = 2;
  repeated AlignmentItem alignments = 3;
  repeated EditSegment edits = 4;
  uint32 remaining_mask_tokens = 5;
  repeated string warnings = 6;
}
```

请求字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `context` | `RequestContext` | 否 | trace/caller 信息；可只设置 `caller` |
| `source_audio` | `AudioInput` | 是 | 原始音频 bytes 或服务端路径 |
| `original_text` | string | 是 | 原始音频对应文本，align/diff 使用这个原始文本 |
| `edited_text` | string | 是 | 修改后的完整文本 |
| `language` | string | 否 | 语言；`zh/ja/ko/yue` 默认字符级 align，`en` 默认词级 align |
| `generation` | `GenerationConfig` | 否 | 生成参数 |
| `output_format` | `AudioFormat` | 否 | 输出格式，常用 `AUDIO_FORMAT_WAV` |
| `padding_ms` | float | 否 | diff 得到的音频区间两侧 padding，单位毫秒 |
| `length_mode` | string | 否 | `auto`、`manual_seconds`、`manual_frames` |
| `manual_duration` | float | 否 | `length_mode=manual_seconds` 时的目标时长，单位秒 |
| `manual_frames` | uint32 | 否 | `length_mode=manual_frames` 时的目标 codec 帧数 |
| `length_scale` | float | 否 | `length_mode=auto` 时的长度缩放 |
| `min_mask_frames` | uint32 | 否 | 最小编辑 mask 帧数 |
| `edit_context_frames` | uint32 | 否 | 编辑区域左右模型上下文帧数 |
| `edit_ref_context_frames` | uint32 | 否 | 参考音色上下文帧数 |
| `preprocess_source_audio` | bool | 否 | 是否预处理源音频 |
| `postprocess_output` | bool | 否 | 是否后处理输出音频 |
| `align_granularity` | string | 否 | 强制对齐粒度：空值自动判断，或 `word` / `char` / `character` |
| `return_debug` | bool | 否 | 是否返回 alignments 和 edits 调试信息 |
| `expand_mask_ratio` | float | 否 | 以原 mask 中心为中心扩大最终编辑 mask |

### Python gRPC：语音克隆

项目内 Python gRPC stub 已生成，可直接使用：

```python
import asyncio
from pathlib import Path

import grpc

from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2_grpc as orch_pb2_grpc


async def main() -> None:
    target = "127.0.0.1:50051"
    async with grpc.aio.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ],
    ) as channel:
        stub = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel)
        request = orch_pb2.SynthesizeRequest(
            context=common_pb2.RequestContext(caller="python_grpc_example"),
            condition=common_pb2.TextCondition(
                text="hello from ViiTorVoice",
                language="en",
                ref_text="this is my prompt voice",
                allow_missing_ref_text=True,
                ref_text_mask_len=10,
            ),
            ref_audio=common_pb2.AudioInput(
                audio_bytes=Path("prompt.wav").read_bytes(),
                sample_rate=0,
                format=common_pb2.AUDIO_FORMAT_WAV,
            ),
            generation=common_pb2.GenerationConfig(
                num_steps=8,
                cfg_scale=0.0,
                position_temperature=1.0,
                class_temperature=0.0,
                t_shift=0.1,
                layer_penalty_factor=5.0,
                preprocess_prompt=True,
                postprocess_output=True,
                request_timeout_sec=600,
            ),
            output_format=common_pb2.AUDIO_FORMAT_WAV,
            return_tokens=False,
        )
        response = await stub.Synthesize(request, timeout=600)
        Path("clone_grpc.wav").write_bytes(response.audio.audio_bytes)
        print("trace_id:", response.context.trace_id)
        print("duration:", response.audio.duration_sec)


asyncio.run(main())
```

### Python gRPC：语音克隆，传 codebook

```python
import asyncio
import json
from pathlib import Path

import grpc

from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2_grpc as orch_pb2_grpc


async def main() -> None:
    payload = json.loads(Path("prompt_codebook.json").read_text())
    if "audio_codebook" in payload:
        payload = payload["audio_codebook"]
    codebook = common_pb2.Int64Tensor(
        values=[int(value) for value in payload["values"]],
        shape=[int(value) for value in payload["shape"]],
    )

    async with grpc.aio.insecure_channel("127.0.0.1:50051") as channel:
        stub = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel)
        request = orch_pb2.SynthesizeRequest(
            context=common_pb2.RequestContext(caller="python_grpc_codebook_example"),
            condition=common_pb2.TextCondition(text="hello from ViiTorVoice", language="en"),
            ref_audio_codebook=codebook,
            generation=common_pb2.GenerationConfig(num_steps=8, request_timeout_sec=600),
            output_format=common_pb2.AUDIO_FORMAT_WAV,
        )
        response = await stub.Synthesize(request, timeout=600)
        Path("clone_grpc_codebook.wav").write_bytes(response.audio.audio_bytes)


asyncio.run(main())
```

### Python gRPC：语音编辑

```python
import asyncio
from pathlib import Path

import grpc

from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2_grpc as orch_pb2_grpc


async def main() -> None:
    async with grpc.aio.insecure_channel(
        "127.0.0.1:50051",
        options=[
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ],
    ) as channel:
        stub = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel)
        request = orch_pb2.TextLocalEditRequest(
            context=common_pb2.RequestContext(caller="python_grpc_edit_example"),
            source_audio=common_pb2.AudioInput(
                audio_bytes=Path("source.wav").read_bytes(),
                sample_rate=0,
                format=common_pb2.AUDIO_FORMAT_WAV,
            ),
            original_text="I like all americans.",
            edited_text="I like all chinese.",
            language="en",
            generation=common_pb2.GenerationConfig(
                num_steps=8,
                cfg_scale=0.0,
                position_temperature=1.0,
                class_temperature=0.0,
                t_shift=0.1,
                layer_penalty_factor=5.0,
                postprocess_output=True,
                request_timeout_sec=900,
            ),
            output_format=common_pb2.AUDIO_FORMAT_WAV,
            padding_ms=250.0,
            expand_mask_ratio=1.5,
            length_mode="auto",
            length_scale=1.1,
            min_mask_frames=6,
            edit_context_frames=40,
            edit_ref_context_frames=120,
            postprocess_output=True,
            align_granularity="word",
            return_debug=True,
        )
        response = await stub.TextLocalEdit(request, timeout=900)
        Path("edited_grpc.wav").write_bytes(response.audio.audio_bytes)
        print("trace_id:", response.context.trace_id)
        print("alignments:", len(response.alignments))
        print("edits:", len(response.edits))


asyncio.run(main())
```
