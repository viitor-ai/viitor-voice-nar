# Speech Edit Inference Deployment

This repository keeps gRPC v2 as the only deployment path. The split services are:

- `encoder`: audio to DualCodec codebook.
- `llm`: text/codebook/semantic/edit-mask to generated codebook.
- `decoder`: codebook to WAV.
- `orchestrator`: product APIs for synthesize, semantic-to-wav, align-for-edit, local edit, and text-driven local edit.
- `http`: FastAPI gateway for end-to-end voice clone and text-driven local edit.
- `provider`: standard `tts.backend.v1.BackendProviderService` adapter for remote gateway integration.

Run commands from this repository root. The launcher clears common proxy environment variables before starting services and uses the repository `.venv`.

## Setup

Create the local virtual environment from the independent repository root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-grpc.txt
```

Model files are expected under `local_models/` by default. This directory is intentionally ignored by git.

## Config

Deployment defaults live in:

```bash
viitorvoice/grpc_server/deploy.env
```

Use a different config when needed:

```bash
./run_grpc_v2.sh --config path/to/deploy.env status all
```

Default endpoints:

```text
encoder:      127.0.0.1:51051
llm:          127.0.0.1:51052
decoder:      127.0.0.1:51053
orchestrator: 0.0.0.0:50051
http:         0.0.0.0:7861
provider:     127.0.0.1:50062
```

Common overrides are set in `deploy.env`: ports, model paths, device id, backend selection, TensorRT cache paths, timeout, log/state directories, aligner path/device/language, and warmup behavior. The default forced-aligner model path is `local_models/aligner/Qwen3-ForcedAligner-0.6B`.

`VIITORVOICE_DEVICE_ID` is a process-local CUDA index after
`CUDA_VISIBLE_DEVICES` remapping. To run on one physical GPU, keep
`VIITORVOICE_DEVICE_ID=0`:

```bash
CUDA_VISIBLE_DEVICES=1 VIITORVOICE_DEVICE_ID=0 ./run_grpc_v2.sh start all
```

The default LLM checkpoint is `local_models/llm/1p7_nvv`, with its ONNX
backbone loaded from `.cache/onnx_backbone_fp32/llm_backbone_dynamic.onnx`.

The encoder defaults to the torch backend with bf16 autocast:

```bash
VIITORVOICE_CODEC_ENCODER_BACKEND=torch
VIITORVOICE_CODEC_ENCODER_PRECISION=bf16
```

The previous ONNX paths remain available by setting
`VIITORVOICE_CODEC_ENCODER_BACKEND=onnx-cuda` or `onnx-trt`.

By default the launcher uses `.venv`, because the local edit aligner
depends on the `qwen-asr` package versions installed there. If you override
`VIITORVOICE_V2_VENV_DIR`, make sure `from qwen_asr import Qwen3ForcedAligner`
works in that environment.

Runtime services are self-contained under this repository and do not import the
training repository's `viitorvoice` Python package.

## Export LLM ONNX

This repository has its own export-only entrypoint for the LLM backbone:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m llm.export_onnx \
  --model outputs/nvv_pair_text_tag_full_finetune_mixed_expansion_4to1_20k/checkpoint-20000 \
  --device cuda:0 \
  --dtype fp32 \
  --attn_implementation eager \
  --onnx_path outputs/nvv_pair_text_tag_full_finetune_mixed_expansion_4to1_20k/checkpoint-20000/.cache/onnx_backbone_fp32/llm_backbone_dynamic.onnx \
  --force_export true \
  --export_only true \
  --export_batch 4 \
  --export_seq_len 256
```

This mirrors the export-only path of `viitorvoice.cli.infer_onnx_trt`, but the runnable entrypoint is local to this repository. Tag-specific CFG can use up to four LLM forward rows for one logical sample, so ONNX/TRT profiles should use `VIITORVOICE_LLM_BATCH_MAX=4` and exported engines should support `--export_batch 4` or larger. Rebuild the TensorRT cache after changing the batch profile.
Exporting a new ONNX backbone still requires the original ViiTorVoice training
package in the export environment because it has to instantiate the training
model class. The serving path only needs the exported model files.

## Start And Stop

Single entrypoint:

```bash
./run_grpc_v2.sh start all
./run_grpc_v2.sh status all
./run_grpc_v2.sh logs orchestrator
./run_grpc_v2.sh stop all
```

Each service can be managed separately:

```bash
./run_grpc_v2.sh start encoder
./run_grpc_v2.sh stop encoder
./run_grpc_v2.sh restart llm -- --no-warmup
./run_grpc_v2.sh restart http
```

PID files are written to `VIITORVOICE_V2_STATE_DIR` and logs to `VIITORVOICE_V2_LOG_DIR`.
When `VIITORVOICE_WARMUP_ON_START=true`, the orchestrator preloads the forced
aligner before it starts listening, so `AlignForEdit` failures surface at
startup rather than on the first request.

## HTTP Gateway

The FastAPI service is started with `start all` and can also be managed as
`http`. It calls the orchestrator through async gRPC and only exposes
end-to-end APIs.

Health:

```bash
curl http://127.0.0.1:7861/health
```

Voice clone from prompt audio:

```bash
curl -X POST http://127.0.0.1:7861/v1/voice-clone \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=hello from ViiTorVoice' \
  -F 'language=en' \
  --output clone.wav
```

Voice clone from prompt codebook:

```bash
curl -X POST http://127.0.0.1:7861/v1/voice-clone \
  -F 'ref_audio_codebook={"values":[...],"shape":[12,305]}' \
  -F 'text=hello from ViiTorVoice' \
  -F 'language=en' \
  --output clone.wav
```

Text-driven local edit:

```bash
curl -X POST http://127.0.0.1:7861/v1/text-local-edit \
  -F 'source_audio=@source.wav' \
  -F 'original_text=original transcript' \
  -F 'edited_text=edited transcript' \
  -F 'language=en' \
  -F 'expand_mask_ratio=1.5' \
  --output edited.wav
```

`expand_mask_ratio` keeps the replacement span and text unchanged, but expands
the final editable audio mask around its center. `1.0` means no expansion.

## Standard Backend Provider

The provider exposes `tts.backend.v1.BackendProviderService` and calls the
existing gRPC v2 orchestrator. Start the current stack first, then start the
provider:

```bash
./run_grpc_v2.sh start all
VIITORVOICE_V2_SERVICE_HOST=127.0.0.1 ./run_grpc_v2.sh start provider
```

By default the provider listens on `127.0.0.1:50062` when started as above and
calls the orchestrator at `127.0.0.1:50051`. Override the orchestrator target
with:

```bash
SPEECH_EDIT_ORCHESTRATOR_TARGET=127.0.0.1:50051 ./run_grpc_v2.sh start provider
```

The first provider version supports unary `Synthesize` and prompt feature
preparation. `SynthesizeStream` returns `UNIMPLEMENTED`, and capabilities report
`supports_true_streaming=false`.

Provider smoke checks:

```bash
.venv/bin/python -m viitorvoice.grpc_server.provider_smoke \
  --target 127.0.0.1:50062 \
  --mode capabilities \
  --output-dir test_outputs/viitorvoice_grpc_server_provider_smoke

.venv/bin/python -m viitorvoice.grpc_server.provider_smoke \
  --target 127.0.0.1:50062 \
  --mode health \
  --output-dir test_outputs/viitorvoice_grpc_server_provider_smoke
```

## Generate Protos

```bash
.venv/bin/python viitorvoice/grpc_server/tools/generate_proto.py
```

## Smoke Tests

Static contract and helper checks:

```bash
.venv/bin/python viitorvoice/grpc_server/tools/static_smoke.py \
  --output-dir test_outputs/viitorvoice_grpc_server_static
```

Health checks after starting the orchestrator:

```bash
.venv/bin/python -m viitorvoice.grpc_server.client_smoke \
  --service orchestrator \
  --target 127.0.0.1:50051 \
  --mode health \
  --output-dir test_outputs/viitorvoice_grpc_server_health
```

Text-driven local edit endpoint:

```text
rpc TextLocalEdit(TextLocalEditRequest) returns (TextLocalEditResponse)
```

`TextLocalEditRequest` accepts `source_audio`, `original_text`, and
`edited_text`. The orchestrator runs `AlignForEdit`, maps the text diff to
alignment indices, then calls the existing `LocalEdit` pipeline and returns the
edited WAV in `audio`.

Smoke client:

```bash
.venv/bin/python -m viitorvoice.grpc_server.client_smoke \
  --target 127.0.0.1:50051 \
  --mode text-local-edit \
  --audio source.wav \
  --text "original transcript" \
  --edited-text "edited transcript" \
  --output-dir test_outputs/viitorvoice_grpc_server_text_local_edit
```
