# ViiTorVoice-NAR

[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface)](https://huggingface.co/ZzWater/ViiTorVoice-NAR)
[![Hugging Face Demo](https://img.shields.io/badge/HuggingFace-Demo-FFD21E?logo=huggingface)](https://huggingface.co/spaces/ZzWater/ViiTorVoice)
[![中文文档](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-blue)](README_zh.md)

ViiTorVoice is a non-autoregressive speech generation system for voice cloning and local speech editing. The current deployment path uses split gRPC v2 services, with an HTTP gateway for end-to-end calls.

Core capabilities:

- Voice cloning: provide prompt audio or a prompt audio codebook, and synthesize speech for the target text.
- Local editing: provide source audio, original text, and edited text; the system locates the changed region and resynthesizes only the local segment.
- Emotion and paralinguistic control: insert emotion tags and paralinguistic information into text conditions, then enhance them with CFG.
- Low-latency inference: supports first block inference, with end-to-end first-frame latency around 60 ms.

For model architecture, features, and technical details, see [Technical Notes](docs/tech_en.md).

## Inference Environment Setup

Run the initialization script from the repository root:

```bash
bash init_env.sh
```

The script creates `.venv` and installs the dependencies required for inference. Service startup uses this virtual environment by default.

## Model Download

Download the model files into `local_models/` under the repository root. Do not use symlinks; make sure the model files really exist under the local `local_models/` directory.

Model page:

```text
https://huggingface.co/ZzWater/ViiTorVoice-NAR
```

```bash
mkdir -p local_models
huggingface-cli download ZzWater/ViiTorVoice-NAR \
  --local-dir local_models \
  --local-dir-use-symlinks False
```

If you use another download tool, keep the same rule: place the downloaded files under `local_models/`, and do not use symlinks.

## Service Startup And Management

Services are managed by `run_grpc_v2.sh`. Use the default all-in-one startup path; `all` starts encoder, llm, decoder, orchestrator, and http services.

```bash
./run_grpc_v2.sh start all
./run_grpc_v2.sh status all
./run_grpc_v2.sh logs orchestrator
./run_grpc_v2.sh stop all
```

The HTTP service listens on `0.0.0.0:7861` by default. Local access uses `http://127.0.0.1:7861`. For other ports, model paths, GPU settings, log directories, and environment variables, see [viitorvoice/grpc_server/deploy.env](viitorvoice/grpc_server/deploy.env).

## HTTP Inference Examples

Default local HTTP endpoint:

```bash
BASE_URL="http://127.0.0.1:7861"
```

### Health Check

```bash
curl "$BASE_URL/health"
```

### Voice Cloning

For no-ref-text cloning, omit `ref_text`:

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=今天天气不错，我们下午一起去公园散步吧。' \
  -F 'language=zh' \
  -F 'allow_missing_ref_text=true' \
  --output clone_no_ref_text.wav
```

### Emotion And Paralinguistic Control

After adding emotion or paralinguistic tags to the text, use CFG parameters to strengthen the control effect:

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=<|emotion-happy|>I finally finished the project, and I feel really happy.' \
  -F 'language=en' \
  -F 'emotion_guidance_scale=6.0' \
  -F 'nvv_guidance_scale=2.0' \
  --output clone_emotion.wav
```

The available tag set depends on the training data and model configuration. If no corresponding tag is present, the related CFG parameters do not take effect.

### Local Editing

Upload source audio, original text, and the complete edited text:

```bash
curl -X POST "$BASE_URL/v1/text-local-edit" \
  -F 'source_audio=@source.wav' \
  -F 'original_text=Please send the meeting notes before Friday.' \
  -F 'edited_text=Please send the meeting notes before Monday.' \
  -F 'language=en' \
  -F 'align_granularity=word' \
  -F 'expand_mask_ratio=1.5' \
  -F 'output_format=wav' \
  --output edited.wav
```

For more HTTP parameters, codebook input, base64 input, and Python examples, see [HTTP API Usage](docs/api_usage/en/http_usage.md). To call the orchestrator gRPC service directly, see [gRPC API Usage](docs/api_usage/en/grpc_usage.md).

## Acknowledgements

The model architecture and training ideas in this project are inspired by:

- [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- [DualCodec](https://dualcodec.github.io)
