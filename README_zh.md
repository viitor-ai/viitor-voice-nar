# ViiTorVoice-NAR

[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface)](https://huggingface.co/ZzWater/ViiTorVoice-NAR)
[![English README](https://img.shields.io/badge/README-English-blue)](README.md)

ViiTorVoice 是一个面向语音克隆与语音局部编辑的非自回归语音生成系统。项目当前以 gRPC v2 拆分服务作为主要部署路径，并提供 HTTP 网关用于端到端调用。

核心能力包括：

- 语音克隆：输入提示音频或提示音 codebook，生成指定文本对应的语音。
- 局部编辑：输入原始音频、原始文本和修改后文本，自动定位差异区域并重合成局部片段。
- 情感与副语言控制：支持在文本条件中插入情感标签和副语言信息，并通过 CFG 增强控制效果。
- 低延迟推理：支持 first block 推理模式，端到端首帧返回时间可以做到约 60ms。

模型结构、特性和技术做法请参考 [技术说明](docs/tech_zh.md)。

## 推理环境安装

请从当前仓库根目录直接执行初始化脚本：

```bash
bash init_env.sh
```

脚本会创建 `.venv` 并安装推理所需依赖。后续服务启动默认使用该虚拟环境。

## 模型下载

模型需要下载到仓库根目录下的 `local_models/`。不要使用软链；请确保模型文件真实存在于本地 `local_models/` 目录中。

模型地址：

```text
https://huggingface.co/ZzWater/ViiTorVoice-NAR
```

```bash
mkdir -p local_models
huggingface-cli download ZzWater/ViiTorVoice-NAR \
  --local-dir local_models \
  --local-dir-use-symlinks False
```

如果使用其他下载工具，也需要保持相同原则：下载结果放在 `local_models/` 下，并且不要使用软链。

## 服务启动与管理

服务通过 `run_grpc_v2.sh` 管理。默认直接启动全部服务即可，`all` 会启动 encoder、llm、decoder、orchestrator 和 http 服务。

```bash
./run_grpc_v2.sh start all
./run_grpc_v2.sh status all
./run_grpc_v2.sh logs orchestrator
./run_grpc_v2.sh stop all
```

默认 HTTP 服务监听 `0.0.0.0:7861`，本机访问地址为 `http://127.0.0.1:7861`。其他端口、模型路径、GPU、日志目录等环境变量请直接查看 [viitorvoice/grpc_server/deploy.env](viitorvoice/grpc_server/deploy.env)。

## HTTP 推理示例

本地 HTTP 服务默认地址：

```bash
BASE_URL="http://127.0.0.1:7861"
```

### 健康检查

```bash
curl "$BASE_URL/health"
```

### 语音克隆

No-ref-text 克隆时可以不传 `ref_text`：

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=今天天气不错，我们下午一起去公园散步吧。' \
  -F 'language=zh' \
  -F 'allow_missing_ref_text=true' \
  --output clone_no_ref_text.wav
```

### 情感控制与副语言控制

在文本中加入情感或副语言标签后，可以通过 CFG 参数增强控制效果：

```bash
curl -X POST "$BASE_URL/v1/voice-clone" \
  -F 'ref_audio=@prompt.wav' \
  -F 'text=<|emotion-happy|>I finally finished the project, and I feel really happy.' \
  -F 'language=en' \
  -F 'emotion_guidance_scale=6.0' \
  -F 'nvv_guidance_scale=2.0' \
  --output clone_emotion.wav
```

具体标签集合以训练数据和模型配置为准。没有对应标签时，相关 CFG 参数不会生效。

### 局部编辑

上传源音频、原始文本和编辑后的完整文本：

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

更多 HTTP 参数、codebook 输入方式、base64 输入方式和 Python 示例请参考 [中文 HTTP 使用文档](docs/api_usage/zh/http_usage.md)。如需直接调用 orchestrator gRPC 服务，请参考 [中文 gRPC 使用文档](docs/api_usage/zh/grpc_usage.md)。

## 致谢

本项目的模型结构和训练思路受到以下工作的启发：

- [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- [DualCodec](https://dualcodec.github.io)
