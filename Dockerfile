# syntax=docker/dockerfile:1.4
FROM nvcr.nju.edu.cn/nvidia/tritonserver:25.03-py3

ARG PIP_MIRROR=https://mirrors.cloud.tencent.com/pypi/simple
ARG DOWNLOAD_MODEL=true
ARG HF_MODEL_REPO=ZzWater/ViiTorVoice-NAR
ARG MODEL_DIR=/workspace/local_models

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl ca-certificates git ffmpeg \
      libopus0 libmp3lame0 libogg0 libvorbis0a && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir uv -i ${PIP_MIRROR}
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_INDEX_URL="${PIP_MIRROR}"
ENV PIP_INDEX_URL="${PIP_MIRROR}"

# Proxy-off defaults for local/internal services
ENV http_proxy=""
ENV https_proxy=""
ENV all_proxy=""
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV ALL_PROXY=""
ENV no_proxy="127.0.0.1,localhost,0.0.0.0"
ENV NO_PROXY="127.0.0.1,localhost,0.0.0.0"

COPY init_env.sh requirements-grpc.txt requirements-alone.txt /tmp/viitorvoice-env/

# Python env and deps. Keep the install order aligned with init_env.sh.
WORKDIR /opt/viitorvoice-runtime
RUN uv venv --python 3.12 && \
    . .venv/bin/activate && \
    uv pip install -r /tmp/viitorvoice-env/requirements-grpc.txt && \
    uv pip install -r /tmp/viitorvoice-env/requirements-alone.txt && \
    uv pip install protobuf==4.25.3 huggingface_hub[cli]

# Match README_zh.md: keep real model files under repository-root local_models/.
# Set --build-arg DOWNLOAD_MODEL=false to skip the build-time Hugging Face check/download.
RUN --mount=type=bind,source=.,target=/tmp/build-context,readonly \
    mkdir -p "${MODEL_DIR}" && \
    if [ "${DOWNLOAD_MODEL}" = "true" ]; then \
      if [ -L /tmp/build-context/local_models ] || \
         { [ -d /tmp/build-context/local_models ] && find /tmp/build-context/local_models -type l -print -quit | grep -q .; }; then \
        echo "Model files under local_models must be real files, not symlinks." >&2; \
        exit 1; \
      fi; \
      if [ ! -d /tmp/build-context/local_models/llm/0p6_emotion ] || \
         [ ! -d /tmp/build-context/local_models/dualcodec/dualcodec_ckpts ] || \
         [ ! -d /tmp/build-context/local_models/aligner/Qwen3-ForcedAligner-0.6B ]; then \
        /opt/viitorvoice-runtime/.venv/bin/huggingface-cli download "${HF_MODEL_REPO}" \
          --local-dir "${MODEL_DIR}" \
          --local-dir-use-symlinks False; \
      else \
        echo "Required model files already exist under local_models; skip download."; \
      fi; \
    fi

# Runtime defaults (can be overridden by docker-compose env)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV VIRTUAL_ENV=/opt/viitorvoice-runtime/.venv
ENV VIITORVOICE_LOCAL_MODELS=${MODEL_DIR}
ENV PATH="/opt/viitorvoice-runtime/.venv/bin:/root/.local/bin:${PATH}"

EXPOSE 50051
EXPOSE 51051
EXPOSE 51052
EXPOSE 51053
EXPOSE 7861

WORKDIR /workspace

CMD ["/bin/bash"]
