FROM nvcr.nju.edu.cn/nvidia/tritonserver:25.03-py3

ARG PIP_MIRROR=https://mirrors.cloud.tencent.com/pypi/simple

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
    uv pip install protobuf==4.25.3

# Runtime defaults (can be overridden by docker-compose env)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV VIRTUAL_ENV=/opt/viitorvoice-runtime/.venv
ENV PATH="/opt/viitorvoice-runtime/.venv/bin:/root/.local/bin:${PATH}"

EXPOSE 50051
EXPOSE 51051
EXPOSE 51052
EXPOSE 51053
EXPOSE 7861

WORKDIR /workspace

CMD ["/bin/bash"]
