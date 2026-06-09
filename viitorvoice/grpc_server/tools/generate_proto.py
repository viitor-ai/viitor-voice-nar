"""Generate Python gRPC stubs for the ViiTorVoice inference v2 protos.

Run from the repository root with:

    uv run python -m viitorvoice.grpc_server.tools.generate_proto
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTO_DIR = REPO_ROOT / "viitorvoice" / "grpc_server" / "proto"
OUT_DIR = REPO_ROOT
PROTO_FILES = [
    PROTO_DIR / "viitorvoice_common.proto",
    PROTO_DIR / "viitorvoice_encoder.proto",
    PROTO_DIR / "viitorvoice_llm.proto",
    PROTO_DIR / "viitorvoice_decoder.proto",
    PROTO_DIR / "viitorvoice_orchestrator.proto",
    PROTO_DIR / "backend_provider.proto",
]


def _clear_proxy_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        env.pop(key, None)
    return env


def main() -> int:
    if find_spec("grpc_tools.protoc") is None:
        print(
            "grpc_tools is not installed in this environment. "
            "Add grpcio-tools, then rerun this script.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={REPO_ROOT}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        *[str(path.relative_to(REPO_ROOT)) for path in PROTO_FILES],
    ]
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_clear_proxy_env())
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
