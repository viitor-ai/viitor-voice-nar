#!/usr/bin/env python3
"""Export the ViiTorVoice LLM backbone ONNX from the inference package.

This mirrors the export-only path of ``viitorvoice.cli.infer_onnx_trt`` while
keeping the runnable entrypoint under ``inference``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


LOGGER = logging.getLogger(__name__)


class LLMBackboneWrapper(nn.Module):
    def __init__(self, llm: nn.Module) -> None:
        super().__init__()
        self.llm = llm

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs[0]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def clear_proxies() -> None:
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
        os.environ.pop(key, None)


def torch_dtype_from_name(name: str) -> torch.dtype:
    value = (name or "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def default_onnx_path(model_path: str | Path, dtype_name: str) -> Path:
    dtype_tag = "fp32" if torch_dtype_from_name(dtype_name) == torch.float32 else dtype_name.strip().lower()
    return Path(model_path).resolve() / ".cache" / f"onnx_backbone_{dtype_tag}" / "llm_backbone_dynamic.onnx"


def export_backbone_onnx(
    *,
    llm: nn.Module,
    output_path: str | Path,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
    force_export: bool,
    export_batch: int,
    export_seq_len: int,
    opset: int = 18,
) -> str:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if output.exists() and not force_export:
                LOGGER.info("ONNX already exists: %s", output)
                return str(output)

            wrapper = LLMBackboneWrapper(llm).eval().to(device=device, dtype=dtype)
            dummy_embeds = torch.randn(
                export_batch,
                export_seq_len,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            dummy_mask = torch.zeros(
                export_batch,
                1,
                export_seq_len,
                export_seq_len,
                device=device,
                dtype=dtype,
            )
            tmp_path = output.with_suffix(output.suffix + ".tmp")
            LOGGER.info(
                "Exporting LLM backbone ONNX to %s batch=%s seq_len=%s dtype=%s",
                output,
                export_batch,
                export_seq_len,
                dtype,
            )
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    (dummy_embeds, dummy_mask),
                    str(tmp_path),
                    input_names=["inputs_embeds", "attention_mask"],
                    output_names=["hidden_states"],
                    dynamic_axes={
                        "inputs_embeds": {0: "batch", 1: "seq_len"},
                        "attention_mask": {
                            0: "batch",
                            2: "seq_len",
                            3: "seq_len",
                        },
                        "hidden_states": {0: "batch", 1: "seq_len"},
                    },
                    do_constant_folding=True,
                    opset_version=opset,
                )
            tmp_path.replace(output)
            LOGGER.info("Wrote ONNX: %s", output)
            return str(output)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ViiTorVoice LLM backbone ONNX from inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Local ViiTorVoice checkpoint directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        help="Attention implementation used when loading the torch LLM.",
    )
    parser.add_argument("--onnx_path", default=None)
    parser.add_argument("--force_export", type=str2bool, default=False)
    parser.add_argument("--export_only", type=str2bool, default=True)
    parser.add_argument("--export_batch", type=int, default=2)
    parser.add_argument("--export_seq_len", type=int, default=256)
    parser.add_argument("--opset", type=int, default=18)
    return parser


def load_model(args: argparse.Namespace) -> tuple[nn.Module, int, torch.device]:
    model_dir = Path(args.model).expanduser().resolve()
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Safetensors checkpoint not found: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    llm_config_dict = dict(config["llm_config"])
    model_type = llm_config_dict.pop("model_type")
    llm_config = AutoConfig.for_model(model_type, **llm_config_dict)
    if args.attn_implementation:
        llm_config._attn_implementation = args.attn_implementation
    dtype = torch_dtype_from_name(args.dtype)
    LOGGER.info(
        "Loading LLM backbone checkpoint=%s device=%s dtype=%s attn=%s",
        model_dir,
        args.device,
        dtype,
        args.attn_implementation,
    )
    llm = AutoModel.from_config(llm_config).to(device=args.device, dtype=dtype)
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("llm."):
                state_dict[key.removeprefix("llm.")] = handle.get_tensor(key)
    if not state_dict:
        raise RuntimeError(f"No llm.* weights found in {weights_path}")
    missing, unexpected = llm.load_state_dict(state_dict, strict=False)
    allowed_missing = {"rotary_emb.inv_freq"}
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load LLM backbone weights: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    llm.eval()
    return llm, int(llm_config.hidden_size), torch.device(args.device)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    clear_proxies()
    args = build_parser().parse_args()
    llm, hidden_size, device = load_model(args)
    onnx_path = Path(args.onnx_path) if args.onnx_path else default_onnx_path(args.model, args.dtype)
    exported = export_backbone_onnx(
        llm=llm,
        output_path=onnx_path,
        hidden_size=hidden_size,
        device=str(device),
        dtype=torch_dtype_from_name(args.dtype),
        force_export=bool(args.force_export),
        export_batch=int(args.export_batch),
        export_seq_len=int(args.export_seq_len),
        opset=int(args.opset),
    )
    print(json.dumps({"onnx_path": exported}, ensure_ascii=False))


if __name__ == "__main__":
    main()
