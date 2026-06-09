#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viitorvoice import paths
from viitorvoice.llm.runtime import LLMOnnxConfig, LLMOnnxStepRunner, clear_proxies, torch_dtype_from_precision


@dataclass
class Snapshot:
    stage: str
    seconds: float
    pid_gpu_mib: int | None
    pid_gpu_delta_mib: int | None
    torch_allocated_mib: float
    torch_reserved_mib: float
    torch_peak_allocated_mib: float
    torch_peak_reserved_mib: float
    note: str = ""


class StagedLLMWeights(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.num_audio_codebook = int(config["num_audio_codebook"])
        self.num_acoustic_codebooks = self.num_audio_codebook - 1
        self.semantic_codebook_size = int(config["audio_codebook_sizes"][0])
        self.acoustic_codebook_size = int(config["audio_codebook_sizes"][1])
        self.hidden_size = int(config["llm_config"]["hidden_size"])
        self.text_vocab_size = int(config["llm_config"]["vocab_size"])

        self.text_embedding = nn.Embedding(self.text_vocab_size, self.hidden_size)
        self.semantic_embedding = nn.Embedding(self.semantic_codebook_size, self.hidden_size)
        self.acoustic_embedding = nn.Embedding(
            self.num_acoustic_codebooks * self.acoustic_codebook_size,
            self.hidden_size,
        )
        self.semantic_head = nn.Linear(self.hidden_size, self.semantic_codebook_size, bias=False)
        self.acoustic_head = nn.Linear(
            self.hidden_size,
            self.num_acoustic_codebooks * self.acoustic_codebook_size,
            bias=False,
        )
        self.register_buffer(
            "acoustic_codebook_offsets",
            torch.arange(self.num_acoustic_codebooks, dtype=torch.long)
            * self.acoustic_codebook_size,
            persistent=False,
        )

    def load_weights(self, checkpoint_dir: Path, device: torch.device) -> None:
        model_path = checkpoint_dir / "model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint safetensors not found: {model_path}")
        required = {
            "llm.embed_tokens.weight": self.text_embedding.weight,
            "semantic_embedding.weight": self.semantic_embedding.weight,
            "acoustic_embedding.weight": self.acoustic_embedding.weight,
            "semantic_head.weight": self.semantic_head.weight,
            "acoustic_head.weight": self.acoustic_head.weight,
        }
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(set(required) - available)
            if missing:
                raise RuntimeError("Missing LLM runtime weights: " + ", ".join(missing))
            for key, parameter in required.items():
                tensor = handle.get_tensor(key).to(device=device, dtype=parameter.dtype)
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"Weight shape mismatch for {key}: checkpoint={tuple(tensor.shape)}, "
                        f"runtime={tuple(parameter.shape)}."
                    )
                parameter.data.copy_(tensor)
            if "acoustic_codebook_offsets" in available:
                offsets = handle.get_tensor("acoustic_codebook_offsets").to(
                    device=device,
                    dtype=torch.long,
                )
                if tuple(offsets.shape) == tuple(self.acoustic_codebook_offsets.shape):
                    self.acoustic_codebook_offsets.copy_(offsets)

    @torch.inference_mode()
    def prepare_embed_inputs(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_embeds = self.text_embedding(input_ids[:, 0, :])
        audio_only_mask = audio_mask.unsqueeze(-1)
        semantic_ids = torch.where(
            audio_mask,
            input_ids[:, 0, :],
            torch.zeros_like(input_ids[:, 0, :]),
        )
        semantic_embeds = self.semantic_embedding(semantic_ids)
        acoustic_ids = torch.where(
            audio_mask.unsqueeze(1),
            input_ids[:, 1:, :],
            torch.zeros_like(input_ids[:, 1:, :]),
        )
        shifted_acoustic_ids = acoustic_ids + self.acoustic_codebook_offsets.view(1, -1, 1)
        acoustic_embeds = self.acoustic_embedding(shifted_acoustic_ids).sum(dim=1)
        audio_embeds = (semantic_embeds + acoustic_embeds) * audio_only_mask
        return torch.where(audio_only_mask, audio_embeds, text_embeds).contiguous()

    @torch.inference_mode()
    def compute_audio_logits(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_dtype = self.semantic_head.weight.dtype
        if hidden_states.dtype != head_dtype:
            hidden_states = hidden_states.to(dtype=head_dtype)
        semantic_logits = self.semantic_head(hidden_states)
        acoustic_logits = self.acoustic_head(hidden_states)
        acoustic_logits = acoustic_logits.view(
            hidden_states.size(0),
            hidden_states.size(1),
            self.num_acoustic_codebooks,
            self.acoustic_codebook_size,
        ).permute(0, 2, 1, 3)
        return semantic_logits, acoustic_logits


def pid_gpu_mib(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    total = 0
    found = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        if parts[0] == str(pid):
            total += int(parts[1])
            found = True
    return total if found else 0


def mib(value: int | float) -> float:
    return float(value) / 1024 / 1024


def synchronize(device: torch.device) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def snapshot(
    *,
    stage: str,
    start: float,
    device: torch.device,
    previous_pid_gpu_mib: int | None,
    note: str = "",
) -> Snapshot:
    synchronize(device)
    current_pid_gpu_mib = pid_gpu_mib(os.getpid())
    delta = (
        None
        if current_pid_gpu_mib is None or previous_pid_gpu_mib is None
        else current_pid_gpu_mib - previous_pid_gpu_mib
    )
    return Snapshot(
        stage=stage,
        seconds=round(time.monotonic() - start, 3),
        pid_gpu_mib=current_pid_gpu_mib,
        pid_gpu_delta_mib=delta,
        torch_allocated_mib=round(mib(torch.cuda.memory_allocated(device)), 2),
        torch_reserved_mib=round(mib(torch.cuda.memory_reserved(device)), 2),
        torch_peak_allocated_mib=round(mib(torch.cuda.max_memory_allocated(device)), 2),
        torch_peak_reserved_mib=round(mib(torch.cuda.max_memory_reserved(device)), 2),
        note=note,
    )


def add_snapshot(
    snapshots: list[Snapshot],
    *,
    stage: str,
    start: float,
    device: torch.device,
    note: str = "",
) -> None:
    previous = snapshots[-1].pid_gpu_mib if snapshots else None
    snapshots.append(
        snapshot(
            stage=stage,
            start=start,
            device=device,
            previous_pid_gpu_mib=previous,
            note=note,
        )
    )


def add_process_start_snapshot(snapshots: list[Snapshot], *, start: float) -> None:
    current_pid_gpu_mib = pid_gpu_mib(os.getpid())
    snapshots.append(
        Snapshot(
            stage="process_start",
            seconds=round(time.monotonic() - start, 3),
            pid_gpu_mib=current_pid_gpu_mib,
            pid_gpu_delta_mib=None,
            torch_allocated_mib=0.0,
            torch_reserved_mib=0.0,
            torch_peak_allocated_mib=0.0,
            torch_peak_reserved_mib=0.0,
        )
    )


def print_table(snapshots: list[Snapshot]) -> None:
    headers = [
        "stage",
        "sec",
        "pid_gpu_mib",
        "delta",
        "torch_alloc",
        "torch_reserved",
        "peak_alloc",
        "peak_reserved",
        "note",
    ]
    rows = [
        [
            item.stage,
            f"{item.seconds:.3f}",
            "" if item.pid_gpu_mib is None else str(item.pid_gpu_mib),
            "" if item.pid_gpu_delta_mib is None else f"{item.pid_gpu_delta_mib:+d}",
            f"{item.torch_allocated_mib:.2f}",
            f"{item.torch_reserved_mib:.2f}",
            f"{item.torch_peak_allocated_mib:.2f}",
            f"{item.torch_peak_reserved_mib:.2f}",
            item.note,
        ]
        for item in snapshots
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the ViiTorVoice LLM with ONNX CUDA and report staged GPU memory usage."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("VIITORVOICE_LLM_CHECKPOINT", paths.llm_checkpoint_dir("0p6_emotion"))),
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Defaults to VIITORVOICE_LLM_ONNX or <checkpoint>/.cache/onnx_backbone_fp32/llm_backbone_dynamic.onnx.",
    )
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("VIITORVOICE_DEVICE_ID", "0")))
    parser.add_argument("--precision", default=os.environ.get("VIITORVOICE_LLM_PRECISION", "fp32"))
    parser.add_argument("--batch", type=int, default=1, help="Dry-run batch size.")
    parser.add_argument("--seq", type=int, default=16, help="Dry-run sequence length.")
    parser.add_argument("--skip-dry-run", action="store_true", help="Only measure load-time memory.")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_proxies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic.")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    onnx_path = args.onnx_path or Path(
        os.environ.get(
            "VIITORVOICE_LLM_ONNX",
            checkpoint_dir / ".cache" / "onnx_backbone_fp32" / "llm_backbone_dynamic.onnx",
        )
    )
    onnx_path = onnx_path.expanduser().resolve()
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    device = torch.device(f"cuda:{args.device_id}")
    start = time.monotonic()
    snapshots: list[Snapshot] = []

    add_process_start_snapshot(snapshots, start=start)
    torch.cuda.set_device(device)
    add_snapshot(snapshots, stage="cuda_set_device", start=start, device=device)

    torch.empty(1, device=device)
    add_snapshot(snapshots, stage="cuda_context_tensor", start=start, device=device)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    add_snapshot(snapshots, stage="config_loaded", start=start, device=device, note=str(config_path))

    model = StagedLLMWeights(config)
    add_snapshot(snapshots, stage="torch_modules_cpu", start=start, device=device)

    dtype = torch_dtype_from_precision(args.precision)
    model.to(device=device, dtype=dtype)
    add_snapshot(snapshots, stage="torch_modules_cuda_empty", start=start, device=device, note=str(dtype))

    model.load_weights(checkpoint_dir, device)
    model.eval()
    add_snapshot(snapshots, stage="torch_weights_loaded", start=start, device=device)

    runner = LLMOnnxStepRunner(
        LLMOnnxConfig(
            onnx_path=onnx_path,
            backend="onnx-cuda",
            precision=args.precision,
            device_id=args.device_id,
            hidden_size=model.hidden_size,
            batch_min=1,
            batch_opt=max(1, args.batch),
            batch_max=max(1, args.batch),
            seq_min=1,
            seq_opt=max(1, args.seq),
            seq_max=max(1, args.seq),
        )
    )
    add_snapshot(
        snapshots,
        stage="onnx_cuda_session",
        start=start,
        device=device,
        note=",".join(runner.active_providers),
    )

    if not args.skip_dry_run:
        batch = max(1, args.batch)
        seq = max(1, args.seq)
        with torch.inference_mode():
            input_ids = torch.zeros(
                (batch, model.num_audio_codebook, seq),
                device=device,
                dtype=torch.long,
            )
            audio_mask = torch.zeros((batch, seq), device=device, dtype=torch.bool)
            attention_mask = torch.ones((batch, 1, seq, seq), device=device, dtype=torch.bool)
            add_snapshot(snapshots, stage="dry_run_inputs", start=start, device=device, note=f"batch={batch},seq={seq}")

            inputs_embeds = model.prepare_embed_inputs(input_ids, audio_mask)
            add_snapshot(snapshots, stage="dry_run_embeds", start=start, device=device)

            hidden_states = runner.run_step(inputs_embeds, attention_mask)
            add_snapshot(snapshots, stage="dry_run_onnx_step", start=start, device=device)

            semantic_logits, acoustic_logits = model.compute_audio_logits(hidden_states)
            add_snapshot(snapshots, stage="dry_run_logits", start=start, device=device)

            del input_ids, audio_mask, attention_mask, inputs_embeds, hidden_states
            del semantic_logits, acoustic_logits
        gc.collect()
        torch.cuda.empty_cache()
        add_snapshot(snapshots, stage="after_dry_run_cleanup", start=start, device=device)

    report = {
        "pid": os.getpid(),
        "backend": "onnx-cuda",
        "precision": args.precision,
        "device": str(device),
        "checkpoint_dir": str(checkpoint_dir),
        "onnx_path": str(onnx_path),
        "snapshots": [asdict(item) for item in snapshots],
    }
    print_table(snapshots)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report: {args.json_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
