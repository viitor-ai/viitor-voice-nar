#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viitorvoice import paths
from viitorvoice.llm.runtime import clear_proxies


INPUT_EMBEDS = "inputs_embeds"
ATTENTION_MASK = "attention_mask"
OUTPUT_NAME = "hidden_states"


@dataclass
class CompareMetrics:
    shape: list[int]
    dtype: str
    max_abs: float
    mean_abs: float
    median_abs: float
    p90_abs: float
    p99_abs: float
    max_rel: float
    mean_rel: float
    cosine: float
    allclose: bool
    atol: float
    rtol: float


def ort_type_to_np_dtype(type_name: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(bool)": np.dtype(np.bool_),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int64)": np.dtype(np.int64),
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported ONNX tensor type: {type_name}")
    return mapping[type_name]


def dtype_name(dtype: np.dtype) -> str:
    if dtype == np.dtype(np.float16):
        return "fp16"
    if dtype == np.dtype(np.float32):
        return "fp32"
    if dtype == np.dtype(np.float64):
        return "fp64"
    if dtype == np.dtype(np.bool_):
        return "bool"
    return str(dtype)


def shape_spec(batch: int, seq: int, hidden: int) -> str:
    return (
        f"{INPUT_EMBEDS}:{batch}x{seq}x{hidden},"
        f"{ATTENTION_MASK}:{batch}x1x{seq}x{seq}"
    )


def parse_dims(dims: str) -> tuple[int, ...]:
    text = dims.strip()
    if text in {"", "scalar"}:
        return ()
    return tuple(int(part) for part in text.split("x") if part)


def make_attention_mask(
    *,
    batch: int,
    seq: int,
    dtype: np.dtype,
    mode: str,
) -> np.ndarray:
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"{ATTENTION_MASK} should be floating point for this ONNX export, got {dtype}.")
    mask = np.zeros((batch, 1, seq, seq), dtype=dtype)
    if mode == "zeros":
        return mask
    if mode == "causal":
        blocked = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        mask_value = np.finfo(dtype).min
        mask[:, :, blocked] = mask_value
        return mask
    raise ValueError(f"Unsupported mask mode: {mode}")


def make_inputs(
    *,
    batch: int,
    seq: int,
    hidden: int,
    embeds_dtype: np.dtype,
    mask_dtype: np.dtype,
    seed: int,
    mask_mode: str,
    input_scale: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    embeds = rng.standard_normal((batch, seq, hidden)).astype(np.float32) * float(input_scale)
    if embeds_dtype == np.dtype(np.float16):
        embeds = embeds.astype(np.float16)
    elif embeds_dtype != np.dtype(np.float32):
        embeds = embeds.astype(embeds_dtype)
    mask = make_attention_mask(batch=batch, seq=seq, dtype=mask_dtype, mode=mask_mode)
    return {
        INPUT_EMBEDS: np.ascontiguousarray(embeds),
        ATTENTION_MASK: np.ascontiguousarray(mask),
    }


def run_command(command: list[str], *, log_path: Path) -> None:
    start = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + f"\n\nELAPSED_SECONDS: {elapsed:.3f}\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}). See log: {log_path}")


def build_engine(
    *,
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    work_dir: Path,
    precision: str,
    hidden: int,
    batch_min: int,
    batch_opt: int,
    batch_max: int,
    seq_min: int,
    seq_opt: int,
    seq_max: int,
    allow_tf32: bool,
    strongly_typed: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={shape_spec(batch_min, seq_min, hidden)}",
        f"--optShapes={shape_spec(batch_opt, seq_opt, hidden)}",
        f"--maxShapes={shape_spec(batch_max, seq_max, hidden)}",
    ]
    precision = precision.lower()
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "bf16":
        command.append("--bf16")
    elif precision != "fp32":
        raise ValueError(f"Unsupported precision: {precision}")
    if not allow_tf32:
        command.append("--noTF32")
    if strongly_typed:
        command.append("--stronglyTyped")
    command.extend(extra_args)
    command.append("--skipInference")
    run_command(command, log_path=work_dir / "trtexec_build.log")
    return command


def run_onnx(onnx_path: Path, inputs: dict[str, np.ndarray], device_id: int) -> np.ndarray:
    providers: list[Any] = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        ("CPUExecutionProvider", {}),
    ]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"CUDAExecutionProvider is not active: {active}")
    outputs = session.run([OUTPUT_NAME], inputs)
    return np.asarray(outputs[0])


def run_trtexec_engine(
    *,
    trtexec: str,
    engine_path: Path,
    inputs: dict[str, np.ndarray],
    output_json: Path,
    work_dir: Path,
    batch: int,
    seq: int,
    hidden: int,
    extra_args: list[str],
) -> tuple[np.ndarray, list[str]]:
    input_files: dict[str, Path] = {}
    for name, value in inputs.items():
        path = work_dir / f"{name}.raw"
        np.ascontiguousarray(value).tofile(path)
        input_files[name] = path

    load_inputs = ",".join(f"{name}:{path}" for name, path in input_files.items())
    command = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--shapes={shape_spec(batch, seq, hidden)}",
        f"--loadInputs={load_inputs}",
        "--iterations=1",
        "--warmUp=0",
        "--duration=0",
        f"--exportOutput={output_json}",
    ]
    command.extend(extra_args)
    run_command(command, log_path=work_dir / "trtexec_run.log")
    output = parse_trtexec_output(output_json)
    return output, command


def parse_trtexec_output(output_json: Path) -> np.ndarray:
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected trtexec output JSON format in {output_json}")
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("name") != OUTPUT_NAME:
            continue
        dims = parse_dims(str(item.get("dimensions", "")))
        values = np.asarray(item.get("values", []), dtype=np.float32)
        expected = math.prod(dims) if dims else 1
        if values.size != expected:
            raise ValueError(
                f"Output size mismatch for {OUTPUT_NAME}: values={values.size}, dims={dims}."
            )
        return values.reshape(dims)
    names = [item.get("name") for item in payload if isinstance(item, dict)]
    raise ValueError(f"Cannot find {OUTPUT_NAME!r} in trtexec output; available outputs: {names}")


def compare_outputs(
    onnx_output: np.ndarray,
    trt_output: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> CompareMetrics:
    if onnx_output.shape != trt_output.shape:
        raise ValueError(f"Output shape mismatch: onnx={onnx_output.shape}, trt={trt_output.shape}")
    onnx_f = onnx_output.astype(np.float64, copy=False)
    trt_f = trt_output.astype(np.float64, copy=False)
    diff = np.abs(onnx_f - trt_f)
    denom = np.maximum(np.abs(onnx_f), 1e-12)
    rel = diff / denom
    flat_a = onnx_f.reshape(-1)
    flat_b = trt_f.reshape(-1)
    norm = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    cosine = float(np.dot(flat_a, flat_b) / norm) if norm > 0 else float("nan")
    return CompareMetrics(
        shape=list(onnx_output.shape),
        dtype=str(onnx_output.dtype),
        max_abs=float(np.max(diff)),
        mean_abs=float(np.mean(diff)),
        median_abs=float(np.median(diff)),
        p90_abs=float(np.percentile(diff, 90)),
        p99_abs=float(np.percentile(diff, 99)),
        max_rel=float(np.max(rel)),
        mean_rel=float(np.mean(rel)),
        cosine=cosine,
        allclose=bool(np.allclose(onnx_f, trt_f, atol=atol, rtol=rtol)),
        atol=float(atol),
        rtol=float(rtol),
    )


def inspect_onnx_io(onnx_path: Path, device_id: int) -> tuple[np.dtype, np.dtype, list[str]]:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=[("CUDAExecutionProvider", {"device_id": device_id}), ("CPUExecutionProvider", {})],
    )
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    for name in (INPUT_EMBEDS, ATTENTION_MASK):
        if name not in inputs:
            raise RuntimeError(f"ONNX input {name!r} not found. Inputs: {sorted(inputs)}")
    if OUTPUT_NAME not in outputs:
        raise RuntimeError(f"ONNX output {OUTPUT_NAME!r} not found. Outputs: {sorted(outputs)}")
    return (
        ort_type_to_np_dtype(inputs[INPUT_EMBEDS].type),
        ort_type_to_np_dtype(inputs[ATTENTION_MASK].type),
        session.get_providers(),
    )


def parse_args() -> argparse.Namespace:
    default_checkpoint = paths.llm_checkpoint_dir("0p6_emotion")
    default_onnx = paths.llm_onnx_path(default_checkpoint)
    parser = argparse.ArgumentParser(
        description="Compare ViiTorVoice LLM backbone outputs from ONNX Runtime CUDA and a trtexec-built TensorRT plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-path", type=Path, default=Path(os.environ.get("VIITORVOICE_LLM_ONNX", default_onnx)))
    parser.add_argument(
        "--reference-onnx-path",
        type=Path,
        default=None,
        help="Optional FP32 reference ONNX. TensorRT plan is still built/loaded from --onnx-path.",
    )
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("test_outputs/llm_trtexec_compare"))
    parser.add_argument("--trtexec", default=shutil.which("trtexec") or "trtexec")
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("VIITORVOICE_DEVICE_ID", "0")))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--profile-batch-min", type=int, default=None)
    parser.add_argument("--profile-batch-opt", type=int, default=None)
    parser.add_argument("--profile-batch-max", type=int, default=None)
    parser.add_argument("--profile-seq-min", type=int, default=None)
    parser.add_argument("--profile-seq-opt", type=int, default=None)
    parser.add_argument("--profile-seq-max", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--allow-tf32", action="store_true", help="Do not pass --noTF32 to trtexec.")
    parser.add_argument("--strongly-typed", action="store_true", help="Pass --stronglyTyped to trtexec.")
    parser.add_argument("--force-build", action="store_true", help="Rebuild the TensorRT plan even if it exists.")
    parser.add_argument("--skip-build", action="store_true", help="Use an existing --engine-path without rebuilding.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask-mode", choices=["causal", "zeros"], default="causal")
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--trtexec-build-extra", nargs="*", default=[])
    parser.add_argument("--trtexec-run-extra", nargs="*", default=[])
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_proxies()
    onnx_path = args.onnx_path.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    reference_onnx_path = (
        args.reference_onnx_path.expanduser().resolve()
        if args.reference_onnx_path
        else onnx_path
    )
    if not reference_onnx_path.is_file():
        raise FileNotFoundError(f"Reference ONNX file not found: {reference_onnx_path}")
    trtexec_path = shutil.which(args.trtexec) or args.trtexec
    if not shutil.which(trtexec_path) and not Path(trtexec_path).is_file():
        raise FileNotFoundError(f"trtexec not found: {args.trtexec}")

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    engine_path = (
        args.engine_path.expanduser().resolve()
        if args.engine_path
        else work_dir / f"llm_b{args.batch}_s{args.seq}_{args.precision}.plan"
    )
    report_json = (
        args.report_json.expanduser().resolve()
        if args.report_json
        else work_dir / "compare_report.json"
    )

    embeds_dtype, mask_dtype, providers = inspect_onnx_io(onnx_path, args.device_id)
    inputs = make_inputs(
        batch=args.batch,
        seq=args.seq,
        hidden=args.hidden,
        embeds_dtype=embeds_dtype,
        mask_dtype=mask_dtype,
        seed=args.seed,
        mask_mode=args.mask_mode,
        input_scale=args.input_scale,
    )

    batch_min = args.profile_batch_min or args.batch
    batch_opt = args.profile_batch_opt or args.batch
    batch_max = args.profile_batch_max or args.batch
    seq_min = args.profile_seq_min or args.seq
    seq_opt = args.profile_seq_opt or args.seq
    seq_max = args.profile_seq_max or args.seq
    if not (batch_min <= args.batch <= batch_max):
        raise ValueError("Runtime batch must be within profile batch min/max.")
    if not (seq_min <= args.seq <= seq_max):
        raise ValueError("Runtime seq must be within profile seq min/max.")

    build_command: list[str] | None = None
    if args.skip_build:
        if not engine_path.is_file():
            raise FileNotFoundError(f"--skip-build was set but engine does not exist: {engine_path}")
    elif args.force_build or not engine_path.is_file():
        build_command = build_engine(
            trtexec=trtexec_path,
            onnx_path=onnx_path,
            engine_path=engine_path,
            work_dir=work_dir,
            precision=args.precision,
            hidden=args.hidden,
            batch_min=batch_min,
            batch_opt=batch_opt,
            batch_max=batch_max,
            seq_min=seq_min,
            seq_opt=seq_opt,
            seq_max=seq_max,
            allow_tf32=args.allow_tf32,
            strongly_typed=args.strongly_typed,
            extra_args=args.trtexec_build_extra,
        )

    onnx_start = time.monotonic()
    onnx_output = run_onnx(reference_onnx_path, inputs, args.device_id)
    onnx_seconds = time.monotonic() - onnx_start

    trt_start = time.monotonic()
    trt_output, run_command_line = run_trtexec_engine(
        trtexec=trtexec_path,
        engine_path=engine_path,
        inputs=inputs,
        output_json=work_dir / "trtexec_output.json",
        work_dir=work_dir,
        batch=args.batch,
        seq=args.seq,
        hidden=args.hidden,
        extra_args=args.trtexec_run_extra,
    )
    trt_seconds = time.monotonic() - trt_start

    metrics = compare_outputs(onnx_output, trt_output, atol=args.atol, rtol=args.rtol)
    report = {
        "onnx_path": str(onnx_path),
        "reference_onnx_path": str(reference_onnx_path),
        "engine_path": str(engine_path),
        "work_dir": str(work_dir),
        "providers": providers,
        "input_dtypes": {
            INPUT_EMBEDS: dtype_name(embeds_dtype),
            ATTENTION_MASK: dtype_name(mask_dtype),
        },
        "shape": {
            "batch": args.batch,
            "seq": args.seq,
            "hidden": args.hidden,
            "profile": {
                "batch_min": batch_min,
                "batch_opt": batch_opt,
                "batch_max": batch_max,
                "seq_min": seq_min,
                "seq_opt": seq_opt,
                "seq_max": seq_max,
            },
        },
        "precision": args.precision,
        "allow_tf32": args.allow_tf32,
        "strongly_typed": args.strongly_typed,
        "seed": args.seed,
        "mask_mode": args.mask_mode,
        "onnx_seconds": round(onnx_seconds, 4),
        "trtexec_seconds": round(trt_seconds, 4),
        "metrics": asdict(metrics),
        "build_command": build_command,
        "run_command": run_command_line,
        "logs": {
            "build": str(work_dir / "trtexec_build.log"),
            "run": str(work_dir / "trtexec_run.log"),
            "trtexec_output": str(work_dir / "trtexec_output.json"),
        },
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"ONNX path: {onnx_path}")
    print(f"Reference ONNX path: {reference_onnx_path}")
    print(f"TensorRT plan: {engine_path}")
    print(f"Shape: batch={args.batch}, seq={args.seq}, hidden={args.hidden}")
    print(f"Input dtypes: {INPUT_EMBEDS}={dtype_name(embeds_dtype)}, {ATTENTION_MASK}={dtype_name(mask_dtype)}")
    print(f"ONNX seconds: {onnx_seconds:.4f}")
    print(f"trtexec seconds: {trt_seconds:.4f}")
    print(
        "Diff: "
        f"max_abs={metrics.max_abs:.8g}, "
        f"mean_abs={metrics.mean_abs:.8g}, "
        f"p99_abs={metrics.p99_abs:.8g}, "
        f"max_rel={metrics.max_rel:.8g}, "
        f"cosine={metrics.cosine:.8g}, "
        f"allclose={metrics.allclose}"
    )
    print(f"Report: {report_json}")
    if args.fail_on_mismatch and not metrics.allclose:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
