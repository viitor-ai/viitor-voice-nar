from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from viitorvoice.llm.runtime import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_ONNX_PATH,
    DEFAULT_TRT_CACHE_ROOT,
    LLMOnnxConfig,
    LLMOnnxStepRunner,
    clear_proxies,
    torch_dtype_from_precision,
)


@dataclass
class LLMModelConfig:
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    onnx_path: Path | None = None
    backend: str = "trt"
    precision: str = "fp16"
    device_id: int = 0
    trt_cache_root: Path | None = None
    batch_min: int = 1
    batch_opt: int = 2
    batch_max: int = 4
    seq_min: int = 16
    seq_opt: int = 512
    seq_max: int = 2048
    ep_options: Mapping[str, Any] = field(default_factory=dict)
    strict_trt: bool = False


@dataclass
class LLMForwardOutput:
    semantic_logits: torch.Tensor
    acoustic_logits: torch.Tensor


class ViiTorVoiceLLMModel(nn.Module):
    """Standalone ViiTorVoice LLM runtime around an exported ONNX backbone."""

    def __init__(
        self,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        onnx_path: str | Path | None = None,
        backend: str = "trt",
        precision: str = "fp16",
        device_id: int = 0,
        trt_cache_root: str | Path | None = None,
        batch_min: int = 1,
        batch_opt: int = 2,
        batch_max: int = 4,
        seq_min: int = 16,
        seq_opt: int = 512,
        seq_max: int = 2048,
        ep_options: Optional[Mapping[str, Any]] = None,
        strict_trt: bool = False,
    ) -> None:
        clear_proxies()
        super().__init__()
        self.runtime_config = LLMModelConfig(
            checkpoint_dir=Path(checkpoint_dir),
            onnx_path=Path(onnx_path) if onnx_path is not None else None,
            backend=backend,
            precision=precision,
            device_id=int(device_id),
            trt_cache_root=Path(trt_cache_root) if trt_cache_root is not None else None,
            batch_min=int(batch_min),
            batch_opt=int(batch_opt),
            batch_max=int(batch_max),
            seq_min=int(seq_min),
            seq_opt=int(seq_opt),
            seq_max=int(seq_max),
            ep_options=dict(ep_options or {}),
            strict_trt=bool(strict_trt),
        )
        self.checkpoint_dir = self.runtime_config.checkpoint_dir.expanduser().resolve()
        self.config = self._load_json(self.checkpoint_dir / "config.json")
        self.device_id = self.runtime_config.device_id
        self.device = torch.device(f"cuda:{self.device_id}")
        self.dtype = torch_dtype_from_precision(self.runtime_config.precision)

        self.num_audio_codebook = int(self.config["num_audio_codebook"])
        self.num_acoustic_codebooks = self.num_audio_codebook - 1
        self.semantic_codebook_size = int(self.config["audio_codebook_sizes"][0])
        self.acoustic_codebook_size = int(self.config["audio_codebook_sizes"][1])
        self.hidden_size = int(self.config["llm_config"]["hidden_size"])
        self.text_vocab_size = int(self.config["llm_config"]["vocab_size"])

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
        self.to(device=self.device, dtype=self.dtype)
        self._load_weights()
        self.eval()

        default_onnx = self.checkpoint_dir / ".cache" / "onnx_backbone" / "llm_backbone_dynamic.onnx"
        onnx_model_path = (
            self.runtime_config.onnx_path.expanduser().resolve()
            if self.runtime_config.onnx_path is not None
            else (default_onnx if default_onnx.is_file() else DEFAULT_ONNX_PATH)
        )
        trt_cache_root = (
            self.runtime_config.trt_cache_root.expanduser().resolve()
            if self.runtime_config.trt_cache_root is not None
            else (self.checkpoint_dir / ".cache" / "trt_cache")
        )
        if not trt_cache_root:
            trt_cache_root = DEFAULT_TRT_CACHE_ROOT
        self.step_runner = LLMOnnxStepRunner(
            LLMOnnxConfig(
                onnx_path=onnx_model_path,
                backend=self.runtime_config.backend,
                precision=self.runtime_config.precision,
                device_id=self.device_id,
                hidden_size=self.hidden_size,
                trt_cache_root=trt_cache_root,
                batch_min=self.runtime_config.batch_min,
                batch_opt=self.runtime_config.batch_opt,
                batch_max=self.runtime_config.batch_max,
                seq_min=self.runtime_config.seq_min,
                seq_opt=self.runtime_config.seq_opt,
                seq_max=self.runtime_config.seq_max,
                ep_options=self.runtime_config.ep_options,
                strict_trt=self.runtime_config.strict_trt,
            )
        )
        self.active_providers = self.step_runner.active_providers

    @torch.inference_mode()
    def forward_step(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> LLMForwardOutput:
        input_ids = input_ids.to(self.device, non_blocking=True)
        audio_mask = audio_mask.to(self.device, non_blocking=True).bool()
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        inputs_embeds = self.prepare_embed_inputs(input_ids, audio_mask)
        hidden_states = self.step_runner.run_step(inputs_embeds, attention_mask)
        semantic_logits, acoustic_logits = self.compute_audio_logits(hidden_states)
        return LLMForwardOutput(
            semantic_logits=semantic_logits,
            acoustic_logits=acoustic_logits,
        )

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
    def compute_audio_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _load_weights(self) -> None:
        model_path = self.checkpoint_dir / "model.safetensors"
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
                tensor = handle.get_tensor(key).to(device=self.device, dtype=parameter.dtype)
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"Weight shape mismatch for {key}: checkpoint={tuple(tensor.shape)}, "
                        f"runtime={tuple(parameter.shape)}."
                    )
                parameter.data.copy_(tensor)
            if "acoustic_codebook_offsets" in available:
                offsets = handle.get_tensor("acoustic_codebook_offsets").to(
                    device=self.device,
                    dtype=torch.long,
                )
                if tuple(offsets.shape) == tuple(self.acoustic_codebook_offsets.shape):
                    self.acoustic_codebook_offsets.copy_(offsets)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


