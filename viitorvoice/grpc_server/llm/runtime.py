from __future__ import annotations

import torch

from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.base_runtime import SingleWorkerRuntime
from viitorvoice.grpc_server.config import norm_backend
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2 as llm_pb2
from viitorvoice.llm import ViiTorVoiceLLMGenerator


class LLMRuntime(SingleWorkerRuntime):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._llm: ViiTorVoiceLLMGenerator | None = None

    def warmup(self) -> None:
        self._get_llm()

    def runtime_info(self, context: common_pb2.RequestContext, message: str = "") -> common_pb2.HealthResponse:
        backends: list[str] = []
        if self._llm is not None:
            backends.extend(f"llm:{provider}" for provider in self._llm.active_providers)
        return common_pb2.HealthResponse(
            context=common.response_context(context, service="llm"),
            state=common_pb2.SERVICE_STATE_READY if self._started else common_pb2.SERVICE_STATE_STARTING,
            message=message or ("ready" if self._started else "stopped"),
            version="inference-grpc-v2-llm",
            active_backends=backends,
            queued_jobs=self.queue_size,
        )

    def generate(
        self,
        request: llm_pb2.LLMGenerateRequest,
        context: common_pb2.RequestContext,
    ) -> llm_pb2.LLMGenerateResponse:
        response: llm_pb2.LLMGenerateResponse
        with common.StageTimer(
            context,
            service="llm",
            rpc="Generate",
            stage="llm_generate",
            input={
                **common.summarize_text(request.condition),
                "ref": common.summarize_tensor(request.ref_audio_codebook),
                "generation": common.summarize_generation_config(request.generation),
            },
        ) as timer:
            llm = self._get_llm()
            ref_tokens = common.normalize_audio_codebook(
                common.tensor_from_proto(request.ref_audio_codebook, name="ref_audio_codebook", required=True),
                name="ref_audio_codebook",
            )
            gen_config = common.generation_config_from_proto(request.generation)
            condition = request.condition
            common.log_event(
                "llm_text_prepared",
                context,
                service="llm",
                rpc="Generate",
                stage="llm_prepare_text",
                input=llm.describe_text_preparation(
                    text=condition.text,
                    language=condition.language or None,
                    ref_text=condition.ref_text or None,
                    instruct=condition.instruct or None,
                    generation_config=gen_config,
                ),
            )
            generated = llm.generate(
                text=condition.text,
                language=condition.language or None,
                ref_text=condition.ref_text or None,
                ref_audio_tokens=ref_tokens,
                instruct=condition.instruct or None,
                duration=common.generation_float(request.generation, "duration"),
                speed=common.generation_float(request.generation, "speed"),
                allow_missing_ref_text=bool(condition.allow_missing_ref_text or not condition.ref_text),
                ref_text_mask_len=int(condition.ref_text_mask_len) if condition.ref_text_mask_len > 0 else None,
                generation_config=gen_config,
            )[0]
            chunks = _strip_generated(generated, llm)
            response = llm_pb2.LLMGenerateResponse(
                context=common.response_context(context, service="llm"),
                generated_audio_codebook=common.tensor_to_proto(chunks[0] if chunks else None),
            )
            response.generated_audio_codebooks.extend(common.tensor_to_proto(chunk) for chunk in chunks)
            timer.output.update({"chunks": len(chunks), **common.summarize_tensor(response.generated_audio_codebook)})
        if timer.metric is not None:
            response.context.metrics.append(timer.metric)
        return response

    def generate_from_semantic(
        self,
        request: llm_pb2.LLMGenerateFromSemanticRequest,
        context: common_pb2.RequestContext,
    ) -> llm_pb2.LLMGenerateFromSemanticResponse:
        response: llm_pb2.LLMGenerateFromSemanticResponse
        with common.StageTimer(
            context,
            service="llm",
            rpc="GenerateFromSemantic",
            stage="llm_generate_from_semantic",
            input={
                **common.summarize_text(request.condition),
                "ref": common.summarize_tensor(request.ref_audio_codebook),
                "semantic": common.summarize_tensor(request.target_semantic_tokens),
                "generation": common.summarize_generation_config(request.generation),
            },
        ) as timer:
            llm = self._get_llm()
            ref_tokens = common.normalize_audio_codebook(
                common.tensor_from_proto(request.ref_audio_codebook, name="ref_audio_codebook", required=True),
                name="ref_audio_codebook",
            )
            semantic = common.normalize_semantic_tokens(
                common.tensor_from_proto(request.target_semantic_tokens, name="target_semantic_tokens", required=True),
                name="target_semantic_tokens",
            )
            condition = request.condition
            gen_config = common.generation_config_from_proto(request.generation)
            common.log_event(
                "llm_text_prepared",
                context,
                service="llm",
                rpc="GenerateFromSemantic",
                stage="llm_prepare_text",
                input=llm.describe_text_preparation(
                    text=condition.text,
                    language=condition.language or None,
                    ref_text=condition.ref_text or None,
                    instruct=condition.instruct or None,
                    generation_config=gen_config,
                ),
            )
            generated = llm.generate_from_semantic_tokens(
                text=condition.text,
                language=condition.language or None,
                ref_text=condition.ref_text or None,
                ref_audio_tokens=ref_tokens,
                semantic_tokens=semantic,
                instruct=condition.instruct or None,
                allow_missing_ref_text=bool(condition.allow_missing_ref_text or not condition.ref_text),
                ref_text_mask_len=int(condition.ref_text_mask_len) if condition.ref_text_mask_len > 0 else None,
                generation_config=gen_config,
            )
            semantic_equal = bool(torch.equal(generated[0].cpu(), semantic.cpu()))
            remaining = common.count_remaining_mask_tokens(generated, llm.audio_mask_ids)
            stripped = common.strip_structural_audio_frames(generated, llm)
            response = llm_pb2.LLMGenerateFromSemanticResponse(
                context=common.response_context(context, service="llm"),
                generated_audio_codebook=common.tensor_to_proto(stripped),
                semantic_matches_target=semantic_equal,
                remaining_mask_tokens=remaining,
            )
            timer.output.update(
                {
                    "semantic_matches_target": semantic_equal,
                    "remaining_mask_tokens": remaining,
                    **common.summarize_tensor(response.generated_audio_codebook),
                }
            )
        if timer.metric is not None:
            response.context.metrics.append(timer.metric)
        return response

    def generate_local_edit(
        self,
        request: llm_pb2.LLMGenerateLocalEditRequest,
        context: common_pb2.RequestContext,
    ) -> llm_pb2.LLMGenerateLocalEditResponse:
        response: llm_pb2.LLMGenerateLocalEditResponse
        with common.StageTimer(
            context,
            service="llm",
            rpc="GenerateLocalEdit",
            stage="llm_generate_local_edit",
            input={
                "full_text": request.full_text,
                "text_chars": len(request.full_text),
                "ref_text": request.ref_text,
                "language": request.language,
                "instruct": request.instruct,
                "ref_text_mask_len": common.optional_int(request, "ref_text_mask_len", None),
                "edit_context_frames": common.optional_int(request, "edit_context_frames", None),
                "target": common.summarize_tensor(request.target_audio_codebook),
                "mask": common.summarize_tensor(request.editable_audio_mask),
                "ref": common.summarize_tensor(request.ref_audio_codebook),
                "generation": common.summarize_generation_config(request.generation),
            },
        ) as timer:
            llm = self._get_llm()
            target = common.normalize_audio_codebook(
                common.tensor_from_proto(request.target_audio_codebook, name="target_audio_codebook", required=True),
                name="target_audio_codebook",
            )
            editable_mask = common.tensor_from_proto(
                request.editable_audio_mask,
                name="editable_audio_mask",
                required=True,
            )
            if editable_mask is None:
                raise ValueError("editable_audio_mask is required.")
            ref_tokens = common.normalize_audio_codebook(
                common.tensor_from_proto(request.ref_audio_codebook, name="ref_audio_codebook", required=True),
                name="ref_audio_codebook",
            )
            gen_config = common.generation_config_from_proto(request.generation)
            common.log_event(
                "llm_text_prepared",
                context,
                service="llm",
                rpc="GenerateLocalEdit",
                stage="llm_prepare_text",
                input=llm.describe_text_preparation(
                    text=request.full_text,
                    language=request.language or None,
                    ref_text=request.ref_text or None,
                    instruct=request.instruct or None,
                    generation_config=gen_config,
                    full_text_field="full_text",
                ),
            )
            generated = llm.generate_edit_masked(
                full_text=request.full_text,
                target_audio_tokens=target,
                editable_audio_mask=editable_mask.bool(),
                language=request.language or None,
                ref_text=request.ref_text or None,
                ref_audio_tokens=ref_tokens,
                instruct=request.instruct or None,
                allow_missing_ref_text=not bool(request.ref_text),
                ref_text_mask_len=common.optional_int(request, "ref_text_mask_len", None),
                generation_config=gen_config,
                edit_context_frames=common.optional_int(request, "edit_context_frames", None),
            )
            remaining = common.count_remaining_mask_tokens(generated, llm.audio_mask_ids)
            stripped = common.strip_structural_audio_frames(generated, llm)
            response = llm_pb2.LLMGenerateLocalEditResponse(
                context=common.response_context(context, service="llm"),
                edited_audio_codebook=common.tensor_to_proto(stripped),
                remaining_mask_tokens=remaining,
            )
            timer.output.update({"remaining_mask_tokens": remaining, **common.summarize_tensor(stripped)})
        if timer.metric is not None:
            response.context.metrics.append(timer.metric)
        return response

    def _get_llm(self) -> ViiTorVoiceLLMGenerator:
        if self._llm is None:
            cfg = self.config.llm
            self._llm = ViiTorVoiceLLMGenerator(
                checkpoint_dir=cfg.checkpoint_dir,
                onnx_path=cfg.onnx_path,
                backend=norm_backend(cfg.backend),
                precision=cfg.precision,
                device_id=self.config.device_id,
                trt_cache_root=cfg.trt_cache_root,
                batch_min=cfg.batch_min,
                batch_opt=cfg.batch_opt,
                batch_max=cfg.batch_max,
                seq_min=cfg.seq_min,
                seq_opt=cfg.seq_opt,
                seq_max=cfg.seq_max,
                strict_trt=cfg.strict_trt,
            )
        return self._llm


def _strip_generated(generated: torch.Tensor | list[torch.Tensor], llm: ViiTorVoiceLLMGenerator) -> list[torch.Tensor]:
    chunks = generated if isinstance(generated, list) else [generated]
    return [common.strip_structural_audio_frames(chunk, llm) for chunk in chunks]
