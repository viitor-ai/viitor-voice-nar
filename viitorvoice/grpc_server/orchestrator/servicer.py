from __future__ import annotations

import asyncio
import difflib
import re
from dataclasses import dataclass
from typing import Any

import grpc

from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.audio_io import pad_audio_input_silence, trim_audio_result_edges
from viitorvoice.grpc_server.config import DEFAULT_FRAME_RATE, OrchestratorTargets
from viitorvoice.grpc_server.orchestrator.clients import ModuleClients
from viitorvoice.grpc_server.orchestrator.runtime import OrchestratorRuntime
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2 as decoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2 as encoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2 as llm_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2_grpc as orch_pb2_grpc
from viitorvoice.grpc_server.servicer_utils import request_timeout, run_rpc
from viitorvoice.llm.text_utils import normalize_nvv_tags

TEXT_LOCAL_EDIT_EDGE_PADDING_SEC = 0.3


class ViiTorVoiceOrchestratorServicer(orch_pb2_grpc.ViiTorVoiceOrchestratorServiceServicer):
    def __init__(
        self,
        runtime: OrchestratorRuntime,
        clients: ModuleClients,
        config,
        targets: OrchestratorTargets,
    ) -> None:
        self.runtime = runtime
        self.clients = clients
        self.config = config
        self.targets = targets

    async def Health(
        self,
        request: common_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.HealthResponse:
        del context
        req_context = self._public_context(request.context)
        return self.runtime.runtime_info(req_context, "ready")

    async def EncodeAudio(
        self,
        request: encoder_pb2.EncodeAudioRequest,
        context: grpc.aio.ServicerContext,
    ) -> encoder_pb2.EncodeAudioResponse:
        async def invoke() -> encoder_pb2.EncodeAudioResponse:
            req_context = self._public_context(request.context)
            proxy = encoder_pb2.EncodeAudioRequest()
            proxy.CopyFrom(request)
            proxy.context.CopyFrom(
                common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
            )
            response = await self.clients.encoder.EncodeAudio(
                proxy,
                timeout=request_timeout(request.context, self.config.request_timeout_sec),
            )
            response.context.service = "orchestrator"
            return response

        return await run_rpc(context, invoke)

    async def Synthesize(
        self,
        request: orch_pb2.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> orch_pb2.SynthesizeResponse:
        async def invoke() -> orch_pb2.SynthesizeResponse:
            req_context = self._public_context(request.context)
            metrics: list[common_pb2.StageMetric] = []
            common.log_event("request_received", req_context, service="orchestrator", rpc="Synthesize")
            encoded_ref: common_pb2.Int64Tensor | None = None
            ref_codebook = request.ref_audio_codebook

            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="Synthesize",
                stage="orchestrator.receive",
                input={
                    **common.summarize_text(request.condition),
                    "generation": common.summarize_generation_config(request.generation),
                },
            ) as timer:
                timer.output["has_ref_tokens"] = bool(ref_codebook.values)
            _append_metric(metrics, timer.metric)

            if not ref_codebook.values:
                child = common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
                encode_req = encoder_pb2.EncodeAudioRequest(
                    context=child,
                    audio=request.ref_audio,
                    preprocess_prompt=common.optional_bool(request.generation, "preprocess_prompt", True),
                    return_audio=False,
                    can_trim_long_audio=True,
                )
                with common.StageTimer(
                    req_context,
                    service="orchestrator",
                    rpc="Synthesize",
                    stage="encoder.encode_ref_audio",
                    input=common.summarize_audio_input(request.ref_audio),
                    target=self.targets.encoder_target,
                ) as timer:
                    encode_resp = await self.clients.encoder.EncodeAudio(
                        encode_req,
                        timeout=request_timeout(request.generation, self.config.request_timeout_sec),
                    )
                    ref_codebook = encode_resp.audio_codebook
                    encoded_ref = encode_resp.audio_codebook
                    timer.output.update(common.summarize_tensor(ref_codebook))
                _append_metric(metrics, timer.metric)
                metrics.extend(encode_resp.context.metrics)

            child = common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
            llm_req = llm_pb2.LLMGenerateRequest(context=child)
            llm_req.condition.CopyFrom(request.condition)
            llm_req.ref_audio_codebook.CopyFrom(ref_codebook)
            llm_req.generation.CopyFrom(request.generation)
            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="Synthesize",
                stage="llm.generate",
                input={
                    **common.summarize_text(request.condition),
                    "generation": common.summarize_generation_config(llm_req.generation),
                },
                target=self.targets.llm_target,
            ) as timer:
                llm_resp = await self.clients.llm.Generate(
                    llm_req,
                    timeout=request_timeout(request.generation, self.config.request_timeout_sec),
                )
                timer.output.update({"chunks": len(llm_resp.generated_audio_codebooks)})
            _append_metric(metrics, timer.metric)
            metrics.extend(llm_resp.context.metrics)

            decode_resp = await self._decode_generated(
                req_context,
                "Synthesize",
                llm_resp.generated_audio_codebooks or [llm_resp.generated_audio_codebook],
                request.generation,
                request.output_format,
                metrics,
            )

            with common.StageTimer(req_context, service="orchestrator", rpc="Synthesize", stage="orchestrator.respond") as timer:
                response = orch_pb2.SynthesizeResponse(
                    context=common.response_context(req_context, service="orchestrator", metrics=metrics),
                    audio=decode_resp.audio,
                )
                if request.return_tokens:
                    response.generated_audio_codebook.CopyFrom(llm_resp.generated_audio_codebook)
                    if encoded_ref is not None:
                        response.ref_audio_codebook.CopyFrom(encoded_ref)
                timer.output.update(common.summarize_audio_result(response.audio))
            _append_metric(metrics, timer.metric)
            del response.context.metrics[:]
            response.context.metrics.extend(metrics)
            common.log_event("request_completed", req_context, service="orchestrator", rpc="Synthesize")
            return response

        return await run_rpc(context, invoke)

    async def SemanticToWav(
        self,
        request: orch_pb2.SemanticToWavRequest,
        context: grpc.aio.ServicerContext,
    ) -> orch_pb2.SemanticToWavResponse:
        async def invoke() -> orch_pb2.SemanticToWavResponse:
            req_context = self._public_context(request.context)
            metrics: list[common_pb2.StageMetric] = []
            common.log_event("request_received", req_context, service="orchestrator", rpc="SemanticToWav")
            ref_codebook = request.ref_audio_codebook
            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="SemanticToWav",
                stage="orchestrator.receive",
                input={
                    **common.summarize_text(request.condition),
                    "generation": common.summarize_generation_config(request.generation),
                },
            ) as timer:
                timer.output.update(common.summarize_tensor(request.target_semantic_tokens))
            _append_metric(metrics, timer.metric)
            if not ref_codebook.values:
                child = common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
                encode_req = encoder_pb2.EncodeAudioRequest(
                    context=child,
                    audio=request.ref_audio,
                    preprocess_prompt=common.optional_bool(request.generation, "preprocess_prompt", True),
                    return_audio=False,
                    can_trim_long_audio=True,
                )
                with common.StageTimer(
                    req_context,
                    service="orchestrator",
                    rpc="SemanticToWav",
                    stage="encoder.encode_ref_audio",
                    input=common.summarize_audio_input(request.ref_audio),
                    target=self.targets.encoder_target,
                ) as timer:
                    encode_resp = await self.clients.encoder.EncodeAudio(
                        encode_req,
                        timeout=request_timeout(request.generation, self.config.request_timeout_sec),
                    )
                    ref_codebook = encode_resp.audio_codebook
                    timer.output.update(common.summarize_tensor(ref_codebook))
                _append_metric(metrics, timer.metric)
                metrics.extend(encode_resp.context.metrics)

            child = common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
            llm_req = llm_pb2.LLMGenerateFromSemanticRequest(context=child)
            llm_req.condition.CopyFrom(request.condition)
            llm_req.ref_audio_codebook.CopyFrom(ref_codebook)
            llm_req.target_semantic_tokens.CopyFrom(request.target_semantic_tokens)
            llm_req.generation.CopyFrom(request.generation)
            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="SemanticToWav",
                stage="llm.generate_from_semantic",
                input={
                    **common.summarize_text(request.condition),
                    "semantic": common.summarize_tensor(request.target_semantic_tokens),
                    "generation": common.summarize_generation_config(llm_req.generation),
                },
                target=self.targets.llm_target,
            ) as timer:
                llm_resp = await self.clients.llm.GenerateFromSemantic(
                    llm_req,
                    timeout=request_timeout(request.generation, self.config.request_timeout_sec),
                )
                timer.output.update(
                    {
                        "semantic_matches_target": llm_resp.semantic_matches_target,
                        "remaining_mask_tokens": llm_resp.remaining_mask_tokens,
                    }
                )
            _append_metric(metrics, timer.metric)
            metrics.extend(llm_resp.context.metrics)

            decode_resp = await self._decode_generated(
                req_context,
                "SemanticToWav",
                [llm_resp.generated_audio_codebook],
                request.generation,
                request.output_format,
                metrics,
            )
            response = orch_pb2.SemanticToWavResponse(
                context=common.response_context(req_context, service="orchestrator", metrics=metrics),
                audio=decode_resp.audio,
                semantic_matches_target=llm_resp.semantic_matches_target,
                remaining_mask_tokens=llm_resp.remaining_mask_tokens,
            )
            if request.return_tokens:
                response.generated_audio_codebook.CopyFrom(llm_resp.generated_audio_codebook)
            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="SemanticToWav",
                stage="orchestrator.respond",
            ) as timer:
                timer.output.update(common.summarize_audio_result(response.audio))
            _append_metric(metrics, timer.metric)
            del response.context.metrics[:]
            response.context.metrics.extend(metrics)
            common.log_event("request_completed", req_context, service="orchestrator", rpc="SemanticToWav")
            return response

        return await run_rpc(context, invoke)

    async def AlignForEdit(
        self,
        request: orch_pb2.AlignForEditRequest,
        context: grpc.aio.ServicerContext,
    ) -> orch_pb2.AlignForEditResponse:
        async def invoke() -> orch_pb2.AlignForEditResponse:
            req_context = self._public_context(request.context)
            proxy = orch_pb2.AlignForEditRequest()
            proxy.CopyFrom(request)
            proxy.granularity = _resolve_alignment_granularity(
                language=request.language,
                text=request.original_text,
                explicit=common.granularity_name(request.granularity),
            )
            return await asyncio.to_thread(self.runtime.align_for_edit, proxy, req_context)

        return await run_rpc(context, invoke)

    async def LocalEdit(
        self,
        request: orch_pb2.LocalEditRequest,
        context: grpc.aio.ServicerContext,
    ) -> orch_pb2.LocalEditResponse:
        async def invoke() -> orch_pb2.LocalEditResponse:
            req_context = self._public_context(request.context)
            return await self._run_local_edit(request, req_context, "LocalEdit")

        return await run_rpc(context, invoke)

    async def TextLocalEdit(
        self,
        request: orch_pb2.TextLocalEditRequest,
        context: grpc.aio.ServicerContext,
    ) -> orch_pb2.TextLocalEditResponse:
        async def invoke() -> orch_pb2.TextLocalEditResponse:
            req_context = self._public_context(request.context)
            metrics: list[common_pb2.StageMetric] = []
            original_text = normalize_nvv_tags(request.original_text)
            edited_text = normalize_nvv_tags(request.edited_text)
            common.log_event("request_received", req_context, service="orchestrator", rpc="TextLocalEdit")

            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="TextLocalEdit",
                stage="orchestrator.diff_text",
                input={
                    "original_chars": len(original_text),
                    "edited_chars": len(edited_text),
                    "original_text": original_text,
                    "edited_text": edited_text,
                    "language": request.language,
                },
            ) as timer:
                if not original_text:
                    raise ValueError("original_text is required.")
                if not edited_text:
                    raise ValueError("edited_text is required.")
                if original_text == edited_text:
                    raise ValueError("edited_text is identical to original_text.")
                timer.output["changed"] = True
                timer.output.update(
                    _summarize_text_diff(
                        original_text,
                        edited_text,
                        language=request.language,
                    )
                )
            _append_metric(metrics, timer.metric)

            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="TextLocalEdit",
                stage="orchestrator.pad_source_audio",
                input={
                    **common.summarize_audio_input(request.source_audio),
                    "edge_padding_sec": TEXT_LOCAL_EDIT_EDGE_PADDING_SEC,
                },
            ) as timer:
                padded_source_audio = await asyncio.to_thread(
                    pad_audio_input_silence,
                    request.source_audio,
                    padding_sec=TEXT_LOCAL_EDIT_EDGE_PADDING_SEC,
                )
                timer.output.update(common.summarize_audio_input(padded_source_audio))
            _append_metric(metrics, timer.metric)

            align_req = orch_pb2.AlignForEditRequest(
                context=common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator"),
                source_audio=padded_source_audio,
                original_text=original_text,
                language=request.language,
                granularity=_resolve_alignment_granularity(
                    language=request.language,
                    text=original_text,
                    explicit=request.align_granularity,
                ),
            )
            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="TextLocalEdit",
                stage="aligner.align",
                input={
                    **common.summarize_audio_input(padded_source_audio),
                    "original_text": original_text,
                    "language": request.language,
                    "granularity": common.granularity_name(align_req.granularity),
                },
            ) as timer:
                align_resp = await asyncio.to_thread(self.runtime.align_for_edit, align_req, align_req.context)
                alignments = list(align_resp.alignments)
                timer.output["alignment_items"] = len(alignments)
                timer.output["alignments"] = _summarize_alignments(alignments)
            _append_metric(metrics, timer.metric)
            metrics.extend(align_resp.context.metrics)

            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="TextLocalEdit",
                stage="orchestrator.diff_to_edits",
                input={
                    "alignment_items": len(alignments),
                    "original_text": original_text,
                    "edited_text": edited_text,
                },
            ) as timer:
                edits = _diff_text_to_edits(
                    original_text=original_text,
                    edited_text=edited_text,
                    alignments=alignments,
                    language=request.language,
                )
                timer.output["edit_segments"] = len(edits)
                timer.output["edits"] = _summarize_edits(edits, alignments)
            _append_metric(metrics, timer.metric)

            local_req = orch_pb2.LocalEditRequest(
                context=req_context,
                source_audio=padded_source_audio,
                original_text=original_text,
                language=request.language,
                alignments=alignments,
                edits=edits,
                return_tokens=bool(request.return_debug),
                output_format=request.output_format,
            )
            local_req.generation.CopyFrom(request.generation)
            for field in (
                "padding_ms",
                "expand_mask_ratio",
                "length_mode",
                "manual_duration",
                "manual_frames",
                "length_scale",
                "min_mask_frames",
                "edit_context_frames",
                "edit_ref_context_frames",
                "preprocess_source_audio",
            ):
                if request.HasField(field):
                    setattr(local_req, field, getattr(request, field))
            # Keep the synthetic 300ms edges intact until TextLocalEdit trims them explicitly.
            local_req.postprocess_output = False

            local_resp = await self._run_local_edit(
                local_req,
                req_context,
                "TextLocalEdit",
                initial_metrics=metrics,
                log_lifecycle=False,
            )
            metrics = list(local_resp.context.metrics)
            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc="TextLocalEdit",
                stage="orchestrator.trim_output_audio",
                input={
                    **common.summarize_audio_result(local_resp.audio),
                    "edge_trim_sec": TEXT_LOCAL_EDIT_EDGE_PADDING_SEC,
                },
            ) as timer:
                trimmed_audio = await asyncio.to_thread(
                    trim_audio_result_edges,
                    local_resp.audio,
                    trim_sec=TEXT_LOCAL_EDIT_EDGE_PADDING_SEC,
                )
                timer.output.update(common.summarize_audio_result(trimmed_audio))
            _append_metric(metrics, timer.metric)
            response = orch_pb2.TextLocalEditResponse(
                context=common.response_context(req_context, service="orchestrator", metrics=metrics),
                audio=trimmed_audio,
                remaining_mask_tokens=local_resp.remaining_mask_tokens,
                warnings=local_resp.warnings,
            )
            if request.return_debug:
                response.alignments.extend(_shift_alignment_times(alignments, -TEXT_LOCAL_EDIT_EDGE_PADDING_SEC))
                response.edits.extend(edits)
            common.log_event("request_completed", req_context, service="orchestrator", rpc="TextLocalEdit")
            return response

        return await run_rpc(context, invoke)

    async def _run_local_edit(
        self,
        request: orch_pb2.LocalEditRequest,
        req_context: common_pb2.RequestContext,
        rpc_name: str,
        *,
        initial_metrics: list[common_pb2.StageMetric] | None = None,
        log_lifecycle: bool = True,
    ) -> orch_pb2.LocalEditResponse:
        metrics: list[common_pb2.StageMetric] = []
        if initial_metrics:
            metrics.extend(initial_metrics)
        if log_lifecycle:
            common.log_event("request_received", req_context, service="orchestrator", rpc=rpc_name)
        with common.StageTimer(
            req_context,
            service="orchestrator",
            rpc=rpc_name,
            stage="orchestrator.receive",
            input={
                "alignment_items": len(request.alignments),
                "edit_segments": len(request.edits),
                "original_text": request.original_text,
                "language": request.language,
                "edits": _summarize_edits(request.edits, request.alignments),
                "edit_params": _summarize_edit_params(request),
                "generation": _summarize_generation(request.generation),
            },
        ) as timer:
            timer.output["has_source_tokens"] = bool(request.source_audio_codebook.values)
        _append_metric(metrics, timer.metric)
        source_codebook = request.source_audio_codebook
        encoded_source: common_pb2.Int64Tensor | None = None
        source_duration = (
            float(request.source_audio_duration_sec)
            if request.HasField("source_audio_duration_sec") and request.source_audio_duration_sec > 0
            else 0.0
        )
        if not source_codebook.values:
            child = common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
            encode_req = encoder_pb2.EncodeAudioRequest(
                context=child,
                audio=request.source_audio,
                preprocess_prompt=common.optional_bool(request, "preprocess_source_audio", False),
                return_audio=False,
                can_trim_long_audio=False,
            )
            with common.StageTimer(
                req_context,
                service="orchestrator",
                rpc=rpc_name,
                stage="encoder.encode_source_audio",
                input=common.summarize_audio_input(request.source_audio),
                target=self.targets.encoder_target,
            ) as timer:
                encode_resp = await self.clients.encoder.EncodeAudio(
                    encode_req,
                    timeout=request_timeout(request.generation, self.config.request_timeout_sec),
                )
                source_codebook = encode_resp.audio_codebook
                encoded_source = encode_resp.audio_codebook
                source_duration = float(source_codebook.shape[-1]) / DEFAULT_FRAME_RATE
                timer.output.update(common.summarize_tensor(source_codebook))
            _append_metric(metrics, timer.metric)
            metrics.extend(encode_resp.context.metrics)
        elif source_duration <= 0:
            source_duration = float(source_codebook.shape[-1]) / DEFAULT_FRAME_RATE

        source_tokens = common.normalize_audio_codebook(
            common.tensor_from_proto(source_codebook, name="source_audio_codebook", required=True),
            name="source_audio_codebook",
        )
        with common.StageTimer(
            req_context,
            service="orchestrator",
            rpc=rpc_name,
            stage="orchestrator.build_edit_plan",
            input={
                "alignment_items": len(request.alignments),
                "edit_segments": len(request.edits),
                "source_duration_sec": source_duration,
                "source_shape": list(source_tokens.shape),
                "edit_params": _summarize_edit_params(request),
            },
        ) as timer:
            plan, items = await asyncio.to_thread(
                self.runtime.build_local_edit_plan,
                request,
                source_tokens,
                source_duration,
            )
            timer.output.update(_summarize_edit_plan(plan, len(items)))
        _append_metric(metrics, timer.metric)

        override_ref = common.tensor_from_proto(request.ref_audio_codebook, name="ref_audio_codebook")
        if override_ref is not None:
            ref_tokens = common.normalize_audio_codebook(override_ref, name="ref_audio_codebook")
        else:
            ref_tokens = plan.source_tokens if request.ref_text else plan.reference_tokens
        child = common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
        llm_req = llm_pb2.LLMGenerateLocalEditRequest(
            context=child,
            full_text=plan.span.edited_full_text,
            target_audio_codebook=common.tensor_to_proto(plan.target_tokens),
            editable_audio_mask=common.tensor_to_proto(plan.editable_audio_mask.long()),
            ref_audio_codebook=common.tensor_to_proto(ref_tokens),
            ref_text=request.ref_text,
            language=request.language,
            instruct=request.instruct,
        )
        llm_req.generation.CopyFrom(request.generation)
        edit_context_frames = common.optional_int(request, "edit_context_frames", None)
        if edit_context_frames is not None:
            llm_req.edit_context_frames = edit_context_frames
        if request.HasField("ref_text_mask_len"):
            llm_req.ref_text_mask_len = request.ref_text_mask_len
        with common.StageTimer(
            req_context,
            service="orchestrator",
            rpc=rpc_name,
            stage="llm.generate_local_edit",
            input={
                "full_text": llm_req.full_text,
                "full_text_chars": len(llm_req.full_text),
                "target": common.summarize_tensor(llm_req.target_audio_codebook),
                "editable_mask": common.summarize_tensor(llm_req.editable_audio_mask),
                "editable_mask_ranges": _mask_ranges(plan.editable_audio_mask),
                "ref": common.summarize_tensor(llm_req.ref_audio_codebook),
                "ref_text": llm_req.ref_text,
                "language": llm_req.language,
                "instruct": llm_req.instruct,
                "edit_context_frames": common.optional_int(llm_req, "edit_context_frames", None),
                "generation": _summarize_generation(llm_req.generation),
            },
            target=self.targets.llm_target,
        ) as timer:
            llm_resp = await self.clients.llm.GenerateLocalEdit(
                llm_req,
                timeout=request_timeout(request.generation, self.config.request_timeout_sec),
            )
            timer.output.update(
                {
                    "remaining_mask_tokens": llm_resp.remaining_mask_tokens,
                    "edited_shape": list(llm_resp.edited_audio_codebook.shape),
                }
            )
        _append_metric(metrics, timer.metric)
        metrics.extend(llm_resp.context.metrics)

        decode_resp = await self._decode_generated(
            req_context,
            rpc_name,
            [llm_resp.edited_audio_codebook],
            request.generation,
            request.output_format,
            metrics,
            postprocess_override=common.optional_bool(request, "postprocess_output", None),
        )
        response = orch_pb2.LocalEditResponse(
            context=common.response_context(req_context, service="orchestrator", metrics=metrics),
            audio=decode_resp.audio,
            edited_text=plan.span.edited_full_text,
            alignments=[common.alignment_item_to_proto(item) for item in items],
            remaining_mask_tokens=llm_resp.remaining_mask_tokens,
        )
        if request.return_tokens:
            response.edited_audio_codebook.CopyFrom(llm_resp.edited_audio_codebook)
            if encoded_source is not None:
                response.source_audio_codebook.CopyFrom(encoded_source)
        with common.StageTimer(
            req_context,
            service="orchestrator",
            rpc=rpc_name,
            stage="orchestrator.respond",
        ) as timer:
            timer.output.update(common.summarize_audio_result(response.audio))
        _append_metric(metrics, timer.metric)
        del response.context.metrics[:]
        response.context.metrics.extend(metrics)
        if log_lifecycle:
            common.log_event("request_completed", req_context, service="orchestrator", rpc=rpc_name)
        return response

    async def _decode_generated(
        self,
        req_context: common_pb2.RequestContext,
        rpc: str,
        codebooks,
        generation: common_pb2.GenerationConfig,
        output_format: int,
        metrics: list[common_pb2.StageMetric],
        postprocess_override: bool | None = None,
    ) -> decoder_pb2.DecodeAudioResponse:
        child = common.child_context(req_context, parent_span_id=req_context.span_id, caller="orchestrator")
        postprocess = (
            common.generation_config_from_proto(generation).postprocess_output
            if postprocess_override is None
            else postprocess_override
        )
        decode_req = decoder_pb2.DecodeAudioRequest(
            context=child,
            postprocess_output=bool(postprocess),
            output_format=output_format,
        )
        decode_req.audio_codebooks.extend(codebooks)
        with common.StageTimer(
            req_context,
            service="orchestrator",
            rpc=rpc,
            stage="decoder.decode",
            input={"chunks": len(decode_req.audio_codebooks)},
            target=self.targets.decoder_target,
        ) as timer:
            decode_resp = await self.clients.decoder.DecodeAudio(
                decode_req,
                timeout=request_timeout(generation, self.config.request_timeout_sec),
            )
            timer.output.update(common.summarize_audio_result(decode_resp.audio))
        _append_metric(metrics, timer.metric)
        metrics.extend(decode_resp.context.metrics)
        return decode_resp

    def _public_context(self, context: common_pb2.RequestContext) -> common_pb2.RequestContext:
        return common.new_span(common.ensure_context(context, caller="orchestrator"))


def _append_metric(metrics: list[common_pb2.StageMetric], metric: common_pb2.StageMetric | None) -> None:
    if metric is not None:
        metrics.append(metric)


def _shift_alignment_times(
    alignments: list[common_pb2.AlignmentItem],
    offset_sec: float,
) -> list[common_pb2.AlignmentItem]:
    shifted: list[common_pb2.AlignmentItem] = []
    for item in alignments:
        copy = common_pb2.AlignmentItem()
        copy.CopyFrom(item)
        copy.start_time = max(0.0, float(copy.start_time) + float(offset_sec))
        copy.end_time = max(0.0, float(copy.end_time) + float(offset_sec))
        shifted.append(copy)
    return shifted


_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_JA_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_KO_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]")
_WORD_DIFF_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class _DiffToken:
    text: str
    start: int
    end: int

    @property
    def key(self) -> str:
        return self.text.casefold()


@dataclass(frozen=True)
class _TextDiff:
    granularity: str
    opcodes: list[tuple[str, int, int, int, int]]
    changed_spans: list[tuple[int, int]]


def _normalize_edit_language(language: str | None) -> str | None:
    if language is None:
        return None
    text = str(language).strip().lower().replace("_", "-")
    aliases = {
        "english": "en",
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "zho": "zh",
        "cmn": "zh",
        "cn": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "japanese": "ja",
        "jpn": "ja",
        "jp": "ja",
        "ja-jp": "ja",
        "korean": "ko",
        "kor": "ko",
        "kr": "ko",
        "ko-kr": "ko",
    }
    return aliases.get(text, text or None)


def _resolve_alignment_granularity(*, language: str | None, text: str, explicit: str | None) -> int:
    explicit_text = (explicit or "").strip().lower()
    if explicit_text in {"word", "words"}:
        return common_pb2.ALIGNMENT_GRANULARITY_WORD
    if explicit_text in {"char", "chars", "character", "characters"}:
        return common_pb2.ALIGNMENT_GRANULARITY_CHARACTER

    lang = _normalize_edit_language(language)
    if lang in {"zh", "yue", "ja", "ko"}:
        return common_pb2.ALIGNMENT_GRANULARITY_CHARACTER
    if lang == "en":
        return common_pb2.ALIGNMENT_GRANULARITY_WORD
    if _CJK_CHAR_RE.search(text) or _JA_KANA_RE.search(text) or _KO_HANGUL_RE.search(text):
        return common_pb2.ALIGNMENT_GRANULARITY_CHARACTER
    if _ASCII_WORD_RE.search(text):
        return common_pb2.ALIGNMENT_GRANULARITY_WORD
    return common_pb2.ALIGNMENT_GRANULARITY_CHARACTER


def _has_field(message: Any, field: str) -> bool:
    try:
        return bool(message.HasField(field))
    except ValueError:
        return False


def _summarize_generation(generation: common_pb2.GenerationConfig) -> dict[str, Any]:
    return common.summarize_generation_config(generation)


def _summarize_edit_params(request: orch_pb2.LocalEditRequest) -> dict[str, Any]:
    optional_fields = (
        "padding_ms",
        "expand_mask_ratio",
        "length_mode",
        "manual_duration",
        "manual_frames",
        "length_scale",
        "min_mask_frames",
        "edit_context_frames",
        "edit_ref_context_frames",
        "ref_text_mask_len",
        "preprocess_source_audio",
        "postprocess_output",
        "source_audio_duration_sec",
    )
    params = {field: getattr(request, field) for field in optional_fields if _has_field(request, field)}
    params["align_granularity"] = request.align_granularity
    params["has_ref_audio_codebook"] = bool(request.ref_audio_codebook.values)
    params["has_ref_text"] = bool(request.ref_text)
    params["instruct_chars"] = len(request.instruct)
    return params


def _summarize_text_diff(original_text: str, edited_text: str, *, language: str | None) -> dict[str, Any]:
    diff = _build_text_diff(original_text=original_text, edited_text=edited_text, language=language)
    return {
        "granularity": diff.granularity,
        "opcodes": [
            {
                "tag": tag,
                "original_span": [start, end],
                "edited_span": [edited_start, edited_end],
                "original": original_text[start:end],
                "edited": edited_text[edited_start:edited_end],
            }
            for tag, start, end, edited_start, edited_end in diff.opcodes
        ],
        "changed_spans": [
            {
                "original_span": [start, end],
                "original": original_text[start:end],
            }
            for start, end in diff.changed_spans
        ],
    }


def _summarize_alignments(alignments: list[common_pb2.AlignmentItem]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(item.index),
            "text": item.text,
            "start_time": round(float(item.start_time), 4),
            "end_time": round(float(item.end_time), 4),
            "start_char": int(item.start_char) if item.has_start_char else None,
            "end_char": int(item.end_char) if item.has_end_char else None,
            "kind": item.kind,
        }
        for item in alignments
    ]


def _summarize_edits(
    edits: list[common_pb2.EditSegment] | Any,
    alignments: list[common_pb2.AlignmentItem] | Any,
) -> list[dict[str, Any]]:
    by_index = {int(item.index): item for item in alignments}
    summaries: list[dict[str, Any]] = []
    for edit in edits:
        indices = [int(idx) for idx in edit.selection.alignment_indices]
        selected = [by_index[idx] for idx in indices if idx in by_index]
        summaries.append(
            {
                "alignment_indices": indices,
                "selected_text": "".join(item.text for item in selected),
                "replacement_text": edit.replacement_text,
                "start_time": round(min((float(item.start_time) for item in selected), default=0.0), 4),
                "end_time": round(max((float(item.end_time) for item in selected), default=0.0), 4),
                "start_char": min((int(item.start_char) for item in selected if item.has_start_char), default=None),
                "end_char": max((int(item.end_char) for item in selected if item.has_end_char), default=None),
            }
        )
    return summaries


def _mask_ranges(mask: Any) -> list[dict[str, int]]:
    values = [bool(item) for item in mask.detach().cpu().tolist()]
    ranges: list[dict[str, int]] = []
    start: int | None = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append({"start": start, "end": idx, "frames": idx - start})
            start = None
    if start is not None:
        ranges.append({"start": start, "end": len(values), "frames": len(values) - start})
    return ranges


def _summarize_edit_plan(plan: Any, alignment_count: int) -> dict[str, Any]:
    segments = []
    for idx, segment in enumerate(plan.span.segments):
        replacement_frames = int(plan.replacement_frames[idx]) if idx < len(plan.replacement_frames) else 0
        segments.append(
            {
                "index": idx,
                "selected_indices": list(segment.selected_indices),
                "selected_text": segment.selected_text,
                "replacement_text": segment.replacement_text,
                "time": [round(float(segment.start_time), 4), round(float(segment.end_time), 4)],
                "padded_time": [
                    round(float(segment.padded_start_time), 4),
                    round(float(segment.padded_end_time), 4),
                ],
                "source_frames": [int(segment.start_frame), int(segment.end_frame)],
                "old_frames": int(segment.old_frames),
                "replacement_frames": replacement_frames,
            }
        )
    return {
        "alignment_items": alignment_count,
        "audio_duration_sec": round(float(plan.audio_duration), 4),
        "source_shape": list(plan.source_tokens.shape),
        "target_shape": list(plan.target_tokens.shape),
        "reference_shape": list(plan.reference_tokens.shape),
        "old_total_frames": int(plan.old_total_frames),
        "new_total_frames": int(plan.new_total_frames),
        "total_replacement_frames": int(plan.total_replacement_frames),
        "editable_frames": int(plan.editable_audio_mask.sum().item()),
        "editable_mask_ranges": _mask_ranges(plan.editable_audio_mask),
        "edited_full_text": plan.span.edited_full_text,
        "segments": segments,
    }


def _diff_text_to_edits(
    *,
    original_text: str,
    edited_text: str,
    alignments: list[common_pb2.AlignmentItem],
    language: str | None,
) -> list[common_pb2.EditSegment]:
    diff = _build_text_diff(original_text=original_text, edited_text=edited_text, language=language)
    opcodes = diff.opcodes
    spans = diff.changed_spans
    index_groups = _merge_overlapping_index_groups(
        [_alignment_indices_for_char_span(alignments, start, end) for start, end in spans]
    )
    edits: list[common_pb2.EditSegment] = []
    for indices in index_groups:
        selected_start, selected_end = _char_span_for_alignment_indices(alignments, indices)
        edited_start, edited_end = _edited_span_for_original_span(
            opcodes,
            selected_start,
            selected_end,
            original_text=original_text,
        )
        edits.append(
            common_pb2.EditSegment(
                selection=common_pb2.EditSelection(alignment_indices=indices),
                replacement_text=edited_text[edited_start:edited_end],
            )
        )
    return edits


def _build_text_diff(*, original_text: str, edited_text: str, language: str | None) -> _TextDiff:
    if _should_use_word_diff(language=language, text=f"{original_text}\n{edited_text}"):
        return _build_word_text_diff(original_text, edited_text)
    opcodes = difflib.SequenceMatcher(a=original_text, b=edited_text, autojunk=False).get_opcodes()
    return _TextDiff(
        granularity="char",
        opcodes=opcodes,
        changed_spans=_changed_text_spans(original_text, opcodes),
    )


def _should_use_word_diff(*, language: str | None, text: str) -> bool:
    lang = _normalize_edit_language(language)
    if lang == "en":
        return True
    if lang in {"zh", "yue", "ja", "ko"}:
        return False
    if _CJK_CHAR_RE.search(text) or _JA_KANA_RE.search(text) or _KO_HANGUL_RE.search(text):
        return False
    return bool(_ASCII_WORD_RE.search(text))


def _build_word_text_diff(original_text: str, edited_text: str) -> _TextDiff:
    original_tokens = _word_diff_tokens(original_text)
    edited_tokens = _word_diff_tokens(edited_text)
    if not original_tokens or not edited_tokens:
        opcodes = difflib.SequenceMatcher(a=original_text, b=edited_text, autojunk=False).get_opcodes()
        return _TextDiff(
            granularity="char",
            opcodes=opcodes,
            changed_spans=_changed_text_spans(original_text, opcodes),
        )

    token_opcodes = difflib.SequenceMatcher(
        a=[token.key for token in original_tokens],
        b=[token.key for token in edited_tokens],
        autojunk=False,
    ).get_opcodes()
    opcodes = [
        (
            tag,
            *_token_opcode_span(original_text, original_tokens, start, end),
            *_token_opcode_span(edited_text, edited_tokens, edited_start, edited_end),
        )
        for tag, start, end, edited_start, edited_end in token_opcodes
    ]
    spans = [
        _expand_word_changed_text_span(
            original_text=original_text,
            edited_text=edited_text,
            original_tokens=original_tokens,
            edited_tokens=edited_tokens,
            start=start,
            end=end,
            edited_start=edited_start,
            edited_end=edited_end,
        )
        for tag, start, end, edited_start, edited_end in token_opcodes
        if tag != "equal"
    ]
    if not spans:
        raise ValueError("edited_text is identical to original_text.")
    return _TextDiff(
        granularity="word",
        opcodes=opcodes,
        changed_spans=[
            (start, end)
            for start, end, _, _ in _merge_overlapping_text_spans(original_text, spans)
        ],
    )


def _word_diff_tokens(text: str) -> list[_DiffToken]:
    return [_DiffToken(match.group(0), match.start(), match.end()) for match in _WORD_DIFF_TOKEN_RE.finditer(text)]


def _token_opcode_span(text: str, tokens: list[_DiffToken], start: int, end: int) -> tuple[int, int]:
    if start < end:
        return tokens[start].start, tokens[end - 1].end
    if start <= 0:
        return 0, 0
    if start >= len(tokens):
        return len(text), len(text)
    return tokens[start].start, tokens[start].start


def _expand_word_changed_text_span(
    *,
    original_text: str,
    edited_text: str,
    original_tokens: list[_DiffToken],
    edited_tokens: list[_DiffToken],
    start: int,
    end: int,
    edited_start: int,
    edited_end: int,
) -> tuple[int, int, int, int]:
    if start < end:
        original_start = original_tokens[start].start
        original_end = original_tokens[end - 1].end
    elif start > 0:
        original_start = original_tokens[start - 1].start
        original_end = original_tokens[start - 1].end
    elif start < len(original_tokens):
        original_start = original_tokens[start].start
        original_end = original_tokens[start].end
    else:
        raise ValueError("original_text cannot be empty for text local edit.")

    if edited_start < edited_end:
        edited_char_start = edited_tokens[edited_start].start
        edited_char_end = edited_tokens[edited_end - 1].end
    elif edited_start > 0:
        edited_char_start = edited_tokens[edited_start - 1].start
        edited_char_end = edited_tokens[edited_start - 1].end
    elif edited_start < len(edited_tokens):
        edited_char_start = edited_tokens[edited_start].start
        edited_char_end = edited_tokens[edited_start].end
    else:
        edited_char_start = edited_char_end = min(len(edited_text), original_start)

    if start == end and edited_start < edited_end:
        if start > 0 and edited_start > 0:
            edited_char_start = edited_tokens[edited_start - 1].start
        elif start < len(original_tokens) and edited_end < len(edited_tokens):
            edited_char_end = edited_tokens[edited_end].end

    if edited_start == edited_end and start < end:
        if start > 0 and edited_start > 0:
            original_start = original_tokens[start - 1].start
            edited_char_start = edited_tokens[edited_start - 1].start
        elif end < len(original_tokens) and edited_start < len(edited_tokens):
            original_end = original_tokens[end].end
            edited_char_end = edited_tokens[edited_start].end

    return original_start, original_end, edited_char_start, edited_char_end


def _changed_text_spans(
    original_text: str,
    opcodes: list[tuple[str, int, int, int, int]],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int, int, int]] = []
    for tag, start, end, edited_start, edited_end in opcodes:
        if tag == "equal":
            continue
        spans.append(
            _expand_changed_text_span(
                original_text=original_text,
                start=start,
                end=end,
                edited_start=edited_start,
                edited_end=edited_end,
            )
        )
    if not spans:
        raise ValueError("edited_text is identical to original_text.")
    return [(start, end) for start, end, _, _ in _merge_overlapping_text_spans(original_text, spans)]


def _expand_changed_text_span(
    *,
    original_text: str,
    start: int,
    end: int,
    edited_start: int,
    edited_end: int,
) -> tuple[int, int, int, int]:
    if start == end:
        if start > 0:
            return start - 1, start, edited_start - 1, edited_end
        if original_text:
            return 0, 1, edited_start, edited_end + 1
        raise ValueError("original_text cannot be empty for text local edit.")

    if edited_start == edited_end:
        if start > 0:
            return start - 1, end, edited_start - 1, edited_end
        if end < len(original_text):
            return start, end + 1, edited_start, edited_end + 1
        raise ValueError("Deleting the entire text is not supported by local edit.")

    return start, end, edited_start, edited_end


def _merge_overlapping_text_spans(
    original_text: str,
    spans: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    ordered = sorted(spans, key=lambda span: (span[0], span[1]))
    merged: list[tuple[int, int, int, int]] = []
    for start, end, edited_start, edited_end in ordered:
        if not merged:
            merged.append((start, end, edited_start, edited_end))
            continue
        separator = original_text[merged[-1][1] : start]
        if start > merged[-1][1] and any(char.isalnum() for char in separator):
            merged.append((start, end, edited_start, edited_end))
            continue
        prev_start, prev_end, prev_edited_start, prev_edited_end = merged[-1]
        merged[-1] = (
            prev_start,
            max(prev_end, end),
            min(prev_edited_start, edited_start),
            max(prev_edited_end, edited_end),
        )
    return merged


def _merge_overlapping_index_groups(groups: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for group in groups:
        values = sorted(set(int(item) for item in group))
        if not values:
            continue
        if not merged:
            merged.append(values)
            continue
        previous = merged[-1]
        if not (set(previous) & set(values)) and values[0] > previous[-1] + 1:
            merged.append(values)
            continue
        merged[-1] = sorted(set(previous) | set(values))
    return merged


def _alignment_indices_for_char_span(
    alignments: list[common_pb2.AlignmentItem],
    start: int,
    end: int,
) -> list[int]:
    if not alignments:
        raise ValueError("Alignment returned no items.")
    indexed: list[tuple[int, int, int]] = []
    for item in alignments:
        if item.has_start_char and item.has_end_char and item.end_char > item.start_char:
            indexed.append((int(item.index), int(item.start_char), int(item.end_char)))
    if not indexed:
        raise ValueError("Alignment items do not include character spans.")

    selected = [idx for idx, item_start, item_end in indexed if item_end > start and item_start < end]
    if selected:
        return sorted(set(selected))

    after = [(item_start, idx) for idx, item_start, _ in indexed if item_start >= end]
    if after:
        return [min(after)[1]]
    before = [(item_end, idx) for idx, _, item_end in indexed if item_end <= start]
    if before:
        return [max(before)[1]]
    raise ValueError(f"Could not map text span [{start}, {end}) to alignment indices.")


def _char_span_for_alignment_indices(
    alignments: list[common_pb2.AlignmentItem],
    indices: list[int],
) -> tuple[int, int]:
    wanted = set(indices)
    selected = [
        item
        for item in alignments
        if int(item.index) in wanted and item.has_start_char and item.has_end_char and item.end_char > item.start_char
    ]
    if not selected:
        raise ValueError(f"Alignment indices do not include character spans: {indices}")
    return min(int(item.start_char) for item in selected), max(int(item.end_char) for item in selected)


def _edited_span_for_original_span(
    opcodes: list[tuple[str, int, int, int, int]],
    start: int,
    end: int,
    *,
    original_text: str = "",
) -> tuple[int, int]:
    edited_ranges: list[tuple[int, int]] = []
    for tag, orig_start, orig_end, edited_start, edited_end in opcodes:
        if tag == "insert":
            if start <= orig_start <= end or (
                orig_start >= end and original_text[end:orig_start].strip() == ""
            ):
                edited_ranges.append((edited_start, edited_end))
            continue
        overlap_start = max(start, orig_start)
        overlap_end = min(end, orig_end)
        if overlap_start >= overlap_end:
            continue
        if tag == "equal":
            edited_ranges.append(
                (
                    edited_start + (overlap_start - orig_start),
                    edited_start + (overlap_end - orig_start),
                )
            )
        else:
            edited_ranges.append((edited_start, edited_end))
    if not edited_ranges:
        raise ValueError(f"Could not map original text span [{start}, {end}) to edited text.")
    return min(item[0] for item in edited_ranges), max(item[1] for item in edited_ranges)


