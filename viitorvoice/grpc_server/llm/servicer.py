from __future__ import annotations

import grpc

from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.llm.runtime import LLMRuntime
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2 as llm_pb2
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2_grpc as llm_pb2_grpc
from viitorvoice.grpc_server.servicer_utils import request_timeout, run_rpc


class ViiTorVoiceLLMServicer(llm_pb2_grpc.ViiTorVoiceLLMServiceServicer):
    def __init__(self, runtime: LLMRuntime, config) -> None:
        self.runtime = runtime
        self.config = config

    async def Health(
        self,
        request: common_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.HealthResponse:
        del context
        req_context = common.new_span(common.ensure_context(request.context, caller="llm"))
        return self.runtime.runtime_info(req_context, "ready")

    async def Generate(
        self,
        request: llm_pb2.LLMGenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> llm_pb2.LLMGenerateResponse:
        async def invoke() -> llm_pb2.LLMGenerateResponse:
            req_context = common.new_span(common.ensure_context(request.context, caller="llm"))
            common.log_event(
                "module_request_received",
                req_context,
                service="llm",
                rpc="Generate",
                input=_summarize_generate_request(request),
            )
            response = await self.runtime.submit(
                lambda: self.runtime.generate(request, req_context),
                timeout=request_timeout(request.generation, self.config.request_timeout_sec),
            )
            common.log_event("module_request_completed", req_context, service="llm", rpc="Generate")
            return response

        return await run_rpc(context, invoke)

    async def GenerateFromSemantic(
        self,
        request: llm_pb2.LLMGenerateFromSemanticRequest,
        context: grpc.aio.ServicerContext,
    ) -> llm_pb2.LLMGenerateFromSemanticResponse:
        async def invoke() -> llm_pb2.LLMGenerateFromSemanticResponse:
            req_context = common.new_span(common.ensure_context(request.context, caller="llm"))
            common.log_event(
                "module_request_received",
                req_context,
                service="llm",
                rpc="GenerateFromSemantic",
                input=_summarize_generate_from_semantic_request(request),
            )
            response = await self.runtime.submit(
                lambda: self.runtime.generate_from_semantic(request, req_context),
                timeout=request_timeout(request.generation, self.config.request_timeout_sec),
            )
            common.log_event("module_request_completed", req_context, service="llm", rpc="GenerateFromSemantic")
            return response

        return await run_rpc(context, invoke)

    async def GenerateLocalEdit(
        self,
        request: llm_pb2.LLMGenerateLocalEditRequest,
        context: grpc.aio.ServicerContext,
    ) -> llm_pb2.LLMGenerateLocalEditResponse:
        async def invoke() -> llm_pb2.LLMGenerateLocalEditResponse:
            req_context = common.new_span(common.ensure_context(request.context, caller="llm"))
            common.log_event(
                "module_request_received",
                req_context,
                service="llm",
                rpc="GenerateLocalEdit",
                input=_summarize_generate_local_edit_request(request),
            )
            response = await self.runtime.submit(
                lambda: self.runtime.generate_local_edit(request, req_context),
                timeout=request_timeout(request.generation, self.config.request_timeout_sec),
            )
            common.log_event("module_request_completed", req_context, service="llm", rpc="GenerateLocalEdit")
            return response

        return await run_rpc(context, invoke)


def _summarize_generate_request(request: llm_pb2.LLMGenerateRequest) -> dict:
    return {
        **common.summarize_text(request.condition),
        "ref": common.summarize_tensor(request.ref_audio_codebook),
        "generation": common.summarize_generation_config(request.generation),
    }


def _summarize_generate_from_semantic_request(request: llm_pb2.LLMGenerateFromSemanticRequest) -> dict:
    return {
        **common.summarize_text(request.condition),
        "ref": common.summarize_tensor(request.ref_audio_codebook),
        "semantic": common.summarize_tensor(request.target_semantic_tokens),
        "generation": common.summarize_generation_config(request.generation),
    }


def _summarize_generate_local_edit_request(request: llm_pb2.LLMGenerateLocalEditRequest) -> dict:
    return {
        "full_text": request.full_text,
        "full_text_chars": len(request.full_text),
        "target": common.summarize_tensor(request.target_audio_codebook),
        "editable_mask": common.summarize_tensor(request.editable_audio_mask),
        "ref": common.summarize_tensor(request.ref_audio_codebook),
        "ref_text": request.ref_text,
        "language": request.language,
        "instruct": request.instruct,
        "ref_text_mask_len": common.optional_int(request, "ref_text_mask_len", None),
        "edit_context_frames": common.optional_int(request, "edit_context_frames", None),
        "generation": common.summarize_generation_config(request.generation),
    }


