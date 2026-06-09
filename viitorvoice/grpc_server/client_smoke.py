from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import grpc

from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2 as decoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2_grpc as decoder_pb2_grpc
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2 as encoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2_grpc as encoder_pb2_grpc
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2 as llm_pb2
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2_grpc as llm_pb2_grpc
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2_grpc as orch_pb2_grpc


DEFAULT_TEXT = (
    "I'm done. You've crossed my line again and again, "
    "treating my trust like a joke."
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal ViiTorVoice gRPC v2 smoke client.")
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument(
        "--service",
        choices=["orchestrator", "encoder", "llm", "decoder"],
        default="orchestrator",
    )
    parser.add_argument(
        "--mode",
        choices=["health", "encode", "synthesize", "semantic-to-wav", "local-edit", "text-local-edit"],
        default="health",
    )
    parser.add_argument("--audio", default="gaoyuliang.wav")
    parser.add_argument("--target-audio", default="test_outputs/without_ref_text_1p7_ft_v2.wav")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--edited-text", default="I'm done. You've crossed my line again and again, treating my patience like a joke.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", default="test_outputs/viitorvoice_grpc_server_smoke")
    parser.add_argument("--num-step", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--emotion-guidance-scale", type=float, default=0.0)
    parser.add_argument("--nvv-guidance-scale", type=float, default=0.0)
    parser.add_argument("--position-temperature", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    async with grpc.aio.insecure_channel(
        args.target,
        options=[
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ],
    ) as channel:
        summary: dict[str, object]
        if args.mode == "health":
            summary = await health(channel, args.service)
        elif args.mode == "encode":
            summary = await encode(channel, args.service, args)
        elif args.mode == "synthesize":
            summary = await synthesize(channel, args)
        elif args.mode == "semantic-to-wav":
            summary = await semantic_to_wav(channel, args)
        elif args.mode == "local-edit":
            summary = await local_edit(channel, args)
        else:
            summary = await text_local_edit(channel, args)
    report_path = output_dir / f"{args.service}_{args.mode}.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "report": str(report_path)}, ensure_ascii=False, indent=2))


async def health(channel: grpc.aio.Channel, service: str) -> dict[str, object]:
    request = common_pb2.HealthRequest(context=context())
    if service == "encoder":
        response = await encoder_pb2_grpc.ViiTorVoiceEncoderServiceStub(channel).Health(request)
    elif service == "llm":
        response = await llm_pb2_grpc.ViiTorVoiceLLMServiceStub(channel).Health(request)
    elif service == "decoder":
        response = await decoder_pb2_grpc.ViiTorVoiceDecoderServiceStub(channel).Health(request)
    else:
        response = await orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel).Health(request)
    return {
        "mode": "health",
        "service": service,
        "state": int(response.state),
        "message": response.message,
        "version": response.version,
        "trace_id": response.context.trace_id,
        "active_backends": list(response.active_backends),
    }


async def encode(channel: grpc.aio.Channel, service: str, args: argparse.Namespace) -> dict[str, object]:
    request = encoder_pb2.EncodeAudioRequest(
        context=context(),
        audio=common_pb2.AudioInput(audio_path=str(Path(args.audio).expanduser().resolve())),
        preprocess_prompt=True,
    )
    if service == "encoder":
        response = await encoder_pb2_grpc.ViiTorVoiceEncoderServiceStub(channel).EncodeAudio(request)
    else:
        response = await orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel).EncodeAudio(request)
    return {
        "mode": "encode",
        "service": service,
        "shape": list(response.audio_codebook.shape),
        "sample_rate": int(response.sample_rate),
        "trace_id": response.context.trace_id,
    }


async def synthesize(channel: grpc.aio.Channel, args: argparse.Namespace) -> dict[str, object]:
    response = await orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel).Synthesize(
        orch_pb2.SynthesizeRequest(
            context=context(),
            condition=common_pb2.TextCondition(
                text=args.text,
                language=args.language,
                allow_missing_ref_text=True,
                ref_text_mask_len=10,
            ),
            ref_audio=common_pb2.AudioInput(audio_path=str(Path(args.audio).expanduser().resolve())),
            generation=generation_config(args),
            return_tokens=True,
            output_format=common_pb2.AUDIO_FORMAT_WAV,
        ),
        timeout=float(args.timeout_sec),
    )
    path = Path(args.output_dir).expanduser().resolve() / "synthesize.wav"
    path.write_bytes(response.audio.audio_bytes)
    return {
        "mode": "synthesize",
        "audio_bytes": len(response.audio.audio_bytes),
        "duration_sec": response.audio.duration_sec,
        "tokens": list(response.generated_audio_codebook.shape),
        "remaining_metrics": len(response.context.metrics),
        "output": str(path),
        "trace_id": response.context.trace_id,
    }


async def semantic_to_wav(channel: grpc.aio.Channel, args: argparse.Namespace) -> dict[str, object]:
    orch = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel)
    enc = await orch.EncodeAudio(
        encoder_pb2.EncodeAudioRequest(
            context=context(),
            audio=common_pb2.AudioInput(audio_path=str(Path(args.audio).expanduser().resolve())),
            preprocess_prompt=True,
        )
    )
    target = await orch.EncodeAudio(
        encoder_pb2.EncodeAudioRequest(
            context=context(),
            audio=common_pb2.AudioInput(audio_path=str(Path(args.target_audio).expanduser().resolve())),
            preprocess_prompt=False,
        )
    )
    frames = int(target.audio_codebook.shape[1])
    semantic = common_pb2.Int64Tensor(values=list(target.audio_codebook.values[:frames]), shape=[frames])
    response = await orch.SemanticToWav(
        orch_pb2.SemanticToWavRequest(
            context=context(),
            condition=common_pb2.TextCondition(
                text=args.text,
                language=args.language,
                allow_missing_ref_text=True,
                ref_text_mask_len=10,
            ),
            ref_audio_codebook=enc.audio_codebook,
            target_semantic_tokens=semantic,
            generation=generation_config(args),
            return_tokens=True,
            output_format=common_pb2.AUDIO_FORMAT_WAV,
        ),
        timeout=float(args.timeout_sec),
    )
    path = Path(args.output_dir).expanduser().resolve() / "semantic_to_wav.wav"
    path.write_bytes(response.audio.audio_bytes)
    return {
        "mode": "semantic-to-wav",
        "audio_bytes": len(response.audio.audio_bytes),
        "duration_sec": response.audio.duration_sec,
        "semantic_matches_target": response.semantic_matches_target,
        "remaining_mask_tokens": response.remaining_mask_tokens,
        "output": str(path),
        "trace_id": response.context.trace_id,
    }


async def local_edit(channel: grpc.aio.Channel, args: argparse.Namespace) -> dict[str, object]:
    orch = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel)
    source = await orch.EncodeAudio(
        encoder_pb2.EncodeAudioRequest(
            context=context(),
            audio=common_pb2.AudioInput(audio_path=str(Path(args.audio).expanduser().resolve())),
            preprocess_prompt=False,
        )
    )
    duration = max(0.1, float(source.audio_codebook.shape[1]) / 25.0)
    response = await orch.LocalEdit(
        orch_pb2.LocalEditRequest(
            context=context(),
            source_audio_codebook=source.audio_codebook,
            source_audio_duration_sec=duration,
            original_text="hello world",
            language="en",
            alignments=[
                common_pb2.AlignmentItem(
                    index=0,
                    text="hello",
                    start_time=0.0,
                    end_time=min(0.8, duration),
                    start_char=0,
                    end_char=5,
                    has_start_char=True,
                    has_end_char=True,
                    kind="word",
                )
            ],
            edits=[
                common_pb2.EditSegment(
                    selection=common_pb2.EditSelection(alignment_indices=[0]),
                    replacement_text="goodbye",
                )
            ],
            generation=generation_config(args),
            min_mask_frames=6,
            edit_context_frames=40,
            edit_ref_context_frames=120,
            postprocess_output=True,
            return_tokens=True,
            output_format=common_pb2.AUDIO_FORMAT_WAV,
        ),
        timeout=float(args.timeout_sec),
    )
    path = Path(args.output_dir).expanduser().resolve() / "local_edit.wav"
    path.write_bytes(response.audio.audio_bytes)
    return {
        "mode": "local-edit",
        "audio_bytes": len(response.audio.audio_bytes),
        "duration_sec": response.audio.duration_sec,
        "edited_text": response.edited_text,
        "remaining_mask_tokens": response.remaining_mask_tokens,
        "output": str(path),
        "trace_id": response.context.trace_id,
    }


async def text_local_edit(channel: grpc.aio.Channel, args: argparse.Namespace) -> dict[str, object]:
    orch = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel)
    response = await orch.TextLocalEdit(
        orch_pb2.TextLocalEditRequest(
            context=context(),
            source_audio=common_pb2.AudioInput(audio_path=str(Path(args.audio).expanduser().resolve())),
            original_text=args.text,
            edited_text=args.edited_text,
            language=args.language,
            generation=generation_config(args),
            min_mask_frames=6,
            edit_context_frames=40,
            edit_ref_context_frames=120,
            postprocess_output=True,
            output_format=common_pb2.AUDIO_FORMAT_WAV,
            return_debug=True,
        ),
        timeout=float(args.timeout_sec),
    )
    path = Path(args.output_dir).expanduser().resolve() / "text_local_edit.wav"
    path.write_bytes(response.audio.audio_bytes)
    return {
        "mode": "text-local-edit",
        "audio_bytes": len(response.audio.audio_bytes),
        "duration_sec": response.audio.duration_sec,
        "remaining_mask_tokens": response.remaining_mask_tokens,
        "alignment_items": len(response.alignments),
        "edit_segments": len(response.edits),
        "output": str(path),
        "trace_id": response.context.trace_id,
    }


def context() -> common_pb2.RequestContext:
    return common_pb2.RequestContext(caller="viitorvoice_grpc_server_smoke")


def generation_config(args: argparse.Namespace) -> common_pb2.GenerationConfig:
    return common_pb2.GenerationConfig(
        num_steps=int(args.num_step),
        cfg_scale=float(args.guidance_scale),
        emotion_guidance_scale=float(args.emotion_guidance_scale),
        nvv_guidance_scale=float(args.nvv_guidance_scale),
        t_shift=0.1,
        layer_penalty_factor=5.0,
        position_temperature=float(args.position_temperature),
        class_temperature=0.0,
        preprocess_prompt=True,
        postprocess_output=True,
        audio_chunk_duration=15.0,
        audio_chunk_threshold=30.0,
        request_timeout_sec=int(args.timeout_sec),
    )


if __name__ == "__main__":
    asyncio.run(main())
