from __future__ import annotations

import numpy as np

from viitorvoice.codec import DualCodecPackedDecoder
from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.audio_utils import cross_fade_chunks
from viitorvoice.grpc_server.audio_io import post_process_wave
from viitorvoice.grpc_server.base_runtime import SingleWorkerRuntime
from viitorvoice.grpc_server.config import DEFAULT_SAMPLE_RATE, codec_backend
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2 as decoder_pb2


class DecoderRuntime(SingleWorkerRuntime):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._decoder: DualCodecPackedDecoder | None = None

    def warmup(self) -> None:
        self._get_decoder()

    def runtime_info(self, context: common_pb2.RequestContext, message: str = "") -> common_pb2.HealthResponse:
        backends: list[str] = []
        if self._decoder is not None:
            backends.extend(f"decoder:{provider}" for provider in self._decoder.active_providers)
        return common_pb2.HealthResponse(
            context=common.response_context(context, service="decoder"),
            state=common_pb2.SERVICE_STATE_READY if self._started else common_pb2.SERVICE_STATE_STARTING,
            message=message or ("ready" if self._started else "stopped"),
            version="inference-grpc-v2-decoder",
            active_backends=backends,
            queued_jobs=self.queue_size,
        )

    def decode_audio(
        self,
        request: decoder_pb2.DecodeAudioRequest,
        context: common_pb2.RequestContext,
    ) -> decoder_pb2.DecodeAudioResponse:
        response: decoder_pb2.DecodeAudioResponse
        with common.StageTimer(
            context,
            service="decoder",
            rpc="DecodeAudio",
            stage="decode_audio",
            input={"chunks": len(request.audio_codebooks)},
        ) as timer:
            if not request.audio_codebooks:
                raise ValueError("At least one audio_codebook is required.")
            tokens = [
                common.normalize_audio_codebook(
                    common.tensor_from_proto(codebook, name=f"audio_codebooks[{idx}]", required=True),
                    name=f"audio_codebooks[{idx}]",
                )
                for idx, codebook in enumerate(request.audio_codebooks)
            ]
            decoder = self._get_decoder()
            waves = [
                post_process_wave(audio, postprocess_output=bool(request.postprocess_output))
                for audio in decoder.decode(tokens)
            ]
            if len(waves) == 1:
                wave = waves[0]
            else:
                channel_waves = [item.reshape(1, -1) for item in waves]
                wave = cross_fade_chunks(channel_waves, DEFAULT_SAMPLE_RATE).reshape(-1).astype(np.float32)
            response = decoder_pb2.DecodeAudioResponse(
                context=common.response_context(context, service="decoder"),
                audio=common.audio_result(wave),
            )
            timer.output.update(common.summarize_audio_result(response.audio))
        if timer.metric is not None:
            response.context.metrics.append(timer.metric)
        return response

    def _get_decoder(self) -> DualCodecPackedDecoder:
        if self._decoder is None:
            cfg = self.config.decoder
            self._decoder = DualCodecPackedDecoder(
                onnx_path=cfg.onnx_path,
                ep_config={
                    "backend": codec_backend(cfg.backend),
                    "device_id": self.config.device_id,
                    "trt_fp16_enable": True,
                },
                trt_cache_root=cfg.trt_cache_root,
                max_frames=cfg.max_frames,
                silence_frames=cfg.silence_frames,
                silence_codec_path=cfg.silence_codec_path,
            )
        return self._decoder
