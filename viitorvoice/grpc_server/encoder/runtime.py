from __future__ import annotations

from viitorvoice.codec import DualCodecOnnxEncoder, DualCodecTorchEncoder
from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.audio_io import encode_wav_bytes, load_audio_for_encoder
from viitorvoice.grpc_server.base_runtime import SingleWorkerRuntime
from viitorvoice.grpc_server.config import DEFAULT_FRAME_RATE, DEFAULT_SAMPLE_RATE, codec_backend
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2 as encoder_pb2


class EncoderRuntime(SingleWorkerRuntime):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._encoder: DualCodecOnnxEncoder | DualCodecTorchEncoder | None = None

    def warmup(self) -> None:
        self._get_encoder()

    def runtime_info(self, context: common_pb2.RequestContext, message: str = "") -> common_pb2.HealthResponse:
        backends: list[str] = []
        if self._encoder is not None:
            backends.extend(f"encoder:{provider}" for provider in self._encoder.active_providers)
        return common_pb2.HealthResponse(
            context=common.response_context(context, service="encoder"),
            state=common_pb2.SERVICE_STATE_READY if self._started else common_pb2.SERVICE_STATE_STARTING,
            message=message or ("ready" if self._started else "stopped"),
            version="inference-grpc-v2-encoder",
            active_backends=backends,
            queued_jobs=self.queue_size,
        )

    def encode_audio(
        self,
        request: encoder_pb2.EncodeAudioRequest,
        context: common_pb2.RequestContext,
    ) -> encoder_pb2.EncodeAudioResponse:
        response: encoder_pb2.EncodeAudioResponse
        with common.StageTimer(
            context,
            service="encoder",
            rpc="EncodeAudio",
            stage="encode_audio",
            input=common.summarize_audio_input(request.audio),
        ) as timer:
            encoder = self._get_encoder()
            loaded = load_audio_for_encoder(
                request.audio,
                preprocess_prompt=bool(request.preprocess_prompt),
                can_trim_long_audio=bool(request.HasField("can_trim_long_audio") and request.can_trim_long_audio),
            )
            tokens = encoder.encode(loaded.encoder_audio)
            response = encoder_pb2.EncodeAudioResponse(
                context=common.response_context(context, service="encoder"),
                audio_codebook=common.tensor_to_proto(tokens),
                sample_rate=DEFAULT_SAMPLE_RATE,
            )
            if request.return_audio:
                response.preprocessed_audio.audio_bytes = encode_wav_bytes(loaded.wave)
                response.preprocessed_audio.sample_rate = DEFAULT_SAMPLE_RATE
                response.preprocessed_audio.format = common_pb2.AUDIO_FORMAT_WAV
                response.preprocessed_audio.channels = 1
                response.preprocessed_audio.duration_sec = loaded.duration_sec
            timer.output.update(common.summarize_tensor(response.audio_codebook))
        if timer.metric is not None:
            response.context.metrics.append(timer.metric)
        return response

    def _get_encoder(self) -> DualCodecOnnxEncoder | DualCodecTorchEncoder:
        if self._encoder is None:
            cfg = self.config.encoder
            backend = str(cfg.backend or "torch").strip().lower().replace("_", "-")
            if backend == "torch":
                self._encoder = DualCodecTorchEncoder(
                    w2v_path=cfg.w2v_path,
                    dualcodec_path=cfg.dualcodec_path,
                    model_id=cfg.model_id,
                    device_id=self.config.device_id,
                    dtype=cfg.precision,
                    max_samples=DEFAULT_SAMPLE_RATE * int(cfg.max_seconds),
                    max_frames=DEFAULT_FRAME_RATE * int(cfg.max_seconds),
                )
            else:
                self._encoder = DualCodecOnnxEncoder(
                    w2v_path=cfg.w2v_path,
                    onnx_path=cfg.onnx_path,
                    ep_config={
                        "backend": codec_backend(cfg.backend),
                        "device_id": self.config.device_id,
                        "trt_fp16_enable": True,
                    },
                    trt_cache_root=cfg.trt_cache_root,
                    max_samples=DEFAULT_SAMPLE_RATE * int(cfg.max_seconds),
                    max_frames=DEFAULT_FRAME_RATE * int(cfg.max_seconds),
                )
        return self._encoder
