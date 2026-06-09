from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.config import ServiceConfig, clear_proxies
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2 as decoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2 as encoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2 as llm_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.orchestrator.servicer import _diff_text_to_edits, _resolve_alignment_granularity
from viitorvoice.local_edit import AlignmentItem, build_multi_local_edit_token_plan
from viitorvoice.llm.text_utils import prepare_text_for_tokenizer, resolve_language


REQUESTS = [
    encoder_pb2.EncodeAudioRequest,
    llm_pb2.LLMGenerateRequest,
    llm_pb2.LLMGenerateFromSemanticRequest,
    llm_pb2.LLMGenerateLocalEditRequest,
    decoder_pb2.DecodeAudioRequest,
    orch_pb2.SynthesizeRequest,
    orch_pb2.SemanticToWavRequest,
    orch_pb2.AlignForEditRequest,
    orch_pb2.LocalEditRequest,
    orch_pb2.TextLocalEditRequest,
]

RESPONSES = [
    common_pb2.HealthResponse,
    encoder_pb2.EncodeAudioResponse,
    llm_pb2.LLMGenerateResponse,
    llm_pb2.LLMGenerateFromSemanticResponse,
    llm_pb2.LLMGenerateLocalEditResponse,
    decoder_pb2.DecodeAudioResponse,
    orch_pb2.SynthesizeResponse,
    orch_pb2.SemanticToWavResponse,
    orch_pb2.AlignForEditResponse,
    orch_pb2.LocalEditResponse,
    orch_pb2.TextLocalEditResponse,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Static smoke checks for gRPC v2.")
    parser.add_argument("--output-dir", default="test_outputs/viitorvoice_grpc_server_static")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    for cls in REQUESTS:
        field = cls.DESCRIPTOR.fields_by_number[1]
        ok = field.name == "context" and field.message_type.full_name == "viitorvoice.inference.v2.RequestContext"
        checks.append({"check": f"{cls.__name__}.field1_context", "ok": ok})
        assert ok, cls.__name__
    for cls in RESPONSES:
        field = cls.DESCRIPTOR.fields_by_number[1]
        ok = field.name == "context" and field.message_type.full_name == "viitorvoice.inference.v2.ResponseContext"
        checks.append({"check": f"{cls.__name__}.field1_context", "ok": ok})
        assert ok, cls.__name__

    tensor = common_pb2.Int64Tensor(values=list(range(24)), shape=[12, 2])
    restored = common.tensor_from_proto(tensor, name="tensor", required=True)
    assert restored is not None and tuple(restored.shape) == (12, 2)
    assert common.normalize_audio_codebook(restored.unsqueeze(0), name="audio").shape == (12, 2)
    semantic = common.normalize_semantic_tokens(common_pb2.Int64Tensor(values=[1, 2], shape=[1, 2]).values, name="semantic")
    assert tuple(semantic.shape) == (2,)
    mask_tokens = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    assert common.count_remaining_mask_tokens(mask_tokens, [0, 2]) == 2
    checks.append({"check": "tensor_helpers", "ok": True})

    edit = _diff_text_to_edits(
        original_text="hello world",
        edited_text="hello brave world",
        alignments=[
            common_pb2.AlignmentItem(
                index=0, text="hello", start_char=0, end_char=5, has_start_char=True, has_end_char=True
            ),
            common_pb2.AlignmentItem(index=1, text=" ", start_char=5, end_char=6, has_start_char=True, has_end_char=True),
            common_pb2.AlignmentItem(
                index=2, text="world", start_char=6, end_char=11, has_start_char=True, has_end_char=True
            ),
        ],
        language="en",
    )[0]
    assert list(edit.selection.alignment_indices) == [0]
    assert edit.replacement_text == "hello brave"
    checks.append({"check": "text_local_edit_diff", "ok": True})

    multi_edits = _diff_text_to_edits(
        original_text="hello world again",
        edited_text="hi world once more",
        alignments=[
            common_pb2.AlignmentItem(
                index=0, text="hello", start_char=0, end_char=5, has_start_char=True, has_end_char=True
            ),
            common_pb2.AlignmentItem(index=1, text=" ", start_char=5, end_char=6, has_start_char=True, has_end_char=True),
            common_pb2.AlignmentItem(
                index=2, text="world", start_char=6, end_char=11, has_start_char=True, has_end_char=True
            ),
            common_pb2.AlignmentItem(
                index=3, text=" ", start_char=11, end_char=12, has_start_char=True, has_end_char=True
            ),
            common_pb2.AlignmentItem(
                index=4, text="again", start_char=12, end_char=17, has_start_char=True, has_end_char=True
            ),
        ],
        language="en",
    )
    selection, replacements = common.edits_to_selection_text(multi_edits)
    assert selection == "0;4"
    assert replacements == ["hi", "once more"]
    checks.append({"check": "text_local_edit_multi_diff", "ok": True})

    same_item_edits = _diff_text_to_edits(
        original_text="trust",
        edited_text="patience",
        alignments=[
            common_pb2.AlignmentItem(
                index=0, text="trust", start_char=0, end_char=5, has_start_char=True, has_end_char=True
            ),
        ],
        language="en",
    )
    same_item_selection, same_item_replacements = common.edits_to_selection_text(same_item_edits)
    assert same_item_selection == "0"
    assert same_item_replacements == ["patience"]
    checks.append({"check": "text_local_edit_same_alignment_diff", "ok": True})

    deletion_original = (
        "Oh, and one last thing, in the interest of keeping a quiet learning space, "
        "only instructors and students are allowed in the classroom when class is in session."
    )
    deletion_edited = (
        "Oh, and one last thing, in the interest of keeping a quiet learning space, "
        "instructors and students are not allowed in the classroom when class is in session."
    )
    deletion_alignments = [
        common_pb2.AlignmentItem(
            index=14, text="only", start_char=75, end_char=79, has_start_char=True, has_end_char=True
        ),
        common_pb2.AlignmentItem(
            index=18, text="are", start_char=105, end_char=108, has_start_char=True, has_end_char=True
        ),
    ]
    deletion_edits = _diff_text_to_edits(
        original_text=deletion_original,
        edited_text=deletion_edited,
        alignments=deletion_alignments,
        language="en",
    )
    deletion_selection, deletion_replacements = common.edits_to_selection_text(deletion_edits)
    assert deletion_selection == "14;18"
    assert deletion_replacements == ["", "are not"]
    checks.append({"check": "text_local_edit_delete_and_insert_diff", "ok": True})

    source_tokens = torch.arange(12 * 20, dtype=torch.long).reshape(12, 20)
    alignment_items = [
        AlignmentItem(index=0, text="hello", start_time=0.0, end_time=0.4, start_char=0, end_char=5),
        AlignmentItem(index=1, text=" ", start_time=0.4, end_time=0.48, start_char=5, end_char=6),
        AlignmentItem(index=2, text="world", start_time=0.48, end_time=0.8, start_char=6, end_char=11),
    ]
    base_plan = build_multi_local_edit_token_plan(
        source_tokens=source_tokens,
        alignment_items=alignment_items,
        original_text="hello world",
        selection="2",
        replacement_text="there",
        audio_duration=0.8,
        padding_ms=0.0,
        min_edit_frames=4,
    )
    expanded_plan = build_multi_local_edit_token_plan(
        source_tokens=source_tokens,
        alignment_items=alignment_items,
        original_text="hello world",
        selection="2",
        replacement_text="there",
        audio_duration=0.8,
        padding_ms=0.0,
        min_edit_frames=4,
        expand_mask_ratio=2.0,
    )
    assert int(base_plan.editable_audio_mask.sum().item()) >= 4
    assert int(expanded_plan.editable_audio_mask.sum().item()) > int(base_plan.editable_audio_mask.sum().item())
    assert expanded_plan.target_tokens.shape == base_plan.target_tokens.shape
    checks.append({"check": "local_edit_expand_mask_ratio", "ok": True})

    deletion_plan = build_multi_local_edit_token_plan(
        source_tokens=torch.arange(12 * 236, dtype=torch.long).reshape(12, 236),
        alignment_items=[
            AlignmentItem(
                index=14,
                text="only",
                start_time=5.12,
                end_time=5.6,
                start_char=75,
                end_char=79,
            ),
            AlignmentItem(
                index=18,
                text="are",
                start_time=7.12,
                end_time=7.2,
                start_char=105,
                end_char=108,
            ),
        ],
        original_text=deletion_original,
        selection=deletion_selection,
        replacement_text=deletion_replacements,
        audio_duration=9.44,
        padding_ms=0.0,
        min_edit_frames=6,
    )
    assert deletion_plan.span.edited_full_text == deletion_edited
    checks.append({"check": "local_edit_delete_replacement_plan", "ok": True})

    assert _resolve_alignment_granularity(language="en", text="hello world", explicit="") == common_pb2.ALIGNMENT_GRANULARITY_WORD
    assert _resolve_alignment_granularity(language="zh", text="你好世界", explicit="") == common_pb2.ALIGNMENT_GRANULARITY_CHARACTER
    assert _resolve_alignment_granularity(language="ja", text="日本語です", explicit="") == common_pb2.ALIGNMENT_GRANULARITY_CHARACTER
    assert _resolve_alignment_granularity(language="ko", text="안녕하세요", explicit="") == common_pb2.ALIGNMENT_GRANULARITY_CHARACTER
    checks.append({"check": "text_local_edit_default_alignment_granularity", "ok": True})

    class DummyTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"<|ja-日|>": 1, "<|ja-本|>": 2}

    assert resolve_language("JA") == "ja"
    assert resolve_language("ja-JP") == "ja"
    assert prepare_text_for_tokenizer("日本語", DummyTokenizer(), resolve_language("ja-JP")) == "<|ja-日|><|ja-本|>語"
    checks.append({"check": "japanese_text_preprocess_tags", "ok": True})

    os.environ["HTTP_PROXY"] = "http://example.invalid"
    os.environ["https_proxy"] = "http://example.invalid"
    clear_proxies()
    assert "HTTP_PROXY" not in os.environ and "https_proxy" not in os.environ
    checks.append({"check": "clear_proxies", "ok": True})

    config = ServiceConfig.from_env()
    assert config.encoder.backend == "torch"
    assert config.encoder.precision == "bf16"
    checks.append({"check": "encoder_default_torch_bf16", "ok": True})

    report = {"status": "ok", "checks": checks}
    report_path = output_dir / "static_smoke.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
