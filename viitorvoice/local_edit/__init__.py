from viitorvoice.local_edit.alignment import AlignmentItem, Qwen3ForcedAlignerWrapper
from viitorvoice.local_edit.duration import estimate_replacement_frames, text_duration_weight
from viitorvoice.local_edit.editor import (
    LocalEditTokenPlan,
    LocalMultiEditTokenPlan,
    build_local_edit_token_plan,
    build_multi_local_edit_token_plan,
    build_multi_reference_tokens,
    build_reference_tokens,
)
from viitorvoice.local_edit.span import (
    EditSegment,
    EditSpan,
    MultiEditSpan,
    parse_replacement_texts,
    parse_selection_groups,
    parse_selection_indices,
    resolve_edit_span,
    resolve_multi_edit_span,
)

__all__ = [
    "AlignmentItem",
    "EditSegment",
    "EditSpan",
    "LocalEditTokenPlan",
    "LocalMultiEditTokenPlan",
    "MultiEditSpan",
    "Qwen3ForcedAlignerWrapper",
    "build_local_edit_token_plan",
    "build_multi_local_edit_token_plan",
    "build_multi_reference_tokens",
    "build_reference_tokens",
    "estimate_replacement_frames",
    "parse_replacement_texts",
    "parse_selection_groups",
    "parse_selection_indices",
    "resolve_edit_span",
    "resolve_multi_edit_span",
    "text_duration_weight",
]
