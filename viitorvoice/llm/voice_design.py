from __future__ import annotations

import re

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")

_INSTRUCT_CATEGORIES = [
    {"male": "男", "female": "女"},
    {
        "child": "儿童",
        "teenager": "少年",
        "young adult": "青年",
        "middle-aged": "中年",
        "elderly": "老年",
    },
    {
        "very low pitch": "极低音调",
        "low pitch": "低音调",
        "moderate pitch": "中音调",
        "high pitch": "高音调",
        "very high pitch": "极高音调",
    },
    {"whisper": "耳语"},
    {
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    },
    {
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    },
]

_INSTRUCT_EN_TO_ZH = {}
_INSTRUCT_ZH_TO_EN = {}
_INSTRUCT_MUTUALLY_EXCLUSIVE = []
for _category in _INSTRUCT_CATEGORIES:
    if isinstance(_category, dict):
        _INSTRUCT_EN_TO_ZH.update(_category)
        _INSTRUCT_ZH_TO_EN.update({value: key for key, value in _category.items()})
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_category) | set(_category.values()))
    else:
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_category))

_INSTRUCT_ALL_VALID = (
    set(_INSTRUCT_EN_TO_ZH)
    | set(_INSTRUCT_ZH_TO_EN)
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-2]
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-1]
)
_INSTRUCT_VALID_EN = frozenset(item for item in _INSTRUCT_ALL_VALID if not _ZH_RE.search(item))
_INSTRUCT_VALID_ZH = frozenset(item for item in _INSTRUCT_ALL_VALID if _ZH_RE.search(item))
