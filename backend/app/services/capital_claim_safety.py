"""Reject unsupported funding-capability assertions in model-authored text.

Registration capital is not evidence of funding capability. Allow only narrow
limitations or independent verification questions; hedges and citations do not
turn an assertion into supported evidence.
"""
from __future__ import annotations

import re
import unicodedata


_CAPABILITY = re.compile(
    r"(?:資金(?:動員|調度|籌措|募集|運用)|籌資|融資|募資)(?:的)?能力|"
    r"(?:資金|財務)(?:實力|彈性)|資金(?:充足|雄厚|充裕)|"
    r"(?:funding|financing|fundmobilization|fundraising)capabilit(?:y|ies)"
)
_SUBJECT = r"(?:(?:該|本|各)?公司(?:的)?|其|實際|目前|當前)*"
_LIMITATION = re.compile(
    r"[^。！？!?；;，,：:]*?"
    r"(?:不能|無法|不足以|不可|不應|不)(?:直接)?"
    r"(?:代表|證明|反映|推論|推斷|判定|認定|衡量|評估)"
    + _SUBJECT + r"$"
)
_UNKNOWN = re.compile(r"(?:仍需另行查核|尚待查核|仍待查核|尚未確認|無法確認|資料未提供)[。.]?$")
_QUESTION = re.compile(
    r"(?:是否|有無)(?:已)?(?:另行|獨立)?(?:查核|確認|核對|評估)"
    + _SUBJECT + r"$"
)
# A prior negation or a question must not launder a new positive assertion.
_UNSAFE_PREFIX = re.compile(
    r"不無|不能不|並非不|不是不|未必不|不僅|不只|並非不能|不是不能|"
    r"但|然而|不過|卻|因此|所以|意味|顯示|反映|暗示|可見|證明|代表|由於|因為|既然"
)


def has_unsupported_funding_claim(text: str, *, is_question: bool = False) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Strip zero-width characters/spacing without losing assertion boundaries.
    normalized = "".join(
        c for c in normalized if not c.isspace() and unicodedata.category(c) != "Cf"
    )
    for clause in re.split(r"[。！？!?；;，,：:]", normalized):
        # Ignore decorative separators inside an obfuscated capability term.
        clause = "".join(c for c in clause if c.isalnum())
        for match in _CAPABILITY.finditer(clause):
            prefix, suffix = clause[:match.start()], clause[match.end():]
            limitation = _LIMITATION.fullmatch(prefix)
            if limitation and not re.search(
                r"不無|不能不|並非不|不是不|未必不|不僅|不只|並非不能|不是不能|但|然而|卻",
                prefix,
            ) and not suffix:
                continue
            if re.fullmatch(_SUBJECT, prefix) and _UNKNOWN.fullmatch(suffix):
                continue
            if (
                is_question and text.rstrip().endswith(("?", "？"))
                and _QUESTION.fullmatch(prefix) and not suffix
                and not _UNSAFE_PREFIX.search(prefix)
            ):
                continue
            return True
    return False
