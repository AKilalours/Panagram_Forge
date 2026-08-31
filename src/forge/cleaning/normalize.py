"""Text normalization. Stages 3 and 4 of the spec pipeline.

Order inside this file matters: markup is stripped before whitespace is collapsed,
because stripping tags leaves ragged whitespace behind that the collapse then cleans
up. Doing it the other way round leaves the ragged whitespace in the output.

ftfy is used when installed but is not required. Mojibake repair improves quality; it
is not worth making the whole Phase 1 pipeline undeployable over.
"""

from __future__ import annotations

import re
import unicodedata

try:  # optional
    from ftfy import fix_text as _ftfy_fix
except ImportError:  # pragma: no cover - exercised only when ftfy is absent
    _ftfy_fix = None

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WS_RUN = re.compile(r"[ \t  - ]+")
_NL_RUN = re.compile(r"\n{3,}")

# Invisible characters. These are also an adversarial attack vector (RAID's
# zero-width-space attack), so stripping them at ingestion means the detector never
# learns to rely on their presence in human text.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E]
)

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
}


def strip_markup(text: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _TAG.sub(" ", text)
    for ent, rep in _HTML_ENTITIES.items():
        text = text.replace(ent, rep)
    return _ENTITY.sub(" ", text)


def strip_invisible(text: str) -> str:
    return text.translate(_INVISIBLE)


def collapse_whitespace(text: str) -> str:
    text = _WS_RUN.sub(" ", text)
    text = _NL_RUN.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize(text: str, *, use_ftfy: bool = True) -> str:
    """Full normalization. Deterministic and idempotent: normalize(normalize(x)) == normalize(x)."""
    if use_ftfy and _ftfy_fix is not None:
        text = _ftfy_fix(text)
    text = unicodedata.normalize("NFKC", text)
    text = strip_invisible(text)
    text = strip_markup(text)
    return collapse_whitespace(text)
