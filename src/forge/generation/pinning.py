"""Rewrite a generator roster's `revision:` values in place.

WHY THIS IS ITS OWN MODULE. This substitution used to live inline in the
`pin-revisions` CLI command as a literal `str.replace` of

    f"model_id: {model_id}\n    revision: TODO_PIN_AT_FIRST_RUN"

which required a newline immediately after the model id. Two roster entries carry
a trailing comment on that line (the gated ones, explaining their fallback), so
for those the literal never matched, `str.replace` returned the text unchanged,
and the command reported success anyway because its count came from how many
HuggingFace lookups had succeeded rather than from how many lines it had
actually rewritten.

Being buried inside a command that requires network access is what kept it
untested. Here it is a pure string function, so the failure it caused is
expressible as a unit test.
"""

from __future__ import annotations

import re

TODO = "TODO_PIN_AT_FIRST_RUN"


def _pattern(model_id: str) -> re.Pattern[str]:
    """Match a family's `model_id:` line and the `revision:` line directly under it.

    Tolerates a trailing `# comment` on the model_id line, and matches whatever the
    current revision value is rather than only the TODO placeholder, so a roster can
    be re-pinned later instead of only pinned once.

    A trailing comment on the REVISION line is consumed rather than preserved. The only
    one the roster ships is the scaffolding hint "`forge pin-revisions` fills these in",
    which stops being true the moment the value is a real sha; leaving it beside a pinned
    revision would be a false statement in the frozen artefact.
    """
    return re.compile(
        r"(^[ \t]*model_id:[ \t]*"
        + re.escape(model_id)
        + r"[ \t]*(?:#[^\n]*)?\n[ \t]*revision:[ \t]*)(\S+)[ \t]*(?:#[^\n]*)?",
        re.MULTILINE,
    )


def pin_revision(text: str, model_id: str, sha: str) -> tuple[str, int]:
    """Return (new text, number of revision lines rewritten).

    The count is the point. A caller must be able to tell a no-op from a write,
    which `str.replace` cannot express.
    """
    return _pattern(model_id).subn(lambda m: m.group(1) + sha, text)


def unpinned_lines(text: str) -> list[str]:
    """Every line still carrying the placeholder, stripped. Empty means fully pinned."""
    return [ln.strip() for ln in text.split("\n") if TODO in ln]
