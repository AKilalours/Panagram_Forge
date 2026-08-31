"""PII detection and redaction. Stage 7.

Regex-based and deliberately conservative. Presidio is better and is an optional
dependency, but a corpus pipeline that only runs when a heavyweight NLP stack is
installed will not get run.

Redaction replaces with a stable typed placeholder rather than deleting. Deleting
changes sentence structure, and structure is exactly the signal the detector reads.
"""

from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "SSN": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "IP": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
}

PLACEHOLDER = "[{kind}]"


def _luhn(digits: str) -> bool:
    """Credit card regexes fire on any long digit run. Luhn cuts the false positives."""
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total, parity = 0, len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def detect(text: str) -> list[str]:
    found: list[str] = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(text):
            if kind == "CREDIT_CARD" and not _luhn(m.group()):
                continue
            found.append(kind)
            break
    return sorted(set(found))


def redact(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    for kind, pat in PATTERNS.items():
        def _sub(m: re.Match[str], kind: str = kind) -> str:
            if kind == "CREDIT_CARD" and not _luhn(m.group()):
                return m.group()
            flags.append(kind)
            return PLACEHOLDER.format(kind=kind)

        text = pat.sub(_sub, text)
    return text, sorted(set(flags))
