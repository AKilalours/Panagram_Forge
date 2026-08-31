from forge.cleaning import pii
from forge.cleaning.filters import check_length, check_quality, repeated_line_ratio, symbol_ratio
from forge.cleaning.normalize import normalize, strip_invisible, strip_markup

PROSE = (
    "The harbour authority published its annual dredging schedule in March, listing "
    "eleven separate operations across the estuary and the two tidal basins. Silt "
    "accumulation had increased noticeably since the previous survey, and the deeper "
    "berths required attention before the autumn shipping season began in earnest."
)


def test_markup_is_removed_before_anything_measures_length():
    html = f"<html><body><script>var x = 1;</script><p>{PROSE}</p></body></html>"
    out = normalize(html)
    assert "<p>" not in out and "var x" not in out
    assert out.startswith("The harbour authority")


def test_zero_width_characters_are_stripped():
    # also an adversarial attack vector, so human text must never carry them
    dirty = "hello​world﻿"
    assert strip_invisible(dirty) == "helloworld"


def test_normalize_is_idempotent():
    once = normalize(PROSE + "   \n\n\n\n  trailing  ")
    assert normalize(once) == once


def test_markdown_links_keep_their_text():
    assert "example" in strip_markup("see [example](https://a.b/c) here")
    assert "https" not in strip_markup("see [example](https://a.b/c) here")


def test_short_text_fails_length():
    assert "too_short_chars" in check_length("Too short.")


def test_symbol_soup_fails_quality():
    junk = "### >>> ||| " * 60
    assert symbol_ratio(junk) > 0.2
    assert check_quality(junk)


def test_repeated_boilerplate_is_detected():
    boiler = "Click here to continue.\n" * 40
    assert repeated_line_ratio(boiler) > 0.9


def test_good_prose_passes_both_gates():
    text = PROSE * 3
    assert not check_length(text)
    assert not check_quality(text)


def test_pii_is_replaced_with_a_typed_placeholder_not_deleted():
    text = "Reach jane.doe@example.com or 555-123-4567 today."
    out, flags = pii.redact(text)
    assert "[EMAIL]" in out and "[PHONE]" in out
    assert "jane.doe" not in out and "555-123-4567" not in out
    assert set(flags) == {"EMAIL", "PHONE"}


def test_luhn_stops_the_credit_card_regex_firing_on_any_long_number():
    # a 16-digit run that fails Luhn is not a card number
    assert "CREDIT_CARD" not in pii.detect("order reference 1234567812345678 shipped")
    # a valid Luhn number is caught
    assert "CREDIT_CARD" in pii.detect("card 4539578763621486 on file")
