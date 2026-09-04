"""The result cards, built in Python instead of in the browser.

WHY THIS EXISTS. `api/forge_app.py` renders its cards in JavaScript from the JSON the API
returns. Streamlit has no such API to call: the analysis happens in the same process as the
page. So the same markup is built here, from the same payload dictionaries, against the
same stylesheet in `forge.ui.theme`.

THE RISK THIS FILE CARRIES, STATED PLAINLY. Two renderers for one payload can disagree, and
this project has shipped that mistake in several forms already. The mitigation is that both
consume `Report.as_dict()` and the scorer's own objects rather than reformatting values
themselves: a wording change belongs upstream, in the report, where both pick it up. When
this file starts deciding what a number means rather than where it sits, it has gone wrong.
"""

from __future__ import annotations

import html

# NO AI DETECTED, never HUMAN. A detector cannot establish that a person made something. It
# reports whether the input resembles the AI distribution it was trained on. "Human" is a
# claim about the world; "no AI detected" is a claim about the model, and only the second is
# true. Kept identical to the FastAPI page's WORD map.
VERDICT_WORD = {"ai": "AI DETECTED", "human": "NO AI DETECTED", "uncertain": "UNCERTAIN"}
VERDICT_CLASS = {"ai": "hot", "human": "ok", "uncertain": "warn"}
VERDICT_MARK = {"ai": "!", "human": "✓", "uncertain": "~"}

DIRECTION = {"toward_ai": "toward AI", "toward_capture": "toward capture", "neutral": ""}


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def banner(*, available: bool, verdict: str | None, probability: float | None,
           reason: str = "", score_label: str = "AI probability", score_note: str = "") -> str:
    """The headline. One shape for text and image, as on the FastAPI page."""
    has_score = probability is not None
    pct = None if (not available or not has_score) else probability * 100
    cls = "pending" if not available else VERDICT_CLASS.get(verdict or "", "warn")
    mark = "?" if not available else VERDICT_MARK.get(verdict or "", "~")
    word = "No verdict" if not available else VERDICT_WORD.get(verdict or "", str(verdict).upper())
    gauge = ""
    if pct is not None:
        position = min(100.0, max(0.0, pct))
        gauge = (
            f'<div class="gauge"><div class="needle" style="left:{position}%"></div></div>'
            '<div class="gaugeends"><span>human</span><span>uncertain</span><span>AI</span></div>'
        )
    score_text = ("not available" if not available else "declared") if pct is None else f"{pct:.1f}%"
    return f"""<div class="assess big {cls}">
  <div class="verdictbox">
    <div class="mark">{mark}</div>
    <div><h2>{esc(word)}</h2><p>{esc(reason)}</p></div>
  </div>
  <div class="scorebox">
    <div class="scorelabel">{esc(score_label)}</div>
    <div class="score {'na' if pct is None else ''}">{esc(score_text)}</div>
    {gauge}
    <div class="scorenote">{esc(score_note)}</div>
  </div></div>"""


def _row(key, value, status=None) -> str:
    shown = esc(value) if value not in (None, "") else '<span class="pill na">not available</span>'
    pill = f' <span class="pill {_pill_class(status)}">{esc(_pill_label(status))}</span>' if status else ""
    return f'<div class="row"><span class="k">{esc(key)}</span><span class="v">{shown}{pill}</span></div>'


def _pill_class(status: str) -> str:
    return {"high": "hot", "medium": "warn", "present": "ok", "low": "ok",
            "not_found": "na", "not_checked": "na"}.get(status, "na")


def _pill_label(status: str) -> str:
    return {"not_found": "not found", "not_checked": "not checked"}.get(status, status)


def _stream_row(stream: dict) -> str:
    width = stream["strength"] if stream.get("available") else 100
    tone = "" if stream.get("available") else " na"
    direction = DIRECTION.get(stream.get("direction") or "", "")
    tag = f' <span class="dir">{esc(direction)}</span>' if stream.get("available") and direction else ""
    # `state` is the word the panel shows INSTEAD of the number: not detected, weak,
    # partial, detected, not available. Reading `strength` here would put "50%" next to a
    # camera-metadata row, which reads as "50% likely to be a photograph" and is not what
    # the number means. Guessing a key name is also how a pill ships empty, so this asserts.
    label = stream.get("state")
    if not label:
        raise KeyError(f"evidence stream {stream.get('key')!r} carries no 'state'")
    return f"""<div class="stream">
  <div class="sname">{esc(stream['label'])}<span class="pill {'na' if not stream.get('available') else ''}">
    {esc(label)}</span></div>
  <div class="bar"><div class="fill{tone}" style="width:{width}%"></div></div>
  <div class="sub">{esc(stream.get('summary'))}{tag}</div></div>"""


def image_result(payload: dict, attribution: dict | None = None) -> str:
    """The image tab's cards, from `Report.as_dict()` plus an optional attribution map."""
    assessment = payload["assessment"]
    evidence = payload["evidence"]
    streams = evidence.get("streams", [])
    primary = next((s for s in streams if s["key"] == "visual_model"), None)
    supporting = [s for s in streams if s["key"] != "visual_model"]

    head = banner(
        available=bool(assessment["available"]),
        verdict=assessment.get("verdict"),
        probability=assessment.get("confidence"),
        reason=assessment.get("reason", ""),
        score_label="Determination" if assessment.get("confidence") is None else "AI probability",
        score_note=assessment.get("detail", ""),
    )

    preview = (f'<img class="prev" src="{payload["preview"]}" alt="preview">'
               if payload.get("preview") else "")
    size_kb = f'{payload["size_bytes"] / 1024:.0f} KB'
    image_card = (
        f'<div class="card"><h3>Image</h3>{preview}<div style="margin-top:11px">'
        f'{_row("File", payload.get("filename"))}{_row("Size", size_kb)}</div></div>'
    )

    evidence_card = (
        '<div class="card"><h3>Evidence</h3><div class="grouplabel">Detector</div>'
        + (_stream_row(primary) if primary else "")
        + '<div class="grouplabel">Supporting signals</div>'
        + "".join(_stream_row(s) for s in supporting)
        + f'<div class="row"><span class="k">Agreement</span>'
          f'<span class="v"><span class="pill {"ok" if evidence.get("conflict") == "low" else "warn"}">'
          f'{esc("consistent" if evidence.get("conflict") == "low" else evidence.get("conflict"))}'
          f'</span></span></div></div>'
    )

    signals_card = (
        '<div class="card"><h3>File signals</h3>'
        + "".join(_row(r["label"], r["value"], r.get("status")) for r in payload["authenticity"])
        + "".join(_row(r["label"], r["value"], r.get("status")) for r in payload["provenance"])
        + "</div>"
    )

    cards = f'<div class="grid">{image_card}{evidence_card}{signals_card}</div>'

    attribution_card = ""
    if attribution and attribution.get("image"):
        peak = attribution["peak"]
        attribution_card = f"""<div class="grid"><div class="card wide">
  <h3>What the verdict rests on <span class="sum">{attribution['grid']}×{attribution['grid']} occlusion</span></h3>
  <div class="attr"><img src="{attribution['image']}" alt="attribution map">
    <div>
      <div class="row"><span class="k">Score with everything visible</span>
        <span class="v">{attribution['base_probability'] * 100:.1f}%</span></div>
      <div class="row"><span class="k">Largest single drop when hidden</span>
        <span class="v">{peak['drop'] * 100:.1f} pts</span></div>
      <p class="caveat">Measured by hiding each region and re-scoring, not by a gradient
      approximation, so it holds for any detector this project loads.</p>
    </div></div></div></div>"""

    limits = "".join(f"<li>{esc(line)}</li>" for line in payload.get("cannot_conclude", []))
    limits_card = (
        f'<div class="grid"><div class="card wide"><h3>What this cannot conclude</h3>'
        f'<ul class="lim">{limits}</ul></div></div>' if limits else ""
    )

    manipulation = "".join(_row(r["label"], r["value"], r.get("status"))
                           for r in payload.get("manipulation", []))
    details = (
        f'<div class="grid"><div class="card wide"><h3>Manipulation analysis</h3>{manipulation}</div></div>'
        if manipulation else ""
    )

    return head + cards + attribution_card + details + limits_card


def _spark(windows: list[float]) -> str:
    """Per-window probabilities as bars. Omitted for a single window, where it says nothing."""
    if len(windows) < 2:
        return ""
    bars = "".join(f'<i style="height:{max(2, round(p * 100))}%"></i>' for p in windows)
    return f'<div class="spark">{bars}</div>'


def _arm_card(arm: dict) -> str:
    tone = {"ai": "hot", "human": "ok"}.get(arm["verdict"], "warn")
    return (
        f'<div class="card"><h3>{esc(arm["label"])}</h3>'
        + _row("Verdict", arm["verdict"], tone)
        + _row("AI probability (mean over windows)", f'{arm["ai_probability"] * 100:.2f}%')
        + _row("Highest single window", f'{arm["max_window_probability"] * 100:.2f}%')
        + _row("Deployed threshold", f'{arm["threshold"]:.6f}')
        + _row("FPR budget", arm["fpr_budget"])
        + _row("Distance from threshold", f'{arm["confidence"] * 100:.1f}%')
        + _row("Windows scored", arm["n_windows"])
        + _row("Its validation FNR", f'{arm["val_fnr"] * 100:.3f}%')
        + _row("Its validation ECE", arm["val_ece"])
        + _spark(arm.get("windows") or [])
        + "</div>"
    )


def text_result(payload: dict) -> str:
    """The text tab, from the payload `forge.inference.text_api.analyse` returns.

    Deliberately the same structure as the FastAPI page's JavaScript, because it is the same
    page: one banner carrying the deployed arm's verdict, both arms side by side beneath it
    because the comparison is the experiment, and the limitations card last. A product has
    one verdict, not two, so the headline is the mirror arm and the other sits beside it.
    """
    unavailable = "".join(
        f'<div class="card err"><h3>{esc(name)} unavailable</h3><p>{esc(reason)}</p></div>'
        for name, reason in (payload.get("unavailable") or {}).items()
    )

    if not payload.get("available"):
        head = banner(
            available=False, verdict=None, probability=None,
            reason="Neither arm could be loaded, so no score is produced. A number here "
                   "would be fabricated.",
            score_label="AI probability, calibrated",
            score_note="Load a trained checkpoint into outputs/ to enable scoring.",
        )
        return head + (f'<div class="grid">{unavailable}'
                       f'<div class="card"><h3>Input</h3>'
                       f'{_row("Words", payload.get("words"))}</div></div>')

    arms = payload["arms"]
    primary = next((a for a in arms if a["arm"] == "mirror"), arms[0])
    if len(arms) > 1:
        agree = arms[0]["verdict"] == arms[1]["verdict"]
        opening = ("Both arms agree." if agree else
                   "THE ARMS DISAGREE: "
                   + ", ".join(f'{a["label"]} says {a["verdict"]}' for a in arms) + ".")
    else:
        opening = "Only one arm is loaded."

    plural = "" if primary["n_windows"] == 1 else "s"
    head = banner(
        available=True,
        verdict=primary["verdict"],
        probability=primary["ai_probability"],
        reason=f'{opening} Mean over {primary["n_windows"]} window{plural}, '
               f'{payload["words"]} words.',
        score_label="AI probability, calibrated",
        score_note=(
            f'{primary["label"]}, threshold {primary["threshold"]:.6f} at a '
            f'{primary["fpr_budget"]} false-positive budget. Validation ECE '
            f'{primary["val_ece"]}. Out of distribution this model is badly miscalibrated: '
            "see the note below."
        ),
    )

    provenance = " · ".join(esc(a["model_version"]) for a in arms)
    limits = (
        '<div class="card wide"><h3>What this score is not</h3>'
        f'<p>{esc(payload["caveat"])}</p><p>{esc(payload["abstention"])}</p>'
        '<p class="tech" style="margin-top:9px">Provenance, for reproducibility rather than '
        f'display: {provenance}</p></div>'
    )

    cards = "".join(_arm_card(a) for a in arms)
    return head + f'<div class="grid">{cards}{unavailable}{limits}</div>'


EXTRA_CSS = """
/* Streamlit paints its own background behind this markup, so the page ground is set here
   rather than on body, which the host controls. */
.forgewrap{background:var(--bg);color:var(--ink);padding:2px 0 10px;
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.forgewrap .grid{margin-bottom:14px}
.forgewrap ul.lim{margin:4px 0 0;padding-left:18px;color:var(--muted);font-size:12.5px}
.forgewrap ul.lim li{margin:5px 0}
"""


def document(body: str) -> str:
    """Wrap rendered cards in the shared stylesheet, for embedding in a Streamlit page."""
    from forge.ui.theme import CSS

    return f"<style>{CSS}{EXTRA_CSS}</style><div class='forgewrap'>{body}</div>"
