"""FORGE on Streamlit Community Cloud. The same detectors, a different shell.

WHY THIS FILE EXISTS. The interface this project was built around is a FastAPI app in
`api/forge_app.py`, and it is still the reference. It cannot be deployed for free any more:
Hugging Face now requires a PRO subscription for Docker Spaces, and every other free host
either caps memory below one model or requires a payment card. Streamlit Community Cloud is
the one that does not, so this file is the interface layer rewritten for it.

Nothing about the detection changes. The verdict, the thresholds, the polarity resolution
and the occlusion attribution all come from `forge.*` exactly as they do in the FastAPI app.
If this page and that one ever disagree, this page is wrong.

MEMORY IS THE DESIGN CONSTRAINT. The free tier is about 2.7 GB of RAM. One text arm is a
float32 DeBERTa-v3-base at roughly 740 MB resident, and the FastAPI page loads both at once
to compare them. Two arms plus torch plus the image detector does not fit, and the failure
mode is the container being killed mid-request, which a reader cannot interpret. So this
page holds ONE arm at a time and says so, and switching arms evicts the other rather than
quietly accumulating. The two-arm comparison is the experiment's result and it is reported
in docs/evaluation.md, where it belongs; a live page is not the place that finding lives.
"""

from __future__ import annotations

import gc
import io
import os
import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

WEIGHTS_REPO = os.getenv("FORGE_WEIGHTS_REPO", "Akilalourdes/forge-detect-weights")
EXPERIMENT = {"baseline": "forge_min_baseline", "mirror": "forge_min_mirror"}
ARM_TITLE = {"mirror": "Arm B, matched mirrors", "baseline": "Arm A, random synthetic"}

st.set_page_config(page_title="FORGE Detect", page_icon="◈", layout="wide")


# ------------------------------------------------------------------ weights on demand

@st.cache_resource(show_spinner=False)
def ensure_weights(arm: str) -> str | None:
    """Fetch one arm's checkpoint from the Hub into the path the loader expects.

    ONE ARM, NOT BOTH. Disk on this tier is as tight as memory, and each checkpoint is
    735 MB. Fetching on demand means a visitor who only uses the image tab never pays for
    either. Returns None on success, or a reason a reader can act on.
    """
    from huggingface_hub import hf_hub_download

    out = pathlib.Path("outputs") / EXPERIMENT[arm]
    if (out / "best.pt").exists() and (out / "summary.json").exists():
        return None
    try:
        for name in ("best.pt", "summary.json"):
            hf_hub_download(
                repo_id=WEIGHTS_REPO,
                filename=f"{EXPERIMENT[arm]}/{name}",
                local_dir="outputs",
            )
    except Exception as error:  # noqa: BLE001 - shown to the reader, never swallowed
        return f"{type(error).__name__}: {error}"
    return None


def load_single_arm(arm: str):
    """Load one arm and evict any other. Returns (arm_object, error_string)."""
    from forge.inference.scorer import ArmUnavailable, load_arm

    if st.session_state.get("resident_arm") not in (None, arm):
        load_arm.cache_clear()
        gc.collect()
    failure = ensure_weights(arm)
    if failure:
        return None, f"could not fetch the {arm} checkpoint: {failure}"
    try:
        loaded = load_arm(arm)
    except ArmUnavailable as error:
        return None, str(error)
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"
    st.session_state["resident_arm"] = arm
    return loaded, None


# ------------------------------------------------------------------------- presentation

VERDICT_WORD = {"ai": "AI DETECTED", "human": "NO AI DETECTED", "uncertain": "UNCERTAIN"}


def verdict_banner(verdict: str, probability: float | None, caption: str = "") -> None:
    """NO AI DETECTED, never HUMAN.

    A detector cannot establish that a person wrote something. It reports whether the input
    resembles the AI distribution it was trained on. "Human" is a claim about the world;
    "no AI detected" is a claim about the model, and only the second one is true.
    """
    word = VERDICT_WORD.get(verdict, verdict.upper())
    left, right = st.columns([3, 1])
    with left:
        if verdict == "ai":
            st.error(f"### {word}")
        elif verdict == "human":
            st.success(f"### {word}")
        else:
            st.warning(f"### {word}")
        if caption:
            st.caption(caption)
    with right:
        st.metric("AI probability", "n/a" if probability is None else f"{probability:.1%}")


# ------------------------------------------------------------------------------ text tab

def text_tab() -> None:
    st.subheader("Text")
    choice = st.radio(
        "Detector arm",
        ["mirror", "baseline"],
        format_func=lambda a: ARM_TITLE[a],
        horizontal=True,
        help="One arm is held in memory at a time on this tier; switching reloads.",
    )
    text = st.text_area(
        "Paste text to analyse",
        height=240,
        placeholder="A few paragraphs works best. Very short passages carry little signal.",
    )

    if not st.button("Analyse text", type="primary"):
        return
    if not text.strip():
        st.warning("Paste some text first.")
        return

    with st.spinner(f"Loading {ARM_TITLE[choice]} and scoring. First load fetches 735 MB."):
        arm, failure = load_single_arm(choice)
    if failure:
        st.error(failure)
        st.caption("No score is shown when an arm cannot load. A number here would be invented.")
        return

    from forge.inference.decision import decide

    scored = arm.score(text)
    decision = decide(scored.mean, arm.policy)

    verdict_banner(
        decision.verdict.value,
        scored.mean,
        f"{ARM_TITLE[choice]} · threshold {arm.policy.threshold:.4f} at a "
        f"{arm.policy.fpr_budget:.1%} false-positive budget",
    )

    left, right = st.columns(2)
    with left:
        st.write("**How the document was scored**")
        st.write(
            {
                "words": len(text.split()),
                "windows": scored.n_windows,
                "mean window probability": round(scored.mean, 4),
                "highest window": round(scored.maximum, 4),
            }
        )
        if scored.n_windows > 1:
            st.caption(
                "The document is scored in overlapping windows and the mean is the "
                "document score. The highest single window is shown because a short "
                "generated passage inside a long human document moves it and not the mean."
            )
    with right:
        st.write("**The threshold this verdict used**")
        st.write(
            {
                "threshold": round(arm.policy.threshold, 6),
                "fitted at FPR budget": arm.policy.fpr_budget,
                "validation FNR": arm.summary["val"].get("fnr"),
                "validation ECE": arm.summary["val"].get("ece"),
                "model version": arm.policy.model_version,
            }
        )

    st.info(
        "**These arms are in-distribution detectors.** On generators they never saw, the "
        "committed evaluation measures a 63% to 96% miss rate and calibration collapsing "
        "from ECE 0.004 to between 0.18 and 0.44. Text from ChatGPT, Claude or Gemini will "
        "usually read as no AI detected. That is the published result, not a malfunction: "
        "see docs/evaluation.md.",
        icon="⚠",
    )


# ----------------------------------------------------------------------------- image tab

def image_tab() -> None:
    st.subheader("Image")
    upload = st.file_uploader(
        "Drop an image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp", "gif", "mpo"],
        help="Analysed in memory. Nothing is written to disk or sent anywhere.",
    )
    if upload is None:
        return

    data = upload.read()
    from forge.image.report import build_report

    with st.spinner("Analysing"):
        report = build_report(data, filename=upload.name, with_stability=False)

    payload = report.as_dict()
    assessment = payload["assessment"]

    left, right = st.columns([1, 2])
    with left:
        st.image(io.BytesIO(data), caption=upload.name, use_container_width=True)
        st.caption(f"{len(data) / 1024:.0f} KB")

    with right:
        if assessment["available"]:
            verdict_banner(
                assessment["verdict"], assessment["confidence"], assessment.get("detail", "")
            )
        else:
            st.warning(f"### No verdict\n{assessment['reason']}")
            st.caption(assessment.get("detail", ""))

        st.write("**Evidence**")
        for stream in payload["evidence"]["streams"]:
            mark = "—" if not stream["available"] else f"{stream['strength']}"
            st.write(f"- **{stream['label']}**: {stream['summary']}  ({mark})")

    from forge.image.detector import load_detector

    try:
        detector = load_detector()
    except Exception:  # noqa: BLE001 - the panel is simply omitted
        detector = None

    if detector is not None:
        from forge.image.attribution import occlusion_attribution

        with st.spinner("Measuring which regions the verdict rests on"):
            attribution = occlusion_attribution(detector, data)
        if attribution is not None:
            st.write("**What the verdict rests on**")
            st.markdown(
                f'<img src="{attribution.png_data_uri}" style="max-width:520px;width:100%">',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{attribution.grid}×{attribution.grid} occlusion. Score with everything "
                f"visible {attribution.base_probability:.1%}; largest single drop when a "
                f"region is hidden {attribution.peak['drop'] * 100:.1f} points. Measured by "
                "hiding each region and re-scoring, not by a gradient approximation."
            )

    with st.expander("File signals"):
        for row in payload["authenticity"] + payload["provenance"]:
            st.write(f"- **{row['label']}**: {row['value'] if row['value'] else 'not found'}")

    with st.expander("What this cannot conclude"):
        for line in payload["cannot_conclude"]:
            st.write(f"- {line}")


# ---------------------------------------------------------------------------------- page

st.title("FORGE Detect")
st.caption(
    "Failure-driven synthetic data generation for robust AI-content detection. "
    "Runs on CPU. Built by Akila Lourdes Miriyala Francis."
)

text_pane, image_pane = st.tabs(["Text", "Image"])
with text_pane:
    text_tab()
with image_pane:
    image_tab()

st.divider()
st.caption(
    "One text arm is held in memory at a time here, because this host allows about 2.7 GB "
    "and each arm is roughly 740 MB in float32. The side-by-side comparison of the two arms "
    "is the experiment's result and is reported in docs/evaluation.md. "
    "Source: github.com/AKilalours/Panagram_Forge"
)
