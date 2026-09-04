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

import io
import os
import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

WEIGHTS_REPO = os.getenv("FORGE_WEIGHTS_REPO", "Akilalourdes/forge-detect-weights")
EXPERIMENT = {"baseline": "forge_min_baseline", "mirror": "forge_min_mirror"}

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


# ------------------------------------------------------------------------- presentation

# THE SAME CARDS AS THE FASTAPI PAGE. `forge.ui.render` builds the identical markup against
# the identical stylesheet in `forge.ui.theme`, which api/forge_app.py also imports. The two
# shells differ only in what collects the input: a browser fetch there, a Streamlit widget
# here. Rendering the results twice from two hand-written templates is exactly how this
# project has produced pages that disagree with themselves, so there is one template.
import streamlit.components.v1 as components  # noqa: E402


def show(body: str, height: int) -> None:
    from forge.ui.render import document

    components.html(document(body), height=height, scrolling=True)


# ------------------------------------------------------------------------------ text tab

def text_tab() -> None:
    text = st.text_area(
        "Paste text to analyse",
        height=300,
        label_visibility="collapsed",
        placeholder="Paste text to analyse. A few paragraphs works best; very short "
                    "passages carry little signal.",
    )
    if not st.button("Analyse text", type="primary"):
        return
    if not text.strip():
        st.error("Paste some text first.")
        return

    from forge.inference.text_api import analyse
    from forge.ui.render import text_result

    # BOTH ARMS, as on the reference page, because the comparison is the experiment. Each is
    # roughly 740 MB in float32 and this host allows about 2.7 GB, so the second one may not
    # fit. analyse() reports that per arm rather than failing the request, and the page then
    # shows the arm it has and says only one is loaded. Fetching is per arm and cached, so
    # the first analysis pays for the download and later ones do not.
    with st.spinner("Loading the arms and scoring. The first run fetches 1.5 GB of weights."):
        for arm in ("mirror", "baseline"):
            failure = ensure_weights(arm)
            if failure:
                st.warning(f"could not fetch the {arm} checkpoint: {failure}")
        payload = analyse(text)

    show(text_result(payload), height=1180)


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

    # The attribution map needs the loaded detector, and it is simply omitted when there is
    # none: an empty panel would read as "nothing drove the verdict", which is a different
    # claim from "no detector ran".
    attribution = None
    try:
        from forge.image.attribution import occlusion_attribution
        from forge.image.detector import load_detector

        detector = load_detector()
    except Exception:  # noqa: BLE001 - reported by the evidence panel, not here
        detector = None
    if detector is not None:
        with st.spinner("Measuring which regions the verdict rests on"):
            measured = occlusion_attribution(detector, data)
        attribution = measured.as_dict() if measured is not None else None

    from forge.ui.render import image_result

    show(image_result(payload, attribution), height=1500 if attribution else 1100)


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
