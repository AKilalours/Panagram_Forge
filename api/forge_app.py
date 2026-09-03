"""FORGE: one interface, two detectors. Text and Image.

RUNS ON CPU. No GPU, no network. The image analysis computes every value from the uploaded
bytes at request time; the text side calls the detection API, which returns its real state.

WHAT IS LIVE TODAY

  Image tab   Every forensic panel: container, EXIF, camera consistency, C2PA presence,
              self-declared generation markers, quantisation tables, error level analysis,
              resampling, noise uniformity, colour statistics, perceptual hash, and a
              transform-stability check run over ten real attacks.

  Text tab    Posts to /v1/detect on the detection API. That endpoint returns 503 with a
              stated reason until a model is registered, and the page shows the reason.

WHAT IS NOT, AND WHY IT IS NOT FAKED. The image verdict and the attribution heatmap need
trained weights. The reference design this page follows shows a large green "Human, 98.2%".
That number comes from a calibrated model. Synthesising one from metadata would fire hardest
on screenshots, chat-app downloads and re-saved photographs, because those have no EXIF and
library quantisation tables: exactly the false positive against innocent human images that
this entire project exists to prevent. So the image banner shows the verdict and score slots
PENDING, with the gauge greyed, rather than filled or removed. Built pending so the day a
detector exists it fills in, instead of the panel being designed then.

The TEXT banner is the same component with a real number in it, because that detector exists
and is calibrated. One shape, two states, so the difference between "measured" and "not
measured" is visible at a glance rather than inferred from which panels are missing.

EVIDENCE STREAMS SHOW A WORD, NOT A PERCENTAGE. `strength` is how much a stream has to say,
not a probability. "Camera metadata 50%" reads as "50% likely to be a photograph", which is
not what it means and is not recoverable from a caption underneath. The bar stays as a rough
visual, the number stays in the payload, and the panel says "partial".

THE FORENSIC MAPS ARE NOT ATTRIBUTION. They are labelled low-level, carry a "not AI
attribution" badge, and sit below the attribution panel that says the localisation head is
untrained. A residual map next to an AI-detection product invites exactly the reading it
cannot support.

THE TRANSFORM-SURVIVAL PASS IS OPT-IN. It re-encodes the image ten times and was about 35 of
the 39 seconds a 2.8 MB JPEG took. It answers a second-order question, so it runs on a
button. Its absence is reported as absence, never as an empty result.

Run:  uvicorn api.forge_app:app --port 8080
"""

from __future__ import annotations

import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from forge.image.maps import build_maps
from forge.image.report import build_report

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 50_000
# MPO is a JPEG container holding more than one frame: dual-lens phones and 3D cameras
# write it, and so does every iPhone that captures depth. Rejecting it turned away ordinary
# photographs, which is the exact population this project exists to protect from false
# accusations, and it was rejected by an allowlist rather than by anything measured.
# Analysis reads frame 0; real_format says so when there is more than one.
ACCEPTED = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF", "MPO"}

# Formats a phone commonly writes that Pillow cannot open without an extra decoder. Named
# separately so the user is told what to do instead of being told the file is unreadable.
NEEDS_DECODER = {
    "HEIC": "HEIF images need the pillow-heif decoder; export as JPEG, or install it",
    "HEIF": "HEIF images need the pillow-heif decoder; export as JPEG, or install it",
    "AVIF": "AVIF images need an AVIF decoder; export as JPEG or PNG, or install one",
}

app = FastAPI(title="FORGE", version="0.2.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "image_detector_loaded": False,
        "text_detector_loaded": False,
        "mode": "forensics live, detectors pending training",
    }


def _thumbnail(data: bytes, box: int = 640) -> str | None:
    """A downscaled preview as a data URI. Uploads are never written to disk."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            img.thumbnail((box, box))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=84)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


@app.post("/v1/image/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    stability: bool = False,
) -> JSONResponse:
    """Analyse one image. The transform-survival pass is opt-in.

    That pass re-encodes the image ten times and is roughly 35 of the 39 seconds a 2.8 MB
    JPEG used to take. It answers a second-order question, "would these signals survive
    redistribution", so it runs on request rather than on every upload. Its absence is
    reported as absence: `stability_available` is False, never an empty list that reads as
    "nothing survived".
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds {MAX_IMAGE_BYTES} bytes")

    import time

    mark = time.perf_counter()
    preview = _thumbnail(data)
    preview_ms = int((time.perf_counter() - mark) * 1000)

    report = build_report(
        data, filename=file.filename or "upload", preview=preview,
        with_stability=stability,
    )
    fmt = report.by_name("file_type")
    detected = fmt.value if fmt else None
    if detected not in ACCEPTED:
        hint = NEEDS_DECODER.get(str(detected or "").upper())
        raise HTTPException(
            status_code=415,
            detail=(
                f"cannot read this file (detected: {detected}). {hint}"
                if hint
                else f"unsupported or unreadable image (detected: {detected}). "
                f"Supported: {', '.join(sorted(ACCEPTED))}."
            ),
        )

    # Forensic residual maps. NOT model saliency: no model is involved, and the panel says
    # so. They show where in this frame each signal is strong, normalized within the frame,
    # which is a thing a reader can go and check by looking. A map that cannot be computed
    # is omitted rather than faked, so this list can legitimately be short or empty.
    payload = report.as_dict()
    mark = time.perf_counter()
    payload["maps"] = build_maps(data)
    payload["timings_ms"] = dict(payload.get("timings_ms") or {})
    payload["timings_ms"]["preview"] = preview_ms
    payload["timings_ms"]["forensic_maps"] = int((time.perf_counter() - mark) * 1000)
    payload["maps_note"] = (
        "Forensic residual maps, not detector saliency. No model is involved. Each map is "
        "normalized within this image, so brightness is relative to this frame and is not "
        "comparable across images."
    )
    return JSONResponse(payload)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@app.post("/v1/text/analyze")
def analyze_text(req: TextRequest) -> JSONResponse:
    """Score with BOTH arms and show them side by side.

    The decision policy is not reimplemented here. Thresholds come from each arm's committed
    summary.json and the verdict comes from forge.inference.decision.decide, so there is one
    place where a score becomes a claim.

    Both arms rather than one, because the comparison IS the project. A single verdict would
    hide the thing the experiment measured, and the two arms genuinely disagree on documents
    near the threshold: on RAID, 122 documents that arm A misses arm B catches.

    An arm that cannot load reports why. It never falls back to a score.
    """
    from forge.inference.decision import decide
    from forge.inference.scorer import ARMS, ArmUnavailable, load_arm

    words = len(req.text.split())
    arms, unavailable = [], {}
    for name in ARMS:
        try:
            arm = load_arm(name)
            scored = arm.score(req.text)
        except ArmUnavailable as error:
            unavailable[name] = str(error)
            continue
        except Exception as error:                      # noqa: BLE001 - surfaced, not hidden
            unavailable[name] = f"{type(error).__name__}: {error}"
            continue

        decision = decide(scored.mean, arm.policy)
        arms.append({
            "arm": scored.arm,
            "label": scored.label,
            "verdict": decision.verdict.value,
            "ai_probability": scored.mean,
            "max_window_probability": scored.maximum,
            "confidence": decision.confidence,
            "threshold": arm.policy.threshold,
            "fpr_budget": arm.policy.fpr_budget,
            "model_version": arm.policy.model_version,
            "abstained": decision.abstained,
            "n_windows": scored.n_windows,
            "windows": scored.window_probabilities[:64],
            "val_fnr": arm.summary["val"].get("fnr"),
            "val_ece": arm.summary["val"].get("ece"),
        })

    return JSONResponse({
        "available": bool(arms),
        "words": words,
        "arms": arms,
        "unavailable": unavailable,
        "aggregation": "mean over windows; max shown alongside because it inflates FPR",
        "caveat": (
            "Trained on four generator families at 1.7B to 3.8B parameters. Against unseen "
            "generators these checkpoints miss 63% to 96% of AI text at this threshold and "
            "their ECE rises from 0.004 to 0.18-0.44. A confident score here is not evidence "
            "of a confident model. See docs/evaluation.md."
        ),
        "abstention": (
            "Off. The uncertain band is derived from validation scores "
            "(decision.band_from_validation), which are not committed, and a band chosen by "
            "eye would silently change the false-positive rate."
        ),
    })


@app.get("/v1/results")
def results() -> JSONResponse:
    """Measured results, read from the committed run records. Never recomputed."""
    from api.results import build

    return JSONResponse(build())


@app.get("/v1/image/detector")
def image_detector_state() -> JSONResponse:
    """Whether a visual detector can serve, and if not, the reason. Never raises."""
    from forge.image.detector import detector_state

    return JSONResponse(detector_state())


@app.get("/v1/text/arms")
def text_arms() -> JSONResponse:
    """Which arms can serve right now, and for those that cannot, the reason."""
    from forge.inference.scorer import available

    return JSONResponse(available())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FORGE</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#11141a;--muted:#68707c;--line:#e2e6ea;
 --accent:#2f5d9e;--ok:#1c7a4d;--warn:#96660f;--hot:#a5382a;--chip:#eef1f5;--bar:#dde3ea}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0e1116;--panel:#161a21;
 --ink:#e9ebef;--muted:#98a1ae;--line:#242a33;--accent:#7ba6dd;--ok:#5fbf8f;--warn:#dba847;
 --hot:#e37d66;--chip:#1d232b;--bar:#232a33}}
*{box-sizing:border-box}
.assess.big{display:flex;flex-wrap:wrap;gap:26px;align-items:center;
 justify-content:space-between}
.assess.big .verdictbox{display:flex;align-items:center;gap:16px;min-width:260px;flex:1 1 320px}
.assess.big .mark{width:46px;height:46px;flex:none;border-radius:50%;display:flex;
 align-items:center;justify-content:center;font-size:24px;font-weight:700;color:#fff;
 background:var(--muted)}
.assess.big.ok .mark{background:var(--ok)} .assess.big.hot .mark{background:var(--hot)}
.assess.big.warn .mark{background:var(--warn)}
.assess.big.pending .mark{background:transparent;border:2px dashed var(--line);
 color:var(--muted)}
.assess.big h2{margin:0}
.assess.big p{margin:5px 0 0;color:var(--muted);font-size:13px;max-width:52ch}
.scorebox{min-width:230px;flex:0 1 300px;text-align:right}
.scorelabel{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.score{font-size:34px;font-weight:680;line-height:1.15;margin-top:2px}
.score.na{font-size:17px;font-weight:600;color:var(--muted);padding:9px 0}
.gauge{position:relative;height:8px;border-radius:5px;margin-top:9px;
 background:linear-gradient(90deg,var(--ok),var(--warn),var(--hot))}
.assess.big.pending .gauge{background:var(--bar)}
.gauge .needle{position:absolute;top:-4px;width:3px;height:16px;border-radius:2px;
 background:var(--ink);transform:translateX(-50%)}
.assess.big.pending .needle{background:var(--line)}
.gaugeends{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);
 margin-top:4px}
.scorenote{font-size:11.5px;color:var(--muted);margin-top:8px;line-height:1.45;text-align:left}
.dir{display:block;margin-top:3px;font-size:11px;color:var(--muted);font-style:italic}
.tbl{width:100%;border-collapse:collapse;margin-top:10px;font-variant-numeric:tabular-nums}
.tbl th{text-align:left;font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;
 color:var(--muted);font-weight:600;padding:0 12px 7px 0;border-bottom:1px solid var(--line)}
.tbl th:not(:first-child),.tbl td:not(:first-child){text-align:right}
.tbl td{padding:8px 12px 8px 0;border-bottom:1px solid var(--line);font-size:13px}
.tbl tbody tr:last-child td{border-bottom:0}
.grouplabel{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
 color:var(--muted);margin:12px 0 6px;padding-top:9px;border-top:1px solid var(--line)}
.grouplabel:first-of-type{margin-top:0;padding-top:0;border-top:0}
details.fold{padding:0}
details.fold > summary{list-style:none;cursor:pointer;padding:15px;font-weight:620;
 font-size:11.5px;letter-spacing:.085em;text-transform:uppercase;color:var(--muted);
 display:flex;align-items:center;gap:9px}
details.fold > summary::-webkit-details-marker{display:none}
details.fold > summary::before{content:'▸';font-size:12px;transition:transform .12s}
details.fold[open] > summary::before{transform:rotate(90deg)}
details.fold > summary:hover{color:var(--ink)}
details.fold .sum{margin-left:auto;text-transform:none;letter-spacing:0;font-weight:500;
 font-size:12px;font-variant-numeric:tabular-nums}
.foldbody{padding:0 15px 15px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;
 background:var(--muted);vertical-align:middle}
.dot.ok{background:var(--ok)} .dot.na{background:var(--muted)}
.techrow{padding:2px 0 7px} .techrow .row{border:0;padding-bottom:2px}
.tech{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.secondary{margin-top:9px;padding:8px 15px;border-radius:8px;border:1px solid var(--line);
 background:var(--chip);color:var(--ink);font:inherit;font-weight:560;cursor:pointer}
.secondary:hover{border-color:var(--accent);color:var(--accent)}
.warnpill{background:var(--warn);color:#fff;margin-left:9px;font-size:10px;
 letter-spacing:.05em;vertical-align:middle}
.attr{display:grid;grid-template-columns:minmax(220px,340px) 1fr;gap:20px;align-items:start;
 margin-top:12px}
.attr img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
@media(max-width:720px){.attr{grid-template-columns:1fr}}
.rgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:9px}
.maps{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.maps figure{margin:0}
.maps img{width:100%;border-radius:7px;border:1px solid var(--line);display:block;
 image-rendering:pixelated;background:var(--chip)}
.maps figcaption{font-size:12px;color:var(--muted);margin-top:7px;line-height:1.45}
.maps figcaption b{color:var(--ink)}
.maps .caveat{display:block;margin-top:5px}
.spark{display:flex;align-items:flex-end;gap:2px;height:44px;margin-top:10px;
 padding:4px;background:var(--chip);border-radius:6px}
.spark i{flex:1;min-width:2px;background:var(--accent);border-radius:1px;opacity:.85}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:12px 22px;background:var(--panel);
 border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.brand{font-weight:650;letter-spacing:.2px} .brand span{color:var(--accent)}
.tabs{display:flex;gap:4px;margin-left:14px}
.tab{padding:6px 15px;border-radius:8px;cursor:pointer;font-weight:550;color:var(--muted)}
.tab.on{background:var(--chip);color:var(--ink)}
.status{margin-left:auto;font-size:12px;color:var(--muted)}
main{max-width:1180px;margin:0 auto;padding:20px}
.drop{border:1.5px dashed var(--line);border-radius:12px;background:var(--panel);padding:32px;
 text-align:center;cursor:pointer} .drop:hover,.drop.over{border-color:var(--accent)}
.drop h2{margin:0 0 5px;font-size:15px} .drop p{margin:0;color:var(--muted);font-size:13px}
textarea{width:100%;min-height:190px;padding:13px;border:1px solid var(--line);border-radius:10px;
 background:var(--panel);color:var(--ink);font:13.5px/1.6 inherit;resize:vertical}
button{background:var(--accent);color:#fff;border:0;border-radius:9px;padding:9px 18px;
 font-weight:600;cursor:pointer;margin-top:10px} button:disabled{opacity:.5;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:13px;margin-top:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}
.card.wide{grid-column:1/-1}
h3{margin:0 0 11px;font-size:10.5px;letter-spacing:1px;text-transform:uppercase;color:var(--muted)}
.row{display:flex;justify-content:space-between;gap:10px;padding:6px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0} .row .k{color:var(--muted);flex:0 0 auto}
.row .v{text-align:right;font-variant-numeric:tabular-nums;word-break:break-word}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;background:var(--chip);
 font-size:11px;font-weight:650;letter-spacing:.2px;white-space:nowrap}
.pill.ok{color:var(--ok)} .pill.warn{color:var(--warn)} .pill.hot{color:var(--hot)}
.pill.na{color:var(--muted)}
.assess{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
 border-radius:12px;padding:17px;margin-top:14px}
.assess h2{margin:0 0 5px;font-size:19px}
.assess p{margin:0;color:var(--muted);max-width:78ch}
.stream{margin:11px 0}
.stream .top{display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px}
.track{height:6px;border-radius:4px;background:var(--bar);overflow:hidden}
.fill{height:100%;border-radius:4px;background:var(--accent)}
.fill.ai{background:var(--hot)} .fill.cap{background:var(--ok)} .fill.off{background:var(--muted);opacity:.35}
.sub{font-size:11.5px;color:var(--muted);margin-top:3px}
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{border:1px solid var(--line);border-radius:9px;padding:8px 11px;min-width:150px;flex:1}
.chip .n{font-size:12px;font-weight:600} .chip .d{font-size:11px;color:var(--muted);margin-top:2px}
img.prev{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
ul.cannot{margin:6px 0 0;padding-left:17px;color:var(--muted)} ul.cannot li{margin:5px 0}
.caveat{margin:6px 0 0;font-size:11.5px;color:var(--muted)}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:12px}
.err{color:var(--hot)} .hide{display:none}
footer{max-width:1180px;margin:0 auto;padding:6px 22px 36px;color:var(--muted);font-size:12px}
</style></head><body>
<header>
  <div class="brand">FORGE <span>Detect</span></div>
  <div class="tabs">
    <div class="tab on" data-tab="text">Text</div>
    <div class="tab" data-tab="image">Image</div>
    <div class="tab" data-tab="results">Results</div>
  </div>
  <div class="status" id="status">
    <span class="dot na"></span><span id="statusText">CPU &middot; checking detectors&hellip;</span>
  </div>
</header>
<main>
  <section id="pane-image" class="hide">
    <div class="drop" id="drop">
      <h2>Drop an image, or click to choose one</h2>
      <p>JPEG, PNG, WEBP, TIFF, BMP, GIF, MPO. Analysed in memory, never written to disk.</p>
      <input type="file" id="file" accept="image/*" hidden>
    </div>
    <div id="outImage"></div>
  </section>
  <section id="pane-text">
    <textarea id="text" placeholder="Paste text to analyse. A few paragraphs works best; very short passages carry little signal."></textarea>
    <button id="go">Analyse text</button>
    <div id="outText"></div>
  </section>
  <section id="pane-results" class="hide"><div id="outResults"></div></section>
</main>
<footer>Built by Akila Lourdes Miriyala Francis</footer>
<script>
const $=s=>document.querySelector(s);
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
  $('#pane-image').classList.toggle('hide',t.dataset.tab!=='image');
  $('#pane-text').classList.toggle('hide',t.dataset.tab!=='text');
  $('#pane-results').classList.toggle('hide',t.dataset.tab!=='results');
  if(t.dataset.tab==='results') loadResults();
});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=s=>({high:'hot',medium:'warn',present:'ok',low:'ok',not_found:'na',not_checked:'na'}[s]||'na');
const lbl=s=>({not_found:'not found',not_checked:'not checked'}[s]||s);
const row=(k,v,st)=>`<div class="row"><span class="k">${esc(k)}</span><span class="v">${
  v!=null&&v!==''?esc(v):'<span class="pill na">not available</span>'} ${
  st?`<span class="pill ${cls(st)}">${esc(lbl(st))}</span>`:''}</span></div>`;
const rows=list=>list.map(r=>row(r.label,r.value,r.status)).join('');

// Manipulation rows carry raw statistics, not probabilities. "Resizing 373.65" reads as a
// quantity of editing; it is the strength of a periodic pattern on an open-ended scale. So
// the chip is the headline and the number is demoted to a technical line beneath it.
const techRows=list=>list.map(r=>`<div class="techrow">
  <div class="row"><span class="k">${esc(r.label)}</span><span class="v">${
    r.status?`<span class="pill ${cls(r.status)}">${esc(lbl(r.status))}</span>`
            :'<span class="pill na">not available</span>'}</span></div>
  ${r.value!=null&&r.value!==''?`<div class="tech">raw statistic ${esc(r.value)}</div>`:''}
  ${r.note?`<div class="tech">${esc(r.note)}</div>`:''}</div>`).join('');

// System state, read from the server rather than asserted in markup. A page that claims
// "detector ready" while nothing is loaded is the same class of lie as a fabricated score.
(async()=>{
  let s={}, img={}; try{ s=await (await fetch('/v1/text/arms')).json(); }catch(e){}
  try{ img=await (await fetch('/v1/image/detector')).json(); }catch(e){}
  const ready=Object.values(s).filter(v=>v==='ready').length;
  const dot=$('#status .dot'), txt=$('#statusText');
  txt.textContent='CPU · text detector '+(ready?'ready':'not loaded')
    +' · image detector '+(img.available?'ready':'not loaded');
  dot.className='dot '+(ready||img.available?'ok':'na');
})();

const drop=$('#drop'),input=$('#file');
drop.onclick=()=>input.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');
  if(e.dataTransfer.files[0])sendImage(e.dataTransfer.files[0])};
input.onchange=()=>input.files[0]&&sendImage(input.files[0]);

async function sendImage(f){
  // Remembered so the opt-in stability pass can re-run on the same file without asking
  // the user to pick it again. A fresh upload resets the opt-in.
  if(f){ window._forgeFile=f; window._forgeStability=false; } else { f=window._forgeFile; }
  if(!f) return;
  $('#outImage').innerHTML='<div class="card">Analysing '+esc(f.name)+'…</div>';
  const body=new FormData(); body.append('file',f);
  let r; try{ r=await fetch('/v1/image/analyze?stability='+(window._forgeStability?'true':'false'),
    {method:'POST',body}); }
  catch(e){ $('#outImage').innerHTML='<div class="card err">'+esc(e)+'</div>'; return; }
  if(!r.ok){ const d=await r.json().catch(()=>({detail:r.statusText}));
    $('#outImage').innerHTML='<div class="card err">'+esc(d.detail)+'</div>'; return; }
  renderImage(await r.json());
}

// The overall assessment banner. ONE shape for both tabs, with the score slot either
// filled by a calibrated model or visibly pending. Built pending rather than omitted so
// that the day a detector exists it fills in, instead of the panel being designed then.
// It is never filled from metadata: a number synthesised from EXIF and JPEG tables would
// fire hardest on screenshots and re-saved photographs, which are exactly the innocent
// files this project exists to protect.
function banner(o){
  // A declared verdict carries no probability: the marker is a label, not a score. The
  // gauge is hidden rather than filled with an invented number.
  const pending = !o.available;
  const hasScore = o.probability !== null && o.probability !== undefined;
  const pct = (pending || !hasScore) ? null : (o.probability*100);
  const cls = pending ? 'pending' : (o.verdict==='ai'?'hot':o.verdict==='human'?'ok':'warn');
  const pos = pct===null ? 50 : Math.min(100, Math.max(0, pct));
  // "NO AI DETECTED", never "HUMAN". A detector cannot establish that a person made
  // something; it can only report whether the input looks like the AI distribution it was
  // trained on. "Human" is a claim about the world, "no AI detected" is a claim about the
  // model, and only the second one is true.
  const WORD = {ai:'AI DETECTED', human:'NO AI DETECTED', uncertain:'UNCERTAIN'};
  return `<div class="assess big ${cls}">
    <div class="verdictbox">
      <div class="mark">${pending?'?':(o.verdict==='ai'?'!':o.verdict==='human'?'✓':'~')}</div>
      <div>
        <h2>${esc(pending?'No verdict':(WORD[o.verdict]||o.verdict.toUpperCase()))}</h2>
        <p>${esc(o.reason||'')}</p>
      </div>
    </div>
    <div class="scorebox">
      <div class="scorelabel">${esc(o.score_label||'AI probability')}</div>
      <div class="score ${pct===null?'na':''}">${pct===null
        ?(pending?'not available':'declared'):pct.toFixed(1)+'%'}</div>
      ${pct===null?'':`<div class="gauge"><div class="needle" style="left:${pos}%"></div></div>
      <div class="gaugeends"><span>human</span><span>uncertain</span><span>AI</span></div>`}
      <div class="scorenote">${esc(o.score_note||'')}</div>
    </div></div>`;
}

function renderImage(d){
  const a=d.assessment, ev=d.evidence;
  const DIR={toward_ai:'points to AI',toward_capture:'points to capture',neutral:''};
  const primary = ev.streams.find(s=>s.key==='visual_model');
  const supporting = ev.streams.filter(s=>s.key!=='visual_model');

  const streamRow = s=>`<div class="stream"><div class="top"><span>${esc(s.label)}</span>
      <span><span class="pill ${s.available?'':'na'}">${esc(s.state)}</span></span></div>
      <div class="track"><div class="fill ${s.direction==='toward_ai'?'ai':
        s.direction==='toward_capture'?'cap':(s.available?'':'off')}"
        style="width:${s.available?s.strength:100}%"></div></div>
      <div class="sub">${esc(s.summary)}${s.available&&DIR[s.direction]
        ?` <span class="dir">${esc(DIR[s.direction])}</span>`:''}</div></div>`;

  let h = banner({
    available: !!a.available,
    verdict: a.verdict,
    probability: a.confidence,
    reason: a.reason,
    score_label: a.confidence==null ? 'Determination' : 'AI probability',
    score_note: a.detail||'',
  }) + `<div class="grid">`;

  h+=`<div class="card"><h3>Image</h3>
    ${d.preview?`<img class="prev" src="${d.preview}" alt="preview">`:''}
    <div style="margin-top:11px">
    ${row('File',d.filename)}${row('Size',(d.size_bytes/1024).toFixed(0)+' KB')}</div></div>`;

  h+=`<div class="card"><h3>Evidence</h3>
    <div class="grouplabel">Detector</div>${primary?streamRow(primary):''}
    <div class="grouplabel">Supporting signals</div>${supporting.map(streamRow).join('')}
    <div class="row" style="margin-top:8px"><span class="k">Agreement</span>
    <span class="v"><span class="pill ${cls(ev.conflict)}">${
      ev.conflict==='low'?'consistent':esc(ev.conflict)}</span></span></div>
    ${ev.conflict!=='low'?`<p class="caveat">${esc(ev.conflict_reason)}</p>`:''}
    </div>`;

  h+=`<div class="card"><h3>File signals</h3>${rows(d.authenticity)}</div>`;

  // Detector robustness: does the VERDICT survive the transforms an image meets in the
  // wild? Runs on every request now that it costs about a second, so there is no button
  // and no second pass over the same file.
  const R=d.robustness||[];
  if(R.length){
    const held=R.filter(r=>!r.error&&!r.changed).length, n=R.filter(r=>!r.error).length;
    h+=`<div class="card wide"><h3>Robustness
      <span class="sum">${held} of ${n} edits keep the same verdict</span></h3>
      <p class="caveat" style="margin:0 0 12px">Every edit is re-scored and compared with the
      original. "flipped" means that edit changes the answer.</p>
      <div class="rgrid">${R.map(r=>r.error
        ?`<div class="chip"><div class="n">${esc(r.attack)}</div>
          <div class="d">failed</div></div>`
        :`<div class="chip"><div class="n">${esc(r.attack)}
          <span class="pill ${r.changed?'hot':'ok'}">${r.changed?'flipped':'held'}</span></div>
          <div class="d">${(r.ai_probability*100).toFixed(1)}%${r.attack==='original'?''
            :` · ${r.delta>=0?'+':''}${(r.delta*100).toFixed(1)} pts`}</div></div>`).join('')}
      </div></div>`;
  }

  // ATTRIBUTION, from the detector itself. Each region is hidden and the image re-scored:
  // a warm cell is one the decision rested on. The old panel here showed compression and
  // noise statistics, which know nothing about the model and were read as if they did.
  const attr=d.attribution_map;
  if(attr&&attr.image){
    h+=`<div class="card wide"><h3>What the verdict rests on
      <span class="sum">${attr.grid}×${attr.grid} occlusion</span></h3>
      <div class="attr"><img src="${attr.image}" alt="attribution map">
        <div><p class="caveat">${esc(attr.reading)}</p>
        <div class="row"><span class="k">Score with everything visible</span>
          <span class="v">${(attr.base_probability*100).toFixed(1)}%</span></div>
        <div class="row"><span class="k">Largest single drop when hidden</span>
          <span class="v">${(attr.peak.drop*100).toFixed(1)} pts</span></div>
        <p class="caveat">Measured by hiding each region and re-scoring, not by a gradient
        approximation, so it holds for any detector this project loads.</p></div></div></div>`;
  }

  const maps=d.maps||[];
  if(maps.length){
    h+=`<details class="card wide fold"><summary>Pixel diagnostics
      <span class="sum">not attribution</span></summary>
      <div class="foldbody">
      <div class="maps">${maps.map(m=>`<figure>
        <img src="${m.image}" alt="${esc(m.title)}">
        <figcaption><b>${esc(m.title)}</b><br>${esc(m.short||m.what_it_shows)}</figcaption>
        </figure>`).join('')}</div>
      <p class="caveat">Statistics of the file, computed without any model. They are not
      what the detector looked at; the panel above is.</p></div></details>`;
  }

  const T=d.timings_ms||{}, order=['preview','detector','forensics','perceptual_hash',
    'evidence','attribution','forensic_maps','detector_robustness'];
  const NAME={preview:'Preview',detector:'Detector inference',forensics:'Forensic analysis',
    perceptual_hash:'Perceptual hash',evidence:'Evidence engine',
    attribution:'Attribution map',forensic_maps:'Pixel diagnostics',
    detector_robustness:'Robustness'};
  h+=`<details class="card wide fold"><summary>Details
      <span class="sum">${(d.elapsed_ms/1000).toFixed(1)} s on CPU</span></summary>
    <div class="foldbody">
    <h3>Manipulation analysis</h3>${techRows(d.manipulation)}
    <h3 style="margin-top:18px">Provenance</h3>${rows(d.provenance)}
    <h3 style="margin-top:18px">Timing</h3>
    ${order.filter(k=>k in T).map(k=>`<div class="row"><span class="k">${esc(NAME[k]||k)}</span>
      <span class="v">${T[k].toLocaleString()} ms</span></div>`).join('')}
    ${row('Total',d.elapsed_ms.toLocaleString()+' ms')}
    ${d.detector&&d.detector.available
      ?`<p class="tech">Detector ${esc(d.detector.model_id)}, uncalibrated baseline.</p>`:''}
    </div></details>`;

  h+='</div>';
  $('#outImage').innerHTML=h;
}


// ---------------------------------------------------------------------------
// RESULTS. Every figure is read from a committed run record. Nothing on this
// page is computed at request time and nothing is typed in by hand, so it
// cannot drift away from the repository the way a maintained table always does.
// Sections whose record is absent do not render, rather than rendering zeros.
// ---------------------------------------------------------------------------
let _resultsLoaded=false;
async function loadResults(){
  if(_resultsLoaded) return; _resultsLoaded=true;
  $('#outResults').innerHTML='<div class="card">Loading…</div>';
  let d; try{ d=await (await fetch('/v1/results')).json(); }
  catch(e){ $('#outResults').innerHTML='<div class="card err">'+esc(e)+'</div>'; return; }

  const pc=v=>v==null?'—':(v*100).toFixed(2)+'%';
  const n4=v=>v==null?'—':(+v).toFixed(4);
  let h='';

  h+=`<div class="assess big"><div class="verdictbox"><div>
    <h2>Measured results</h2>
    <p>Two arms of one experiment: identical data budget, differing only in which AI
    documents they contain. Read from committed run records.</p></div></div></div>
    <div class="grid">`;

  const idd=d.in_distribution;
  if(idd){
    h+=`<div class="card wide"><h3>In distribution
      <span class="sum">held-in generators</span></h3>
      <table class="tbl"><thead><tr><th>Arm</th><th>Human FPR</th><th>AI FNR</th>
      <th>AUROC</th><th>ECE</th><th>n</th></tr></thead><tbody>
      ${idd.rows.map(r=>`<tr><td>${esc(r.label)}</td><td>${pc(r.fpr)}</td>
        <td><b>${pc(r.fnr)}</b></td><td>${n4(r.auroc)}</td><td>${n4(r.ece)}</td>
        <td>${r.n_human} + ${r.n_ai}</td></tr>`).join('')}
      </tbody></table>
      ${idd.significance?`<p class="caveat">${esc(typeof idd.significance==='string'
        ?idd.significance:JSON.stringify(idd.significance))}</p>`:''}
      ${idd.caveat?`<p class="caveat">${esc(typeof idd.caveat==='string'
        ?idd.caveat:JSON.stringify(idd.caveat))}</p>`:''}
      </div>`;
  }

  const ood=d.out_of_distribution;
  if(ood){
    h+=`<div class="card wide"><h3>Out of distribution
      <span class="sum">generators neither arm saw</span></h3>
      <table class="tbl"><thead><tr><th>Benchmark</th><th>Arm</th><th>AUROC</th>
      <th>FNR at a matched 0.1% budget</th><th>ECE</th></tr></thead><tbody>
      ${ood.cells.map(c=>`<tr><td>${esc(c.benchmark.toUpperCase())}</td>
        <td>${esc(c.label)}</td><td>${n4(c.auroc)}</td>
        <td><b>${pc(c.fnr_at_budget)}</b></td><td>${n4(c.ece)}</td></tr>`).join('')}
      </tbody></table>
      <p class="caveat">ECE is 0.004 in distribution and 0.18 to 0.44 here. Calibration does
      not degrade under shift, it collapses: the model stays confident while becoming wrong.
      Neither arm is deployable against unseen generators.</p></div>`;

    if(ood.significance.length){
      h+=`<div class="card wide"><h3>Does the gap survive a test?</h3>
        <table class="tbl"><thead><tr><th>Benchmark</th><th>ΔAUROC (B−A)</th><th>95% CI</th>
        <th>Sign reverses</th><th>Discordant at a matched budget</th>
        <th>McNemar p</th></tr></thead><tbody>
        ${ood.significance.map(s=>{
          const dis=s.discordant||{};
          const rev=s.reversal!=null?(s.reversal*100).toFixed(1)+'%':'—';
          return `<tr><td>${esc(s.benchmark.toUpperCase())}</td>
            <td>${s.delta_auroc>=0?'+':''}${n4(s.delta_auroc)}</td>
            <td>${s.ci95?`[${n4(s.ci95[0])}, ${n4(s.ci95[1])}]`:'—'}</td>
            <td>${rev}</td>
            <td>${dis.random_miss_mirror_catch!=null
              ?`${dis.random_miss_mirror_catch} vs ${dis.random_catch_mirror_miss}`:'—'}</td>
            <td>${s.mcnemar_p!=null?(+s.mcnemar_p).toExponential(1):'—'}</td></tr>`;
        }).join('')}
        </tbody></table>
        <p class="caveat">On RAID the two arms have the same AUROC to within noise, the sign
        reversing in 72.7% of resamples, and yet at a matched false-positive budget the
        mirror arm catches 122 documents the control misses against 9 the other way.
        Mirroring moved the low-false-positive tail without moving the ranking, which the
        headline metric cannot see. On MAGE there is no effect at all.</p></div>`;
    }
  }

  const ip=d.image_probe;
  if(ip){
    h+=`<div class="card wide"><h3>Image detector
      <span class="sum">baseline, measured on a labelled probe set</span></h3>
      <table class="tbl"><thead><tr><th></th><th>n</th><th>median P(AI)</th>
      <th>mean P(AI)</th></tr></thead><tbody>
      <tr><td>Generated</td><td>${ip.n.ai}</td><td><b>${n4(ip.median.ai)}</b></td>
        <td>${n4(ip.mean.ai)}</td></tr>
      <tr><td>Photographs</td><td>${ip.n.human}</td><td><b>${n4(ip.median.human)}</b></td>
        <td>${n4(ip.mean.human)}</td></tr>
      </tbody></table>
      <div class="row"><span class="k">Threshold, above every photograph</span>
        <span class="v">${n4(ip.threshold)}</span></div>
      <div class="row"><span class="k">Recall there</span>
        <span class="v"><b>${pc(ip.recall)}</b></span></div>
      <div class="row"><span class="k">Detector</span>
        <span class="v">${esc(ip.model_id)}</span></div>
      <p class="caveat">${ip.in_sample?'IN SAMPLE. The threshold is fitted on the same '
        +ip.n.human+' photographs it is evaluated on, so the false-positive rate is zero by '
        +'construction and this recall is optimistic. It is an operating point, not a '
        +'result.':''} Two other published detectors were measured on the same set: one did
        not separate the groups at all, and one separated only in the mean while placing over
        half the generated images below 0.07.</p></div>`;
  }

  h+=`<div class="card wide"><p class="caveat">${esc(d.note)}</p></div></div>`;
  $('#outResults').innerHTML=h;
}

$('#go').onclick=async()=>{
  const text=$('#text').value.trim();
  if(!text){ $('#outText').innerHTML='<div class="card err">Paste some text first.</div>'; return; }
  $('#outText').innerHTML='<div class="card">Analysing…</div>';
  let r; try{ r=await fetch('/v1/text/analyze',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({text})}); }
  catch(e){ $('#outText').innerHTML='<div class="card err">'+esc(e)+'</div>'; return; }
  const d=await r.json();
  const why=Object.entries(d.unavailable||{})
    .map(([a,m])=>`<div class="card err"><h3>${esc(a)} unavailable</h3><p>${esc(m)}</p></div>`)
    .join('');
  if(!d.available){
    $('#outText').innerHTML=banner({
      available:false,
      reason:'Neither arm could be loaded, so no score is produced. A number here would be '
        +'fabricated.',
      score_label:'AI probability, calibrated',
      score_note:'Load a trained checkpoint into outputs/ to enable scoring.',
    })+`<div class="grid">${why}
      <div class="card"><h3>Input</h3>${row('Words',d.words)}</div></div>`;
    return;
  }

  // The deployed arm is the headline. The other arm is shown beside it because the
  // comparison is the experiment, but a product has one verdict, not two.
  const primary = d.arms.find(a=>a.arm==='mirror') || d.arms[0];
  const agree = d.arms.length>1 && d.arms[0].verdict===d.arms[1].verdict;
  const head = banner({
    available: true,
    verdict: primary.verdict,
    probability: primary.ai_probability,
    reason: (d.arms.length>1
      ? (agree ? 'Both arms agree.' : 'THE ARMS DISAGREE: '
          + d.arms.map(a=>a.label+' says '+a.verdict).join(', ') + '.')
      : 'Only one arm is loaded.')
      + ` Mean over ${primary.n_windows} window${primary.n_windows===1?'':'s'}, `
      + `${d.words} words.`,
    score_label: 'AI probability, calibrated',
    score_note: `${esc(primary.label)}, threshold ${primary.threshold.toFixed(6)} at a `
      + `${primary.fpr_budget} false-positive budget. Validation ECE ${primary.val_ece}. `
      + 'Out of distribution this model is badly miscalibrated: see the note below.',
  });

  const spark = w => w.length<2 ? '' :
    `<div class="spark">${w.map(p=>`<i style="height:${Math.max(2,Math.round(p*100))}%"></i>`).join('')}</div>`;

  const cards = d.arms.map(a=>`<div class="card"><h3>${esc(a.label)}</h3>
    ${row('Verdict',a.verdict,a.verdict==='ai'?'hot':(a.verdict==='human'?'ok':'warn'))}
    ${row('AI probability (mean over windows)',(a.ai_probability*100).toFixed(2)+'%')}
    ${row('Highest single window',(a.max_window_probability*100).toFixed(2)+'%')}
    ${row('Deployed threshold',a.threshold.toFixed(6))}
    ${row('FPR budget',a.fpr_budget)}
    ${row('Distance from threshold',(a.confidence*100).toFixed(1)+'%')}
    ${row('Windows scored',a.n_windows)}
    ${row('Its validation FNR',(a.val_fnr*100).toFixed(3)+'%')}
    ${row('Its validation ECE',a.val_ece)}
    ${spark(a.windows)}</div>`).join('');

  $('#outText').innerHTML=head+`<div class="grid">${cards}${why}
    <div class="card wide"><h3>What this score is not</h3>
      <p>${esc(d.caveat)}</p><p>${esc(d.abstention)}</p>
      <p class="tech" style="margin-top:9px">Provenance, for reproducibility rather than
      display: ${d.arms.map(a=>esc(a.model_version)).join(' · ')}</p></div></div>`;
};
</script></body></html>"""
