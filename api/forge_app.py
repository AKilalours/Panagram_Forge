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

WHAT IS NOT, AND WHY IT IS NOT FAKED. The overall verdict and the attribution heatmap need
trained weights. The reference design this page follows shows a large green "Human, 98.2%".
That number comes from a calibrated model. Synthesising one from metadata would fire hardest
on screenshots, chat-app downloads and re-saved photographs, because those have no EXIF and
library quantisation tables: exactly the false positive against innocent human images that
this entire project exists to prevent. So the assessment panel states its absence and the
reason, and the evidence panel shows each stream separately rather than averaging them.

The layout keeps every panel of the target design. The panels that need a model are present,
labelled, and will populate the moment weights exist.

Run:  uvicorn api.forge_app:app --port 8080
"""

from __future__ import annotations

import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from forge.image.report import build_report

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 50_000
ACCEPTED = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF"}

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
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds {MAX_IMAGE_BYTES} bytes")

    report = build_report(data, filename=file.filename or "upload", preview=_thumbnail(data))
    fmt = report.by_name("file_type")
    if fmt is None or fmt.value not in ACCEPTED:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported or unreadable image (detected: {fmt.value if fmt else None})",
        )
    return JSONResponse(report.as_dict())


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
    <div class="tab on" data-tab="image">Image</div>
    <div class="tab" data-tab="text">Text</div>
  </div>
  <div class="status">CPU &middot; forensics live &middot; detectors pending training</div>
</header>
<main>
  <section id="pane-image">
    <div class="drop" id="drop">
      <h2>Drop an image, or click to choose one</h2>
      <p>JPEG, PNG, WEBP, TIFF, BMP, GIF. Analysed in memory, never written to disk.</p>
      <input type="file" id="file" accept="image/*" hidden>
    </div>
    <div id="outImage"></div>
  </section>
  <section id="pane-text" class="hide">
    <textarea id="text" placeholder="Paste text to analyse."></textarea>
    <button id="go">Analyse text</button>
    <div id="outText"></div>
  </section>
</main>
<footer>Every value is computed from what you uploaded. Nothing is inferred, defaulted or
filled in; where a signal is unavailable the panel says so.</footer>
<script>
const $=s=>document.querySelector(s);
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
  $('#pane-image').classList.toggle('hide',t.dataset.tab!=='image');
  $('#pane-text').classList.toggle('hide',t.dataset.tab!=='text');
});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=s=>({high:'hot',medium:'warn',present:'ok',low:'ok',not_found:'na',not_checked:'na'}[s]||'na');
const lbl=s=>({not_found:'not found',not_checked:'not checked'}[s]||s);
const row=(k,v,st)=>`<div class="row"><span class="k">${esc(k)}</span><span class="v">${
  v!=null&&v!==''?esc(v):'<span class="pill na">not available</span>'} ${
  st?`<span class="pill ${cls(st)}">${esc(lbl(st))}</span>`:''}</span></div>`;
const rows=list=>list.map(r=>row(r.label,r.value,r.status)).join('');

const drop=$('#drop'),input=$('#file');
drop.onclick=()=>input.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');
  if(e.dataTransfer.files[0])sendImage(e.dataTransfer.files[0])};
input.onchange=()=>input.files[0]&&sendImage(input.files[0]);

async function sendImage(f){
  $('#outImage').innerHTML='<div class="card">Analysing '+esc(f.name)+'…</div>';
  const body=new FormData(); body.append('file',f);
  let r; try{ r=await fetch('/v1/image/analyze',{method:'POST',body}); }
  catch(e){ $('#outImage').innerHTML='<div class="card err">'+esc(e)+'</div>'; return; }
  if(!r.ok){ const d=await r.json().catch(()=>({detail:r.statusText}));
    $('#outImage').innerHTML='<div class="card err">'+esc(d.detail)+'</div>'; return; }
  renderImage(await r.json());
}

function renderImage(d){
  const a=d.assessment, at=d.attribution, ev=d.evidence;
  let h=`<div class="assess"><h2>${esc(a.verdict||'No verdict')}</h2>
    <p>${esc(a.reason)}</p></div><div class="grid">`;

  h+=`<div class="card"><h3>Image</h3>
    ${d.preview?`<img class="prev" src="${d.preview}" alt="preview">`:''}
    <div style="margin-top:11px">
    ${row('File',d.filename)}${row('Size',(d.size_bytes/1024).toFixed(0)+' KB')}
    ${row('Perceptual hash',d.phash)}</div></div>`;

  h+='<div class="card"><h3>Evidence breakdown</h3>'+ev.streams.map(s=>{
    const k=s.direction==='toward_ai'?'ai':s.direction==='toward_capture'?'cap':
            (s.available?'':'off');
    return `<div class="stream"><div class="top"><span>${esc(s.label)}</span>
      <span>${s.available?s.strength+'%':'<span class="pill na">unavailable</span>'}</span></div>
      <div class="track"><div class="fill ${k}" style="width:${s.available?s.strength:100}%"></div></div>
      <div class="sub">${esc(s.summary)}${s.note?' — '+esc(s.note):''}</div></div>`;
  }).join('')+`<div class="row" style="margin-top:8px"><span class="k">Evidence conflict</span>
    <span class="v"><span class="pill ${cls(ev.conflict)}">${esc(ev.conflict)}</span></span></div>
    <p class="caveat">${esc(ev.conflict_reason)}</p></div>`;

  h+=`<div class="card"><h3>Authenticity signals</h3>${rows(d.authenticity)}
    <p class="caveat">Each line is read from the file. None of them establishes authorship.</p></div>`;

  h+=`<div class="card"><h3>AI attribution</h3>
    ${row('Attributed area',at.ai_area_fraction)}
    ${row('Mixed content',at.mixed_content)}
    ${row('Heatmap',at.heatmap)}
    <p class="caveat">${esc(at.reason)}</p></div>`;

  h+=`<div class="card"><h3>Manipulation analysis</h3>${rows(d.manipulation)}</div>`;
  h+=`<div class="card"><h3>Provenance &amp; forensics</h3>${rows(d.provenance)}</div>`;

  h+=`<div class="card wide"><h3>Signal stability under transformation</h3>
    <div class="chips">${d.stability.map(s=>`<div class="chip">
      <div class="n">${esc(s.label)} <span class="pill ${cls(s.status)}">${esc(lbl(s.status))}</span></div>
      <div class="d">${s.value?esc(s.value):'not checked'}</div></div>`).join('')}</div>
    <p class="caveat">This is signal survival, measured on your image, not detector
    robustness: that needs the detector. A "high" chip means this transform destroys a
    signal shown above.</p></div>`;

  h+=`<div class="card wide"><h3>What this cannot tell you</h3>
    <ul class="cannot">${d.cannot_conclude.map(l=>'<li>'+esc(l)+'</li>').join('')}</ul></div>`;

  h+=`<div class="card wide"><h3>Run</h3>${row('Analysis time',d.elapsed_ms+' ms (CPU)')}
    ${row('Report version',d.report_version)}
    ${row('Image detector','not loaded','not_checked')}</div></div>`;
  $('#outImage').innerHTML=h;
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
    $('#outText').innerHTML=`<div class="assess"><h2>No verdict</h2>
      <p>Neither arm could be loaded, so no score is produced. A number here would be
      fabricated.</p></div><div class="grid">${why}
      <div class="card"><h3>Input</h3>${row('Words',d.words)}</div></div>`;
    return;
  }

  // The two arms disagree near the threshold, and that disagreement is the experiment.
  const agree = d.arms.length>1 && d.arms[0].verdict===d.arms[1].verdict;
  const head = d.arms.length>1
    ? (agree ? `Both arms say ${esc(d.arms[0].verdict)}`
             : `The arms disagree: ${d.arms.map(a=>esc(a.arm)+' '+esc(a.verdict)).join(', ')}`)
    : `${esc(d.arms[0].label)}: ${esc(d.arms[0].verdict)}`;

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
    ${row('Model',a.model_version)}
    ${row('Its validation FNR',(a.val_fnr*100).toFixed(3)+'%')}
    ${row('Its validation ECE',a.val_ece)}
    ${spark(a.windows)}</div>`).join('');

  $('#outText').innerHTML=`<div class="assess"><h2>${head}</h2>
    <p>${esc(d.aggregation)}. ${d.arms.length} of 2 arms loaded, ${d.words} words.</p></div>
    <div class="grid">${cards}${why}
    <div class="card wide"><h3>What this score is not</h3>
      <p>${esc(d.caveat)}</p><p>${esc(d.abstention)}</p></div></div>`;
};
</script></body></html>"""
