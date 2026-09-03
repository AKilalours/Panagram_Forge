"""FORGE Image: upload an image, see what the file itself can tell you.

RUNS ENTIRELY ON CPU. No GPU, no network, no model weights. Every number on the page is
computed from the uploaded bytes at request time.

WHAT THIS IS AND IS NOT. It is a forensic reader: container format, EXIF, C2PA presence,
quantisation tables, compression residual, resampling evidence, perceptual hash. It is NOT
an AI detector, because FORGE-Image has not been trained. The verdict panel says exactly
that, and the page ends with a list of the conclusions its own signals cannot support.

That last part is the design. A page like this is read by someone who wants a yes or a no,
and the temptation is to manufacture one from whatever is available. Stripped EXIF plus
library quantisation tables would give a confident-looking "likely AI" that fires hardest
on screenshots and re-saved holiday photographs, which is precisely the false positive a
detector must never produce. So the verdict stays empty until there is a model behind it.

Run:  uvicorn api.image_main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from forge.image.report import build_report

MAX_BYTES = 25 * 1024 * 1024
ACCEPTED = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF"}

app = FastAPI(title="FORGE Image", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "detector_loaded": False, "mode": "forensics-only"}


@app.post("/v1/image/analyze")
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds {MAX_BYTES} bytes")

    report = build_report(data, filename=file.filename or "upload")
    fmt = report.by_name("file_type")
    if fmt is None or fmt.value not in ACCEPTED:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported or unreadable image (detected: {fmt.value if fmt else None})",
        )

    payload = report.as_dict()
    payload["what_this_cannot_say"] = report.what_this_cannot_say
    payload["preview"] = _thumbnail(data)
    return JSONResponse(payload)


def _thumbnail(data: bytes, box: int = 520) -> str | None:
    """A downscaled preview, returned as a data URI so the page needs no file storage.

    Uploads are never written to disk. The bytes live in memory for the length of one
    request and the only thing that leaves is this thumbnail and a set of measurements.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            img.thumbnail((box, box))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FORGE Image</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#12141a; --muted:#666e7a; --line:#e3e6ea;
  --accent:#2f4f7f; --ok:#1f7a4d; --warn:#9a6a12; --hot:#a33a2a; --chip:#eef1f5;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#0f1115; --panel:#161a21; --ink:#e8eaee; --muted:#98a1ae; --line:#252b34;
  --accent:#7aa2d8; --ok:#5fbf8f; --warn:#d9a441; --hot:#e0765f; --chip:#1e242d;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 22px;
  background:var(--panel);border-bottom:1px solid var(--line)}
.brand{font-weight:650;letter-spacing:.2px}
.brand span{color:var(--accent)}
.tag{margin-left:auto;font-size:12px;color:var(--muted)}
main{max-width:1120px;margin:0 auto;padding:22px}
.drop{border:1.5px dashed var(--line);border-radius:12px;background:var(--panel);
  padding:34px;text-align:center;cursor:pointer;transition:border-color .15s}
.drop:hover,.drop.over{border-color:var(--accent)}
.drop h2{margin:0 0 6px;font-size:16px}
.drop p{margin:0;color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-top:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h3{margin:0 0 12px;font-size:11px;letter-spacing:.9px;text-transform:uppercase;color:var(--muted)}
.row{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.row .k{color:var(--muted)}
.row .v{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;background:var(--chip);
  font-size:11.5px;font-weight:600;letter-spacing:.2px}
.pill.ok{color:var(--ok)} .pill.warn{color:var(--warn)} .pill.hot{color:var(--hot)}
.pill.na{color:var(--muted)}
.verdict{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:12px;padding:18px;margin-top:16px}
.verdict h2{margin:0 0 6px;font-size:19px}
.verdict p{margin:0;color:var(--muted);max-width:70ch}
.caveat{margin:7px 0 0;font-size:12px;color:var(--muted);line-height:1.5}
img.preview{width:100%;border-radius:9px;display:block;border:1px solid var(--line)}
ul.cannot{margin:8px 0 0;padding-left:18px;color:var(--muted)}
ul.cannot li{margin:5px 0}
footer{max-width:1120px;margin:0 auto;padding:8px 22px 40px;color:var(--muted);font-size:12px}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:12px}
.err{color:var(--hot)}
</style></head><body>
<header>
  <div class="brand">FORGE <span>Image</span></div>
  <div class="tag">forensics-only build &middot; CPU &middot; no model loaded</div>
</header>
<main>
  <div class="drop" id="drop">
    <h2>Drop an image, or click to choose one</h2>
    <p>JPEG, PNG, WEBP, TIFF, BMP or GIF. Analysed in memory, never written to disk.</p>
    <input type="file" id="file" accept="image/*" hidden>
  </div>
  <div id="out"></div>
</main>
<footer>Every value shown is read from the uploaded file. Nothing is inferred, defaulted or
filled in. Where a signal is unavailable the panel says so.</footer>
<script>
const drop=document.getElementById('drop'), input=document.getElementById('file'),
      out=document.getElementById('out');
drop.onclick=()=>input.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');
  if(e.dataTransfer.files[0]) send(e.dataTransfer.files[0])};
input.onchange=()=>input.files[0]&&send(input.files[0]);

const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=s=>({high:'hot',medium:'warn',present:'ok',low:'ok',
               not_found:'na',not_checked:'na'}[s]||'na');
const label=s=>({not_found:'not found',not_checked:'not checked'}[s]||s);

function row(k,v,status){
  const pill=status?`<span class="pill ${cls(status)}">${esc(label(status))}</span>`:'';
  return `<div class="row"><span class="k">${esc(k)}</span>
          <span class="v">${v??'<span class="pill na">not available</span>'} ${pill}</span></div>`;
}

async function send(f){
  out.innerHTML='<div class="card">Analysing '+esc(f.name)+'…</div>';
  const body=new FormData(); body.append('file',f);
  let r;
  try{ r=await fetch('/v1/image/analyze',{method:'POST',body}); }
  catch(e){ out.innerHTML='<div class="card err">Request failed: '+esc(e)+'</div>'; return; }
  if(!r.ok){ const d=await r.json().catch(()=>({detail:r.statusText}));
    out.innerHTML='<div class="card err">'+esc(d.detail||'failed')+'</div>'; return; }
  render(await r.json());
}

function find(d,n){ return d.findings.find(x=>x.name===n); }

function render(d){
  const fmt=find(d,'file_type'), ex=find(d,'exif'), cam=find(d,'camera_consistency'),
        c2=find(d,'c2pa'), q=find(d,'jpeg_tables'), el=find(d,'error_level'),
        rs=find(d,'resample'), sm=find(d,'manipulation_summary');
  const dim=fmt&&fmt.detail?`${fmt.detail.width} × ${fmt.detail.height}`:null;
  const exf=ex&&ex.value?ex.value:{};

  let html=`<div class="verdict">
    <h2>${esc(d.detector.verdict||'No detector verdict')}</h2>
    <p>${esc(d.detector.reason)}</p></div>`;

  html+='<div class="grid">';

  html+=`<div class="card"><h3>Image</h3>
    ${d.preview?`<img class="preview" src="${d.preview}" alt="uploaded image preview">`:''}
    <div style="margin-top:12px">
    ${row('File',esc(d.filename))}
    ${row('Format',fmt&&fmt.value?esc(fmt.value):null,fmt&&fmt.status)}
    ${row('Dimensions',dim)}
    ${row('Size',(d.size_bytes/1024).toFixed(0)+' KB')}
    ${row('Perceptual hash',d.phash?'<code>'+esc(d.phash)+'</code>':null)}
    </div></div>`;

  html+=`<div class="card"><h3>Capture metadata</h3>
    ${row('Camera make',exf.camera_make?esc(exf.camera_make):null)}
    ${row('Camera model',exf.camera_model?esc(exf.camera_model):null)}
    ${row('Captured',exf.datetime_original?esc(exf.datetime_original):(exf.datetime?esc(exf.datetime):null))}
    ${row('Software',exf.software?esc(exf.software):null)}
    ${row('Exposure',exf.exposure_time?esc(exf.exposure_time):null)}
    ${row('ISO',exf.iso?esc(exf.iso):null)}
    ${row('EXIF block',ex&&ex.status==='present'?(ex.detail.field_count+' fields'):null,ex&&ex.status)}
    ${row('Internal consistency',cam&&cam.value?esc(cam.value):null,cam&&cam.status)}
    <p class="caveat">${esc(ex?ex.caveat:'')}</p></div>`;

  const qv=q&&q.value?q.value:null;
  html+=`<div class="card"><h3>Encoding history</h3>
    ${row('C2PA provenance',c2&&c2.value?esc(c2.value):null,c2&&c2.status)}
    ${row('Estimated JPEG quality',qv&&qv.estimated_quality!=null?qv.estimated_quality:null,q&&q.status)}
    ${row('Quantisation tables',qv?(qv.standard_tables?'library defaults':'custom / camera'):null)}
    <p class="caveat">${esc(q?q.caveat:'')}</p>
    <p class="caveat">${esc(c2?c2.caveat:'')}</p></div>`;

  const ev=el&&el.value?el.value:null, rv=rs&&rs.value?rs.value:null;
  html+=`<div class="card"><h3>Manipulation signals</h3>
    ${row('Compression residual',ev?ev.mean_residual:null,el&&el.status)}
    ${row('Residual patchiness',ev?ev.patchiness:null)}
    ${row('Resampling periodicity',rv?rv.peak_to_median:null,rs&&rs.status)}
    ${row('Strongest signal',sm&&sm.value?esc(sm.value):null,sm&&sm.status)}
    <p class="caveat">${esc(el?el.caveat:'')}</p>
    <p class="caveat">${esc(rs?rs.caveat:'')}</p>
    <p class="caveat">${esc(sm?sm.caveat:'')}</p></div>`;

  html+='</div>';

  html+=`<div class="card" style="margin-top:14px"><h3>What this analysis cannot tell you</h3>
    <ul class="cannot">${d.what_this_cannot_say.map(l=>'<li>'+esc(l)+'</li>').join('')}</ul></div>`;

  html+=`<div class="card" style="margin-top:14px"><h3>Run</h3>
    ${row('Analysis time',d.elapsed_ms+' ms (CPU)')}
    ${row('Forensics version','<code>'+esc(d.forensics_version)+'</code>')}
    ${row('Detector','<span class="pill na">not loaded</span>')}</div>`;

  out.innerHTML=html;
}
</script></body></html>"""
