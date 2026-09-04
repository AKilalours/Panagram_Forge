"""ONE stylesheet, two pages.

The FastAPI page in `api/forge_app.py` and the Streamlit page in `streamlit_app.py` are
two shells over the same detectors, and they must not look like two different products.
This CSS used to live inline in the FastAPI page; copying it into the second page would
have created the failure this project keeps hitting, where one copy is corrected and the
other quietly is not. Both import it from here.
"""

CSS = r""":root{--bg:#f4f6f8;--panel:#fff;--ink:#11141a;--muted:#68707c;--line:#e2e6ea;
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
/* THE DECISION BOUNDARY, drawn on the gauge. Without it the bar implies a boundary in the
   middle, and a document scoring 79.8% against a deployed threshold of 0.992 puts the
   needle deep in the red beside the words NO AI DETECTED. Both were correct and the pair
   read as a bug. The tick is where the verdict actually changes. */
.gauge .bound{position:absolute;top:-3px;width:2px;height:14px;background:var(--ink);
 opacity:.55;transform:translateX(-50%)}
.gauge .bound::after{content:'threshold';position:absolute;top:15px;left:50%;
 transform:translateX(-50%);font-size:9.5px;letter-spacing:.02em;color:var(--muted);
 white-space:nowrap}
.gaugeends{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);
 margin-top:4px}
.gauge.hasbound + .gaugeends{margin-top:16px}
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
"""
