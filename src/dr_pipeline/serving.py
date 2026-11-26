"""FastAPI inference service — Step 3 of the MLOps pipeline.

    uvicorn dr_pipeline.serving:app --port 8080

Endpoints:
    GET  /health   -> liveness (process is up)
    GET  /ready    -> readiness (model loaded)
    GET  /model    -> which registry version is serving
    POST /predict  -> grade + referable flag + abstain (HITL routing)

Model source (in order of precedence):
    MODEL_URI env var (e.g. models:/dr-effnetb0@production)
    -> falls back to local models/*.pt checkpoints (dev / no MLflow server)
"""

from __future__ import annotations

import io
import json
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config as C

app = FastAPI(
    title="Diabetic Retinopathy Grading API",
    version="1.0.0",
    docs_url=None,  # overridden with the branded console below (GET /docs)
    description=(
        "Screens a retinal (fundus) photograph for diabetic retinopathy (DR).\n\n"
        "**How to use:** expand `POST /predict` → `Try it out` → `Choose file` "
        "(any fundus photo) → `Execute`.\n\n"
        "**Response fields:**\n"
        "- `grade` — predicted severity: 0 = no DR, 1 = mild, 2 = moderate, "
        "3 = severe, 4 = proliferative\n"
        "- `probs` — model probability for each grade [0..4], sums to 1\n"
        "- `confidence` — the highest of those probabilities (model certainty)\n"
        "- `referable_score` — P(grade >= 2): chance the case needs a specialist\n"
        "- `referable` — clinical decision: `referable_score >= threshold` "
        "→ refer to an ophthalmologist\n"
        "- `threshold` — frozen referral cutoff (0.3537), enforced by the eval gate\n"
        "- `abstain` — `true` when confidence < 0.55: the model refuses to answer "
        "and routes the case to human review (HITL)\n"
        "- `model` — which model/registry version answered\n"
        "- `latency_ms` — server-side inference time\n\n"
        "Screening aid only — not a diagnosis."
    ),
)

# --------------------------------------------------------------------------
# model loading (once, at startup)
# --------------------------------------------------------------------------
_state = {"model": None, "source": None, "loaded_at": None}

# Validation evidence surfaced by /model and the landing page. Lives in models/
# so it ships in the image (reports/ is dockerignored).
try:
    MODEL_CARD = json.loads((C.MODELS_DIR / "model_card.json").read_text(encoding="utf-8"))
except Exception:
    MODEL_CARD = {}

GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
GRADE_MEANINGS = {g["grade"]: g["meaning"]
                  for g in MODEL_CARD.get("grade_scale", [])
                  if "grade" in g} or {i: "" for i in range(5)}


class RegistryEnsemble:
    """Wraps the MLflow-logged EnsembleModule (normalised tensor -> probs)."""

    def __init__(self, model):
        self.model = model

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(x).float()
            if torch.cuda.is_available():
                t = t.cuda()
                self.model = self.model.cuda()
            out = self.model(t)
        return out.cpu().numpy()


class LocalEnsemble:
    """Wraps the local .pt seed checkpoints (dev fallback, no server needed).

    DR_SEEDS env var selects which seeds to load (comma-separated).
    Free-tier instances have 1 GB RAM -> default is single-seed there;
    locally/defaults follow config.SEEDS (full 3-model ensemble).
    """

    def __init__(self):
        import os
        from .predict import DREnsemble
        seeds_env = os.environ.get("DR_SEEDS", "")
        seeds = (tuple(int(s) for s in seeds_env.split(",")) if seeds_env
                 else C.SEEDS)
        self.inner = DREnsemble(seeds=seeds)

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        import torch
        # (N,5): batch of normalised tensors -> ensemble probabilities
        return np.stack(
            [self.inner.predict_proba(torch.from_numpy(xi[None]))
             for xi in x])

    def predict_probs_per_seed(self, x: np.ndarray) -> list[list[float]]:
        """Per-seed probability vectors for one image, for agreement reporting."""
        import torch
        return [[float(p) for p in vec]
                for vec in self.inner.predict_per_seed(torch.from_numpy(x))]


def _load_model() -> None:
    uri = os.environ.get("MODEL_URI")
    if uri:
        import mlflow.pyfunc
        _state["model"] = RegistryEnsemble(mlflow.pyfunc.load_model(uri))
        _state["source"] = uri
    else:
        _state["model"] = LocalEnsemble()
        _state["source"] = "local checkpoints (set MODEL_URI for registry)"


@app.on_event("startup")
def startup() -> None:
    try:
        _load_model()
        _state["loaded_at"] = time.time()
    except Exception as e:  # /ready will report, /health still responds
        _state["model"] = None
        _state["source"] = f"FAILED: {e}"


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Never depends on the model."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the model is loaded and can serve. 503 while loading or if loading failed."""
    if _state["model"] is None:
        raise HTTPException(503, detail=f"model not loaded: {_state['source']}")
    return {"ready": True, "model": _state["source"]}


@app.get("/model")
def model_info() -> dict:
    """Which model/source is serving, which threshold, and the validation card."""
    return {"source": _state["source"], "loaded_at": _state["loaded_at"],
            "threshold": C.THRESHOLD_REFERABLE, "card": MODEL_CARD}


# --------------------------------------------------------------------------
# interactive landing page (POST /predict from a drag-and-drop box)
# --------------------------------------------------------------------------
# no f-strings here: the CSS braces stay literal
_LANDING_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diabetic Retinopathy Grading — live demo</title>
<style>
  :root{
    --bg:#0f172a; --card:#1e293b; --card2:#263449; --ink:#e2e8f0;
    --muted:#94a3b8; --accent:#38bdf8; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:"Segoe UI",Calibri,Arial,sans-serif;color:var(--ink);
       background:radial-gradient(1200px 600px at 80% -10%,#1e3a5f 0%,var(--bg) 55%);}
  .wrap{max-width:880px;margin:0 auto;padding:34px 20px 60px}
  header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
  .logo{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#38bdf8,#0ea5e9);
        display:flex;align-items:center;justify-content:center;font-size:24px}
  h1{font-size:22px;margin:0}
  .sub{color:var(--muted);font-size:13.5px;margin:2px 0 22px}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
  .chip{font-size:12px;padding:5px 11px;border-radius:999px;border:1px solid #334155;background:var(--card)}
  .chip.ok{color:var(--ok);border-color:#14532d}
  .chip.down{color:var(--bad);border-color:#7f1d1d}
  .chip b{color:inherit}
  .card{background:var(--card);border:1px solid #29364a;border-radius:16px;padding:22px;margin-bottom:18px}
  .drop{border:2px dashed #3b82f6;border-radius:12px;padding:36px 20px;text-align:center;cursor:pointer;
        transition:background .18s,border-color .18s}
  .drop:hover,.drop.drag{background:#1b2637;border-color:var(--accent)}
  .drop .big{font-size:17px;color:var(--accent);font-weight:600}
  .drop .hint{color:var(--muted);font-size:12.5px;margin-top:6px}
  #file{display:none}
  .row{display:flex;gap:10px;align-items:center;margin:14px 0 0;flex-wrap:wrap}
  button{background:linear-gradient(135deg,#0ea5e9,#38bdf8);border:0;color:#04121f;font-weight:700;
         padding:11px 22px;border-radius:10px;cursor:pointer;font-size:14px}
  button:disabled{opacity:.45;cursor:not-allowed}
  .spinner{width:16px;height:16px;border:2.5px solid #94a3b8;border-top-color:transparent;border-radius:50%;
           animation:spin .7s linear infinite;display:none}
  .spin .spinner{display:inline-block}
  @keyframes spin{to{transform:rotate(360deg)}}
  .thumb{max-width:180px;max-height:130px;border-radius:8px;border:1px solid #334155;vertical-align:middle}
  .verdict{display:none;padding:16px 18px;border-radius:12px;font-size:16px;font-weight:700;margin-top:4px}
  .verdict.referable{background:rgba(239,68,68,.14);color:var(--bad);border:1px solid #7f1d1d}
  .verdict.calm{background:rgba(34,197,94,.13);color:var(--ok);border:1px solid #14532d}
  .verdict.abstain{background:rgba(245,158,11,.13);color:var(--warn);border:1px solid #78350f}
  .grade{display:flex;align-items:center;gap:18px;margin-top:18px;flex-wrap:wrap}
  .gradeN{font-size:52px;font-weight:800;line-height:1;padding:10px 22px;border-radius:14px}
  .gradelabel{font-size:13px;color:var(--muted)}
  .gradelabel b{font-size:16px;color:var(--ink)}
  .bars{margin-top:16px;display:grid;gap:8px}
  .bar{display:grid;grid-template-columns:150px 1fr 52px;gap:10px;align-items:center;font-size:12.5px}
  .track{background:#0b1524;border-radius:999px;height:12px;overflow:hidden}
  .fill{height:100%;border-radius:999px;width:0;transition:width .5s ease}
  .pct{text-align:right;color:var(--muted)}
  .meta{color:var(--muted);font-size:12px;margin-top:14px;display:flex;gap:14px;flex-wrap:wrap}
  .err{background:rgba(239,68,68,.15);color:var(--bad);border:1px solid #7f1d1d;padding:12px 14px;
       border-radius:10px;display:none;margin-top:12px;font-size:13.5px}
  footer{color:var(--muted);font-size:12px;margin-top:26px;line-height:1.5}
  footer a{color:var(--accent)}
  h2{font-size:15px;margin:0 0 10px}
  .sub2{color:var(--muted);font-size:12.5px;margin:6px 0 10px;line-height:1.5}
  ul.why{margin:0 0 12px 18px;padding:0;font-size:13.5px}
  ul.why li{margin:4px 0}
  table.val{width:100%;border-collapse:collapse;font-size:12.5px}
  table.val th,table.val td{border-bottom:1px solid #24344e;padding:7px 8px;text-align:left;vertical-align:top}
  table.val th{color:var(--muted);font-weight:600}
  table.val td:nth-child(2){color:var(--ok);font-variant-numeric:tabular-nums;white-space:nowrap}
  .seedrow{display:grid;grid-template-columns:70px 1fr 84px;gap:10px;align-items:center;
           font-size:12.5px;margin:5px 0}
  .seedrow .track{height:8px}
  details{margin-top:12px}
  summary{cursor:pointer;color:var(--accent);font-size:12.5px}
</style></head>
<body><div class="wrap">
  <header>
    <div class="logo">👁️</div>
    <div>
      <h1>Diabetic Retinopathy Grading — live API</h1>
    </div>
  </header>
  <div class="sub">Upload a retinal fundus photograph: the model grades severity 0–4 and decides
     whether the patient should be referred to a specialist.</div>
  <div class="chips">
    <span class="chip" id="chip-model">model: loading…</span>
    <span class="chip" id="chip-ready">readiness: checking…</span>
    <span class="chip"><b>API:</b> <a href="/docs" style="color:var(--accent)">/docs Swagger console</a></span>
  </div>

  <div class="card">
    <label class="drop" id="drop" for="file">
      <div class="big">Drag a fundus image here — or click to choose</div>
      <div class="hint">.png / .jpg · max 20 MB · image never stored, only a hash is logged</div>
    </label>
    <input id="file" type="file" accept="image/*">
    <div class="row" id="row-preview" style="display:none">
      <img id="thumb" class="thumb" alt="selected fundus image">
      <button id="go">Grade this image</button><span class="spinner"></span>
    </div>
    <div class="err" id="err"></div>
  </div>

  <div class="card" id="result" style="display:none">
    <div class="verdict" id="verdict"></div>
    <div class="grade">
      <div class="gradeN" id="graden">–</div>
      <div class="gradelabel">predicted grade <b id="gradename">–</b><br>
        confidence <span id="conf" style="color:var(--ink)">–</span></div>
    </div>
    <div class="bars" id="bars"></div>
    <div class="meta">
      <span id="latency"></span><span id="refscore"></span><span id="modelinfo"></span>
    </div>
  </div>

  <div class="card" id="why" style="display:none">
    <h2>Why this grade</h2>
    <ul class="why" id="why-list"></ul>
    <div id="seeds"></div>
  </div>

  <div class="card">
    <h2>How this model was validated</h2>
    <div class="sub2">The clinically actionable output is the <b>binary referral decision</b>
      (grade ≥ 2), not the exact 5-way grade.</div>
    <table class="val">
      <tr><th>Metric</th><th>Value</th><th>What it means</th></tr>
      <tr><td>Referable AUC</td><td id="v-auc">–</td><td>Ranking quality for "needs referral", held-out test</td></tr>
      <tr><td>Sensitivity</td><td id="v-sens">–</td><td>Share of truly referable cases that get caught</td></tr>
      <tr><td>Specificity</td><td id="v-spec">–</td><td>Share of non-referable cases correctly cleared</td></tr>
      <tr><td>Threshold</td><td id="v-thr">–</td><td>Frozen on validation (Youden's J); test scored once</td></tr>
      <tr><td>5-class accuracy</td><td id="v-acc">–</td><td>Exact grade agreement — advisory only</td></tr>
    </table>
    <details>
      <summary>Per-grade recall — the honest breakdown</summary>
      <table class="val" style="margin-top:8px">
        <tr><th>Grade</th><th>Recall</th><th>Reading</th></tr>
        <tbody id="v-grades"></tbody>
      </table>
      <div class="sub2">Mild, severe and proliferative grades are often confused with their
        neighbours. That is precisely why referral is decided by the binary score rather than
        the exact grade — and why the service abstains below 0.55 confidence.</div>
    </details>
    <div class="sub2" id="v-src"></div>
  </div>

  <footer>
    <b>Screening aid only — not a diagnosis.</b> Grade names: 0 No DR · 1 Mild · 2 Moderate ·
    3 Severe · 4 Proliferative. <code>referable</code> = referable_score ≥ 0.3537 (frozen
    clinical threshold from the eval gate). <code>abstain: true</code> = confidence &lt; 0.55 →
    the model refuses and routes the case to human review (HITL).
  </footer>
</div>
<script>
const GRADES = ["No DR","Mild","Moderate","Severe","Proliferative"];
const COLORS = ["#22c55e","#84cc16","#eab308","#f97316","#ef4444"];
const $ = id => document.getElementById(id);
const drop = $("drop"), fileInput = $("file");
let currentFile = null;

fetch("/ready").then(r => r.ok ? r.json() : {ready:false}).then(d => {
  const c = $("chip-ready");
  if (d.ready) { c.classList.add("ok"); c.innerHTML = "<b>ready</b> · " + (d.model||""); }
  else { c.classList.add("down"); c.textContent = "not ready"; }
});
const pct1 = x => (x == null ? "–" : (x * 100).toFixed(1) + "%");
fetch("/model").then(r => r.json()).then(d => {
  $("chip-model").innerHTML = "<b>threshold</b> " + d.threshold + " · " + (d.source || "");
  const c = d.card || {}, v = c.validation || {}, rec = c.per_grade_recall || {};
  if (c.architecture) {
    $("v-src").textContent = c.architecture + " · " + (c.dataset || "") + " · input " + (c.input || "");
  }
  $("v-auc").textContent = v.referable_auc != null ? v.referable_auc.toFixed(3) : "–";
  $("v-sens").textContent = pct1(v.sensitivity);
  $("v-spec").textContent = pct1(v.specificity);
  $("v-thr").textContent = d.threshold != null ? d.threshold : "–";
  $("v-acc").textContent = pct1(v.accuracy_5class);
  const tb = $("v-grades"); tb.innerHTML = "";
  Object.keys(rec).forEach(g => {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + g + " " + GRADES[g] + "</td><td>" + pct1(rec[g]) + "</td><td>" +
      (rec[g] < 0.5 ? "often confused with neighbouring grades" : "reliable") + "</td>";
    tb.appendChild(tr);
  });
}).catch(() => {});
fetch("/health").catch(() => {
  const c = $("chip-ready"); c.classList.add("down"); c.textContent = "service down";
});

function showFile(f){
  currentFile = f;
  $("thumb").src = URL.createObjectURL(f);
  $("row-preview").style.display = "flex";
}
drop.addEventListener("dragover", e => {e.preventDefault(); drop.classList.add("drag");});
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("drag");
  if (e.dataTransfer.files[0]) showFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) showFile(fileInput.files[0]); });

$("go").addEventListener("click", async () => {
  if (!currentFile) return;
  $("go").disabled = true; $("go").classList.add("spin");
  $("err").style.display = "none"; $("result").style.display = "none";
  const fd = new FormData(); fd.append("file", currentFile, currentFile.name || "fundus.png");
  try {
    const r = await fetch("/predict", {method:"POST", body: fd});
    if (!r.ok) {
      let msg = r.status + " " + r.statusText;
      try { const j = await r.json(); if (j.detail) msg = j.detail; } catch(e){}
      throw new Error(msg);
    }
    render(await r.json());
  } catch (e) {
    $("err").textContent = "error: " + e.message; $("err").style.display = "block";
  } finally {
    $("go").disabled = false; $("go").classList.remove("spin");
  }
});

function render(d){
  $("result").style.display = "block";
  const v = $("verdict"); v.style.display = "block";
  if (d.abstain) { v.className = "verdict abstain";
    v.textContent = "ABSTAINED — confidence too low, case routed to human review (HITL)"; }
  else if (d.referable) { v.className = "verdict referable";
    v.textContent = "REFERABLE — specialist review recommended (score " + d.referable_score +
      " ≥ threshold " + d.threshold + ")"; }
  else { v.className = "verdict calm";
    v.textContent = "NOT referable at the frozen threshold (score " + d.referable_score +
      " < " + d.threshold + ")"; }

  $("graden").textContent = d.grade;
  $("graden").style.background = COLORS[d.grade]; $("graden").style.color = "#04121f";
  $("gradename").textContent = d.grade + " — " + (d.grade_label || GRADES[d.grade]);
  if (d.grade_description) {
    $("gradename").textContent += " · " + d.grade_description;
  }
  $("conf").textContent = Math.round(d.confidence * 100) + "% confident";

  const bars = $("bars"); bars.innerHTML = "";
  d.probs.forEach((p, i) => {
    const row = document.createElement("div"); row.className = "bar";
    row.innerHTML =
      "<span>" + i + " " + GRADES[i] + "</span>" +
      "<div class='track'><div class='fill' style='background:" + COLORS[i] + "'></div></div>" +
      "<span class='pct'>" + (p * 100).toFixed(1) + "%</span>";
    bars.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      row.querySelector(".fill").style.width = (p * 100) + "%";
    }));
  });
  $("latency").textContent = "inference: " + d.latency_ms + " ms";
  $("refscore").textContent = "referable score: " + d.referable_score;
  $("modelinfo").textContent = "model: " + d.model;

  // "Why this grade": ensemble agreement, margin, threshold, abstain reason.
  $("why").style.display = "block";
  const list = $("why-list"); list.innerHTML = "";
  (d.explanation || []).forEach(t => {
    const li = document.createElement("li"); li.textContent = t; list.appendChild(li);
  });
  const seeds = $("seeds"); seeds.innerHTML = "";
  const sp = d.seed_probs || [];
  if (!sp.length) {
    seeds.innerHTML = "<div class='sub2'>Per-seed detail unavailable for registry models.</div>";
  }
  sp.forEach((s, i) => {
    const top = Math.max.apply(null, s), gi = s.indexOf(top);
    const row = document.createElement("div"); row.className = "seedrow";
    row.innerHTML = "<span>seed " + i + "</span>" +
      "<div class='track'><div class='fill' style='width:" + (top * 100) +
      "%;background:" + COLORS[gi] + "'></div></div>" +
      "<span class='pct'>g" + gi + " · " + Math.round(top * 100) + "%</span>";
    seeds.appendChild(row);
  });
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    """Interactive demo page: drag & drop a fundus image, see the verdict live."""
    return HTMLResponse(_LANDING_HTML)


@app.get("/docs.css", response_class=HTMLResponse, include_in_schema=False)
def docs_css() -> HTMLResponse:
    """Dark theme for the Swagger console — matches the landing page."""
    return HTMLResponse("""
    /* ---------- page scaffold ---------- */
    html, body { background: #0f172a !important; }
    .swagger-ui {
      background: #0f172a; color: #e2e8f0;
      font-family: "Segoe UI", Calibri, Arial, sans-serif;
    }
    .topbar { display: none; }
    .swagger-ui .wrapper { max-width: 1100px; }
    .scheme-container { background: #1e293b !important; box-shadow: none !important; }

    /* ---------- header / info ---------- */
    .swagger-ui .info .title, .swagger-ui .info h2 { color: #e2e8f0 !important; }
    .swagger-ui .info .title small pre { color: #94a3b8 !important; }
    .swagger-ui .info p, .swagger-ui .info li, .swagger-ui .info div {
      color: #94a3b8 !important;
    }
    .swagger-ui .info a { color: #38bdf8 !important; }
    .swagger-ui .info table { color: #e2e8f0; }
    .swagger-ui .info td, .swagger-ui .info th { color: #94a3b8 !important; }

    /* ---------- endpoints ---------- */
    .swagger-ui .opblock { border-radius: 12px; box-shadow: none !important;
                           background: #1e293b; }
    .swagger-ui .opblock .opblock-summary-path,
    .swagger-ui .opblock .opblock-summary-path__deprecated {
      color: #e2e8f0 !important; font-weight: 600;
    }
    .swagger-ui .opblock .opblock-summary-description {
      color: #94a3b8 !important;
    }
    .swagger-ui .opblock-description-wrapper, .swagger-ui .opblock-external-docs-wrapper,
    .swagger-ui .opblock-title_normal { color: #94a3b8 !important; }
    .swagger-ui .opblock-description-wrapper p, .swagger-ui .opblock-title_normal p {
      color: #94a3b8 !important;
    }
    .swagger-ui .opblock-tag { color: #e2e8f0 !important;
      border-bottom: 1px solid #334155 !important; }
    .swagger-ui .opblock-tag small { color: #94a3b8 !important; }

    /* ---------- forms / inputs ---------- */
    .swagger-ui input[type="text"], .swagger-ui input[type="password"],
    .swagger-ui input[type="search"], .swagger-ui input[type="email"],
    .swagger-ui textarea, .swagger-ui select {
      background: #0b1524 !important; color: #e2e8f0 !important;
      border: 1px solid #334155 !important; border-radius: 8px;
    }
    .swagger-ui select { background-image: none !important; }
    .swagger-ui .parameters-col_description, .swagger-ui .parameter__name,
    .swagger-ui .parameter__type, .swagger-ui .parameter__in {
      color: #94a3b8 !important;
    }
    .swagger-ui table thead tr td, .swagger-ui table thead tr th {
      color: #94a3b8 !important; border-bottom: 1px solid #334155 !important;
    }
    .swagger-ui table tbody tr td { color: #e2e8f0 !important;
      border-bottom: 1px solid #29364a !important; }

    /* ---------- buttons ---------- */
    .swagger-ui .btn { color: #e2e8f0 !important; border-color: #334155 !important;
      background: #263449 !important; box-shadow: none !important; }
    .swagger-ui .btn:hover { background: #334155 !important; }
    .swagger-ui .btn.execute {
      background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
      color: #04121f !important; font-weight: 700; border-radius: 10px;
    }
    .swagger-ui .btn.cancel { color: #ef4444 !important; }
    .swagger-ui .btn.authorize { background: #14532d !important; color: #22c55e !important; }

    /* ---------- responses / code panes ---------- */
    .swagger-ui .responses-inner, .swagger-ui .responses { color: #e2e8f0; }
    .swagger-ui .response-col_description, .swagger-ui .response-col_status {
      color: #94a3b8 !important;
    }
    .swagger-ui code, .swagger-ui .code, .swagger-ui pre {
      background: #0b1524 !important; color: #d7e3f4 !important;
      border-radius: 8px;
    }
    .swagger-ui .microlight { background: #0b1524 !important;
      color: #d7e3f4 !important; }
    .swagger-ui .microlight code span { color: #d7e3f4 !important; }
    .swagger-ui .live-responses-table .response__body { color: #d7e3f4 !important; }

    /* ---------- models section ---------- */
    .swagger-ui section.models { background: #1e293b !important;
      border: 1px solid #29364a !important; border-radius: 12px; }
    .swagger-ui section.models h4, .swagger-ui section.models h5 {
      color: #e2e8f0 !important; }
    .swagger-ui section.models .model-container { background: #263449 !important;
      border-radius: 8px; }
    .swagger-ui .model { color: #e2e8f0 !important; }
    .swagger-ui .model-title { color: #94a3b8 !important; }
    .swagger-ui .prop-type, .swagger-ui .prop-format { color: #38bdf8 !important; }
    .swagger-ui .prop_ENUM { color: #f59e0b !important; }
    .swagger-ui span.brace-close, .swagger-ui span.brace-open { color: #64748b !important; }
    """, media_type="text/css")


# this FastAPI has no custom_css_url param on get_swagger_ui_html, so the
# console HTML is built here directly (no f-strings: braces stay literal)
_DOCS_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Diabetic Retinopathy Grading — API console</title>
<link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
<link type="text/css" rel="stylesheet" href="/docs.css">
<link rel="shortcut icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>👁️</text></svg>">
</head><body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({
  url: "/openapi.json",
  dom_id: "#swagger-ui",
  docExpansion: "none",
  tryItOutEnabled: true
});
</script>
</body></html>"""


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def custom_docs() -> HTMLResponse:
    """Swagger console, mildly branded + Try-it-out pre-enabled."""
    return HTMLResponse(_DOCS_HTML)


class PredictResponse(BaseModel):
    grade: int = Field(description="Predicted DR severity: 0=none, 1=mild, "
                                   "2=moderate, 3=severe, 4=proliferative")
    confidence: float = Field(description="Highest class probability — how "
                                          "certain the model is (0..1)")
    probs: list[float] = Field(description="Probability per grade [0,1,2,3,4], "
                                           "sums to 1")
    referable: bool = Field(description="Clinical decision: needs specialist "
                                        "referral (referable_score >= threshold)")
    referable_score: float = Field(description="P(grade >= 2) = 1 - P(0) - P(1); "
                                               "the chance of referable disease")
    threshold: float = Field(description="Frozen referral threshold (0.3537) "
                                         "enforced by the eval gate")
    abstain: bool = Field(description="true = confidence < 0.55, routed to human "
                                      "review (HITL) instead of an answer")
    model: str = Field(description="Which model / registry version answered")
    latency_ms: float = Field(description="Server-side inference time")
    grade_label: str = Field(default="", description="Human label for the grade")
    grade_description: str = Field(default="", description="One-line clinical meaning")
    margin: float = Field(default=0.0, description="Top-1 minus top-2 probability")
    seed_probs: list[list[float]] = Field(default_factory=list,
                                          description="Per-seed probability vectors")
    agreement: float | None = Field(default=None,
                                    description="Share of seeds whose argmax matches the ensemble grade")
    explanation: list[str] = Field(default_factory=list,
                                   description="Plain-language reasons for this decision")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Grade a fundus photograph (multipart file upload) -> DR severity,
    referral decision, and abstain/HITL flag."""
    # validate the upload FIRST (cheap, model-independent) so clients get a
    # precise 4xx even while the model is still loading; 503 only for the
    # actual inference path.
    t0 = time.time()
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "image too large (>20MB)")
    if not raw:
        raise HTTPException(400, "empty upload")

    # decode -> model input (same preprocessing as training)
    import cv2
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(400, "not a decodable image")
    img = cv2.resize(img, C.IMG_SIZE)
    if img.ndim != 3 or img.shape[2] != 3:
        raise HTTPException(400, "not a color image")

    if _state["model"] is None:
        raise HTTPException(503, "model not loaded")

    x = np.ascontiguousarray(img[:, :, ::-1]).transpose(2, 0, 1)[None]
    x = x.astype(np.float32) / 255.0
    mean = np.array(C.IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
    std = np.array(C.IMAGENET_STD, dtype=np.float32)[None, :, None, None]
    x = (x - mean) / std

    try:
        probs = _state["model"].predict_probs(x)[0]
    except Exception as e:
        raise HTTPException(500, f"inference failed: {e}")

    grade = int(probs.argmax())
    confidence = float(probs.max())
    ref_score = float(1 - probs[0] - probs[1])
    abstain = confidence < 0.55
    referable = ref_score >= C.THRESHOLD_REFERABLE

    # Per-seed evidence. The registry path has one model, so this is best-effort.
    seed_probs: list[list[float]] = []
    try:
        seed_probs = _state["model"].predict_probs_per_seed(x)
    except Exception:
        seed_probs = []

    agreement = None
    if seed_probs:
        agreeing = sum(1 for s in seed_probs if int(np.argmax(s)) == grade)
        agreement = round(agreeing / len(seed_probs), 3)

    ordered = sorted((float(p) for p in probs), reverse=True)
    margin = round(ordered[0] - ordered[1], 4)

    explanation = []
    if seed_probs and agreement is not None:
        explanation.append(
            f"{int(round(agreement * len(seed_probs)))} of {len(seed_probs)} seed models "
            f"predict grade {grade} ({GRADE_LABELS[grade]}).")
    explanation.append(
        f"Top probability {ordered[0]:.2f} vs next {ordered[1]:.2f} — margin {margin:.2f}.")
    explanation.append(
        (f"Referable score {ref_score:.2f} ≥ threshold {C.THRESHOLD_REFERABLE}: "
         "specialist referral recommended.")
        if referable else
        (f"Referable score {ref_score:.2f} < threshold {C.THRESHOLD_REFERABLE}: "
         "no referral at this threshold."))
    if abstain:
        explanation.append(
            f"Confidence {confidence:.2f} is below 0.55, so the case is routed to human "
            "review instead of being answered.")
    if seed_probs:
        spread = float(np.std(seed_probs, axis=0).max())
        explanation.append(
            f"Largest disagreement across seeds {spread:.2f} (0 = all seeds identical).")

    # audit + drift log: image hash only, never the image itself
    try:
        import hashlib
        from .monitor import log_prediction
        log_prediction({
            "image_sha256": hashlib.sha256(raw).hexdigest(),
            "predicted_grade": grade,
            "confidence": round(confidence, 4),
            "referable_score": round(ref_score, 4),
            "abstain": bool(abstain),
        })
    except Exception:
        pass  # monitoring must never break inference

    return PredictResponse(
        grade=grade,
        confidence=round(confidence, 4),
        probs=[round(float(p), 4) for p in probs],
        referable=referable,
        referable_score=round(ref_score, 4),
        threshold=C.THRESHOLD_REFERABLE,
        # low confidence -> service asks for human review instead of answering
        abstain=abstain,
        model=_state["source"],
        latency_ms=round((time.time() - t0) * 1000, 1),
        grade_label=GRADE_LABELS[grade],
        grade_description=GRADE_MEANINGS.get(grade, ""),
        margin=margin,
        seed_probs=[[round(float(p), 4) for p in s] for s in seed_probs],
        agreement=agreement,
        explanation=explanation,
    )
