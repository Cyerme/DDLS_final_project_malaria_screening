// /static/app.js — defensive, and compatible with the server responses.
// - Install fastapi and uvicorn==0.30.*
// - Works with single or multiple files
// - Never sets img/src or a link/href to "undefined" (prevents GET /undefined 404s)
// - Adapts to both new keys (input_png, cam_overlay_png) and old aliases (input_url, overlay_url)
// - Gracefully no-ops if certain elements aren’t present in index.html

// /static/app.js — matches current index.html IDs and defends against nulls.

(() => {
  "use strict";
  console.log("[app] app.js loaded");

  // ---- DOM lookup (IDs must match index.html) ----
  const fileInput = document.getElementById("fileInput");
  const runBtn    = document.getElementById("runBtn");
  const cfgEl     = document.getElementById("cfg");
  const resultsEl = document.getElementById("results");

  const rocFig  = document.getElementById("rocFig");
  const relFig  = document.getElementById("relFig");
  const gcamFig = document.getElementById("gcamFig");

  // ---- State ----
  let CONFIG = null;

  // ---- Helpers ----
  function humanBytes(n) {
    if (!Number.isFinite(n)) return "";
    const UNITS = ["B","KB","MB","GB"];
    let i = 0, x = n;
    while (x >= 1024 && i < UNITS.length - 1) { x /= 1024; i++; }
    return `${x.toFixed(x >= 10 ? 0 : 1)} ${UNITS[i]}`;
  }

  // Safely pick an image source from known keys (URL, data:, or base64). Returns null if none found.
  function pickImageSrc(obj, keys, mime = "image/png") {
    for (const k of keys) {
      const v = obj?.[k];
      if (typeof v === "string" && v.length > 0) {
        // Accept already-valid URLs: data:, http(s):, or absolute path
        if (/^(data:|\/|https?:)/i.test(v)) return v;
        // Otherwise assume base64 payload
        return `data:${mime};base64,${v}`;
      }
    }
    return null;
  }

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = Array.isArray(v) ? v.join(" ") : String(v);
      else if (k in node) node[k] = v;
      else node.setAttribute(k, v);
    }
    (Array.isArray(children) ? children : [children]).forEach(c => {
      if (c == null) return;
      if (typeof c === "string") node.appendChild(document.createTextNode(c));
      else node.appendChild(c);
    });
    return node;
  }

  function clearResults() {
    if (resultsEl) resultsEl.innerHTML = "";
  }

  function setCfgText(text) {
    if (cfgEl) cfgEl.textContent = text ?? "";
  }

  // ---- Render one result card ----
  function renderResultCard(res, fileLike) {
    const filename = (fileLike && fileLike.name) || res.filename || "uploaded image";

    // Numbers
    const prob = (typeof res.probability === "number") ? res.probability
                : (typeof res.prob === "number") ? res.prob
                : (typeof res.p === "number") ? res.p
                : null;
    const probTxt = (prob != null) ? prob.toFixed(3) : "—";

    const threshold = (CONFIG && CONFIG.threshold != null)
      ? CONFIG.threshold
      : (typeof res.threshold === "number" ? res.threshold : null);
    const thrTxt = (threshold != null) ? Number(threshold).toFixed(3) : "—";

    // Labels from backend (new keys first)
    const decision = res.final_decision || res.decision || res.prediction;
    const labelAtThr = res.label_at_threshold || res.pred_label || res.predicted_label || res.label;
    const predLabel = decision || labelAtThr || "—";

    const abstain = !!(res.abstain || res.abstained || res.defer);
    // res.ood is an object: { is_ood, msp, tau }
    const oodFlag = !!(res.is_ood || res.ood_flag || (res.ood && res.ood.is_ood));

    // Images: handle both URL/data and raw base64 keys
    const inputSrc = pickImageSrc(res, [
      "input_png", "input_url", "preproc_url", "input_png_b64", "input_b64", "preproc_png_b64"
    ]);
    const overlaySrc = pickImageSrc(res, [
      "cam_overlay_png", "overlay_url", "overlay_png_b64", "overlay_b64"
    ]);

    // Build card
    const header = el("div", { class: "card-title" }, [
      el("strong", {}, filename),
      " ",
      abstain ? el("span", { class: "flag abstain" }, "abstained") : null,
      oodFlag ? el("span", { class: "flag ood" }, "OOD") : null
    ].filter(Boolean));

    const kv = el("div", { class: "kv" }, [
      el("div", {}, [el("span", {}, "Prediction: "), String(predLabel)]),
      labelAtThr && decision && labelAtThr !== decision
        ? el("div", {}, [el("span", {}, "Label @ threshold: "), String(labelAtThr)])
        : null,
      el("div", {}, [el("span", {}, "p: "), probTxt, "  |  ", el("span", {}, "thr: "), thrTxt]),
      res.message ? el("div", { class: "hint" }, String(res.message)) : null,
    ].filter(Boolean));

    const imgrow = el("div", { class: "imgrow" }, [
      (inputSrc
        ? el("img", { alt: "preprocessed", src: inputSrc })
        : el("div", { class: "hint" }, "No input image returned")),
      (overlaySrc
        ? el("img", { alt: "Grad-CAM overlay", src: overlaySrc })
        : el("div", { class: "hint" }, "No overlay returned"))
    ]);

    const jsonBtn = (() => {
      const json = JSON.stringify(res, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url  = URL.createObjectURL(blob);
      const a = el("a", { href: url, download: `${filename.replace(/\W+/g, "_")}_result.json` }, "Download JSON");
      // best-effort revoke later
      const obs = new MutationObserver(() => {
        if (!document.body.contains(a)) {
          URL.revokeObjectURL(url);
          obs.disconnect();
        }
      });
      obs.observe(document.body, { childList: true, subtree: true });
      return a;
    })();

    return el("div", { class: "card" }, [header, kv, imgrow, jsonBtn]);
  }

  // ---- Networking ----
  async function fetchConfig() {
    try {
      const r = await fetch("/api/config", { cache: "no-store" });
      if (!r.ok) throw new Error(`config HTTP ${r.status}`);
      CONFIG = await r.json();
      console.log("[app] /api/config →", CONFIG);

      // Show a concise summary
      const t = (CONFIG.temperature_T != null) ? Number(CONFIG.temperature_T).toFixed(4) : "—";
      const thr = (CONFIG.threshold != null) ? Number(CONFIG.threshold).toFixed(4) : "—";
      setCfgText(`T=${t}, threshold=${thr}`);

      // Static figures (only set if provided)
      if (rocFig && CONFIG.assets && CONFIG.assets.roc) rocFig.src = CONFIG.assets.roc;
      if (relFig && CONFIG.assets && CONFIG.assets.reliability) relFig.src = CONFIG.assets.reliability;
      if (gcamFig && CONFIG.assets && CONFIG.assets.gradcam_panel) gcamFig.src = CONFIG.assets.gradcam_panel;
    } catch (err) {
      console.warn("[app] Failed to fetch /api/config", err);
      setCfgText("Could not load configuration.");
    }
  }

  async function analyseOne(file) {
    const fd = new FormData();
    fd.append("files", file, file.name); // key must match FastAPI param 'files'

    console.log("[app] analysing:", file.name, humanBytes(file.size));
    setCfgText(`Analysing “${file.name}”… (${humanBytes(file.size)})`);

    try {
      const r = await fetch("/api/predict", { method: "POST", body: fd });
      if (!r.ok) throw new Error(`predict HTTP ${r.status}`);
      const res = await r.json();         // { results: [ ... ] } or single object
      const payload = Array.isArray(res?.results) ? res.results[0] : res;
      console.log("[app] result payload:", payload);

      const card = renderResultCard(payload, file);
      if (resultsEl) resultsEl.prepend(card);
      setCfgText("Done.");
    } catch (err) {
      console.error("[app] analyseOne failed:", err);
      setCfgText("Something went wrong — check console.");
    }
  }

  async function analyseFiles(files) {
    if (!files || files.length === 0) {
      setCfgText("Pick an image first.");
      return;
    }
    if (runBtn) runBtn.disabled = true;
    try {
      for (const f of files) {
        await analyseOne(f);
      }
    } finally {
      if (runBtn) runBtn.disabled = false;
    }
  }

  // ---- Wire UI ----
  function wire() {
    if (!fileInput || !runBtn) {
      console.warn("[app] Missing #fileInput or #runBtn — check index.html IDs.");
    }

    runBtn?.addEventListener("click", (ev) => {
      ev.preventDefault();
      console.log("[app] Run clicked");
      analyseFiles(fileInput?.files);
    });

    fileInput?.addEventListener("change", () => {
      const f = fileInput.files?.[0];
      if (f) setCfgText(`Selected “${f.name}” (${humanBytes(f.size)})`);
    });
  }

  // ---- Boot ----
  async function boot() {
    await fetchConfig();
    wire();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();