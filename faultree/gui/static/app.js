// faultree GUI: state, editor wiring, requests to the local server and the
// results dashboard. Plain ES modules, no build step, nothing from the network.
import { t, lang, setLang, locale, applyI18n, initLang } from "./i18n.js";
import { $, el, C, api, fmtP, fmtSci, fmtNum, fmtPct, fmtInt, fmtMs, parseNum, numText, isNum, esc, trunc,
  probScale, downloadJson, exportButtons, bindTip, hideTip, svgEl } from "./util.js";
import * as M from "./model.js";
import { renderTree, layout, legend, gateDesc } from "./treeview.js";
import * as Ch from "./charts.js";

// ------------------------------------------------------------- state ---
const S = {
  doc: null, version: 0, issues: [], selUid: null, file: null,
  hist: { past: [], future: [], key: null },
  success: false, maxOrder: 6,
  point: null, pointVer: -1, pointErr: null,
  whatif: { forced: {}, res: null, ver: -1, err: null },
  unc: { n: 2000, seed: 0, res: null, ver: -1, err: null, kind: "hist", log: true, bins: 40, measure: "criticality", started: 0 },
  ui: { ltab: "tree", rtab: "tree", zoomEdit: null, zoomRes: null, impSort: "criticality", scatter: "rawrrw",
    cutSort: { key: "p", dir: -1 }, highlight: null },
  examples: [], info: null, jobs: new Set(),
};
const MEASURES = () => [
  { key: "birnbaum", label: t("imp.birnbaum") },
  { key: "criticality", label: t("imp.criticality"), pct: true },
  { key: "raw", label: "RAW", log: true },
  { key: "rrw", label: "RRW", log: true },
];
const doc = () => S.doc;
const pFresh = () => S.point && S.pointVer === S.version && !S.issues.length;
const nameOf = (id) => { const e = doc().events[id]; if (e) return M.dispName(e); const g = M.gates(doc()).get(id); return g ? M.dispName(g) : id; };
const fileStem = () => (S.file ? S.file.replace(/\.json$/i, "") : (doc().root.id || "fault-tree"));

// ------------------------------------------------------------ status ---
let tick = null;
function busy(job, on) {
  if (on) S.jobs.add(job); else S.jobs.delete(job);
  renderStatus();
  if (S.jobs.has("unc") && !tick) tick = setInterval(renderStatus, 250);
  if (!S.jobs.has("unc") && tick) { clearInterval(tick); tick = null; }
}
function renderStatus(msg) {
  const st = $("status");
  st.className = "status" + (S.jobs.size ? " busy" : "");
  if (msg) { st.textContent = msg; st.classList.add("err"); return; }
  if (S.jobs.has("unc")) st.textContent = t("status.sampling", { s: ((performance.now() - S.unc.started) / 1000).toLocaleString(locale(), { maximumFractionDigits: 1 }) });
  else if (S.jobs.size) st.textContent = t("status.computing");
  else st.textContent = S.point && pFresh() ? t("status.done", { ms: fmtMs(S.point.elapsed_ms) }) : "";
}

// ----------------------------------------------------------- history ---
function snapshot() { return JSON.stringify({ root: doc().root, events: doc().events, meta: doc().meta, sel: S.selUid }); }
function pushHist(key = null) {
  if (key && S.hist.key === key) return;
  S.hist.past.push(snapshot());
  if (S.hist.past.length > 200) S.hist.past.shift();
  S.hist.future.length = 0;
  S.hist.key = key;
}
function restore(s) { const o = JSON.parse(s); S.doc = { root: o.root, events: o.events, meta: o.meta || {} }; S.selUid = o.sel; M.bumpUid(S.doc); }
function undo() { if (!S.hist.past.length) return; S.hist.future.push(snapshot()); restore(S.hist.past.pop()); S.hist.key = null; changed(); }
function redo() { if (!S.hist.future.length) return; S.hist.past.push(snapshot()); restore(S.hist.future.pop()); S.hist.key = null; changed(); }

// ---------------------------------------------------- change pipeline ---
function changed(opts = {}) {
  M.gcEvents(doc());
  S.version++;
  S.issues = M.validate(doc());
  for (const id of Object.keys(S.whatif.forced)) if (!doc().events[id]) delete S.whatif.forced[id];
  if (S.selUid != null && !M.findUid(doc(), S.selUid)) S.selUid = null;
  $("btnUndo").disabled = !S.hist.past.length;
  $("btnRedo").disabled = !S.hist.future.length;
  renderIssues();
  renderEditor();
  if (opts.side !== false) renderSide();
  if (opts.events !== false && S.ui.ltab === "probs") renderEvents();
  if (S.ui.ltab === "json" && opts.json !== false) fillJson();
  scheduleAnalysis();
  renderResults();
}
function renderIssues() {
  const iss = $("issues");
  iss.hidden = !S.issues.length;
  iss.innerHTML = S.issues.length ? `<strong>${esc(t("issue.title"))}</strong><ul>${S.issues.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>` : "";
}
function loadDoc(d, { note = null } = {}) {
  S.doc = d;
  M.bumpUid(d);
  S.selUid = null;
  S.whatif = { forced: {}, res: null, ver: -1, err: null };
  S.unc.res = null; S.unc.err = null;
  S.hist.past.length = 0; S.hist.future.length = 0; S.hist.key = null;
  S.ui.zoomEdit = null; S.ui.zoomRes = null;
  showNote(note);
  changed();
}
function showNote(extra) {
  const m = doc().meta || {};
  const note = (lang() === "pt" && m.note_pt) || m.note || "";
  const text = [note, extra].filter(Boolean).join(" ");
  $("modelNote").hidden = !text;
  $("modelNote").textContent = text;
}

// ---------------------------------------------------------- analysis ---
let pointTimer = null, pointCtl = null, whatCtl = null;
function scheduleAnalysis() {
  clearTimeout(pointTimer);
  if (S.issues.length) { pointCtl?.abort(); busy("point", false); return; }
  pointTimer = setTimeout(runPoint, 180);
}
async function runPoint() {
  const ver = S.version;
  pointCtl?.abort();
  const ctl = (pointCtl = new AbortController());
  busy("point", true);
  try {
    const res = await api("/api/analyze", { tree: M.exportTree(doc(), { engine: true }), success_mode: S.success, max_order: S.maxOrder }, ctl.signal);
    if (ver !== S.version) return;
    S.point = res; S.pointVer = ver; S.pointErr = null;
  } catch (e) {
    if (e.name === "AbortError") return;
    if (ver !== S.version) return;
    S.point = null; S.pointErr = e.message;
  } finally {
    if (pointCtl === ctl) busy("point", false);
  }
  renderResults();
  if (Object.keys(S.whatif.forced).length) runWhatIf();
}
async function runWhatIf() {
  const forced = { ...S.whatif.forced };
  if (!Object.keys(forced).length || S.issues.length) { S.whatif.res = null; renderResults(); return; }
  const ver = S.version;
  whatCtl?.abort();
  const ctl = (whatCtl = new AbortController());
  busy("whatif", true);
  try {
    const res = await api("/api/whatif", { tree: M.exportTree(doc(), { engine: true }), success_mode: S.success, forced }, ctl.signal);
    if (ver !== S.version || JSON.stringify(forced) !== JSON.stringify(S.whatif.forced)) return;
    S.whatif.res = res; S.whatif.ver = ver; S.whatif.err = null;
  } catch (e) {
    if (e.name === "AbortError") return;
    S.whatif.err = e.message;
  } finally {
    if (whatCtl === ctl) busy("whatif", false);
  }
  renderResults();
}
function uncInputs() {
  const distributions = {}, samples = {};
  for (const id of M.eventIds(doc())) {
    const e = doc().events[id];
    if (e.dist) distributions[id] = e.dist;
    else if (e.samples) samples[id] = e.samples;
  }
  return { distributions, samples };
}
async function runUnc() {
  if (S.issues.length || S.jobs.has("unc")) return;
  const { distributions, samples } = uncInputs();
  const ver = S.version;
  S.unc.started = performance.now();
  busy("unc", true);
  renderResults();
  try {
    S.unc.res = await api("/api/uncertainty", { tree: M.exportTree(doc(), { engine: true }), success_mode: S.success,
      distributions, samples, n: S.unc.n, seed: S.unc.seed });
    S.unc.ver = ver; S.unc.err = null;
  } catch (e) {
    S.unc.err = e.message;
  } finally {
    busy("unc", false);
  }
  renderResults();
}

// ------------------------------------------------------------ editor ---
// Explicit zoom, or fit: the editor fits the width (and scrolls down);
// result trees fit the whole tree when it stays legible.
function fitZoom(box, d, explicit, mode = "edit") {
  if (explicit) return explicit;
  const L = layout(d, mode);
  const wFit = (box.clientWidth - 8) / L.width;
  if (mode === "edit") return Math.max(0.62, Math.min(1, wFit));
  return Math.max(0.6, Math.min(1.15, wFit, (box.clientHeight - 8) / L.height));
}
function renderEditor() {
  const box = $("editBox"), svg = $("editSvg");
  const sl = box.scrollLeft, st = box.scrollTop;
  renderTree(svg, doc(), {
    mode: "edit", zoom: fitZoom(box, doc(), S.ui.zoomEdit), selUid: S.selUid, highlight: S.ui.highlight,
    toolbar: nodeAction, onNode: (n) => select(n.uid), tip: editTip, success: S.success,
  });
  box.scrollLeft = sl; box.scrollTop = st;
}
function editTip(n) {
  if (n.t === "ref") return [t("tree.cloneOf", { id: n.ref }), nameOf(n.ref)];
  if (n.t === "gate") {
    const p = pFresh() ? S.point.probabilities[n.id] : null;
    return [`${n.id} — ${M.dispName(n)}`, gateDesc(n.gate), p != null ? [S.success ? "R" : "P", fmtP(p, 6)] : null];
  }
  const e = doc().events[n.id];
  const occ = M.occurrences(doc()).get(n.id) || 1;
  return [`${n.id} — ${M.dispName(e)}`, [S.success ? t("probs.r") : "p", fmtP(e.prob, 6)],
    e.dist ? [t("probs.dist"), distText(e.dist)] : null, occ > 1 ? t("tree.sharedN", { n: occ }) : null];
}
function distText(d) {
  if (!d) return "—";
  const names = { lognormal: t("dist.lognormal"), beta: "beta", uniform: t("dist.uniform"), loguniform: t("dist.loguniform") };
  return `${names[d.dist]} (${M.DISTS[d.dist].map((k) => `${t("param." + k)} ${fmtNum(d[k], 4)}`).join(", ")})`;
}
function select(uid) { S.selUid = uid; renderEditor(); renderSide(); }
function nodeAction(act, uid) {
  const f = M.findUid(doc(), uid);
  if (!f) return;
  const n = f.n;
  pushHist();
  if (act === "addEvent") { const c = M.newEvent(doc()); n.children.push(c); S.selUid = c.uid; }
  else if (act === "addGate") { const c = M.newGate(doc()); n.children.push(c); S.selUid = c.uid; }
  else if (act === "cycle") {
    if (n.t === "gate") { n.gate = M.GATES[(M.GATES.indexOf(n.gate) + 1) % M.GATES.length]; if (n.gate === "K_OF_N") n.k = Math.min(Math.max(1, n.k || 2), Math.max(1, n.children.length)); }
    else if (n.t === "leaf") { const e = doc().events[n.id]; e.kind = e.kind === "basic" ? "undeveloped" : "basic"; }
  } else if (act === "delete" && f.p) { f.p.children.splice(f.i, 1); S.selUid = f.p.uid; }
  changed();
}
function sideMsg(s) { const m = $("fMsg"); if (m) m.textContent = s; }
function renderSide() {
  const box = $("sideEdit");
  const f = S.selUid != null ? M.findUid(doc(), S.selUid) : null;
  if (!f) {
    box.innerHTML = `<h3>${esc(t("side.title"))}</h3><p class="hint">${esc(t("side.help"))}</p><p class="hint">${esc(t("side.symbols"))}</p>`;
    return;
  }
  const n = f.n;
  const move = f.p ? `<button type="button" class="btn ghost sm" data-side="left" title="${esc(t("side.left"))}">◀</button><button type="button" class="btn ghost sm" data-side="right" title="${esc(t("side.right"))}">▶</button>` : "";
  const remove = f.p ? `<button type="button" class="btn danger sm" data-side="delete">${esc(t("side.remove"))}</button>` : "";
  const opt = (id, label) => `<option value="${esc(id)}">${esc(id)} — ${esc(trunc(label, 26))}</option>`;
  if (n.t === "gate") {
    const p = pFresh() ? S.point.probabilities[n.id] : null;
    const evOpts = Object.keys(doc().events).map((id) => opt(id, M.dispName(doc().events[id]))).join("");
    const clones = M.cloneTargets(doc(), n).map((id) => opt(id, nameOf(id))).join("");
    box.innerHTML = `<h3>${esc(f.p ? t("side.intermediate") : t("side.top"))}</h3>
      <div class="row2"><label class="field"><span>${esc(t("side.gate"))}</span><select id="fGate">${M.GATES.map((g) => `<option value="${g}"${g === n.gate ? " selected" : ""}>${esc(t("gate.name." + g))}</option>`).join("")}</select></label>
      <label class="field"${n.gate === "K_OF_N" ? "" : " hidden"}><span>k (${esc(t("side.of"))} ${n.children.length})</span><input type="number" id="fK" min="0" max="${n.children.length}" step="1" value="${n.k}"></label></div>
      <p class="hint" style="margin-top:-4px">${esc(gateDesc(n.gate))}</p>
      <div class="row2"><label class="field"><span>Id</span><input type="text" id="fId" value="${esc(n.id)}" spellcheck="false"></label>
      <label class="field"><span>${esc(t("side.name"))}</span><input type="text" id="fName" value="${esc(M.dispName(n))}" style="font-family:var(--sans)"></label></div>
      <p class="mono" style="margin:0 0 6px">${S.success ? "R" : "P"} = ${p != null ? esc(fmtP(p, 6)) : "—"}</p>
      <div class="btnrow"><button type="button" class="btn ghost sm" data-side="addEvent">+ ${esc(t("side.addEvent"))}</button><button type="button" class="btn ghost sm" data-side="addGate">+ ${esc(t("side.addGate"))}</button></div>
      ${evOpts ? `<div class="btnrow"><select id="fExisting" aria-label="${esc(t("side.existing"))}" style="flex:1;min-width:0">${evOpts}</select><button type="button" class="btn ghost sm" data-side="addExisting">+ ${esc(t("side.shared"))}</button></div>` : ""}
      ${clones ? `<div class="btnrow"><select id="fClone" aria-label="${esc(t("side.cloneOf"))}" style="flex:1;min-width:0">${clones}</select><button type="button" class="btn ghost sm" data-side="addClone">+ ${esc(t("side.clone"))}</button></div>` : ""}
      <div class="btnrow">${move}<button type="button" class="btn ghost sm" data-side="wrap">${esc(t("side.wrap"))}</button>${remove}</div>
      <p class="msg err" id="fMsg"></p>`;
  } else if (n.t === "ref") {
    const targets = M.cloneTargets(doc(), f.p).map((id) => `<option value="${esc(id)}"${id === n.ref ? " selected" : ""}>${esc(id)} — ${esc(trunc(nameOf(id), 26))}</option>`).join("");
    box.innerHTML = `<h3>${esc(t("side.cloneTitle"))}</h3>
      <p class="hint">${esc(t("side.cloneHelp"))}</p>
      <label class="field"><span>${esc(t("side.cloneOf"))}</span><select id="fRef">${targets}</select></label>
      <div class="btnrow">${move}${remove}</div><p class="msg err" id="fMsg"></p>`;
  } else {
    const e = doc().events[n.id];
    const occ = M.occurrences(doc()).get(n.id) || 1;
    const others = Object.keys(doc().events).filter((id) => id !== n.id);
    box.innerHTML = `<h3>${esc(e.kind === "undeveloped" ? t("side.undeveloped") : t("side.basic"))} ${occ > 1 ? `<span class="tag">${esc(t("side.sharedTag", { n: occ }))}</span>` : ""}</h3>
      <div class="row2"><label class="field"><span>Id</span><input type="text" id="fId" value="${esc(n.id)}" spellcheck="false"></label>
      <label class="field"><span>${esc(t("side.name"))}</span><input type="text" id="fName" value="${esc(M.dispName(e))}" style="font-family:var(--sans)"></label></div>
      <p class="hint" style="margin-top:-4px">${esc(t("side.idHelp"))}</p>
      <div class="row2"><label class="field"><span>${esc(S.success ? t("probs.r") : t("probs.p"))}</span><input type="text" id="fProb" inputmode="decimal" value="${esc(numText(e.prob))}" spellcheck="false"${e.samples ? " disabled" : ""}></label>
      <label class="field"><span>${esc(t("side.kind"))}</span><select id="fKind"><option value="basic"${e.kind === "basic" ? " selected" : ""}>${esc(t("side.kindBasic"))}</option><option value="undeveloped"${e.kind === "undeveloped" ? " selected" : ""}>${esc(t("side.kindUndev"))}</option></select></label></div>
      <p class="hint">${esc(t("probs.dist"))}: ${esc(e.dist ? distText(e.dist) : e.samples ? t("probs.samples", { n: e.samples.length }) : "—")} · <a href="#" data-side="toProbs">${esc(t("side.editProbs"))}</a></p>
      ${others.length ? `<div class="btnrow"><select id="fExisting" aria-label="${esc(t("side.useOther"))}" style="flex:1;min-width:0">${others.map((id) => opt(id, M.dispName(doc().events[id]))).join("")}</select><button type="button" class="btn ghost sm" data-side="useExisting">${esc(t("side.useThis"))}</button></div>` : ""}
      <div class="btnrow">${move}${occ > 1 ? `<button type="button" class="btn ghost sm" data-side="detach">${esc(t("side.detach"))}</button>` : ""}<button type="button" class="btn ghost sm" data-side="wrap">${esc(t("side.wrap"))}</button>${remove}</div>
      <p class="msg err" id="fMsg"></p>`;
  }
  bindSide(n, f);
}
function bindSide(n, f) {
  const box = $("sideEdit");
  box.querySelectorAll("[data-side]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); sideAction(b.dataset.side, n, f); }));
  if (n.t === "ref") {
    $("fRef").addEventListener("change", (ev) => { pushHist(); n.ref = ev.target.value; changed(); });
    return;
  }
  const idIn = $("fId");
  idIn.addEventListener("change", () => {
    const v = idIn.value.trim();
    if (v === n.id) return;
    if (!/^[A-Za-z0-9_.\-]+$/.test(v)) { sideMsg(t("side.badId")); idIn.value = n.id; return; }
    if (n.t === "gate") {
      if (M.allIds(doc()).has(v)) { sideMsg(t("side.idUsed", { id: v })); idIn.value = n.id; return; }
      pushHist();
      const old = n.id;
      n.id = v;
      M.walk(doc().root, (x) => { if (x.t === "ref" && x.ref === old) x.ref = v; });
      changed();
    } else if (doc().events[v]) {
      pushHist(); n.id = v; changed();
    } else {
      if (M.gates(doc()).has(v)) { sideMsg(t("side.isGate", { id: v })); idIn.value = n.id; return; }
      pushHist();
      const old = n.id;
      doc().events[v] = doc().events[old];
      M.walk(doc().root, (x) => { if (x.t === "leaf" && x.id === old) x.id = v; });
      if (S.whatif.forced[old]) { S.whatif.forced[v] = S.whatif.forced[old]; delete S.whatif.forced[old]; }
      changed();
    }
  });
  $("fName").addEventListener("input", (ev) => {
    pushHist("name" + n.uid);
    M.setDispName(n.t === "gate" ? n : doc().events[n.id], ev.target.value);
    changed({ side: false });
  });
  if (n.t === "gate") {
    $("fGate").addEventListener("change", (ev) => { pushHist(); n.gate = ev.target.value; if (n.gate === "K_OF_N") n.k = Math.min(Math.max(1, n.k || 2), Math.max(1, n.children.length)); changed(); });
    $("fK").addEventListener("input", (ev) => { const k = Number(ev.target.value); if (!Number.isInteger(k)) return; pushHist("k" + n.uid); n.k = k; changed({ side: false }); });
  } else {
    $("fProb").addEventListener("input", (ev) => {
      const p = parseNum(ev.target.value);
      if (!(p >= 0 && p <= 1)) { sideMsg(t("side.badProb")); return; }
      sideMsg("");
      pushHist("p" + n.id); doc().events[n.id].prob = p; changed({ side: false });
    });
    $("fKind").addEventListener("change", (ev) => { pushHist(); doc().events[n.id].kind = ev.target.value; changed(); });
  }
}
function sideAction(act, n, f) {
  if (act === "addEvent" || act === "addGate" || act === "delete") return nodeAction(act, n.uid);
  if (act === "toProbs") { setLtab("probs"); return; }
  pushHist();
  if (act === "addExisting") { const c = M.leafNode($("fExisting").value); n.children.push(c); S.selUid = c.uid; }
  else if (act === "addClone") { const c = M.refNode($("fClone").value); n.children.push(c); S.selUid = c.uid; }
  else if (act === "useExisting") { n.id = $("fExisting").value; }
  else if (act === "detach") {
    let id = n.id + "_b", i = 2; const ids = M.allIds(doc());
    while (ids.has(id)) id = n.id + "_" + String.fromCharCode(97 + i++);
    doc().events[id] = JSON.parse(JSON.stringify(doc().events[n.id]));
    n.id = id;
  } else if (act === "left" || act === "right") {
    const j = f.i + (act === "left" ? -1 : 1);
    if (j < 0 || j >= f.p.children.length) { S.hist.past.pop(); return; }
    [f.p.children[f.i], f.p.children[j]] = [f.p.children[j], f.p.children[f.i]];
  } else if (act === "wrap") {
    const g = M.gateNode(M.freshId(doc(), "G"), t("model.newGate"), "OR", [n]);
    if (f.p) f.p.children[f.i] = g; else { g.name = M.dispName(n) || t("model.top"); doc().root = g; }
    S.selUid = g.uid;
  }
  changed();
}

// ------------------------------------------------- events (inputs) tab ---
const DEFAULT_DIST = {
  lognormal: (p) => ({ dist: "lognormal", median: p > 0 ? p : 1e-3, ef: 3 }),
  beta: (p) => ({ dist: "beta", a: 1, b: p > 0 && p < 1 ? +((1 - p) / p).toPrecision(4) : 99 }),
  uniform: (p) => ({ dist: "uniform", low: +(p * 0.5).toPrecision(4), high: Math.min(1, +(p * 1.5).toPrecision(4)) || 0.01 }),
  loguniform: (p) => ({ dist: "loguniform", low: p > 0 ? +(p / 3).toPrecision(4) : 1e-4, high: p > 0 ? Math.min(1, +(p * 3).toPrecision(4)) : 1e-2 }),
};
function renderEvents() {
  const table = $("eventsTable");
  const occ = M.occurrences(doc());
  const head = `<thead><tr><th>${esc(t("probs.event"))}</th><th>${esc(S.success ? t("probs.r") : t("probs.p"))}</th><th>${esc(t("probs.dist"))}</th><th>${esc(t("probs.params"))}</th><th class="num">${esc(t("probs.mean"))}</th></tr></thead>`;
  table.innerHTML = head + "<tbody></tbody>";
  const tb = table.tBodies[0];
  for (const id of M.eventIds(doc())) {
    const e = doc().events[id];
    const tr = el("tr", {}, tb);
    const td0 = el("td", {}, tr);
    el("b", { class: "mono" }, td0, id);
    if (occ.get(id) > 1) el("span", { class: "tag", style: "margin-left:6px" }, td0, "×" + occ.get(id));
    if (M.dispName(e) !== id) el("span", { class: "name", title: M.dispName(e) }, td0, M.dispName(e));
    const td1 = el("td", {}, tr);
    const pin = el("input", { type: "text", value: e.samples ? fmtP(e.prob, 4) : numText(e.prob), inputmode: "decimal", spellcheck: "false", "aria-label": `p ${id}`, disabled: !!e.samples }, td1);
    pin.addEventListener("input", () => {
      const p = parseNum(pin.value);
      pin.classList.toggle("bad", !(p >= 0 && p <= 1));
      if (!(p >= 0 && p <= 1)) return;
      pushHist("p" + id); e.prob = p; changed({ events: false });
    });
    const td2 = el("td", {}, tr);
    if (e.samples) {
      el("span", { class: "hint" }, td2, t("probs.samples", { n: e.samples.length }) + " ");
      el("button", { type: "button", class: "btn ghost xs", onclick: () => { pushHist(); e.samples = null; changed(); } }, td2, t("probs.dropSamples"));
      el("td", {}, tr); el("td", { class: "num mono" }, tr, fmtP(e.prob, 4));
      continue;
    }
    const sel = el("select", { "aria-label": t("probs.dist") + " " + id }, td2);
    for (const [v, label] of [["", "—"], ["lognormal", t("dist.lognormal")], ["beta", "beta"], ["uniform", t("dist.uniform")], ["loguniform", t("dist.loguniform")]])
      el("option", { value: v, selected: (e.dist ? e.dist.dist : "") === v }, sel, label);
    sel.addEventListener("change", () => { pushHist(); e.dist = sel.value ? DEFAULT_DIST[sel.value](e.prob ?? 0.01) : null; changed(); });
    const td3 = el("td", {}, tr);
    const td4 = el("td", { class: "num mono" }, tr, e.dist && M.validDist(e.dist) ? fmtP(M.distMean(e.dist), 4) : "");
    if (e.dist) {
      const box = el("div", { class: "params" }, td3);
      for (const k of M.DISTS[e.dist.dist]) {
        const lab = el("label", {}, box, t("param." + k));
        const inp = el("input", { type: "text", value: numText(e.dist[k]), inputmode: "decimal", spellcheck: "false" }, lab);
        inp.addEventListener("input", () => {
          const v = parseNum(inp.value);
          if (!isNum(v)) { inp.classList.add("bad"); return; }
          pushHist("d" + id + k);
          e.dist = { ...e.dist, [k]: v };
          const ok = M.validDist(e.dist);
          box.querySelectorAll("input").forEach((x) => x.classList.toggle("bad", !ok));
          td4.textContent = ok ? fmtP(M.distMean(e.dist), 4) : "";
          changed({ events: false });
        });
      }
    }
  }
}
async function importProbFile(file) {
  const msg = $("probMsg");
  try {
    const buf = new Uint8Array(await file.arrayBuffer());
    let bin = "";
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    const res = await api("/api/probfile", { filename: file.name, content_b64: btoa(bin) });
    pushHist();
    const notes = M.applyProbs(doc(), res.columns);
    msg.className = "msg ok";
    msg.textContent = t("probs.loaded", { file: file.name, rows: res.rows }) + (notes.length ? " " + notes.join(" ") : "");
    changed();
  } catch (e) {
    msg.className = "msg err";
    msg.textContent = t("probs.loadFail") + " " + e.message;
  }
}

// -------------------------------------------------------------- json ---
function fillJson() { $("jsonText").value = JSON.stringify(M.exportTree(doc()), null, 2); $("jsonMsg").textContent = ""; }
async function loadJsonData(data, source) {
  const res = await api("/api/import", { data });
  const { doc: d, notes } = M.importTree(res.tree, res.probs || null);
  if (res.prob_file_missing) notes.push(t("import.probFileMissing", { file: res.prob_file_missing }));
  S.file = source || null;
  loadDoc(d, { note: notes.join(" ") || null });
  return notes;
}

// ----------------------------------------------------------- results ---
function renderResults() {
  renderSummary();
  renderStatus();
  const tab = S.ui.rtab;
  ({ tree: renderTreeTab, cuts: renderCuts, imp: renderImp, whatif: renderWhatIf, unc: renderUnc, expr: renderExpr })[tab]();
  const badge = document.querySelector('[data-rtab="whatif"] .badge');
  const nf = Object.keys(S.whatif.forced).length;
  if (badge) badge.remove();
  if (nf) el("span", { class: "badge" }, document.querySelector('[data-rtab="whatif"]'), String(nf));
}
function chip(parent, k, v, s, cls = "") {
  const c = el("div", { class: "chip " + cls }, parent);
  el("div", { class: "k" }, c, k);
  el("div", { class: "v" }, c, v);
  if (s) el("div", { class: "s" }, c, s);
  return c;
}
function renderSummary() {
  const box = $("summary");
  box.replaceChildren();
  const d = doc();
  const nBasic = M.eventIds(d).length, nGates = M.gates(d).size;
  if (S.issues.length) {
    chip(box, t("sum.model"), t("sum.incomplete"), t("sum.incompleteHint"), "hero");
    return;
  }
  if (S.pointErr) {
    const c = chip(box, t("sum.error"), "—", S.pointErr, "hero");
    c.style.maxWidth = "100%";
    return;
  }
  const P = S.point;
  if (!P) { chip(box, S.success ? t("sum.R") : t("sum.Q"), "…", t("status.computing"), "hero"); return; }
  const stale = pFresh() ? "" : "stale";
  chip(box, S.success ? t("sum.R") : t("sum.Q"), fmtSci(P.Q, 4), t("sum.exact"), "hero " + stale);
  chip(box, S.success ? t("sum.Qc") : t("sum.Rc"), fmtSci(1 - P.Q, 6), S.success ? t("sum.QcHint") : t("sum.RcHint"), stale);

  if (P.cut_sets && P.cut_sets.available) {
    const cs = P.cut_sets;
    const c = chip(box, t("sum.cuts"), fmtInt(cs.rows.length), "", stale);
    el("span", { class: "tag " + (cs.complete ? "ok" : "warn") }, c.querySelector(".s") || el("div", { class: "s" }, c), cs.complete ? t("cuts.complete") : t("cuts.truncated"));
  } else if (!S.success && P.cut_sets) chip(box, t("sum.cuts"), "—", t("sum.cutsNA"), stale);
  chip(box, t("sum.model"), `${nBasic} · ${nGates} · ${P.bdd.top_nodes}`, t("sum.modelHint"), stale);
  const U = S.unc.res;
  if (U && U.mode === (S.success ? "success" : "failure")) {
    const old = S.unc.ver !== S.version;
    chip(box, t("sum.unc"), `${fmtP(U.Q_stats.p05, 3)} – ${fmtP(U.Q_stats.p95, 3)}`, t("sum.uncHint", { n: fmtInt(U.n) }) + (old ? " · " + t("unc.staleShort") : ""), old ? "stale" : "");
  }
}
function needPoint(host) {
  if (S.issues.length) { host.replaceChildren(el("p", { class: "empty" }, null, t("res.fixModel"))); return false; }
  if (S.pointErr) { host.replaceChildren(el("p", { class: "empty" }, null, S.pointErr)); return false; }
  if (!S.point) { host.replaceChildren(el("p", { class: "empty" }, null, t("status.computing"))); return false; }
  return true;
}
function card(host, title, exportName) {
  const c = el("div", { class: "card" }, host);
  const h = el("h3", {}, c);
  el("span", {}, h, title);
  const plot = el("div", { class: "plot" });
  if (exportName) exportButtons(h, () => plot.querySelector("svg"), () => `${fileStem()}-${exportName}`);
  c.appendChild(plot);
  return { card: c, head: h, plot };
}

// tree with probabilities ------------------------------------------------
function resultTip(n, probs, extra = {}) {
  const P = S.point;
  const pl = S.success ? "R" : "P";
  if (n.t === "ref") return [t("tree.cloneOf", { id: n.ref }), nameOf(n.ref), [pl, fmtP(probs[n.ref], 6)]];
  if (n.t === "gate") {
    const lines = [`${n.id} — ${M.dispName(n)}`, gateDesc(n.gate), [pl, fmtP(probs[n.id], 6)]];
    if (extra.base) lines.push([t("whatif.baseShort"), fmtP(extra.base[n.id], 6)]);
    const U = S.unc.res;
    if (!extra.base && U && S.unc.ver === S.version && U.probabilities[n.id]) lines.push(["p5 – p95", `${fmtP(U.probabilities[n.id].p05, 3)} – ${fmtP(U.probabilities[n.id].p95, 3)}`]);
    return lines;
  }
  const e = doc().events[n.id];
  const lines = [`${n.id} — ${M.dispName(e)}`, [S.success ? t("probs.r") : "p", fmtP(probs[n.id] ?? e.prob, 6)]];
  if (extra.forced && extra.forced[n.id]) lines.push(t("whatif.state." + extra.forced[n.id]));
  if (P && P.importance && P.importance[n.id] && !extra.forced) {
    const I = P.importance[n.id];
    lines.push([t("imp.birnbaum"), fmtNum(I.birnbaum, 4)], [t("imp.criticality"), fmtPct(I.criticality)], ["RAW", fmtNum(I.raw, 4)], ["RRW", fmtNum(I.rrw, 4)]);
  } else if (P && P.birnbaum && P.birnbaum[n.id] != null && !extra.forced) lines.push([t("imp.birnbaum"), fmtNum(P.birnbaum[n.id], 4)]);
  if (extra.hint) lines.push(extra.hint);
  return lines;
}
function treeCanvas(host, id) {
  let box = host.querySelector(".canvasBox");
  if (!box) {
    box = el("div", { class: "canvasBox result", id }, host);
    const z = el("div", { class: "zoom" }, box);
    for (const [k, g] of [["-", "−"], ["fit", "⤢"], ["+", "+"]]) el("button", { type: "button", "data-zoom": `res:${k}` }, z, g);
    svgEl("svg", { role: "img" }, box);
  }
  return box;
}
function renderTreeTab() {
  const host = $("rtab-tree");
  if (!needPoint(host)) return;
  let top = host.querySelector(".treebar");
  if (!top) {
    host.replaceChildren();
    top = el("div", { class: "bar treebar" }, host);
  }
  top.replaceChildren();
  const P = S.point;
  const probs = { ...P.probabilities };
  const scale = probScale(Object.values(probs));
  const lg = el("span", { class: "legendbox" }, top);
  legend(lg, scale, S.success ? t("tree.legendR") : t("tree.legendP"));
  el("span", { class: "hint", style: "flex:1" }, top, t("tree.hint"));
  const box = treeCanvas(host, "resBox");
  const svg = box.querySelector("svg");
  exportButtons(top, () => svg, () => `${fileStem()}-tree`);
  const sl = box.scrollLeft, st = box.scrollTop;
  renderTree(svg, doc(), { mode: "result", zoom: fitZoom(box, doc(), S.ui.zoomRes, "result"), probs, color: scale, success: S.success,
    tip: (n) => resultTip(n, probs) });
  box.scrollLeft = sl; box.scrollTop = st;
}

// cut sets -------------------------------------------------------------
function renderCuts() {
  const host = $("rtab-cuts");
  host.replaceChildren();
  if (S.success) { host.appendChild(el("p", { class: "empty" }, null, t("cuts.successMode"))); return; }
  if (!needPoint(host)) return;
  const cs = S.point.cut_sets;
  const top = el("div", { class: "bar" }, host);
  const lab = el("label", { class: "inline" }, top, t("cuts.maxOrder") + " ");
  const mo = el("input", { type: "number", min: 1, max: 12, value: S.maxOrder, style: "width:58px" }, lab);
  mo.addEventListener("change", () => { const v = Number(mo.value); if (Number.isInteger(v) && v >= 1 && v <= 12) { S.maxOrder = v; S.version++; scheduleAnalysis(); } });
  if (!cs.available) {
    const why = /XOR|monotone/i.test(cs.error || "") ? t("cuts.xor")
      : /max_basic_events/.test(cs.error || "") ? t("cuts.tooMany", { n: cs.limit ?? 32 }) : t("cuts.error", { msg: cs.error });
    el("p", { class: "empty" }, host, why);
    return;
  }
  el("span", { class: "tag " + (cs.complete ? "ok" : "warn") }, top, cs.complete ? t("cuts.complete") : t("cuts.truncated"));
  el("span", { class: "hint" }, top, cs.complete ? t("cuts.completeHint") : t("cuts.truncatedHint", { reason: t("cuts.reason." + cs.reason) }));
  const sums = el("div", { class: "summary", style: "padding-top:0;margin-bottom:12px" }, host);
  chip(sums, t("cuts.count"), fmtInt(cs.rows.length), Object.entries(cs.by_order).map(([o, n]) => t("cuts.byOrder", { o, n })).join(" · "));
  chip(sums, t("cuts.exactQ"), fmtP(S.point.Q, 4), t("cuts.exactHint"));
  const rel = (x) => { const d = S.point.Q > 0 ? x / S.point.Q - 1 : NaN; return t("cuts.vsExact", { r: (d >= 0 ? "+" : "−") + fmtPct(Math.abs(d), 2) }); };
  chip(sums, t("cuts.rare"), fmtP(cs.rare_event, 4), rel(cs.rare_event));
  chip(sums, t("cuts.mcub"), fmtP(cs.mcub, 4), rel(cs.mcub));
  const label = (r) => "{" + r.events.join(", ") + "}";
  const hover = (ids) => { S.ui.highlight = ids ? new Set(ids) : null; renderEditor(); };
  const grid = el("div", { class: "grid2" }, host);
  const left = el("div", {}, grid), right = el("div", {}, grid);
  // table
  const tc = el("div", { class: "card" }, left);
  el("h3", {}, tc, t("cuts.table"));
  const box = el("div", { class: "tablebox", style: "max-height:calc(100vh - 360px)" }, tc);
  const table = el("table", {}, box);
  const cols = [["rank", "#", true], ["events", t("cuts.events"), false], ["order", t("cuts.order"), true], ["p", "P", true], ["share", t("cuts.share"), true]];
  const thr = el("tr", {}, el("thead", {}, table));
  const rank = new Map([...cs.rows].sort((a, b) => b.p - a.p).map((r, i) => [r, i + 1]));
  const key = S.ui.cutSort.key, dir = S.ui.cutSort.dir;
  for (const [k, labTxt, num] of cols) {
    const th = el("th", { class: "sortable" + (num ? " num" : ""), title: t("imp.sortBy", { what: labTxt }) }, thr, labTxt + (k === key ? (dir < 0 ? " ▼" : " ▲") : ""));
    th.addEventListener("click", () => { S.ui.cutSort = { key: k, dir: k === key ? -dir : (k === "events" || k === "rank" || k === "order" ? 1 : -1) }; renderCuts(); });
  }
  const val = (r) => (key === "rank" ? rank.get(r) : key === "events" ? label(r) : key === "order" ? r.order : key === "share" ? r.share ?? 0 : r.p);
  const rows = [...cs.rows].sort((a, b) => { const x = val(a), y = val(b); return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir || b.p - a.p; });
  const maxShare = Math.max(...cs.rows.map((r) => r.share || 0), 1e-12);
  const tbody = el("tbody", {}, table);
  for (const r of rows) {
    const tr = el("tr", {}, tbody);
    el("td", { class: "num" }, tr, String(rank.get(r)));
    const tdE = el("td", { class: "mono" }, tr, label(r));
    tdE.title = r.events.map((id) => `${id}: ${nameOf(id)}`).join("\n");
    el("td", { class: "num" }, tr, String(r.order));
    el("td", { class: "num mono" }, tr, fmtP(r.p, 4));
    const tdS = el("td", { class: "num mono", style: "white-space:nowrap" }, tr);
    el("span", { class: "sharebar", style: `width:${Math.max(1, 70 * (r.share || 0) / maxShare)}px` }, tdS);
    tdS.appendChild(document.createTextNode(fmtPct(r.share)));
    tr.addEventListener("pointerenter", () => { tr.classList.add("hl"); hover(r.events); });
    tr.addEventListener("pointerleave", () => { tr.classList.remove("hl"); hover(null); });
  }
  el("p", { class: "hint" }, tc, t("cuts.shareNote"));
  // contributions chart
  const top15 = [...cs.rows].sort((a, b) => b.p - a.p).slice(0, 15).map((r) => ({ label: label(r), share: r.share, p: r.p, events: r.events }));
  const ch = card(right, t("cuts.chart"), "cut-sets");
  Ch.cutBars(ch.plot, top15, { caption: t("cuts.chartCaption", { n: top15.length, total: cs.rows.length }), onHover: hover });
  const ex = el("div", { class: "card explain" }, host);
  ex.innerHTML = `<h3>${esc(t("cuts.whatTitle"))}</h3><p>${esc(t("cuts.what1"))}</p><p>${esc(t("cuts.what2"))}</p>`;
}

// importance -------------------------------------------------------------
function renderImp() {
  const host = $("rtab-imp");
  host.replaceChildren();
  if (!needPoint(host)) return;
  const P = S.point;
  const ids = P.basic_events;
  const hover = (id) => { S.ui.highlight = id ? new Set([id]) : null; renderEditor(); };
  if (!P.importance) {
    const c = card(host, t("imp.birnbaumOnly"), "birnbaum");
    const rows = ids.map((id) => ({ id, label: nameOf(id), values: { birnbaum: P.birnbaum[id] } }));
    Ch.importanceBars(c.plot, rows, [{ key: "birnbaum", label: t("imp.birnbaum") }], { sort: "birnbaum", onHover: hover });
    el("p", { class: "hint" }, c.card, t("imp.successNote"));
    return;
  }
  const measures = MEASURES();
  const rows = ids.map((id) => ({ id, label: nameOf(id), values: P.importance[id] }));
  const c = card(host, t("imp.title"), "importance");
  el("span", { class: "hint" }, c.head, t("imp.clickHeader"));
  const draw = () => Ch.importanceBars(c.plot, rows, measures, { sort: S.ui.impSort, onSort: (k) => { S.ui.impSort = k; draw(); }, onHover: hover });
  draw();
  const grid = el("div", { class: "grid2" }, host);
  const sc = card(grid, t("imp.scatter"), "importance-scatter");
  const sel = el("select", { "aria-label": t("imp.axes") }, sc.head);
  for (const [v, labTxt] of [["rawrrw", "RAW × RRW"], ["bircrit", t("imp.birnbaum") + " × " + t("imp.criticality")]])
    el("option", { value: v, selected: S.ui.scatter === v }, sel, labTxt);
  sel.addEventListener("change", () => { S.ui.scatter = sel.value; renderImp(); });
  if (S.ui.scatter === "rawrrw") {
    Ch.scatter(sc.plot, rows.map((r) => ({ id: r.id, label: r.label, x: r.values.rrw === null ? Infinity : r.values.rrw, y: r.values.raw })),
      { xLabel: t("imp.rrwAxis"), yLabel: t("imp.rawAxis"), xLog: true, yLog: true, onHover: hover,
        refs: [{ axis: "y", v: 2, label: "RAW = 2" }, { axis: "x", v: 1.005, label: t("imp.rrwRef") }] });
  } else {
    Ch.scatter(sc.plot, rows.map((r) => ({ id: r.id, label: r.label, x: r.values.birnbaum, y: r.values.criticality })),
      { xLabel: t("imp.birnbaum"), yLabel: t("imp.criticality"), onHover: hover });
  }
  const ex = el("div", { class: "card explain" }, grid);
  ex.innerHTML = `<h3>${esc(t("imp.whatTitle"))}</h3><dl>
    <dt>${esc(t("imp.birnbaum"))}</dt><dd>${esc(t("imp.defB"))}</dd>
    <dt>${esc(t("imp.criticality"))}</dt><dd>${esc(t("imp.defC"))}</dd>
    <dt>RAW</dt><dd>${esc(t("imp.defRAW"))}</dd>
    <dt>RRW</dt><dd>${esc(t("imp.defRRW"))}</dd></dl>
    <p>${esc(S.ui.scatter === "rawrrw" ? t("imp.refNote") : t("imp.bcNote"))}</p>`;
}

// what-if --------------------------------------------------------------
function cycleForced(id) {
  const cur = S.whatif.forced[id];
  if (!cur) S.whatif.forced[id] = "failed";
  else if (cur === "failed") S.whatif.forced[id] = "working";
  else delete S.whatif.forced[id];
  if (!Object.keys(S.whatif.forced).length) S.whatif.res = null;
  renderResults();
  runWhatIf();
}
function renderWhatIf() {
  const host = $("rtab-whatif");
  if (!needPoint(host)) return;
  let top = host.querySelector(".wbar");
  if (!top) { host.replaceChildren(); top = el("div", { class: "wbar" }, host); }
  top.replaceChildren();
  const P = S.point;
  const forced = S.whatif.forced;
  const W = Object.keys(forced).length && S.whatif.res && S.whatif.ver === S.version ? S.whatif.res : null;
  const sums = el("div", { class: "summary", style: "padding-top:0;margin-bottom:8px" }, top);
  const ql = S.success ? "R" : "Q";
  chip(sums, t("whatif.base", { q: ql }), fmtSci(P.Q, 4), t("whatif.baseHint"));
  if (W) {
    chip(sums, t("whatif.cond", { q: ql }), fmtSci(W.Q, 4), t("whatif.condHint"), "hero");
    chip(sums, t("whatif.ratio"), P.Q > 0 ? "× " + fmtNum(W.Q / P.Q, 4) : "—", t("whatif.ratioHint"));
    chip(sums, t("whatif.delta"), (W.Q >= P.Q ? "+" : "−") + fmtP(Math.abs(W.Q - P.Q), 4), "");
  } else chip(sums, t("whatif.cond", { q: ql }), "—", Object.keys(forced).length ? t("status.computing") : t("whatif.none"));
  const pills = el("div", { class: "forced bar" }, top);
  el("span", { class: "hint", style: "margin:0" }, pills, t("whatif.forcedLabel"));
  if (!Object.keys(forced).length) el("span", { class: "hint", style: "margin:0" }, pills, "—");
  for (const [id, st] of Object.entries(forced)) {
    const p = el("span", { class: "pill " + st }, pills, `${id}: ${t("whatif.state." + st)}`);
    el("button", { type: "button", "aria-label": t("whatif.release", { id }), title: t("whatif.release", { id }), onclick: () => { delete forced[id]; if (!Object.keys(forced).length) S.whatif.res = null; renderResults(); runWhatIf(); } }, p, "×");
  }
  if (Object.keys(forced).length) el("button", { type: "button", class: "btn ghost xs", onclick: () => { S.whatif.forced = {}; S.whatif.res = null; renderResults(); } }, pills, t("whatif.clear"));
  if (S.whatif.err) el("p", { class: "msg err" }, top, S.whatif.err);
  const probs = W ? W.probabilities : P.probabilities;
  const scale = probScale([...Object.values(P.probabilities), ...(W ? Object.values(W.probabilities) : [])]);
  el("span", { class: "hint grow", style: "margin:0" }, pills, t("whatif.help"));
  const lgb = el("span", { class: "legendbox" }, pills);
  legend(lgb, scale, S.success ? t("tree.legendR") : t("tree.legendP"));
  const box = treeCanvas(host, "whatBox");
  box.classList.add("whatif");
  const svg = box.querySelector("svg");
  exportButtons(lgb, () => svg, () => `${fileStem()}-what-if`);
  const sl = box.scrollLeft, st = box.scrollTop;
  renderTree(svg, doc(), { mode: "whatif", zoom: fitZoom(box, doc(), S.ui.zoomRes, "result"), probs, color: scale, forced, success: S.success,
    tip: (n) => resultTip(n, probs, { base: W ? W.base_probabilities : null, forced, hint: n.t === "leaf" ? t("whatif.clickHint") : null }),
    onNode: (n) => { if (n.t === "leaf") cycleForced(n.id); } });
  box.scrollLeft = sl; box.scrollTop = st;
}

// uncertainty ----------------------------------------------------------
function renderUnc() {
  const host = $("rtab-unc");
  host.replaceChildren();
  if (S.issues.length) { host.appendChild(el("p", { class: "empty" }, null, t("res.fixModel"))); return; }
  const { distributions, samples } = uncInputs();
  const nIn = Object.keys(distributions).length + Object.keys(samples).length;
  const ctl = el("div", { class: "card" }, host);
  const bar = el("div", { class: "bar", style: "margin:0" }, ctl);
  const lenFixed = Object.values(samples)[0]?.length;
  const nl = el("label", { class: "inline" }, bar, t("unc.n") + " ");
  const nIn_ = el("input", { type: "number", min: 2, max: S.info?.limits?.max_samples || 50000, step: 1, value: lenFixed || S.unc.n, style: "width:84px", disabled: !!lenFixed }, nl);
  nIn_.addEventListener("change", () => { const v = Number(nIn_.value); if (Number.isInteger(v) && v >= 2) S.unc.n = v; });
  const sl = el("label", { class: "inline" }, bar, t("unc.seed") + " ");
  const sIn = el("input", { type: "number", min: 0, step: 1, value: S.unc.seed, style: "width:84px" }, sl);
  sIn.addEventListener("change", () => { const v = Number(sIn.value); if (Number.isInteger(v) && v >= 0) S.unc.seed = v; });
  const run = el("button", { type: "button", class: "btn sm", disabled: !nIn || S.jobs.has("unc"), onclick: runUnc }, bar, S.jobs.has("unc") ? t("unc.running") : t("unc.run"));
  run.id = "btnRunUnc";
  el("span", { class: "hint", style: "margin:0" }, bar, nIn ? t("unc.inputsN", { n: nIn }) : "");
  if (!nIn) {
    const p = el("p", { class: "hint" }, ctl, t("unc.noInputs") + " ");
    el("button", { type: "button", class: "btn ghost xs", onclick: () => setLtab("probs") }, p, t("unc.goProbs"));
  }
  el("p", { class: "hint", style: "margin-bottom:0" }, ctl, lenFixed ? t("unc.lenFixed", { n: lenFixed }) : t("unc.how"));
  if (S.unc.err) el("p", { class: "msg err" }, ctl, S.unc.err);
  const U = S.unc.res;
  if (!U) return;
  if (S.unc.ver !== S.version) el("p", { class: "msg err" }, ctl, t("unc.stale"));
  const ql = S.success ? "R" : "Q";
  const Q = U.Q_stats;
  const sums = el("div", { class: "summary", style: "padding-top:0;margin-bottom:12px" }, host);
  chip(sums, t("unc.mean", { q: ql }), fmtSci(Q.mean, 3), t("unc.meanHint", { n: fmtInt(U.n), seed: U.seed }), "hero");
  chip(sums, "p5", fmtSci(Q.p05, 3), "");
  chip(sums, t("unc.median"), fmtSci(Q.p50, 3), "");
  chip(sums, "p95", fmtSci(Q.p95, 3), "");
  if (Q.p05 > 0) chip(sums, t("unc.ef"), fmtNum(Math.sqrt(Q.p95 / Q.p05), 3), t("unc.efHint"));
  if (S.point && pFresh()) chip(sums, t("unc.point", { q: ql }), fmtSci(S.point.Q, 3), t("unc.pointHint"));
  // distribution of Q
  const dc = card(host, t("unc.distTitle", { q: ql }), S.unc.kind === "cdf" ? "uncertainty-cdf" : "uncertainty-histogram");
  const ctr = el("span", { class: "seg" }, dc.head);
  for (const [k, labTxt] of [["hist", t("unc.hist")], ["cdf", t("unc.cdf")]])
    el("button", { type: "button", class: S.unc.kind === k ? "on" : "", onclick: () => { S.unc.kind = k; renderUnc(); } }, ctr, labTxt);
  const lg = el("label", { class: "inline" }, dc.head);
  const cb = el("input", { type: "checkbox", checked: S.unc.log }, lg);
  lg.appendChild(document.createTextNode(t("unc.logX")));
  cb.addEventListener("change", () => { S.unc.log = cb.checked; renderUnc(); });
  if (S.unc.kind === "hist") {
    const bs = el("select", { "aria-label": t("unc.bins") }, dc.head);
    for (const b of [20, 40, 80]) el("option", { value: b, selected: S.unc.bins === b }, bs, t("unc.binsN", { n: b }));
    bs.addEventListener("change", () => { S.unc.bins = Number(bs.value); renderUnc(); });
  }
  const marks = [
    { v: Q.p05, label: "p5", color: C.s3, dash: "6 4" },
    { v: Q.p50, label: "p50", color: C.s3 },
    { v: Q.p95, label: "p95", color: C.s3, dash: "6 4" },
    { v: Q.mean, label: t("unc.meanShort"), color: C.s2, width: 2.2 },
  ];
  if (S.point && pFresh()) marks.push({ v: S.point.Q, label: t("unc.pointShort"), color: C.ink, dash: "2 3" });
  Ch.distChart(dc.plot, U.Q, { kind: S.unc.kind, log: S.unc.log, bins: S.unc.bins, marks, xLabel: ql, title: t("unc.distTitle", { q: ql }) });
  // importance spread
  const measures = U.mode === "failure" ? MEASURES() : [{ key: "birnbaum", label: t("imp.birnbaum") }];
  if (!measures.some((m) => m.key === S.unc.measure)) S.unc.measure = measures[0].key;
  const m = measures.find((x) => x.key === S.unc.measure);
  const ic = card(host, t("unc.impTitle"), "importance-spread");
  const ms = el("select", { "aria-label": t("unc.measure") }, ic.head);
  for (const x of measures) el("option", { value: x.key, selected: x.key === m.key }, ms, x.label);
  ms.addEventListener("change", () => { S.unc.measure = ms.value; renderUnc(); });
  const pointImp = S.point && pFresh() ? (S.point.importance || null) : null;
  const rowsI = Object.entries(U.importance).map(([id, I]) => {
    const s = I[m.key] || {};
    const pt = pointImp ? pointImp[id]?.[m.key] : (m.key === "birnbaum" && S.point && pFresh() ? S.point.birnbaum[id] : null);
    return { id, label: nameOf(id), p05: s.p05 ?? null, p50: s.p50 ?? null, p95: s.p95 ?? (s.n ? null : null), point: pt === null && m.key === "rrw" ? Infinity : pt };
  }).sort((a, b) => (b.p50 ?? -1) - (a.p50 ?? -1));
  Ch.intervalPlot(ic.plot, rowsI, { log: !!m.log, xLabel: m.label, ref: m.log ? 1 : 0 });
  el("p", { class: "hint" }, ic.card, t("unc.impNote"));
  // inputs
  const tc = el("div", { class: "card" }, host);
  el("h3", {}, tc, t("unc.inputs"));
  if (Object.keys(U.clipped).length) el("p", { class: "msg err" }, tc, t("unc.clipped", { list: Object.entries(U.clipped).map(([id, n]) => `${id} (${fmtInt(n)})`).join(", ") }));
  const tb = el("table", {}, el("div", { class: "tablebox" }, tc));
  const hr = el("tr", {}, el("thead", {}, tb));
  for (const h of [t("probs.event"), t("probs.dist"), "p5", "p50", "p95", t("probs.mean"), t("unc.shape")]) el("th", { class: ["p5", "p50", "p95", t("probs.mean")].includes(h) ? "num" : "" }, hr, h);
  const body = el("tbody", {}, tb);
  for (const [id, inp] of Object.entries(U.inputs)) {
    const tr = el("tr", {}, body);
    const e = doc().events[id];
    el("td", { class: "mono" }, tr, id);
    el("td", {}, tr, inp.kind === "samples" ? t("probs.samples", { n: inp.stats.n }) : e && e.dist ? distText(e.dist) : inp.kind);
    for (const k of ["p05", "p50", "p95", "mean"]) el("td", { class: "num mono" }, tr, fmtP(inp.stats[k], 3));
    el("td", {}, tr).appendChild(Ch.sparkHist(inp.preview));
  }
  el("p", { class: "hint" }, tc, t("unc.rng", { seed: U.seed }));
}

// expression and BDD ---------------------------------------------------
function renderExpr() {
  const host = $("rtab-expr");
  host.replaceChildren();
  if (!needPoint(host)) return;
  const P = S.point;
  const c1 = el("div", { class: "card" }, host);
  for (const [title, text] of [[t("expr.expression"), P.expression], [t("expr.symbolic"), P.symbolic]]) {
    const h = el("h3", {}, c1, title);
    el("button", { type: "button", class: "btn ghost xs", onclick: async (ev) => { try { await navigator.clipboard.writeText(text); ev.target.textContent = t("expr.copied"); } catch { ev.target.textContent = t("expr.copyFail"); } } }, h, t("expr.copy"));
    el("div", { class: "exprbox" }, c1, text);
  }
  el("p", { class: "hint" }, c1, S.success ? t("expr.successNote") : t("expr.note"));
  const c2 = card(host, t("bdd.title"), "bdd");
  el("p", { class: "hint", style: "margin:2px 0 6px" }, c2.card, t("bdd.stats", { top: P.bdd.top_nodes, all: P.bdd.nodes, order: P.bdd.ordering.join(" < ") }));
  c2.card.appendChild(c2.plot);
  if (P.bdd.graph) {
    const names = {};
    for (const id of P.basic_events) names[id] = nameOf(id);
    Ch.bddChart(c2.plot, P.bdd.graph, { names, success: S.success });
    el("p", { class: "hint" }, c2.card, t("bdd.help"));
  } else c2.plot.appendChild(el("p", { class: "empty" }, null, t("bdd.tooBig", { limit: S.info?.limits?.bdd_draw_limit ?? 60 })));
}

// --------------------------------------------------------- examples ---
async function loadExamples() {
  try { S.examples = await api("/api/examples"); } catch { S.examples = []; }
  fillExamples();
}
function fillExamples() {
  const sel = $("exampleSel");
  sel.replaceChildren();
  el("option", { value: "" }, sel, t("file.examplesPick"));
  const list = [...S.examples].sort((a, b) => (a.file === "pressure_tank.json" ? -1 : b.file === "pressure_tank.json" ? 1 : a.file.localeCompare(b.file)));
  for (const x of list) {
    const title = (lang() === "pt" && x.title_pt) || x.title;
    const label = `${x.file.replace(/\.json$/, "")} — ${trunc(title, 34)} (${x.basic_events} ${t("file.events")}${x.success_mode ? ", " + t("file.successTag") : ""})`;
    el("option", { value: x.file }, sel, label);
  }
}
async function loadExample(file) {
  try {
    const res = await api("/api/examples/" + encodeURIComponent(file));
    const { doc: d, notes } = M.importTree(res.tree, res.probs || null);
    S.file = file;
    setMode(res.success_mode ? "success" : "failure", false);
    loadDoc(d, { note: res.prob_file ? t("file.withProbs", { file: res.prob_file }) + (notes.length ? " " + notes.join(" ") : "") : notes.join(" ") || null });
  } catch (e) { renderStatus(e.message); }
}

// ------------------------------------------------------------- shell ---
function setLtab(tab) {
  S.ui.ltab = tab;
  document.querySelectorAll("[data-ltab]").forEach((b) => { const on = b.dataset.ltab === tab; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); });
  for (const k of ["tree", "probs", "json"]) $("ltab-" + k).hidden = k !== tab;
  if (tab === "probs") renderEvents();
  if (tab === "json") fillJson();
  if (tab === "tree") renderEditor();
}
function setRtab(tab) {
  S.ui.rtab = tab;
  hideTip();
  document.querySelectorAll("[data-rtab]").forEach((b) => { const on = b.dataset.rtab === tab; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); });
  for (const k of ["tree", "cuts", "imp", "whatif", "unc", "expr"]) $("rtab-" + k).hidden = k !== tab;
  renderResults();
}
function setMode(mode, rerun = true) {
  S.success = mode === "success";
  document.querySelectorAll("[data-mode]").forEach((b) => { const on = b.dataset.mode === mode; b.classList.toggle("on", on); b.setAttribute("aria-checked", on); });
  if (!rerun) return;
  S.unc.res = null; S.whatif.res = null;
  S.point = null;
  S.version++;
  renderSide();
  if (S.ui.ltab === "probs") renderEvents();
  scheduleAnalysis();
  renderResults();
}
function renderAll() {
  applyI18n(document);
  document.querySelectorAll("[data-lang]").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang()));
  fillExamples();
  showNote(null);
  renderEditor(); renderSide();
  if (S.ui.ltab === "probs") renderEvents();
  S.issues = M.validate(doc());
  renderIssues();
  renderResults();
}
function zoomAction(spec) {
  const [which, op] = spec.split(":");
  const key = which === "edit" ? "zoomEdit" : "zoomRes";
  const box = which === "edit" ? $("editBox") : document.querySelector(".rtab:not([hidden]) .canvasBox");
  const cur = S.ui[key] || (box ? fitZoom(box, doc(), null, which === "edit" ? "edit" : "result") : 1);
  S.ui[key] = op === "fit" ? null : Math.max(0.3, Math.min(1.8, cur + (op === "+" ? 0.15 : -0.15)));
  if (which === "edit") renderEditor(); else renderResults();
}
function exportResults() {
  const P = S.point && pFresh() ? S.point : null;
  downloadJson(`${fileStem()}-results.json`, {
    generator: "faultree GUI", faultree_version: S.info?.version, exported: new Date().toISOString(),
    mode: S.success ? "success" : "failure",
    model: M.exportTree(doc()),
    settings: { max_order: S.maxOrder, samples: S.unc.n, seed: S.unc.seed },
    point_analysis: P,
    what_if: Object.keys(S.whatif.forced).length ? { forced: S.whatif.forced, result: S.whatif.ver === S.version ? S.whatif.res : null } : null,
    uncertainty: S.unc.res && S.unc.ver === S.version ? S.unc.res : null,
  });
}
function wire() {
  document.querySelectorAll("[data-ltab]").forEach((b) => b.addEventListener("click", () => setLtab(b.dataset.ltab)));
  document.querySelectorAll("[data-rtab]").forEach((b) => b.addEventListener("click", () => setRtab(b.dataset.rtab)));
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { if ((b.dataset.mode === "success") !== S.success) setMode(b.dataset.mode); }));
  document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => { setLang(b.dataset.lang); renderAll(); }));
  document.addEventListener("click", (e) => { const z = e.target.closest("[data-zoom]"); if (z) zoomAction(z.dataset.zoom); });
  $("editBox").addEventListener("click", (e) => { if (!e.target.closest(".nd, .tb, .zoom") && S.selUid != null) select(null); });
  $("btnUndo").addEventListener("click", undo);
  $("btnRedo").addEventListener("click", redo);
  $("btnNew").addEventListener("click", () => { S.file = null; loadDoc(M.emptyDoc()); $("exampleSel").value = ""; });
  $("btnOpen").addEventListener("click", () => $("fileOpen").click());
  $("fileOpen").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    try { await loadJsonData(JSON.parse(await file.text()), file.name); $("exampleSel").value = ""; }
    catch (err) { renderStatus(t("file.openFail", { msg: err.message })); }
  });
  $("btnSave").addEventListener("click", () => downloadJson(`${fileStem()}.json`, M.exportTree(doc())));
  $("btnSaveFlat").addEventListener("click", () => downloadJson(`${fileStem()}-flat.json`, M.exportFlat(doc())));
  $("btnExport").addEventListener("click", exportResults);
  $("exampleSel").addEventListener("change", (e) => { if (e.target.value) loadExample(e.target.value); });
  $("btnProbFile").addEventListener("click", () => $("probFile").click());
  $("probFile").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) importProbFile(f); });
  $("btnAllLogn").addEventListener("click", () => {
    const ef = parseNum($("allEf").value);
    if (!(ef >= 1)) { $("probMsg").className = "msg err"; $("probMsg").textContent = t("probs.badEf"); return; }
    pushHist();
    for (const id of M.eventIds(doc())) { const e = doc().events[id]; if (!e.samples && e.prob > 0) e.dist = { dist: "lognormal", median: e.prob, ef }; }
    $("probMsg").className = "msg ok"; $("probMsg").textContent = t("probs.allDone", { ef: fmtNum(ef, 3) });
    changed();
  });
  $("btnNoDist").addEventListener("click", () => { pushHist(); for (const e of Object.values(doc().events)) e.dist = null; changed(); });
  $("jsonApply").addEventListener("click", async () => {
    const msg = $("jsonMsg");
    try {
      const notes = await loadJsonData(JSON.parse($("jsonText").value), S.file);
      msg.className = "msg ok"; msg.textContent = t("json.applied") + (notes.length ? " " + notes.join(" ") : "");
    } catch (err) { msg.className = "msg err"; msg.textContent = t("json.fail", { msg: err.message }); }
  });
  $("jsonReset").addEventListener("click", fillJson);
  $("btnCollapse").addEventListener("click", () => { $("layout").classList.add("collapsed"); $("btnExpand").hidden = false; renderResults(); });
  $("btnExpand").addEventListener("click", () => { $("layout").classList.remove("collapsed"); $("btnExpand").hidden = true; renderEditor(); renderResults(); });
  const sp = $("splitter");
  sp.addEventListener("pointerdown", (e) => {
    sp.setPointerCapture(e.pointerId); sp.classList.add("drag");
    const move = (ev) => { const pct = Math.max(25, Math.min(70, (ev.clientX / window.innerWidth) * 100)); document.documentElement.style.setProperty("--left", pct + "%"); };
    const up = () => { sp.classList.remove("drag"); sp.removeEventListener("pointermove", move); sp.removeEventListener("pointerup", up); renderEditor(); renderResults(); };
    sp.addEventListener("pointermove", move); sp.addEventListener("pointerup", up);
  });
  sp.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--left")) || 42;
    document.documentElement.style.setProperty("--left", Math.max(25, Math.min(70, cur + (e.key === "ArrowLeft" ? -2 : 2))) + "%");
    renderEditor(); renderResults();
  });
  let rz = null;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { renderEditor(); renderResults(); }, 150); });
  document.addEventListener("keydown", (e) => {
    const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName);
    if (inField) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); redo(); }
    else if ((e.key === "Delete" || e.key === "Backspace") && S.ui.ltab === "tree" && S.selUid != null) {
      const f = M.findUid(doc(), S.selUid);
      if (f && f.p) { e.preventDefault(); nodeAction("delete", S.selUid); }
    }
  });
}

async function start() {
  initLang();
  applyI18n(document);
  document.querySelectorAll("[data-lang]").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang()));
  wire();
  S.doc = M.emptyDoc();
  try { S.info = await api("/api/info"); } catch { S.info = null; }
  await loadExamples();
  const params = new URLSearchParams(location.search);
  const first = params.get("example") || (S.examples.some((x) => x.file === "pressure_tank.json") ? "pressure_tank.json" : null);
  if (first) await loadExample(first);
  else loadDoc(M.emptyDoc());
  if (first) $("exampleSel").value = first;
  const tab = params.get("tab");
  if (tab && $("rtab-" + tab)) setRtab(tab);
}
// Read-only hook for automated checks.
window.faultreeGui = { state: S, exportTree: () => M.exportTree(doc()), runUnc, cycleForced, setRtab, setLtab };
start();
