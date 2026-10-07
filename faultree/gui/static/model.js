// Editor model of a fault tree, adapted from the lab's fault-tree page.
//
// doc.root   gate {t:"gate", uid, id, name, name_pt?, gate, k, children}
//            leaf {t:"leaf", uid, id}            (a basic event occurrence)
//            ref  {t:"ref", uid, ref}            (a clone of a gate: faultree's {"ref": id})
// doc.events id -> {name, name_pt?, prob, kind, dist, samples}
//            one definition per basic event; repeated leaves share it.
//            prob: point probability (number or null); dist: optional
//            uncertainty {dist, ...params}; samples: optional sample vector
//            from a file (prob is then its mean).
// doc.meta   {note?, note_pt?} carried through load/save.
import { t, lang } from "./i18n.js";

export const GATES = ["AND", "OR", "XOR", "K_OF_N"];
export const DISTS = {
  lognormal: ["median", "ef"],
  beta: ["a", "b"],
  uniform: ["low", "high"],
  loguniform: ["low", "high"],
};
let UID = 0;

export function gateNode(id, name, gate, children = [], k = 2) { return { t: "gate", uid: ++UID, id, name, gate, k, children }; }
export function leafNode(id) { return { t: "leaf", uid: ++UID, id }; }
export function refNode(ref) { return { t: "ref", uid: ++UID, ref }; }

export function walk(n, fn, parent = null, idx = 0, depth = 0) {
  fn(n, parent, idx, depth);
  if (n.t === "gate") n.children.forEach((c, i) => walk(c, fn, n, i, depth + 1));
}
export function maxUid(n) { let m = 0; walk(n, (x) => { m = Math.max(m, x.uid); }); return m; }
export function bumpUid(doc) { UID = Math.max(UID, maxUid(doc.root)); }
export function gates(doc) { const m = new Map(); walk(doc.root, (n) => { if (n.t === "gate" && !m.has(n.id)) m.set(n.id, n); }); return m; }
export function occurrences(doc) {
  const m = new Map();
  walk(doc.root, (n) => { if (n.t === "leaf") m.set(n.id, (m.get(n.id) || 0) + 1); });
  return m;
}
export function refCounts(doc) {
  const m = new Map();
  walk(doc.root, (n) => { if (n.t === "ref") m.set(n.ref, (m.get(n.ref) || 0) + 1); });
  return m;
}
export function eventIds(doc) { return [...occurrences(doc).keys()]; }
export function findUid(doc, uid) { let r = null; walk(doc.root, (n, p, i) => { if (n.uid === uid) r = { n, p, i }; }); return r; }
export function allIds(doc) { const s = new Set(Object.keys(doc.events)); walk(doc.root, (n) => { if (n.t === "gate") s.add(n.id); }); return s; }
export function freshId(doc, prefix) { const ids = allIds(doc); let i = 1; while (ids.has(prefix + i)) i++; return prefix + i; }
export function gcEvents(doc) { const occ = occurrences(doc); for (const id of Object.keys(doc.events)) if (!occ.has(id)) delete doc.events[id]; }
// The name shown in the current language (name_pt when it exists).
export function dispName(x) { return (lang() === "pt" && x && x.name_pt) || (x && x.name) || ""; }
export function setDispName(x, v) { if (lang() === "pt" && x.name_pt != null) x.name_pt = v; else x.name = v; }
export function newEvent(doc) {
  const id = freshId(doc, "E");
  doc.events[id] = { name: t("model.newEvent"), prob: 0.01, kind: "basic", dist: null, samples: null };
  return leafNode(id);
}
export function newGate(doc) {
  const g = gateNode(freshId(doc, "G"), t("model.newGate"), "OR", []);
  g.children = [newEvent(doc), newEvent(doc)];
  return g;
}
export function emptyDoc() {
  const doc = { root: null, events: {}, meta: {} };
  doc.root = gateNode("TOP", t("model.top"), "OR", []);
  doc.root.children = [newEvent(doc), newEvent(doc)];
  return doc;
}

// Gate ids reachable from gate `id` through children and refs.
function reachableGates(doc, id, gmap = gates(doc)) {
  const seen = new Set();
  const stack = [id];
  while (stack.length) {
    const g = gmap.get(stack.pop());
    if (!g || seen.has(g.id)) continue;
    seen.add(g.id);
    walk(g, (n) => { if (n.t === "gate" && n !== g) stack.push(n.id); if (n.t === "ref") stack.push(n.ref); });
  }
  return seen;
}
// Gates a clone placed under `parent` may point to (no cycles).
export function cloneTargets(doc, parent) {
  const gmap = gates(doc);
  const out = [];
  for (const id of gmap.keys()) if (id !== doc.root.id && !reachableGates(doc, id, gmap).has(parent.id)) out.push(id);
  return out;
}

export function validDist(d) {
  if (!d) return true;
  const v = (k) => d[k];
  const fin = (x) => typeof x === "number" && Number.isFinite(x);
  if (!DISTS[d.dist] || !DISTS[d.dist].every((k) => fin(v(k)))) return false;
  if (d.dist === "lognormal") return v("median") > 0 && v("median") <= 1 && v("ef") >= 1;
  if (d.dist === "beta") return v("a") > 0 && v("b") > 0;
  if (d.dist === "uniform") return v("low") >= 0 && v("low") <= v("high") && v("high") <= 1;
  return v("low") > 0 && v("low") <= v("high") && v("high") <= 1;
}
export function distMean(d) {
  const z = 1.6448536269514722;
  if (d.dist === "lognormal") { const s = Math.log(d.ef) / z; return d.median * Math.exp(s * s / 2); }
  if (d.dist === "beta") return d.a / (d.a + d.b);
  if (d.dist === "uniform") return (d.low + d.high) / 2;
  return d.low === d.high ? d.low : (d.high - d.low) / Math.log(d.high / d.low);
}

export function validate(doc) {
  const out = [];
  const gateIds = new Set();
  const gmap = gates(doc);
  walk(doc.root, (n, p) => {
    if (n.t === "ref") {
      if (!gmap.has(n.ref)) out.push(t("issue.refMissing", { id: n.ref }));
      else if (p && reachableGates(doc, n.ref, gmap).has(p.id)) out.push(t("issue.refCycle", { id: n.ref }));
      return;
    }
    if (n.t !== "gate") return;
    if (!n.id) out.push(t("issue.gateNoId"));
    else if (gateIds.has(n.id)) out.push(t("issue.gateDup", { id: n.id }));
    else gateIds.add(n.id);
    if (doc.events[n.id]) out.push(t("issue.gateIsEvent", { id: n.id }));
    if (!n.children.length) out.push(t("issue.gateEmpty", { id: n.id }));
    if (n.gate === "K_OF_N" && (!Number.isInteger(n.k) || n.k < 0 || n.k > n.children.length))
      out.push(t("issue.kRange", { id: n.id, n: n.children.length }));
  });
  for (const [id, e] of Object.entries(doc.events)) {
    if (!id) out.push(t("issue.eventNoId"));
    if (e.prob == null) out.push(t("issue.probMissing", { id }));
    else if (!(e.prob >= 0 && e.prob <= 1)) out.push(t("issue.probRange", { id }));
    if (e.dist && !validDist(e.dist)) out.push(t("issue.distBad", { id }));
  }
  return out;
}

// ---------------------------------------------------------- import ------
// `tree` is the recursive form returned by the server (/api/import), so
// flat models arrive already converted; refs and identical repeated
// definitions are kept as shared events and gate clones.
export function importTree(tree, probs = null) {
  const notes = [];
  const registry = new Map();
  (function reg(n) {
    if (!n || typeof n !== "object" || Array.isArray(n)) throw new Error(t("import.notObject"));
    if ("ref" in n) return;
    if (n.id != null && !registry.has(String(n.id))) registry.set(String(n.id), n);
    (n.children || []).forEach(reg);
  })(tree);
  const isLeaf = (n) => !(n.children || []).length &&
    (n.event_type === "basic" || n.event_type === "undeveloped" || n.gate == null || n.gate === "BASIC");
  const events = {};
  const seenGates = new Set();
  function eventFrom(n) {
    const id = String(n.id);
    const e = { name: n.name != null ? String(n.name) : id, prob: null, kind: n.event_type === "undeveloped" ? "undeveloped" : "basic", dist: null, samples: null };
    if (n.name_pt != null) e.name_pt = String(n.name_pt);
    setProb(e, n.prob);
    const u = n.uncertainty;
    if (u && typeof u === "object" && DISTS[u.dist]) {
      const d = { dist: u.dist };
      for (const k of DISTS[u.dist]) d[k] = Number(u[k]);
      if (validDist(d)) e.dist = d; else notes.push(t("import.badDist", { id }));
    }
    return e;
  }
  function conv(n, path) {
    if ("ref" in n) {
      const id = String(n.ref);
      const target = registry.get(id);
      if (!target) throw new Error(t("import.unknownRef", { id }));
      if (isLeaf(target)) { if (!events[id]) events[id] = eventFrom(target); return leafNode(id); }
      return refNode(id);
    }
    if (isLeaf(n)) {
      if (n.id == null || n.id === "") throw new Error(t("import.leafNoId"));
      const id = String(n.id);
      if (!events[id]) events[id] = eventFrom(n);
      return leafNode(id);
    }
    const gate = String(n.gate || "").toUpperCase();
    if (!GATES.includes(gate)) throw new Error(t("import.badGate", { gate: n.gate }));
    const id = n.id != null && n.id !== "" ? String(n.id) : null;
    if (!id) throw new Error(t("import.gateNoId"));
    if (path.includes(id)) throw new Error(t("import.cycle", { id }));
    if (seenGates.has(id)) return refNode(id); // an identical repeated definition
    seenGates.add(id);
    const g = gateNode(id, n.name != null ? String(n.name) : "", gate, [], Number.isInteger(n.k) ? n.k : 2);
    if (n.name_pt != null) g.name_pt = String(n.name_pt);
    g.children = (n.children || []).map((c) => conv(c, [...path, id]));
    return g;
  }
  let root = conv(tree, []);
  if (root.t !== "gate") { root = gateNode("TOP", t("model.top"), "OR", [root]); notes.push(t("import.wrapped")); }
  const doc = { root, events, meta: {} };
  for (const k of ["note", "note_pt"]) if (typeof tree[k] === "string") doc.meta[k] = tree[k];
  if (probs) notes.push(...applyProbs(doc, probs));
  return { doc, notes };
}
function setProb(e, p) {
  if (Array.isArray(p)) {
    const arr = p.map(Number);
    if (arr.length === 1) { e.prob = arr[0]; e.samples = null; }
    else { e.samples = arr; e.prob = arr.reduce((a, b) => a + b, 0) / (arr.length || 1); }
  } else if (p == null) { e.prob = null; e.samples = null; }
  else { e.prob = Number(p); e.samples = null; }
}
// Probability columns (from a file or a request's "probs") onto events.
export function applyProbs(doc, probs) {
  const notes = [];
  const unknown = [];
  let vectors = 0;
  for (const [id, p] of Object.entries(probs)) {
    const e = doc.events[id];
    if (!e) { unknown.push(id); continue; }
    setProb(e, p);
    if (e.samples) { vectors++; e.dist = null; }
  }
  if (unknown.length) notes.push(t("import.unknownColumns", { ids: unknown.join(", ") }));
  if (vectors) notes.push(t("import.vectors", { n: vectors }));
  return notes;
}

// ---------------------------------------------------------- export ------
// Recursive faultree JSON. The first occurrence of a basic event carries its
// definition and later ones are {"ref": id}; gate clones are {"ref": id}.
// engine: point probabilities only, without GUI-only fields.
export function exportTree(doc, { engine = false } = {}) {
  const emitted = new Set();
  function emit(n, isRoot) {
    if (n.t === "ref") return { ref: n.ref };
    if (n.t === "leaf") {
      if (emitted.has(n.id)) return { ref: n.id };
      emitted.add(n.id);
      const e = doc.events[n.id];
      const o = { id: n.id, name: e.name };
      if (!engine && e.name_pt != null) o.name_pt = e.name_pt;
      o.event_type = e.kind;
      o.gate = null;
      o.prob = !engine && e.samples ? e.samples : e.prob;
      if (!engine && e.dist) o.uncertainty = { ...e.dist };
      o.children = [];
      return o;
    }
    const o = { id: n.id, name: n.name };
    if (!engine && n.name_pt != null) o.name_pt = n.name_pt;
    if (isRoot && !engine) for (const k of ["note", "note_pt"]) if (doc.meta[k]) o[k] = doc.meta[k];
    o.event_type = isRoot ? "top" : "intermediate";
    o.gate = n.gate;
    if (n.gate === "K_OF_N") o.k = n.k;
    o.children = n.children.map((c) => emit(c, false));
    return o;
  }
  return emit(doc.root, true);
}
// Flat faultree JSON (ft_nodes / be_nodes); the unique unreferenced gate is the root.
export function exportFlat(doc) {
  const ft = [], seen = new Set();
  walk(doc.root, (n) => {
    if (n.t !== "gate" || seen.has(n.id)) return;
    seen.add(n.id);
    const o = { label: n.id, gate: n.gate };
    if (n.gate === "K_OF_N") o.k = n.k;
    o.branches = n.children.map((c) => (c.t === "ref" ? c.ref : c.id));
    ft.push(o);
  });
  const be = eventIds(doc).map((id) => ({ label: id, prob: doc.events[id].samples || doc.events[id].prob }));
  return { ft_nodes: ft, be_nodes: be };
}
