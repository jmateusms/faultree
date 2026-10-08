// Editor model of a fault tree, adapted from the lab's fault-tree page.
//
// doc.root   gate {t:"gate", uid, id, name, name_pt?, gate, k, children}
//            leaf {t:"leaf", uid, id}            (a basic event occurrence)
//            ref  {t:"ref", uid, ref}            (a clone of a gate: faultree's {"ref": id})
// doc.events id -> {name, name_pt?, prob, kind, dist, samples, model?}
//            one definition per basic event; repeated leaves share it.
//            prob: point probability (number or null); dist: optional
//            uncertainty {dist, ...params}; samples: optional sample vector
//            from a file (prob is then its mean); model: optional failure
//            model {dist, ...params} from which prob is computed at the
//            mission time (see refreshModels).
// doc.meta   {note?, note_pt?, mission_time?, time_unit?} carried through load/save.
import { t, lang } from "./i18n.js";

export const GATES = ["AND", "OR", "XOR", "K_OF_N"];
export const DISTS = {
  lognormal: ["median", "ef"],
  beta: ["a", "b"],
  uniform: ["low", "high"],
  loguniform: ["low", "high"],
};
// Failure models: p = F(t), the probability that the event has occurred by
// the mission time t, from a time-to-failure distribution, or from a count
// of failures (binomial over n demands, Poisson over [0, t]) reaching k.
// Rates are per time unit; eta, mu, sigma, tmed, theta, a, b are times.
export const MODELS = {
  exponential: ["lambda"],
  weibull: ["beta", "eta"],
  normal: ["mu", "sigma"],
  lognormal: ["tmed", "s"],
  gamma: ["alpha", "theta"],
  uniform: ["a", "b"],
  binomial: ["q", "n", "k"],
  poisson: ["lambda", "k"],
};
export const LIFETIME = ["exponential", "weibull", "normal", "lognormal", "gamma", "uniform"];
export const COUNTS = ["binomial", "poisson"];
export const TIME_PARAMS = new Set(["eta", "mu", "sigma", "tmed", "theta", "a", "b"]);
export const DEFAULT_T = 1000;
export const TIME_UNITS = ["h", "d", "y", "min", "cycles"];
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
// Shared items: basic events placed more than once and gates that have
// clones. Each gets a letter (A, B, …, Z, AA, …) and a colour index in
// order of first appearance, so every occurrence carries the same marker.
export function letterOf(i) {
  let s = "";
  for (i += 1; i > 0; i = Math.floor((i - 1) / 26)) s = String.fromCharCode(65 + ((i - 1) % 26)) + s;
  return s;
}
export function sharedMarks(doc) {
  const occ = occurrences(doc), refs = refCounts(doc);
  const out = new Map();
  walk(doc.root, (n) => {
    const key = n.t === "ref" ? n.ref : n.id;
    if (out.has(key)) return;
    const count = n.t === "leaf" ? occ.get(key) : (refs.get(key) || 0) + 1;
    if (count > 1) out.set(key, { letter: letterOf(out.size), index: out.size, count, gate: n.t !== "leaf" });
  });
  return out;
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

// ------------------------------------------------------ failure models ---
// ln Γ(x), Lanczos (g = 7, 9 terms), relative error about 1e-15.
const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lnGamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lnGamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const tt = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(tt) - tt + Math.log(a);
}
const EPS = 1e-15;
// Regularized incomplete gamma: P(a, x) by its series (x < a + 1) and
// Q(a, x) = 1 − P(a, x) by its continued fraction (modified Lentz), so the
// smaller of the two is always computed directly, without cancellation.
function gser(a, x) {
  let ap = a, sum = 1 / a, del = sum;
  for (let i = 0; i < 10000 && Math.abs(del) > Math.abs(sum) * EPS; i++) { ap += 1; del *= x / ap; sum += del; }
  return sum * Math.exp(-x + a * Math.log(x) - lnGamma(a));
}
function gcf(a, x) {
  const tiny = 1e-300;
  let b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
  for (let i = 1; i < 10000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return Math.exp(-x + a * Math.log(x) - lnGamma(a)) * h;
}
export function gammaP(a, x) { return x <= 0 ? 0 : x === Infinity ? 1 : x < a + 1 ? gser(a, x) : 1 - gcf(a, x); }
export function gammaQ(a, x) { return x <= 0 ? 1 : x === Infinity ? 0 : x < a + 1 ? 1 - gser(a, x) : gcf(a, x); }
// Standard normal CDF; Φ(−|z|) = Q(1/2, z²/2) / 2 keeps the tails accurate.
export function normCdf(z) {
  if (Number.isNaN(z)) return NaN;
  const tail = 0.5 * gammaQ(0.5, (z * z) / 2);
  return z < 0 ? tail : 1 - tail;
}
export function normInv(p) {
  if (!(p > 0 && p < 1)) return p === 0 ? -Infinity : p === 1 ? Infinity : NaN;
  let lo = -40, hi = 40;
  for (let i = 0; i < 200 && hi - lo > 1e-13; i++) { const m = (lo + hi) / 2; if (normCdf(m) < p) lo = m; else hi = m; }
  return (lo + hi) / 2;
}
// ln C(n, k), summing logs when the smaller side is short (exact for large n).
function lnChoose(n, k) {
  const m = Math.min(k, n - k);
  if (m <= 2000) { let s = 0; for (let j = 1; j <= m; j++) s += Math.log((n - m + j) / j); return s; }
  return lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);
}
// [P(X ≥ k), P(X < k)] for X ~ Binomial(n, q). The tail away from the mode
// is summed directly from its first term (recurrence on the term ratio),
// and the other side is its complement.
function binomTail(q, n, k) {
  if (q === 0) return [0, 1];
  if (q === 1) return [1, 0];
  if (k === 1) { const l = n * Math.log1p(-q); return [-Math.expm1(l), Math.exp(l)]; }
  const r = q / (1 - q), mode = Math.floor((n + 1) * q);
  const term = (i) => Math.exp(lnChoose(n, i) + i * Math.log(q) + (n - i) * Math.log1p(-q));
  if (k > mode) {
    let s = 0;
    for (let i = k, x = term(k); i <= n; i++) { s += x; x *= ((n - i) / (i + 1)) * r; if (x <= s * 1e-17) break; }
    return [Math.min(1, s), Math.max(0, 1 - s)];
  }
  let s = 0;
  for (let i = k - 1, x = term(k - 1); i >= 0; i--) { s += x; x *= (i / (n - i + 1)) / r; if (x <= s * 1e-17) break; }
  return [Math.max(0, 1 - s), Math.min(1, s)];
}

const fin = (x) => typeof x === "number" && Number.isFinite(x);
const isInt = (x) => Number.isInteger(x);
export function validModel(m) {
  if (!m || !MODELS[m.dist] || !MODELS[m.dist].every((k) => fin(m[k]))) return false;
  switch (m.dist) {
    case "exponential": return m.lambda >= 0;
    case "weibull": return m.beta > 0 && m.eta > 0;
    case "normal": return m.sigma > 0;
    case "lognormal": return m.tmed > 0 && m.s > 0;
    case "gamma": return m.alpha > 0 && m.theta > 0;
    case "uniform": return m.a >= 0 && m.a < m.b;
    case "binomial": return m.q >= 0 && m.q <= 1 && isInt(m.n) && m.n >= 1 && m.n <= 1e9 && isInt(m.k) && m.k >= 1 && m.k <= m.n;
    case "poisson": return m.lambda >= 0 && isInt(m.k) && m.k >= 1;
  }
  return false;
}
export function validTime(x) { return fin(x) && x >= 0; }
// [F(t), 1 − F(t)] of a failure model, each computed directly.
export function modelPair(m, time) {
  if (!validModel(m) || !validTime(time)) return [NaN, NaN];
  switch (m.dist) {
    case "exponential": { const x = m.lambda * time; return [-Math.expm1(-x), Math.exp(-x)]; }
    case "weibull": { const x = Math.pow(time / m.eta, m.beta); return [-Math.expm1(-x), Math.exp(-x)]; }
    case "normal": { const z = (time - m.mu) / m.sigma; return [normCdf(z), normCdf(-z)]; }
    case "lognormal": {
      if (time === 0) return [0, 1];
      const z = Math.log(time / m.tmed) / m.s;
      return [normCdf(z), normCdf(-z)];
    }
    case "gamma": { const x = time / m.theta; return [gammaP(m.alpha, x), gammaQ(m.alpha, x)]; }
    case "uniform": { const f = Math.min(1, Math.max(0, (time - m.a) / (m.b - m.a))); return [f, 1 - f]; }
    case "binomial": return binomTail(m.q, m.n, m.k);
    case "poisson": { const x = m.lambda * time; return m.k === 1 ? [-Math.expm1(-x), Math.exp(-x)] : [gammaP(m.k, x), gammaQ(m.k, x)]; }
  }
  return [NaN, NaN];
}
// The event's input value: the failure probability F(t), or in success mode
// the reliability 1 − F(t).
export function modelProb(m, time, success = false) { return modelPair(m, time)[success ? 1 : 0]; }
export function missionTime(doc) { const v = doc.meta && doc.meta.mission_time; return v == null ? DEFAULT_T : v; }
export function timeUnit(doc) { return (doc.meta && doc.meta.time_unit) || "h"; }
export function hasModels(doc) { return Object.values(doc.events).some((e) => e.model); }
// Recompute prob for every event with a failure model.
export function refreshModels(doc, success = false) {
  const time = missionTime(doc);
  for (const e of Object.values(doc.events)) {
    if (!e.model) continue;
    const p = modelProb(e.model, time, success);
    e.prob = Number.isFinite(p) ? p : null;
  }
}
// Parameters of a new model chosen so that F(t) is close to p (a failure
// probability), rounded to 4 significant digits.
export function defaultModel(dist, p, time) {
  if (!(p > 0 && p < 1)) p = 0.01;
  if (!(time > 0)) time = DEFAULT_T;
  const r4 = (x) => +x.toPrecision(4);
  const h = -Math.log1p(-p); // cumulative hazard −ln(1 − p)
  const z = normInv(p);
  switch (dist) {
    case "exponential": return { dist, lambda: r4(h / time) };
    case "weibull": return { dist, beta: 2, eta: r4(time / Math.sqrt(h)) };
    case "normal": return p < 0.5 ? { dist, mu: r4(2 * time), sigma: r4(-time / z) } : { dist, mu: r4(time / 2), sigma: r4(time / 2 / Math.max(z, 1e-3)) };
    case "lognormal": return { dist, tmed: r4(time * Math.exp(-z)), s: 1 };
    case "gamma": {
      let lo = 0, hi = 1;
      while (gammaP(2, hi) < p) hi *= 2;
      for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (gammaP(2, m) < p) lo = m; else hi = m; }
      return { dist, alpha: 2, theta: r4(time / ((lo + hi) / 2)) };
    }
    case "uniform": return { dist, a: 0, b: r4(time / p) };
    case "binomial": return { dist, q: r4(-Math.expm1(Math.log1p(-p) / 10)), n: 10, k: 1 };
    case "poisson": return { dist, lambda: r4(h / time), k: 1 };
  }
  return null;
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
  if (hasModels(doc) && !validTime(missionTime(doc))) out.push(t("issue.timeBad"));
  for (const [id, e] of Object.entries(doc.events)) {
    if (!id) out.push(t("issue.eventNoId"));
    if (e.model) { if (!validModel(e.model)) out.push(t("issue.modelBad", { id })); }
    else if (e.prob == null) out.push(t("issue.probMissing", { id }));
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
    const fm = n.failure_model;
    if (fm && typeof fm === "object" && MODELS[fm.dist]) {
      const m = { dist: fm.dist };
      for (const k of MODELS[fm.dist]) m[k] = Number(fm[k]);
      if (validModel(m)) { e.model = m; e.samples = null; } else notes.push(t("import.badModel", { id }));
    }
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
  for (const k of ["note", "note_pt", "time_unit"]) if (typeof tree[k] === "string") doc.meta[k] = tree[k];
  if (validTime(tree.mission_time)) doc.meta.mission_time = tree.mission_time;
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
    e.model = null;
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
      if (!engine && e.model) o.failure_model = { ...e.model };
      if (!engine && e.dist) o.uncertainty = { ...e.dist };
      o.children = [];
      return o;
    }
    const o = { id: n.id, name: n.name };
    if (!engine && n.name_pt != null) o.name_pt = n.name_pt;
    if (isRoot && !engine) {
      for (const k of ["note", "note_pt"]) if (doc.meta[k]) o[k] = doc.meta[k];
      if (hasModels(doc) || doc.meta.mission_time != null) { o.mission_time = missionTime(doc); o.time_unit = timeUnit(doc); }
    }
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
