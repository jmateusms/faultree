// Small helpers shared by the GUI modules: DOM/SVG, number formats,
// tooltip, downloads (JSON, SVG, PNG) and the JSON API.
import { t, locale } from "./i18n.js";
import { SET } from "./settings.js";

export const $ = (id) => document.getElementById(id);
export const SVGNS = "http://www.w3.org/2000/svg";

// Palette. UI chrome and text use the ink/accent family; chart series use
// the fixed data order blue, amber, wine, teal (teal only with labels).
export const C = {
  ink: "#2B2B2B", accent: "#1A4E80", wine: "#6D2E46", green: "#2C5F2D",
  amber: "#E8B923", muted: "#8A8A86", rule: "#D8D4CC", paper: "#FBFAF7",
  accentLight: "#E4E9EB", card: "#FFFFFF",
  s1: "#2F6DB5", s2: "#C98A12", s3: "#B0456E", s4: "#2A9D8F",
};
// Markers of shared events: distinct dark hues away from the blue of the
// probability scale and the selection, all readable under white text.
export const SHARED = ["#B03A2E", "#2C6E2F", "#6B3FA0", "#8A5A00", "#0F6E6E", "#A13D74", "#5E6B1F", "#4A5568"];
export const sharedColor = (mark) => SHARED[mark.index % SHARED.length];
export const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
export const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';

export function el(tag, attrs = {}, parent = null, text = null) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
export function svgEl(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
export function svgText(parent, x, y, s, attrs = {}) {
  const e = svgEl("text", { x, y, fill: C.ink, ...attrs }, parent);
  e.textContent = s;
  return e;
}
export function newSvg(w, h, label) {
  const s = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img",
    "aria-label": label || "", "font-family": FONT, "font-size": 12 });
  s.style.fontVariantNumeric = "tabular-nums";
  return s;
}
export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
export function trunc(s, n) { s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

// ------------------------------------------------------------- numbers ---
export function fmtP(p, sig = 4) {
  if (p === null || p === undefined) return "—";
  if (p === Infinity) return "∞";
  if (!Number.isFinite(p)) return "—";
  if (p === 0) return "0";
  const a = Math.abs(p);
  if (a >= 1e-3 && a < 1e5 && (SET.notation !== "sci" || a >= 1)) return p.toLocaleString(locale(), { maximumSignificantDigits: sig });
  const [m, e] = p.toExponential(sig - 1).split("e");
  return Number(m).toLocaleString(locale(), { maximumFractionDigits: sig - 1 }) + "e" + e.replace("+", "");
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
// Scientific notation with a superscript exponent, for the large readouts.
export function fmtSci(p, sig = 4) {
  if (!isNum(p)) return fmtP(p, sig);
  if (p === 0) return "0";
  if (Math.abs(p) >= (SET.notation === "sci" ? 1 : 1e-2) && Math.abs(p) < 1e4) return fmtP(p, sig);
  const [m, e] = p.toExponential(sig - 1).split("e");
  const exp = String(Number(e)).split("").map((c) => SUP[c] || c).join("");
  return Number(m).toLocaleString(locale(), { minimumFractionDigits: sig - 1, maximumFractionDigits: sig - 1 }) + " × 10" + exp;
}
export function fmtNum(v, digits = 3) {
  if (v === Infinity) return "∞";
  if (!isNum(v)) return "—";
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6)) return fmtP(v, digits);
  return v.toLocaleString(locale(), { maximumSignificantDigits: digits });
}
export function fmtPct(x, digits = 1) {
  if (!isNum(x)) return "—";
  const v = x * 100;
  const d = Math.abs(v) < 0.1 && v !== 0 ? 3 : digits;
  return v.toLocaleString(locale(), { maximumFractionDigits: d, minimumFractionDigits: Math.min(d, 1) }) + " %";
}
export function fmtInt(n) { return isNum(n) ? n.toLocaleString(locale()) : "—"; }
export function fmtMs(ms) { return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toLocaleString(locale(), { maximumFractionDigits: 1 })} s`; }
// Accepts 0.01, 0,01, 1e-4, 2% (and returns NaN for anything else).
export function parseNum(s) {
  s = String(s ?? "").trim().replace(/\s/g, "").replace(",", ".");
  let pct = false;
  if (s.endsWith("%")) { pct = true; s = s.slice(0, -1); }
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(s)) return NaN;
  const v = Number(s);
  return pct ? v / 100 : v;
}
export function numText(v) {
  if (!isNum(v)) return "";
  const s = Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential() : String(+v.toPrecision(12));
  return locale() === "pt-BR" ? s.replace(".", ",") : s;
}

// ------------------------------------------------------------ scales -----
export function linScale(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.inv = (x) => d0 + ((x - r0) / (r1 - r0 || 1)) * (d1 - d0);
  return f;
}
export function logScale(d0, d1, r0, r1) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  const f = (v) => r0 + ((Math.log10(v) - l0) / (l1 - l0 || 1)) * (r1 - r0);
  f.inv = (x) => Math.pow(10, l0 + ((x - r0) / (r1 - r0 || 1)) * (l1 - l0));
  return f;
}
export function niceTicks(lo, hi, count = 5) {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
export function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    const v = Math.pow(10, e);
    if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
  }
  if (out.length < 2) { // narrow range: add 2 and 5 multiples
    const more = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
      for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) more.push(v); }
    return more.length ? more : [lo, hi];
  }
  return out;
}
export function quantile(sorted, p) {
  if (!sorted.length) return NaN;
  const h = (sorted.length - 1) * p, i = Math.floor(h);
  return i + 1 < sorted.length ? sorted[i] + (h - i) * (sorted[i + 1] - sorted[i]) : sorted[i];
}

// ----------------------------------------------------------- colours -----
function hex(c) { return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)); }
function toHex(rgb) { return "#" + rgb.map((v) => Math.round(v).toString(16).padStart(2, "0")).join(""); }
export function mix(a, b, f) { const x = hex(a), y = hex(b); return toHex(x.map((v, i) => v + (y[i] - v) * f)); }
export function isDark(c) { const [r, g, b] = hex(c); return 0.2126 * r + 0.7152 * g + 0.0722 * b < 140; }
// One-hue probability scale (light to dark accent), logarithmic in p.
export function probScale(values) {
  const pos = values.filter((v) => isNum(v) && v > 0);
  let lo = pos.length ? Math.min(...pos) : 1e-6, hi = pos.length ? Math.max(...pos) : 1;
  lo = Math.max(lo, 1e-15);
  if (hi / lo < 10) { lo = hi / 10; }
  const l0 = Math.log10(lo), l1 = Math.log10(hi);
  const f = (p) => {
    if (!isNum(p) || p <= 0) return C.card;
    const x = Math.min(1, Math.max(0, (Math.log10(Math.max(p, lo)) - l0) / (l1 - l0 || 1)));
    return mix(C.accentLight, C.accent, 0.08 + 0.92 * x);
  };
  f.lo = lo; f.hi = hi;
  return f;
}

// ----------------------------------------------------------- tooltip -----
let TIP = null;
export function showTip(evt, lines) {
  if (!TIP) { TIP = el("div", { class: "tip", role: "tooltip" }, document.body); }
  TIP.replaceChildren();
  lines.forEach((ln, i) => {
    if (ln == null) return;
    const row = el("div", { class: i === 0 ? "tip-h" : "tip-r" }, TIP);
    if (Array.isArray(ln)) { el("span", { class: "tip-k" }, row, ln[0]); el("b", {}, row, ln[1]); }
    else row.textContent = ln;
  });
  TIP.hidden = false;
  let cx = evt.clientX, cy = evt.clientY;
  if (cx === undefined || evt.type === "focus") { const r = evt.target.getBoundingClientRect(); cx = r.left + r.width / 2; cy = r.bottom; }
  const w = TIP.offsetWidth, h = TIP.offsetHeight, pad = 14;
  let x = cx + pad, y = cy + pad;
  if (x + w > window.innerWidth - 8) x = cx - w - pad;
  if (y + h > window.innerHeight - 8) y = cy - h - pad;
  TIP.style.left = Math.max(8, x) + "px";
  TIP.style.top = Math.max(8, y) + "px";
}
export function hideTip() { if (TIP) TIP.hidden = true; }
export function bindTip(node, lines) {
  node.addEventListener("pointermove", (e) => showTip(e, typeof lines === "function" ? lines() : lines));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", (e) => showTip(e, typeof lines === "function" ? lines() : lines));
  node.addEventListener("blur", hideTip);
}

// ---------------------------------------------------------- downloads ----
export function download(name, blob) {
  const a = el("a", { href: URL.createObjectURL(blob), download: name }, document.body);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
export function downloadJson(name, obj) {
  download(name, new Blob([JSON.stringify(obj, null, 2) + "\n"], { type: "application/json" }));
}
function standaloneSvg(svg) {
  const clone = svg.cloneNode(true);
  const vb = (svg.getAttribute("viewBox") || "").split(/\s+/).map(Number);
  const w = vb[2] || svg.clientWidth, h = vb[3] || svg.clientHeight;
  clone.setAttribute("xmlns", SVGNS);
  clone.setAttribute("width", w);
  clone.setAttribute("height", h);
  clone.setAttribute("font-family", FONT);
  clone.querySelectorAll("[data-noexport]").forEach((n) => n.remove());
  const bg = svgEl("rect", { x: vb[0] || 0, y: vb[1] || 0, width: w, height: h, fill: C.paper });
  clone.insertBefore(bg, clone.firstChild);
  const style = svgEl("style");
  style.textContent = `text{font-family:${FONT};font-variant-numeric:tabular-nums}.mono{font-family:${MONO}}`;
  clone.insertBefore(style, clone.firstChild);
  return { text: new XMLSerializer().serializeToString(clone), w, h };
}
export function downloadSvg(svg, name) {
  const { text } = standaloneSvg(svg);
  download(name + ".svg", new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + text], { type: "image/svg+xml" }));
}
export function downloadPng(svg, name, scale = 2) {
  const { text, w, h } = standaloneSvg(svg);
  const img = new Image();
  img.onload = () => {
    const canvas = el("canvas", { width: Math.round(w * scale), height: Math.round(h * scale) });
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, w, h);
    canvas.toBlob((blob) => download(name + ".png", blob), "image/png");
  };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text);
}
// Two small SVG/PNG buttons for a chart card; `get` returns the <svg>.
export function exportButtons(parent, get, name) {
  const box = el("span", { class: "export" }, parent);
  for (const kind of ["SVG", "PNG"]) {
    el("button", { type: "button", class: "btn ghost xs", title: t("export.chart", { kind }),
      onclick: () => { const s = get(); if (s) (kind === "SVG" ? downloadSvg : downloadPng)(s, name()); } }, box, kind);
  }
  return box;
}

// ---------------------------------------------------------------- api ----
export async function api(path, body, signal) {
  const opts = body === undefined ? { signal } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  };
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}
