// Result charts in hand-written SVG (no libraries), in the manner of
// drum-lab's charts.js: bar tables, scatter, histogram/CDF, interval plot,
// cut-set contributions and the BDD drawing. Colours are SVG attributes so
// an exported file looks the same as the page.
import { svgEl, svgText, newSvg, C, fmtP, fmtNum, fmtPct, isNum, linScale, logScale,
  niceTicks, logTicks, bindTip, showTip, hideTip, trunc } from "./util.js";
import { t } from "./i18n.js";

const GRID = "#ECE9E2";

function fmtAxis(v, log) { return log ? fmtP(v, 2) : fmtNum(v, 3); }
// Chart width from the space its card offers (charts redraw on resize).
function widthOf(host, min, max) {
  const w = host && host.clientWidth ? host.clientWidth - 2 : max;
  return Math.round(Math.max(min, Math.min(max, w)));
}

// ------------------------------------------------- importance bar table --
// rows: [{id, label, values: {key: number|Infinity|null}}]
// measures: [{key, label, log, pct}]; opts.sort: key or "id"; opts.onSort(key);
// opts.onHover(id|null).
export function importanceBars(host, rows, measures, opts = {}) {
  const W = widthOf(host, 700, 1100), labelW = 200, rowH = 24, headH = 40, gap = 16;
  const pw = (W - labelW - gap * (measures.length - 1)) / measures.length;
  const H = headH + rows.length * rowH + 8;
  const svg = newSvg(W, H, t("imp.chartLabel"));
  const sortKey = opts.sort || measures[0].key;
  const idChars = Math.min(16, Math.max(3, ...rows.map((r) => r.id.length)));
  const idW = Math.min(labelW - 10, idChars * 7.6 + 10);
  const val = (r, k) => r.values[k];
  const sorted = [...rows].sort((a, b) => {
    if (sortKey === "id") return a.id.localeCompare(b.id, undefined, { numeric: true });
    const x = val(a, sortKey), y = val(b, sortKey);
    const nx = x == null || Number.isNaN(x) ? -Infinity : x, ny = y == null || Number.isNaN(y) ? -Infinity : y;
    return ny - nx || a.id.localeCompare(b.id);
  });
  // header
  const hdr = (x, w, text, key, anchor = "start") => {
    const g = svgEl("g", { class: "sorthead", tabindex: 0, role: "button" }, svg);
    const on = key === sortKey;
    svgText(g, anchor === "start" ? x : x + w / 2, 16, text + (on ? " ▼" : ""), { "font-size": 12.5, "font-weight": 700, fill: on ? C.accent : C.ink, "text-anchor": anchor });
    svgEl("rect", { x, y: 2, width: w, height: 20, fill: "transparent" }, g);
    const fire = () => opts.onSort && opts.onSort(key);
    g.addEventListener("click", fire);
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fire(); } });
    svgEl("title", {}, g).textContent = t("imp.sortBy", { what: text });
  };
  hdr(4, labelW - 14, t("imp.event"), "id");
  const panels = measures.map((m, i) => {
    const x0 = labelW + i * (pw + gap), x1 = x0 + pw - 58;
    const vals = rows.map((r) => val(r, m.key)).filter((v) => v != null && !Number.isNaN(v));
    const finite = vals.filter((v) => Number.isFinite(v));
    const hasInf = vals.some((v) => v === Infinity);
    let sc, base, ticks;
    if (m.log) {
      const pos = finite.filter((v) => v > 0);
      let lo = Math.min(1, ...pos), hi = Math.max(1.01, ...pos);
      lo = lo / 1.15; hi = hi * 1.15;
      sc = logScale(lo, hi, x0, hasInf ? x1 - 16 : x1);
      base = 1;
      ticks = logTicks(lo, hi);
    } else {
      const lo = Math.min(0, ...finite), hi = Math.max(1e-12, ...finite);
      sc = linScale(lo, hi, x0, x1);
      base = 0;
      ticks = niceTicks(lo, hi, 3);
    }
    hdr(x0, pw, m.label, m.key);
    svgText(svg, x0, 33, m.log ? t("imp.logAxis") : m.pct ? t("imp.fraction") : "", { "font-size": 10.5, fill: C.muted });
    for (const tk of ticks) {
      const x = sc(tk);
      if (x < x0 - 0.5 || x > x1 + 0.5) continue;
      svgEl("line", { x1: x, x2: x, y1: headH - 2, y2: H - 6, stroke: GRID }, svg);
    }
    svgEl("line", { x1: sc(base), x2: sc(base), y1: headH - 2, y2: H - 6, stroke: C.muted }, svg);
    return { m, x0, x1, sc, base, hasInf };
  });
  sorted.forEach((r, i) => {
    const y = headH + i * rowH;
    const g = svgEl("g", { class: "barrow" }, svg);
    const bg = svgEl("rect", { x: 0, y, width: W, height: rowH, fill: i % 2 ? "transparent" : "#F4F2EC" }, g);
    svgText(g, 4, y + 16, trunc(r.id, idChars), { "font-size": 12, "font-weight": 700, class: "mono" });
    if (r.label && r.label !== r.id) svgText(g, 4 + idW, y + 16, trunc(r.label, Math.floor((labelW - idW - 12) / 6.3)), { "font-size": 11.5, fill: "#4A4A46" });
    for (const P of panels) {
      const v = val(r, P.m.key);
      if (v == null || Number.isNaN(v)) { svgText(g, P.x0 + pw - 4, y + 16, "—", { "font-size": 11.5, "text-anchor": "end" }); continue; }
      const xb = P.sc(P.base);
      const xv = v === Infinity ? P.x1 : Math.max(P.x0, Math.min(P.x1, P.m.log ? P.sc(Math.max(v, 1e-300)) : P.sc(v)));
      svgEl("rect", { x: Math.min(xb, xv), y: y + 5, width: Math.max(1, Math.abs(xv - xb)), height: rowH - 10, fill: C.s1, rx: 1.5 }, g);
      if (v === Infinity) svgText(g, P.x1 + 3, y + 16, "∞", { "font-size": 12, "font-weight": 700, fill: C.ink });
      svgText(g, P.x0 + pw - 4, y + 16, P.m.pct ? fmtPct(v) : fmtNum(v, 3), { "font-size": 11.5, "text-anchor": "end", class: "mono" });
    }
    g.addEventListener("pointerenter", () => { bg.setAttribute("fill", "#E4E9EB"); opts.onHover && opts.onHover(r.id); });
    g.addEventListener("pointerleave", () => { bg.setAttribute("fill", i % 2 ? "transparent" : "#F4F2EC"); opts.onHover && opts.onHover(null); });
    bindTip(g, () => [`${r.id} — ${r.label}`, ...measures.map((m) => [m.label, m.pct ? fmtPct(val(r, m.key)) : fmtNum(val(r, m.key), 5)])]);
  });
  host.replaceChildren(svg);
  return svg;
}

// ------------------------------------------------------------- scatter --
// rows: [{id, label, x, y}]; opts: {xLabel, yLabel, xLog, yLog, refs: [{axis, v, label}]}
export function scatter(host, rows, opts = {}) {
  const W = widthOf(host, 420, 720), H = Math.round(W * 0.66), m = { l: 66, r: 30, t: 20, b: 52 };
  const svg = newSvg(W, H, opts.title || "");
  const pts = rows.filter((r) => r.x != null && r.y != null && !Number.isNaN(r.x) && !Number.isNaN(r.y));
  const dom = (vals, log, refs) => {
    const fin = vals.filter((v) => Number.isFinite(v) && (!log || v > 0)).concat(refs);
    let lo = Math.min(...fin), hi = Math.max(...fin);
    if (!fin.length) { lo = log ? 1 : 0; hi = log ? 10 : 1; }
    if (log) { if (hi / lo < 4) { lo /= 2; hi *= 2; } return [lo / 1.3, hi * 1.3]; }
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
    return [Math.min(0, lo - pad), hi + pad];
  };
  const refs = opts.refs || [];
  const xInf = pts.some((p) => p.x === Infinity), yInf = pts.some((p) => p.y === Infinity);
  const [x0, x1] = dom(pts.map((p) => p.x), opts.xLog, refs.filter((r) => r.axis === "x").map((r) => r.v));
  const [y0, y1] = dom(pts.map((p) => p.y), opts.yLog, refs.filter((r) => r.axis === "y").map((r) => r.v));
  const xr = [m.l, W - m.r - (xInf ? 34 : 0)], yr = [H - m.b, m.t + (yInf ? 30 : 0)];
  const xs = (opts.xLog ? logScale : linScale)(x0, x1, ...xr);
  const ys = (opts.yLog ? logScale : linScale)(y0, y1, ...yr);
  const xt = opts.xLog ? logTicks(x0, x1) : niceTicks(x0, x1, 5);
  const yt = opts.yLog ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
  for (const v of xt) { const x = xs(v); if (x < xr[0] - 1 || x > xr[1] + 1) continue;
    svgEl("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: GRID }, svg);
    svgText(svg, x, H - m.b + 16, fmtAxis(v, opts.xLog), { "text-anchor": "middle", "font-size": 11 }); }
  for (const v of yt) { const y = ys(v); if (y > yr[0] + 1 || y < yr[1] - 1) continue;
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: GRID }, svg);
    svgText(svg, m.l - 6, y + 4, fmtAxis(v, opts.yLog), { "text-anchor": "end", "font-size": 11 }); }
  svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "none", stroke: C.rule }, svg);
  svgText(svg, (m.l + W - m.r) / 2, H - 12, opts.xLabel || "", { "text-anchor": "middle", "font-size": 12.5, "font-weight": 600 });
  const yl = svgText(svg, 16, (m.t + H - m.b) / 2, opts.yLabel || "", { "text-anchor": "middle", "font-size": 12.5, "font-weight": 600 });
  yl.setAttribute("transform", `rotate(-90 16 ${(m.t + H - m.b) / 2})`);
  if (xInf) svgText(svg, W - m.r - 15, H - m.b + 16, "∞", { "text-anchor": "middle", "font-size": 13, "font-weight": 700 });
  if (yInf) svgText(svg, m.l - 6, m.t + 16, "∞", { "text-anchor": "end", "font-size": 13, "font-weight": 700 });
  const placed = [];
  for (const r of refs) {
    const w = r.label.length * 6.4;
    if (r.axis === "x") {
      const x = xs(r.v);
      svgEl("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: C.s3, "stroke-dasharray": "5 4", "stroke-width": 1.3 }, svg);
      const ym = (m.t + H - m.b) / 2;
      const lab = svgText(svg, x - 5, ym, r.label, { "font-size": 11, fill: C.ink, "text-anchor": "middle" });
      lab.setAttribute("transform", `rotate(-90 ${x - 5} ${ym})`);
      placed.push({ x0: x - 16, x1: x - 2, y0: ym - w / 2, y1: ym + w / 2 });
    } else {
      const y = ys(r.v);
      svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: C.s3, "stroke-dasharray": "5 4", "stroke-width": 1.3 }, svg);
      svgText(svg, W - m.r - 4, y - 5, r.label, { "font-size": 11, fill: C.ink, "text-anchor": "end" });
      placed.push({ x0: W - m.r - 6 - w, x1: W - m.r, y0: y - 17, y1: y - 1 });
    }
  }
  const fits = (b) => b.x0 >= m.l && b.x1 <= W - m.r && b.y0 >= m.t && b.y1 <= H - m.b &&
    !placed.some((q) => b.x0 < q.x1 && b.x1 > q.x0 && b.y0 < q.y1 && b.y1 > q.y0);
  const sortedPts = [...pts].sort((a, b) => (b.y === Infinity ? 1e308 : b.y) - (a.y === Infinity ? 1e308 : a.y));
  const at = (p) => [p.x === Infinity ? W - m.r - 15 : xs(Math.max(p.x, x0)), p.y === Infinity ? m.t + 12 : ys(Math.max(p.y, y0))];
  // dots first, so that no label is placed over a dot drawn after it
  for (const p of sortedPts) {
    const [cx, cy] = at(p);
    const dot = svgEl("circle", { cx, cy, r: 5.5, fill: C.s1, stroke: "#FFFFFF", "stroke-width": 1.5, tabindex: 0 }, svg);
    placed.push({ x0: cx - 6, x1: cx + 6, y0: cy - 6, y1: cy + 6 });
    bindTip(dot, () => [`${p.id} — ${p.label}`, [opts.xLabel, fmtNum(p.x, 5)], [opts.yLabel, fmtNum(p.y, 5)]]);
    if (opts.onHover) {
      dot.addEventListener("pointerenter", () => opts.onHover(p.id));
      dot.addEventListener("pointerleave", () => opts.onHover(null));
    }
  }
  for (const p of sortedPts) {
    const [cx, cy] = at(p);
    const w = p.id.length * 7.2 + 4;
    const tries = [[8, 4, "start"], [8, -7, "start"], [8, 15, "start"], [-8, 4, "end"], [-8, -7, "end"], [-8, 15, "end"], [8, -18, "start"], [8, 26, "start"], [-8, -18, "end"], [-8, 26, "end"]];
    for (const [dx, dy, anchor] of tries) {
      const bx0 = anchor === "start" ? cx + dx : cx + dx - w;
      const box = { x0: bx0, x1: bx0 + w, y0: cy + dy - 10, y1: cy + dy + 2 };
      if (fits(box)) {
        placed.push(box);
        svgText(svg, cx + dx, cy + dy, p.id, { "font-size": 11.5, "font-weight": 600, class: "mono", "text-anchor": anchor });
        break;
      }
    }
  }
  host.replaceChildren(svg);
  return svg;
}

// ------------------------------------------------- histogram and CDF ----
// values: samples; opts: {kind: "hist"|"cdf", log, bins, marks: [{v, label, color, dash, width}], xLabel}
export function distChart(host, values, opts = {}) {
  const W = widthOf(host, 560, 1100), H = 330, m = { l: 62, r: 24, t: 46, b: 48 };
  const svg = newSvg(W, H, opts.title || "");
  const v = values.filter((x) => isNum(x)).sort((a, b) => a - b);
  if (!v.length) { host.replaceChildren(svg); return svg; }
  const log = !!opts.log && v[v.length - 1] > 0;
  const pos = v.filter((x) => x > 0);
  let lo = log ? pos[0] : v[0], hi = v[v.length - 1];
  if (hi <= lo) { const d = Math.abs(hi) * 0.1 || 1e-9; lo = log ? hi / 2 : hi - d; hi = log ? hi * 2 : hi + d; }
  const xs = (log ? logScale : linScale)(lo, hi, m.l, W - m.r);
  const X = (x) => xs(log ? Math.max(x, lo) : x);
  const xt = log ? logTicks(lo, hi) : niceTicks(lo, hi, 6);
  const plotTop = m.t, plotBot = H - m.b;
  let ys, ymax;
  if (opts.kind === "cdf") {
    ys = linScale(0, 1, plotBot, plotTop);
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: ys(p), y2: ys(p), stroke: GRID }, svg);
      svgText(svg, m.l - 6, ys(p) + 4, fmtNum(p, 2), { "text-anchor": "end", "font-size": 11 });
    }
    let d = `M${m.l},${ys(0)}`;
    v.forEach((x, i) => { d += `H${X(x)}V${ys((i + 1) / v.length)}`; });
    d += `H${W - m.r}`;
    svgEl("path", { d, fill: "none", stroke: C.s1, "stroke-width": 2 }, svg);
  } else {
    const bins = opts.bins || 40;
    const counts = new Array(bins).fill(0);
    const pos01 = (x) => (log ? (Math.log10(Math.max(x, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (x - lo) / (hi - lo));
    v.forEach((x) => { counts[Math.min(bins - 1, Math.max(0, Math.floor(pos01(x) * bins)))]++; });
    const frac = counts.map((c) => c / v.length);
    ymax = Math.max(...frac) * 1.1 || 1;
    ys = linScale(0, ymax, plotBot, plotTop);
    for (const p of niceTicks(0, ymax, 4)) {
      if (p > ymax) continue;
      svgEl("line", { x1: m.l, x2: W - m.r, y1: ys(p), y2: ys(p), stroke: GRID }, svg);
      svgText(svg, m.l - 6, ys(p) + 4, fmtPct(p, 0), { "text-anchor": "end", "font-size": 11 });
    }
    const bw = (W - m.r - m.l) / bins;
    frac.forEach((f, i) => {
      if (!f) return;
      svgEl("rect", { x: m.l + i * bw + 0.5, y: ys(f), width: Math.max(0.5, bw - 1), height: plotBot - ys(f), fill: C.s1, "fill-opacity": 0.8 }, svg);
    });
    opts._bins = { counts, frac, bins, pos01 };
  }
  for (const tk of xt) {
    const x = xs(tk);
    if (x < m.l - 1 || x > W - m.r + 1) continue;
    svgEl("line", { x1: x, x2: x, y1: plotBot, y2: plotBot + 5, stroke: C.ink }, svg);
    svgText(svg, x, plotBot + 18, fmtAxis(tk, log), { "text-anchor": "middle", "font-size": 11 });
  }
  svgEl("line", { x1: m.l, x2: W - m.r, y1: plotBot, y2: plotBot, stroke: C.ink }, svg);
  svgText(svg, (m.l + W - m.r) / 2, H - 10, opts.xLabel || "", { "text-anchor": "middle", "font-size": 12.5, "font-weight": 600 });
  const yl = svgText(svg, 16, (plotTop + plotBot) / 2, opts.kind === "cdf" ? t("unc.cdfAxis") : t("unc.freqAxis"), { "text-anchor": "middle", "font-size": 12, "font-weight": 600 });
  yl.setAttribute("transform", `rotate(-90 16 ${(plotTop + plotBot) / 2})`);
  // marks (percentiles, mean, point value), labels staggered on two rows
  const rowsUsed = [[], []];
  for (const mk of opts.marks || []) {
    if (!isNum(mk.v) || mk.v < lo || mk.v > hi) continue;
    const x = X(mk.v);
    svgEl("line", { x1: x, x2: x, y1: plotTop - 4, y2: plotBot, stroke: mk.color || C.ink, "stroke-width": mk.width || 1.6, "stroke-dasharray": mk.dash || null }, svg);
    const w = mk.label.length * 6.6;
    let row = rowsUsed.findIndex((r) => !r.some(([a, b]) => x - w / 2 < b && x + w / 2 > a));
    if (row < 0) row = 1;
    rowsUsed[row].push([x - w / 2, x + w / 2]);
    svgText(svg, Math.min(W - m.r - w / 2, Math.max(m.l + w / 2, x)), plotTop - 10 - row * 15, mk.label, { "text-anchor": "middle", "font-size": 11, fill: C.ink, "font-weight": 600 });
  }
  // hover crosshair
  const cross = svgEl("line", { x1: 0, x2: 0, y1: plotTop, y2: plotBot, stroke: C.ink, "stroke-width": 1, opacity: 0, "data-noexport": "1" }, svg);
  const hit = svgEl("rect", { x: m.l, y: plotTop, width: W - m.l - m.r, height: plotBot - plotTop, fill: "transparent", "data-noexport": "1" }, svg);
  hit.addEventListener("pointermove", (evt) => {
    const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    const xv = xs.inv(loc.x);
    cross.setAttribute("x1", loc.x); cross.setAttribute("x2", loc.x); cross.setAttribute("opacity", 0.5);
    let below = 0, a = 0, b = v.length;
    while (a < b) { const mid = (a + b) >> 1; if (v[mid] <= xv) a = mid + 1; else b = mid; }
    below = a / v.length;
    const lines = [opts.xLabel + " ≈ " + fmtP(xv, 3), [t("unc.below"), fmtPct(below)]];
    if (opts._bins) {
      const B = opts._bins; const i = Math.min(B.bins - 1, Math.max(0, Math.floor(B.pos01(xv) * B.bins)));
      lines.push([t("unc.inBin"), fmtPct(B.frac[i])]);
    }
    showTip(evt, lines);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  host.replaceChildren(svg);
  return svg;
}

// ------------------------------------------------------ interval plot ---
// rows: [{id, label, p05, p50, p95, point}]; opts: {log, xLabel, ref}
export function intervalPlot(host, rows, opts = {}) {
  const W = widthOf(host, 600, 1100), rowH = 26, m = { l: 200, r: 30, t: 30, b: 44 };
  const H = m.t + rows.length * rowH + m.b;
  const svg = newSvg(W, H, opts.title || "");
  const idChars = Math.min(16, Math.max(3, ...rows.map((r) => r.id.length)));
  const idW = Math.min(m.l - 10, idChars * 7.6 + 10);
  const vals = rows.flatMap((r) => [r.p05, r.p50, r.p95, r.point]).filter((v) => isNum(v) && (!opts.log || v > 0));
  const hasInf = rows.some((r) => r.p95 === Infinity || r.p95 === null && r.p50 != null);
  let lo = Math.min(...vals, opts.log ? 1 : 0), hi = Math.max(...vals, opts.log ? 1.01 : 1e-12);
  if (!vals.length) { lo = opts.log ? 1 : 0; hi = opts.log ? 10 : 1; }
  if (opts.log) { lo /= 1.2; hi *= 1.2; } else { const pad = (hi - lo) * 0.04; lo = Math.min(lo, 0); hi += pad; }
  const xEnd = W - m.r - (hasInf ? 22 : 0);
  const xs = (opts.log ? logScale : linScale)(lo, hi, m.l, xEnd);
  const X = (v) => (v === Infinity || v === null ? xEnd + 12 : xs(opts.log ? Math.max(v, lo) : v));
  const ticks = opts.log ? logTicks(lo, hi) : niceTicks(lo, hi, 6);
  for (const tk of ticks) {
    const x = xs(tk);
    if (x < m.l - 1 || x > xEnd + 1) continue;
    svgEl("line", { x1: x, x2: x, y1: m.t - 6, y2: H - m.b, stroke: GRID }, svg);
    svgText(svg, x, H - m.b + 16, fmtAxis(tk, opts.log), { "text-anchor": "middle", "font-size": 11 });
  }
  if (hasInf) svgText(svg, xEnd + 12, H - m.b + 16, "∞", { "text-anchor": "middle", "font-size": 13, "font-weight": 700 });
  if (isNum(opts.ref)) svgEl("line", { x1: xs(opts.ref), x2: xs(opts.ref), y1: m.t - 6, y2: H - m.b, stroke: C.muted }, svg);
  svgText(svg, (m.l + W - m.r) / 2, H - 10, opts.xLabel || "", { "text-anchor": "middle", "font-size": 12.5, "font-weight": 600 });
  // legend
  const lg = svgEl("g", {}, svg);
  svgEl("rect", { x: m.l, y: 6, width: 26, height: 8, fill: C.s1, "fill-opacity": 0.35 }, lg);
  svgText(lg, m.l + 32, 14, t("unc.legendBand"), { "font-size": 11 });
  svgEl("circle", { cx: m.l + 190, cy: 10, r: 4.5, fill: C.s1 }, lg);
  svgText(lg, m.l + 199, 14, t("unc.legendMedian"), { "font-size": 11 });
  svgEl("line", { x1: m.l + 290, x2: m.l + 290, y1: 3, y2: 17, stroke: C.ink, "stroke-width": 2 }, lg);
  svgText(lg, m.l + 296, 14, t("unc.legendPoint"), { "font-size": 11 });
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + rowH / 2;
    const g = svgEl("g", {}, svg);
    svgEl("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: i % 2 ? "transparent" : "#F4F2EC" }, g);
    svgText(g, 4, y + 4, trunc(r.id, idChars), { "font-size": 12, "font-weight": 700, class: "mono" });
    if (r.label && r.label !== r.id) svgText(g, 4 + idW, y + 4, trunc(r.label, Math.floor((m.l - idW - 12) / 6.3)), { "font-size": 11.5, fill: "#4A4A46" });
    if (r.p50 == null && r.p05 == null) return;
    const a = X(r.p05), b = X(r.p95);
    svgEl("rect", { x: Math.min(a, b), y: y - 5, width: Math.max(2, Math.abs(b - a)), height: 10, fill: C.s1, "fill-opacity": 0.35, rx: 2 }, g);
    if (r.p50 != null) svgEl("circle", { cx: X(r.p50), cy: y, r: 4.5, fill: C.s1 }, g);
    if (isNum(r.point) || r.point === Infinity) svgEl("line", { x1: X(r.point), x2: X(r.point), y1: y - 8, y2: y + 8, stroke: C.ink, "stroke-width": 2 }, g);
    bindTip(g, () => [`${r.id} — ${r.label}`, ["p5", fmtNum(r.p05, 4)], ["p50", fmtNum(r.p50, 4)], ["p95", fmtNum(r.p95, 4)], [t("unc.legendPoint"), fmtNum(r.point, 4)]]);
  });
  host.replaceChildren(svg);
  return svg;
}

// ------------------------------------------------ small input histogram --
export function sparkHist(preview, w = 120, h = 26) {
  const svg = newSvg(w, h, "");
  if (!preview || !preview.counts) return svg;
  const max = Math.max(...preview.counts) || 1, n = preview.counts.length, bw = w / n;
  preview.counts.forEach((c, i) => {
    if (!c) return;
    const bh = (c / max) * (h - 2);
    svgEl("rect", { x: i * bw, y: h - bh, width: Math.max(0.6, bw - 0.6), height: bh, fill: C.s1, "fill-opacity": 0.75 }, svg);
  });
  svgEl("line", { x1: 0, x2: w, y1: h - 0.5, y2: h - 0.5, stroke: C.muted }, svg);
  return svg;
}

// ------------------------------------------------ cut-set contributions --
// rows: [{label, share, p}] (already sorted, top N)
export function cutBars(host, rows, opts = {}) {
  const W = widthOf(host, 340, 1000), rowH = 24, m = { l: Math.min(230, Math.round(W * 0.3)), r: 62, t: 26, b: 34 };
  const H = m.t + rows.length * rowH + m.b;
  const svg = newSvg(W, H, opts.title || "");
  const hi = Math.max(0.01, ...rows.map((r) => r.share || 0));
  const xs = linScale(0, hi, m.l, W - m.r);
  for (const tk of niceTicks(0, hi, 5)) {
    if (tk > hi * 1.0001) continue;
    svgEl("line", { x1: xs(tk), x2: xs(tk), y1: m.t - 6, y2: H - m.b, stroke: GRID }, svg);
    svgText(svg, xs(tk), H - m.b + 16, fmtPct(tk, 0), { "text-anchor": "middle", "font-size": 11 });
  }
  svgText(svg, m.l, 14, opts.caption || "", { "font-size": 11.5, fill: C.muted });
  rows.forEach((r, i) => {
    const y = m.t + i * rowH;
    const g = svgEl("g", {}, svg);
    svgEl("rect", { x: 0, y, width: W, height: rowH, fill: i % 2 ? "transparent" : "#F4F2EC" }, g);
    svgText(g, m.l - 8, y + 16, trunc(r.label, Math.floor((m.l - 12) / 7.2)), { "font-size": 12, "text-anchor": "end", class: "mono", "font-weight": 600 });
    svgEl("rect", { x: m.l, y: y + 5, width: Math.max(1, xs(r.share || 0) - m.l), height: rowH - 10, fill: C.s1, rx: 1.5 }, g);
    svgText(g, W - 4, y + 16, fmtPct(r.share), { "font-size": 11.5, "text-anchor": "end", class: "mono" });
    bindTip(g, () => [r.label, ["P", fmtP(r.p, 5)], [t("cuts.share"), fmtPct(r.share, 2)]]);
    if (opts.onHover) {
      g.addEventListener("pointerenter", () => opts.onHover(r.events));
      g.addEventListener("pointerleave", () => opts.onHover(null));
    }
  });
  svgEl("line", { x1: m.l, x2: m.l, y1: m.t - 6, y2: H - m.b, stroke: C.ink }, svg);
  host.replaceChildren(svg);
  return svg;
}

// ------------------------------------------------------------- the BDD --
// graph: {root, nodes: [{id, var, level, high, low, q, p}], levels}
// Decision nodes sit on one row per variable; an edge into a terminal ends
// in its own small 0/1 box beside the node, which keeps chains readable.
export function bddChart(host, graph, opts = {}) {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const levels = [...new Set(graph.nodes.map((n) => n.level))].sort((a, b) => a - b);
  const rank = new Map(levels.map((l, i) => [l, i]));
  const rows = levels.map(() => []);
  const seen = new Set();
  (function dfs(id) {
    if (seen.has(id) || !byId.has(id)) return;
    seen.add(id);
    const n = byId.get(id);
    rows[rank.get(n.level)].push(n);
    dfs(n.low); dfs(n.high);
  })(graph.root);
  const colW = 128, rowH = 80, padT = 30;
  const widest = Math.max(1, ...rows.map((r) => r.length));
  const W = Math.max(560, widest * colW + 200), H = padT + levels.length * rowH + 20;
  const svg = newSvg(W, H, t("bdd.label"));
  const pos = new Map();
  rows.forEach((r, i) => {
    if (i > 0) {
      const parentX = (n) => {
        const xs = graph.nodes.filter((p) => p.high === n.id || p.low === n.id).map((p) => pos.get(p.id)?.x).filter(isNum);
        return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : W / 2;
      };
      r.sort((a, b) => parentX(a) - parentX(b));
    }
    const span = (r.length - 1) * colW;
    r.forEach((n, j) => pos.set(n.id, { x: W / 2 - span / 2 + j * colW, y: padT + 18 + i * rowH }));
  });
  const edges = svgEl("g", {}, svg);
  const terms = svgEl("g", {}, svg);
  for (const n of graph.nodes) {
    const a = pos.get(n.id);
    for (const [kid, hi] of [[n.high, true], [n.low, false]]) {
      const style = { fill: "none", stroke: hi ? C.s1 : C.muted, "stroke-width": hi ? 1.8 : 1.4, "stroke-dasharray": hi ? null : "5 4" };
      if (kid === "0" || kid === "1") {
        const tx = a.x + (hi ? 40 : -40), ty = a.y + 42;
        svgEl("line", { x1: a.x + (hi ? 11 : -11), y1: a.y + 12, x2: tx, y2: ty - 10, ...style }, edges);
        svgEl("rect", { x: tx - 11, y: ty - 10, width: 22, height: 20, rx: 3, fill: kid === "1" ? C.s3 : C.card, stroke: C.ink, "stroke-width": 1.2 }, terms);
        svgText(terms, tx, ty + 5, kid, { "text-anchor": "middle", "font-size": 12.5, "font-weight": 700, fill: kid === "1" ? "#FFFFFF" : C.ink });
        continue;
      }
      const b = pos.get(kid);
      if (!b) continue;
      const dx = hi ? 7 : -7;
      svgEl("path", { d: `M${a.x + dx},${a.y + 15} C${a.x + dx * 3},${(a.y + b.y) / 2} ${b.x + dx},${(a.y + b.y) / 2} ${b.x + dx / 2},${b.y - 16}`, ...style }, edges);
    }
  }
  const val = (id) => (id === "1" ? 1 : id === "0" ? 0 : byId.get(id).p);
  for (const n of graph.nodes) {
    const p = pos.get(n.id);
    const g = svgEl("g", { tabindex: 0 }, svg);
    svgEl("circle", { cx: p.x, cy: p.y, r: 16, fill: n.id === graph.root ? C.accentLight : C.card, stroke: C.ink, "stroke-width": n.id === graph.root ? 2.2 : 1.4 }, g);
    svgText(g, p.x, p.y + 4, trunc(n.var, 5), { "text-anchor": "middle", "font-size": 11.5, "font-weight": 700, class: "mono" });
    svgText(g, p.x - 22, p.y + 4, fmtP(n.p, 3), { "text-anchor": "end", "font-size": 10.5, class: "mono", fill: "#4A4A46" });
    bindTip(g, () => [`${n.var}${opts.names && opts.names[n.var] ? " — " + opts.names[n.var] : ""}`,
      ["q", fmtP(n.q, 5)],
      [t("bdd.recursion"), `${fmtP(n.q, 3)} × ${fmtP(val(n.high), 3)} + (1 − ${fmtP(n.q, 3)}) × ${fmtP(val(n.low), 3)}`],
      ["p", fmtP(n.p, 6)]]);
  }
  svgEl("line", { x1: 12, x2: 40, y1: 14, y2: 14, stroke: C.s1, "stroke-width": 1.8 }, svg);
  svgText(svg, 46, 18, t("bdd.high"), { "font-size": 11 });
  svgEl("line", { x1: 12, x2: 40, y1: 30, y2: 30, stroke: C.muted, "stroke-width": 1.4, "stroke-dasharray": "5 4" }, svg);
  svgText(svg, 46, 34, t("bdd.low"), { "font-size": 11 });
  host.replaceChildren(svg);
  return svg;
}
