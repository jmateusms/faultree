// Tree drawing (SVG), shared by the editor, the results tree and what-if.
// The layout and symbols come from the lab's fault-tree page: leaves on
// slots, parents centred over their children; rectangles for top and
// intermediate events, circles for basic events, diamonds for undeveloped
// events and triangles for gate clones (faultree refs). Shared events and
// cloned gates carry a marker (see settings.js) in a colour of their own.
import { svgEl, svgText, trunc, fmtP, C, FONT, bindTip, isDark, sharedColor } from "./util.js";
import { t } from "./i18n.js";
import { walk, sharedMarks, dispName } from "./model.js";
import { SET } from "./settings.js";

const SLOT = 132, LEVEL = 116, PADX = 14;

// Two lines of at most n characters, breaking at spaces when possible.
export function wrap2(text, n) {
  const s = String(text || "").trim();
  if (s.length <= n) return [s];
  let cut = s.lastIndexOf(" ", n);
  if (cut < n * 0.45) cut = n;
  const first = s.slice(0, cut).trim(), rest = s.slice(cut).trim();
  return [first, trunc(rest, n)];
}

export function layout(doc, mode = "edit") {
  const TOPPAD = mode === "edit" ? 40 : mode === "preview" ? 4 : 14;
  const slotW = mode === "preview" ? 70 : SLOT; // the settings preview shows symbols only
  let slot = 0, maxDepth = 0;
  const items = [];
  (function place(n, depth, parent) {
    const it = { n, depth, parent, y: TOPPAD + depth * LEVEL, kids: [] };
    items.push(it);
    maxDepth = Math.max(maxDepth, depth);
    if (n.t === "gate" && n.children.length) {
      it.kids = n.children.map((c) => place(c, depth + 1, it));
      it.x = (it.kids[0].x + it.kids[it.kids.length - 1].x) / 2;
    } else { it.x = PADX + slot * slotW + slotW / 2; slot++; }
    return it;
  })(doc.root, 0, null);
  return { items, width: PADX * 2 + Math.max(slot, 2) * slotW, height: TOPPAD + maxDepth * LEVEL + 104 };
}

function gateSymbol(g, x, ys, n, stroke, fill) {
  const yb = ys + 30, h = 16;
  let d;
  if (n.gate === "AND") d = `M${x - h},${yb} L${x - h},${ys + h} A${h},${h} 0 0 1 ${x + h},${ys + h} L${x + h},${yb} Z`;
  else d = `M${x - h},${yb} Q${x},${yb - 9} ${x + h},${yb} Q${x + h - 2},${ys + 12} ${x},${ys} Q${x - h + 2},${ys + 12} ${x - h},${yb} Z`;
  svgEl("path", { d, fill, stroke, "stroke-width": 1.5 }, g);
  let bottom = yb - (n.gate === "AND" ? 0 : 4);
  if (n.gate === "XOR") { svgEl("path", { d: `M${x - h},${yb + 5} Q${x},${yb - 4} ${x + h},${yb + 5}`, fill: "none", stroke, "stroke-width": 1.5 }, g); bottom = yb + 1; }
  const lab = { AND: "·", OR: "+", XOR: "=1", K_OF_N: `${n.k}/${n.children.length}` }[n.gate];
  svgText(g, x, yb - (n.gate === "AND" ? 7 : 9), lab, { "text-anchor": "middle", "font-size": n.gate === "AND" ? 16 : n.gate === "OR" ? 12 : 10, "font-weight": 700 });
  return bottom;
}
export function gateDesc(g) { return t("gate.desc." + g); }

// The marker badge of a shared item, centred at (x, y): its letter or its
// number of occurrences; the "color" style has the coloured outline only.
function markBadge(g, x, y, mk, color, style) {
  if (style === "color") return;
  const label = style === "count" ? "×" + mk.count : mk.letter;
  const w = Math.max(18, 8 + label.length * 7.5);
  svgEl("rect", { x: x - w / 2, y: y - 9, width: w, height: 18, rx: 9, fill: color, stroke: "#FFFFFF", "stroke-width": 1.2 }, g);
  svgText(g, x, y + 4.5, label, { "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "#FFFFFF" });
}
// A halo shown (by CSS) on every occurrence of the hovered or selected item.
function twinRing(tag, attrs, color, g) {
  svgEl(tag, { ...attrs, class: "twinRing", fill: "none", stroke: color, "stroke-width": 5, "stroke-opacity": 0.45,
    visibility: "hidden", "data-noexport": "1" }, g);
}

// opts: mode ("edit" | "result" | "whatif" | "preview"), zoom, selUid,
// probs (id -> p), base (id -> p, what-if baseline), color (p -> fill),
// forced (id -> state), highlight (Set of event ids), success (bool),
// sharedStyle (overrides the setting), onNode(node, evt), tip(node).
export function renderTree(svg, doc, opts = {}) {
  const mode = opts.mode || "edit";
  svg.replaceChildren();
  const L = layout(doc, mode);
  const zoom = opts.zoom || 1;
  svg.setAttribute("viewBox", `0 0 ${L.width} ${L.height}`);
  svg.setAttribute("width", Math.round(L.width * zoom));
  svg.setAttribute("height", Math.round(L.height * zoom));
  svg.setAttribute("font-family", FONT);
  const marks = sharedMarks(doc);
  const style = opts.sharedStyle || SET.shared;
  const keyOf = (n) => (n.t === "ref" ? n.ref : n.id);
  let selKey = null;
  if (mode === "edit" && opts.selUid != null) walk(doc.root, (n) => { if (n.uid === opts.selUid) selKey = keyOf(n); });
  const probs = opts.probs || null;
  const color = opts.color || null;
  const forced = opts.forced || {};
  const hl = opts.highlight || null;
  const edges = svgEl("g", {}, svg);
  const nodes = svgEl("g", {}, svg);
  const pLabel = opts.success ? "R" : "P";
  let selItem = null;
  for (const it of L.items) {
    const n = it.n;
    const g = svgEl("g", { class: "nd", "data-uid": n.uid, tabindex: 0, role: "button" }, nodes);
    const sel = mode === "edit" && n.uid === opts.selUid;
    if (sel) selItem = it;
    const key = keyOf(n);
    const mk = marks.get(key);
    const mc = mk ? sharedColor(mk) : null;
    const msw = style === "color" ? 3.4 : 2.4;
    if (mk) {
      g.dataset.twin = key;
      if (key === selKey && n.uid !== opts.selUid) g.classList.add("twinSel");
      if (SET.twins && mode !== "preview") {
        const twins = (on) => svg.querySelectorAll(`.nd[data-twin="${CSS.escape(key)}"]`).forEach((x) => x.classList.toggle("twin", on));
        g.addEventListener("pointerenter", () => twins(true));
        g.addEventListener("pointerleave", () => twins(false));
        g.addEventListener("focus", () => twins(true));
        g.addEventListener("blur", () => twins(false));
      }
    }
    const p = probs ? probs[key] : undefined;
    const fill = color && p != null ? color(p) : C.card;
    const txt = isDark(fill) ? "#FFFFFF" : C.ink;
    if (n.t === "gate") {
      const bw = 128, bh = 56, bx = it.x - bw / 2, by = it.y;
      svgEl("rect", { class: "focusRing", x: bx - 4, y: by - 4, width: bw + 8, height: bh + 42, rx: 6, fill: "none", stroke: "none" }, g);
      if (mk) twinRing("rect", { x: bx - 5, y: by - 5, width: bw + 10, height: bh + 10, rx: 7 }, mc, g);
      svgEl("rect", { x: bx, y: by, width: bw, height: bh, rx: 3, fill,
        stroke: sel ? C.accent : mc || C.ink, "stroke-width": sel ? 2.6 : mk ? msw : it.depth === 0 ? 2.2 : 1.2 }, g);
      const lines = wrap2(dispName(n) || n.id, 20);
      lines.forEach((ln, i) => svgText(g, it.x, by + (lines.length > 1 ? 15 + i * 14 : 21), ln, { "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: txt }));
      const pTxt = p != null ? `${pLabel} = ${fmtP(p, 3)}` : "";
      svgText(g, it.x, by + 47, trunc(n.id, pTxt ? 7 : 16) + (pTxt ? " · " + pTxt : ""), { "text-anchor": "middle", "font-size": 11, class: "mono", fill: txt });
      svgEl("line", { x1: it.x, y1: by + bh, x2: it.x, y2: by + bh + 6, stroke: C.ink, "stroke-width": 1.5 }, g);
      const bottom = gateSymbol(g, it.x, by + bh + 6, n, C.ink, C.card);
      for (const k of mode === "preview" ? [] : it.kids) {
        const bus = bottom + (k.y - bottom) * 0.45;
        svgEl("path", { d: `M${it.x},${bottom} V${bus} H${k.x} V${k.y}`, fill: "none", stroke: C.muted, "stroke-width": 1.2 }, edges);
      }
      if (!n.children.length) svgText(g, it.x, by + bh + 56, t("tree.noInputs"), { "text-anchor": "middle", "font-size": 11, fill: C.wine });
      if (mk) markBadge(g, bx + bw - 2, by, mk, mc, style);
    } else if (n.t === "ref") {
      const cx = it.x, top = it.y + 2, h = 46, w = 54;
      svgEl("rect", { class: "focusRing", x: cx - 50, y: it.y - 3, width: 100, height: 92, rx: 6, fill: "none", stroke: "none" }, g);
      if (mk) twinRing("path", { d: `M${cx},${top - 9} L${cx + w / 2 + 8},${top + h + 5} L${cx - w / 2 - 8},${top + h + 5} Z`, "stroke-linejoin": "round" }, mc, g);
      svgEl("path", { d: `M${cx},${top} L${cx + w / 2},${top + h} L${cx - w / 2},${top + h} Z`, fill, stroke: sel ? C.accent : mc || C.ink, "stroke-width": sel ? 2.6 : mk ? msw : 1.4, "stroke-linejoin": "round" }, g);
      svgText(g, cx, top + h - 9, trunc(n.ref, 7), { "text-anchor": "middle", "font-size": 11, "font-weight": 700, class: "mono", fill: txt });
      svgText(g, cx, top + h + 15, t("tree.clone"), { "text-anchor": "middle", "font-size": 11, fill: C.muted });
      if (p != null) svgText(g, cx, top + h + 29, `${pLabel} = ${fmtP(p, 3)}`, { "text-anchor": "middle", "font-size": 11, class: "mono", fill: "#4A4A46" });
      if (mk) markBadge(g, cx + w / 2 + 2, top + 10, mk, mc, style);
    } else {
      const e = doc.events[n.id] || { name: n.id, prob: NaN, kind: "basic" };
      const cx = it.x, cy = it.y + 26, r = 24;
      const state = forced[n.id];
      const hot = hl && hl.has(n.id);
      svgEl("rect", { class: "focusRing", x: cx - 56, y: it.y - 3, width: 112, height: 98, rx: 6, fill: "none", stroke: "none" }, g);
      if (hot) svgEl("circle", { cx, cy, r: r + 7, fill: "none", stroke: C.s2, "stroke-width": 4 }, g);
      if (mk) twinRing("circle", { cx, cy, r: r + 6 }, mc, g);
      let f = fill, stroke = sel ? C.accent : mc || C.ink, sw = sel ? 2.6 : mk ? msw : 1.4, tc = txt;
      if (state === "failed") { f = C.s3; stroke = C.wine; tc = "#FFFFFF"; sw = 2.4; }
      if (state === "working") { f = C.s1; stroke = C.accent; tc = "#FFFFFF"; sw = 2.4; }
      if (e.kind === "undeveloped") svgEl("path", { d: `M${cx},${cy - r - 3} L${cx + r + 5},${cy} L${cx},${cy + r + 3} L${cx - r - 5},${cy} Z`, fill: f, stroke, "stroke-width": sw }, g);
      else svgEl("circle", { cx, cy, r, fill: f, stroke, "stroke-width": sw }, g);
      svgText(g, cx, cy + 4, trunc(n.id, 7), { "text-anchor": "middle", "font-size": 12, "font-weight": 700, class: "mono", fill: tc });
      const nl = wrap2(dispName(e), 21);
      nl.forEach((ln, i) => svgText(g, cx, cy + r + 14 + i * 13, ln, { "text-anchor": "middle", "font-size": 11 }));
      let below;
      if (state) below = state === "failed" ? t("whatif.failed") : t("whatif.working");
      else if (mode === "edit") below = e.samples ? `${fmtP(e.prob, 3)} (n=${e.samples.length})`
        : e.model ? `${fmtP(e.prob, 3)} · ${t("model.short." + e.model.dist)}` : fmtP(e.prob, 3);
      else below = p != null ? fmtP(p, 3) : fmtP(e.prob, 3);
      svgText(g, cx, cy + r + 15 + nl.length * 13, below, { "text-anchor": "middle", "font-size": 11, class: state ? "" : "mono",
        "font-weight": state ? 700 : 400, fill: state === "failed" ? C.wine : state === "working" ? C.accent : "#4A4A46" });
      if (e.dist && mode === "edit") {
        const bx = cx - r + 2, by = cy - r + 3;
        svgEl("circle", { cx: bx, cy: by, r: 8.5, fill: C.accent }, g);
        svgEl("path", { d: `M${bx - 5.5},${by + 3} C${bx - 3},${by + 3} ${bx - 2.2},${by - 4.5} ${bx},${by - 4.5} C${bx + 2.2},${by - 4.5} ${bx + 3},${by + 3} ${bx + 5.5},${by + 3}`,
          fill: "none", stroke: "#FFFFFF", "stroke-width": 1.6, "stroke-linecap": "round" }, g);
      }
      if (mk) markBadge(g, cx + r + 3, cy - r + 3, mk, mc, style);
    }
    if (opts.tip) bindTip(g, () => opts.tip(n));
    if (opts.onNode) {
      g.addEventListener("click", (evt) => { evt.stopPropagation(); opts.onNode(n, evt); });
      g.addEventListener("keydown", (evt) => { if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); opts.onNode(n, evt); } });
    }
  }
  if (selItem && opts.toolbar) drawToolbar(nodes, selItem, opts.toolbar);
  return L;
}

function drawToolbar(parent, it, onAct) {
  const n = it.n;
  const acts = [];
  if (n.t === "gate") {
    acts.push(["addEvent", "+○", t("tb.addEvent")]);
    acts.push(["addGate", "+□", t("tb.addGate")]);
    acts.push(["cycle", "⇄", t("tb.cycleGate")]);
  } else if (n.t === "leaf") acts.push(["cycle", "⇄", t("tb.cycleKind")]);
  if (it.parent) acts.push(["delete", "✕", t("tb.delete")]);
  const bw = 30, gap = 4, total = acts.length * bw + (acts.length - 1) * gap;
  let x = it.x - total / 2;
  const y = it.y - 31;
  for (const [act, glyph, tip] of acts) {
    const b = svgEl("g", { class: "tb" + (act === "delete" ? " del" : ""), tabindex: 0, role: "button", "aria-label": tip, "data-noexport": "1" }, parent);
    svgEl("title", {}, b).textContent = tip;
    svgEl("rect", { x, y, width: bw, height: 24, rx: 5, "stroke-width": 1.2 }, b);
    svgText(b, x + bw / 2, y + 17, glyph, { "text-anchor": "middle" });
    const fire = (evt) => { evt.stopPropagation(); onAct(act, n.uid); };
    b.addEventListener("click", fire);
    b.addEventListener("keydown", (evt) => { if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); fire(evt); } });
    x += bw + gap;
  }
}

// Colour legend for the probability scale (a gradient with decade ticks).
export function legend(svgHost, scale, label) {
  const w = 260, h = 40;
  const s = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, "font-family": FONT, role: "img", "aria-label": label }, svgHost);
  const id = "lg" + Math.random().toString(36).slice(2, 8);
  const defs = svgEl("defs", {}, s);
  const grad = svgEl("linearGradient", { id, x1: 0, x2: 1, y1: 0, y2: 0 }, defs);
  for (let i = 0; i <= 10; i++) {
    const p = Math.pow(10, Math.log10(scale.lo) + (i / 10) * (Math.log10(scale.hi) - Math.log10(scale.lo)));
    svgEl("stop", { offset: i / 10, "stop-color": scale(p) }, grad);
  }
  const x0 = 10, x1 = w - 14;
  svgText(s, x0, 11, label, { "font-size": 11, fill: C.muted });
  svgEl("rect", { x: x0, y: 15, width: x1 - x0, height: 10, fill: `url(#${id})`, stroke: C.rule }, s);
  const l0 = Math.log10(scale.lo), l1 = Math.log10(scale.hi);
  const ticks = [];
  for (let e = Math.ceil(l0); e <= Math.floor(l1); e++) ticks.push(e);
  const step = Math.max(1, Math.ceil(ticks.length / 6));
  ticks.filter((_, i) => i % step === 0).forEach((e) => {
    const x = x0 + ((e - l0) / (l1 - l0 || 1)) * (x1 - x0);
    svgEl("line", { x1: x, x2: x, y1: 25, y2: 29, stroke: C.ink }, s);
    svgText(s, x, 39, fmtP(Math.pow(10, e), 1), { "text-anchor": "middle", "font-size": 10 });
  });
  return s;
}
export { walk };
