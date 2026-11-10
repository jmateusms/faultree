// Block diagram drawing (SVG): the Blocks view of the editor, the results
// and what-if, drawn from the same document as treeview.js (M.rbdOf maps
// it). Series groups run from left to right, parallel groups stack between
// two rails, and a k-out-of-n group ends in a voter k/n. Every group is a
// frame named after its gate, with the gate's value in its header in the
// results. A house event is a switch (open: the event has occurred, the
// path is cut), a conditioning event a dashed rounded block in parallel with
// the INHIBIT gate's input, a clone a repeat of a group drawn as one block
// with side bars, and XOR, PAND and K_OF_N with k = 0 module blocks with a
// double border (their logic is seen in the tree view). Shared events and
// cloned gates carry the tree's markers.
import { svgEl, svgText, trunc, fmtP, fmtR, fmtRq, C, FONT, bindTip, isDark, sharedColor } from "./util.js";
import { t } from "./i18n.js";
import { walk, sharedMarks, dispName, rbdOf } from "./model.js";
import { SET } from "./settings.js";
import { markBadge, twinRing } from "./treeview.js";

const BW = 116, BH = 44, CELL = BH + 20; // block, and block with its value line
const SW = 84, SPY = 26;                 // switch width and port height
const GAP = 22, VGAP = 12, RAIL = 16;    // series link, parallel spacing, rail margin
const FPAD = 9, FHEAD = 20, FTOP = 6;    // group frame padding and header
const VR = 13, MARGIN = 30;              // voter radius, input/output margin

// Width of a group header: id, name and (results) the value.
function headNeed(n, values) {
  const name = dispName(n);
  return 16 + n.id.length * 7 + (name ? 8 + Math.min(name.length, 24) * 6 : 0) + (values ? 14 + 16 * 6.6 : 0); // "R = 1 − 5.6e-180"
}
// {w, h, py, place(x, y, out)} of an item: its size, the height of its
// input and output, and where its parts go once its corner is known.
function box(item, values) {
  if (item.type !== "series" && item.type !== "parallel" && item.type !== "kofn") {
    const sw = item.type === "switch";
    return { w: sw ? SW : BW, h: CELL, py: sw ? SPY : BH / 2, place: (x, y, out) => out.cells.push({ item, x, y }) };
  }
  const kids = item.items.map((c) => box(c, values));
  const uid = item.n.uid;
  const inner = !kids.length ? { w: 110, h: 28, py: 14, place: (x, y, out) => out.empty.push({ x: x + 55, y: y + 18, uid }) }
    : item.type === "series" ? seriesBox(kids) : parallelBox(kids, item.type === "kofn" ? `${item.k ?? "?"}/${kids.length}` : null, uid);
  const w = Math.max(inner.w + 2 * FPAD, headNeed(item.n, values));
  const top = FHEAD + FTOP, py = top + inner.py, ox = (w - inner.w) / 2;
  return {
    w, h: top + inner.h + FPAD, py,
    place(x, y, out) {
      out.frames.push({ item, x, y, w, h: this.h });
      inner.place(x + ox, y + top, out);
      out.edges.push([x, y + py, x + ox, y + py], [x + ox + inner.w, y + py, x + w, y + py]);
    },
  };
}
function seriesBox(kids) {
  const py = Math.max(...kids.map((k) => k.py));
  const below = Math.max(...kids.map((k) => k.h - k.py));
  return {
    w: kids.reduce((s, k) => s + k.w, 0) + GAP * (kids.length - 1), h: py + below, py,
    place(x, y, out) {
      let cx = x;
      kids.forEach((k, i) => {
        k.place(cx, y + py - k.py, out);
        if (i < kids.length - 1) out.edges.push([cx + k.w, y + py, cx + k.w + GAP, y + py]);
        cx += k.w + GAP;
      });
    },
  };
}
function parallelBox(kids, vote, uid) {
  const maxW = Math.max(...kids.map((k) => k.w));
  const ys = [];
  let h = 0;
  for (const k of kids) { ys.push(h); h += k.h + VGAP; }
  h -= VGAP;
  const ports = kids.map((k, i) => ys[i] + k.py);
  const py = (ports[0] + ports[ports.length - 1]) / 2;
  const xr = 2 * RAIL + maxW;
  return {
    w: xr + (vote ? 8 + 2 * VR : 0), h, py,
    place(x, y, out) {
      kids.forEach((k, i) => {
        const cx = x + RAIL + (maxW - k.w) / 2, p = y + ports[i];
        k.place(cx, y + ys[i], out);
        out.edges.push([x, p, cx, p], [cx + k.w, p, x + xr, p]);
      });
      out.edges.push([x, y + ports[0], x, y + ports[ports.length - 1]], [x + xr, y + ports[0], x + xr, y + ports[ports.length - 1]]);
      if (vote) {
        out.edges.push([x + xr, y + py, x + xr + 8, y + py]);
        out.voters.push({ cx: x + xr + 8 + VR, cy: y + py, label: vote, uid });
      }
    },
  };
}

// Editing on the diagram: a block (or a whole group) gets a block in series
// or in parallel with it, a group a new block of its own, a series,
// parallel or voting group another of those structures, a basic or
// undeveloped block the other kind. The app turns each into tree edits.
// parent: the gate the element is an input of (null for the top); the
// inputs of an INHIBIT group stay its input and its condition.
export function blockActs(doc, item, parent = null) {
  const n = item.n, acts = [];
  const inInhibit = !!parent && parent.gate === "INHIBIT";
  const group = item.type === "series" || item.type === "parallel" || item.type === "kofn";
  const e = n.t === "leaf" ? doc.events[n.id] : null;
  // An INHIBIT group keeps its input and its conditioning event: nothing is
  // added to it, and the conditioning event gets nothing next to it.
  if (group && n.gate !== "INHIBIT") acts.push(["addBlock", "+▭", t("tb.addBlock")]);
  if (!(e && e.kind === "conditioning")) acts.push(["series", "+⟶", t("tb.addSeries")], ["parallel", "+‖", t("tb.addParallel")]);
  if (group && ["OR", "AND", "K_OF_N"].includes(n.gate)) acts.push(["structure", "⇄", t("tb.cycleStructure")]);
  if (e && (e.kind === "basic" || e.kind === "undeveloped")) acts.push(["cycle", "⇄", t("tb.cycleKind")]);
  if (n !== doc.root && !inInhibit && !(e && e.kind === "conditioning")) acts.push(["delete", "✕", t("tb.delete")]);
  return acts;
}
// keep: the corner [0, keep.x] x [0, keep.y] that the zoom buttons cover.
// width: the drawing's; a toolbar that cannot clear the zoom buttons there
// goes below the element's top edge instead.
function drawToolbar(parent, cx, y, acts, onAct, uid, keep, width, height) {
  const bw = 32, gap = 4, total = acts.length * bw + (acts.length - 1) * gap;
  let x = cx - total / 2;
  if (y < keep.y && x < keep.x) {
    if (keep.x + total <= width - 4) x = keep.x;
    else { x = Math.max(4, Math.min(x, width - total - 4)); y = Math.min(Math.max(y, keep.y), height - 26); }
  }
  for (const [act, glyph, tip] of acts) {
    const b = svgEl("g", { class: "tb" + (act === "delete" ? " del" : ""), tabindex: 0, role: "button", "aria-label": tip, "data-noexport": "1", "data-act": act }, parent);
    svgEl("title", {}, b).textContent = tip;
    svgEl("rect", { x, y, width: bw, height: 24, rx: 5, "stroke-width": 1.2 }, b);
    svgText(b, x + bw / 2, y + 17, glyph, { "text-anchor": "middle" });
    const fire = (evt) => { evt.stopPropagation(); onAct(act, uid); };
    b.addEventListener("click", fire);
    b.addEventListener("keydown", (evt) => { if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); fire(evt); } });
    x += bw + gap;
  }
}

export function layoutBlocks(doc, mode = "edit") {
  const root = box(rbdOf(doc), mode !== "edit");
  const top = 52; // the zoom buttons cover the corner (see drawToolbar)
  return { root, top, width: root.w + 2 * MARGIN, height: top + root.h + 14 };
}

// opts as renderTree's: mode ("edit" | "result" | "whatif"), zoom, selUid,
// probs, color, forced, highlight, success, sharedStyle, onNode, tip, and
// toolbar(act, uid): the buttons above the selected element in the editor
// (see blockActs); complement: show R = 1 - P of failure-mode values (the
// colours stay those of P).
export function renderBlocks(svg, doc, opts = {}) {
  const mode = opts.mode || "edit";
  svg.replaceChildren();
  const L = layoutBlocks(doc, mode);
  const zoom = opts.zoom || 1;
  svg.setAttribute("viewBox", `0 0 ${L.width} ${L.height}`);
  svg.setAttribute("width", Math.round(L.width * zoom));
  svg.setAttribute("height", Math.round(L.height * zoom));
  svg.setAttribute("font-family", FONT);
  const out = { frames: [], cells: [], edges: [], voters: [], empty: [] };
  L.root.place(MARGIN, L.top, out);
  const py = L.top + L.root.py, xr = MARGIN + L.root.w;
  out.edges.push([MARGIN - 16, py, MARGIN, py], [xr, py, xr + 16, py]);

  const marks = sharedMarks(doc);
  const style = opts.sharedStyle || SET.shared;
  const keyOf = (n) => (n.t === "ref" ? n.ref : n.id);
  let selKey = null;
  if (mode === "edit" && opts.selUid != null) walk(doc.root, (n) => { if (n.uid === opts.selUid) selKey = keyOf(n); });
  const probs = opts.probs || null, color = opts.color || null, forced = opts.forced || {}, hl = opts.highlight || null;
  const complement = !!opts.complement && !opts.success;
  const pLabel = opts.success || complement ? "R" : "P";
  const shown = (p) => (complement ? fmtRq(p) : opts.success ? fmtR(p) : fmtP(p, 3));
  const msw = style === "color" ? 3.4 : 2.4;

  const edges = svgEl("g", { fill: "none", stroke: C.muted, "stroke-width": 1.4 }, svg);
  for (const [x1, y1, x2, y2] of out.edges) if (x1 !== x2 || y1 !== y2) svgEl("path", { d: `M${x1},${y1} L${x2},${y2}` }, edges);
  for (const x of [MARGIN - 18, xr + 18]) svgEl("circle", { cx: x, cy: py, r: 4, fill: C.ink, stroke: "none" }, edges);
  const nodes = svgEl("g", {}, svg);

  // The node group of an element: tooltip, click, keyboard and the
  // highlight of every occurrence of a shared item.
  function group(n) {
    const g = svgEl("g", { class: "nd", "data-uid": n.uid, tabindex: 0, role: "button" }, nodes);
    const key = keyOf(n), mk = marks.get(key);
    if (mk) {
      g.dataset.twin = key;
      if (key === selKey && n.uid !== opts.selUid) g.classList.add("twinSel");
      if (SET.twins) {
        const twins = (on) => svg.querySelectorAll(`.nd[data-twin="${CSS.escape(key)}"]`).forEach((x) => x.classList.toggle("twin", on));
        g.addEventListener("pointerenter", () => twins(true));
        g.addEventListener("pointerleave", () => twins(false));
        g.addEventListener("focus", () => twins(true));
        g.addEventListener("blur", () => twins(false));
      }
    }
    if (opts.tip) bindTip(g, () => opts.tip(n));
    if (opts.onNode) {
      g.addEventListener("click", (evt) => { evt.stopPropagation(); opts.onNode(n, evt); });
      g.addEventListener("keydown", (evt) => { if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); opts.onNode(n, evt); } });
    }
    return { g, key, mk, mc: mk ? sharedColor(mk) : null, sel: mode === "edit" && n.uid === opts.selUid };
  }
  const valueOf = (key) => (probs ? probs[key] : undefined);
  const fillOf = (p) => (color && p != null ? color(p) : C.card);

  for (const f of out.frames) {
    const n = f.item.n, { g, key, mk, mc, sel } = group(n);
    const p = valueOf(key), head = mode === "edit" || p == null ? "#F1EFEA" : fillOf(p), txt = isDark(head) ? "#FFFFFF" : C.ink;
    if (hl && hl.has(key)) svgEl("rect", { x: f.x - 5, y: f.y - 5, width: f.w + 10, height: f.h + 10, rx: 9, fill: "none", stroke: C.s2, "stroke-width": 4 }, g);
    if (mk) twinRing("rect", { x: f.x - 4, y: f.y - 4, width: f.w + 8, height: f.h + 8, rx: 8 }, mc, g);
    svgEl("rect", { class: "focusRing", x: f.x, y: f.y, width: f.w, height: f.h, rx: 6, fill: "transparent",
      stroke: sel ? C.accent : mc || (n === doc.root ? C.ink : C.rule), "stroke-width": sel ? 2.6 : mk ? msw : n === doc.root ? 1.6 : 1.2 }, g);
    svgEl("path", { d: `M${f.x + 0.6},${f.y + FHEAD} V${f.y + 6} Q${f.x + 0.6},${f.y + 0.6} ${f.x + 6},${f.y + 0.6} H${f.x + f.w - 6} Q${f.x + f.w - 0.6},${f.y + 0.6} ${f.x + f.w - 0.6},${f.y + 6} V${f.y + FHEAD} Z`, fill: head, stroke: "none" }, g);
    const pTxt = mode !== "edit" && p != null ? `${pLabel} = ${shown(p)}` : "";
    const idW = n.id.length * 7 + 4, pW = pTxt ? 14 + pTxt.length * 6.6 : 0;
    svgText(g, f.x + 8, f.y + 14, trunc(n.id, Math.max(4, Math.floor((f.w - 16 - pW) / 7))), { "font-size": 11, "font-weight": 700, class: "mono", fill: txt });
    const room = Math.floor((f.w - 16 - idW - 8 - pW) / 6);
    if (room >= 4 && dispName(n)) svgText(g, f.x + 8 + idW + 8, f.y + 14, trunc(dispName(n), room), { "font-size": 11, fill: txt });
    if (pTxt) svgText(g, f.x + f.w - 8, f.y + 14, pTxt, { "font-size": 11, class: "mono", "text-anchor": "end", fill: txt });
    if (mk) markBadge(g, f.x + f.w - 2, f.y, mk, mc, style);
  }
  // Voters and "no inputs" belong to their group: they select it.
  const owner = (uid) => nodes.querySelector(`.nd[data-uid="${uid}"]`) || nodes;
  for (const v of out.voters) {
    svgEl("circle", { cx: v.cx, cy: v.cy, r: VR, fill: C.card, stroke: C.ink, "stroke-width": 1.4 }, owner(v.uid));
    svgText(owner(v.uid), v.cx, v.cy + 4, v.label, { "text-anchor": "middle", "font-size": 10.5, "font-weight": 700 });
  }
  for (const e of out.empty) svgText(owner(e.uid), e.x, e.y, t("tree.noInputs"), { "text-anchor": "middle", "font-size": 11, fill: C.wine });
  for (const c of out.cells) (c.item.type === "switch" ? drawSwitch : drawBlock)(c);
  if (mode === "edit" && opts.toolbar && opts.selUid != null) {
    const f = out.frames.find((x) => x.item.n.uid === opts.selUid);
    const c = f ? null : out.cells.find((x) => x.item.n.uid === opts.selUid);
    const keep = { x: 124 / zoom, y: 46 / zoom };
    let parent = null;
    walk(doc.root, (n, p) => { if (n.uid === opts.selUid) parent = p; });
    if (f) drawToolbar(nodes, f.x + Math.min(f.w, 260) / 2, f.y - 30, blockActs(doc, f.item, parent), opts.toolbar, f.item.n.uid, keep, L.width, L.height);
    else if (c) drawToolbar(nodes, c.x + (c.item.type === "switch" ? SW : BW) / 2, c.y - 30, blockActs(doc, c.item, parent), opts.toolbar, c.item.n.uid, keep, L.width, L.height);
  }

  function drawSwitch({ item, x, y }) {
    const n = item.n, e = doc.events[n.id] || { name: n.id, state: false }, { g, mk, mc, sel } = group(n);
    const open = e.state === true, x1 = x + 14, x2 = x + SW - 14, p = y + SPY;
    svgEl("rect", { class: "focusRing", x: x + 2, y, width: SW - 4, height: CELL, rx: 6, fill: "transparent", stroke: "none" }, g);
    if (hl && hl.has(n.id)) svgEl("rect", { x: x + 2, y: y - 2, width: SW - 4, height: CELL, rx: 6, fill: "none", stroke: C.s2, "stroke-width": 4 }, g);
    if (mk) twinRing("rect", { x: x + 4, y: y + 2, width: SW - 8, height: CELL - 6, rx: 6 }, mc, g);
    const stroke = sel ? C.accent : mc || C.ink;
    svgEl("path", { d: `M${x},${p} H${x1} M${x2},${p} H${x + SW}`, fill: "none", stroke: C.muted, "stroke-width": 1.4 }, g);
    const a = open ? (-28 * Math.PI) / 180 : 0, len = x2 - x1;
    svgEl("line", { x1, y1: p, x2: x1 + len * Math.cos(a), y2: p + len * Math.sin(a), stroke, "stroke-width": sel ? 3 : 2.2, "stroke-linecap": "round" }, g);
    for (const cx of [x1, x2]) svgEl("circle", { cx, cy: p, r: 3.2, fill: C.card, stroke, "stroke-width": 1.6 }, g);
    svgText(g, x + SW / 2, y + 11, trunc(n.id, 11), { "text-anchor": "middle", "font-size": 11, "font-weight": 700, class: "mono" });
    svgText(g, x + SW / 2, y + 46, trunc(dispName(e), 13), { "text-anchor": "middle", "font-size": 11 });
    svgText(g, x + SW / 2, y + 60, t(open ? "rbd.open" : "rbd.closed"), { "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: open ? C.wine : C.accent });
    if (mk) markBadge(g, x + SW - 6, y + 8, mk, mc, style);
  }

  function drawBlock({ item, x, y }) {
    const n = item.n, { g, key, mk, mc, sel } = group(n);
    const e = n.t === "leaf" ? doc.events[n.id] || { name: n.id, prob: NaN, kind: "basic" } : null;
    const p = valueOf(key), state = forced[key];
    let fill = fillOf(p), txt = isDark(fill) ? "#FFFFFF" : C.ink, stroke = sel ? C.accent : mc || C.ink, sw = sel ? 2.6 : mk ? msw : 1.4;
    if (state) { fill = state === "failed" ? C.s3 : C.s1; stroke = state === "failed" ? C.wine : C.accent; txt = "#FFFFFF"; sw = 2.4; }
    const cond = e && e.kind === "conditioning", rx = cond ? BH / 2 : 4;
    svgEl("rect", { class: "focusRing", x: x - 3, y: y - 3, width: BW + 6, height: CELL + 4, rx: 6, fill: "none", stroke: "none" }, g);
    if (hl && hl.has(key)) svgEl("rect", { x: x - 6, y: y - 6, width: BW + 12, height: BH + 12, rx: rx + 4, fill: "none", stroke: C.s2, "stroke-width": 4 }, g);
    if (mk) twinRing("rect", { x: x - 5, y: y - 5, width: BW + 10, height: BH + 10, rx: rx + 3 }, mc, g);
    svgEl("rect", { x, y, width: BW, height: BH, rx, fill, stroke, "stroke-width": sw, "stroke-dasharray": cond && !state ? "5 3" : null }, g);
    if (item.type === "module") svgEl("rect", { x: x + 3.5, y: y + 3.5, width: BW - 7, height: BH - 7, rx: 2, fill: "none", stroke, "stroke-width": 1 }, g);
    if (item.type === "clone") for (const bx of [x + 8, x + BW - 8]) svgEl("line", { x1: bx, y1: y, x2: bx, y2: y + BH, stroke, "stroke-width": 1.2 }, g);
    if (e && e.kind === "undeveloped") svgEl("path", { d: `M${x},${y + BH - 7} L${x + 7},${y + BH} L${x},${y + BH + 7} L${x - 7},${y + BH} Z`, fill: C.card, stroke: C.ink, "stroke-width": 1.2 }, g);
    svgText(g, x + BW / 2, y + 18, trunc(key, item.type === "clone" ? 11 : 14), { "text-anchor": "middle", "font-size": 12, "font-weight": 700, class: "mono", fill: txt });
    const second = item.type === "clone" ? t("rbd.repeat")
      : item.type === "module" ? [n.gate === "K_OF_N" ? `K_OF_N k=${n.k}` : n.gate, dispName(n)].filter(Boolean).join(" · ") : dispName(e);
    svgText(g, x + BW / 2, y + 34, trunc(second, item.type === "module" ? 15 : 16), { "text-anchor": "middle", "font-size": 11, fill: txt });
    let below = "";
    if (state) below = t("whatif." + state);
    else if (mode !== "edit" && p != null) below = (complement ? "R " : "") + shown(p);
    else if (mode === "edit" && e) below = e.samples ? `${fmtP(e.prob, 3)} (n=${e.samples.length})` : e.model ? `${fmtP(e.prob, 3)} · ${t("model.short." + e.model.dist)}` : fmtP(e.prob, 3);
    else if (mode === "edit" && item.type === "module") below = t("rbd.module");
    if (below) svgText(g, x + BW / 2, y + BH + 14, below, { "text-anchor": "middle", "font-size": 11, class: state || !e ? "" : "mono",
      "font-weight": state ? 700 : 400, fill: state === "failed" ? C.wine : state === "working" ? C.accent : "#4A4A46" });
    if (e && e.dist && mode === "edit") {
      const bx = x + 2, by = y + 2;
      svgEl("circle", { cx: bx, cy: by, r: 8.5, fill: C.accent }, g);
      svgEl("path", { d: `M${bx - 5.5},${by + 3} C${bx - 3},${by + 3} ${bx - 2.2},${by - 4.5} ${bx},${by - 4.5} C${bx + 2.2},${by - 4.5} ${bx + 3},${by + 3} ${bx + 5.5},${by + 3}`,
        fill: "none", stroke: "#FFFFFF", "stroke-width": 1.6, "stroke-linecap": "round" }, g);
    }
    if (mk) markBadge(g, x + BW - 2, y, mk, mc, style);
  }
  return L;
}
