// CM Point Editor: points / boxes per object per keyframe, kijai-style markers, live SAM2 mask preview.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LOADER, LOADER_CF, wv, hideWidget, loaderSettings, qs, el } from "./cm_common.js";

const NODE = "CMPointEditor";
const PALETTE = ["#ff4d4d", "#4dabff", "#4dff88", "#ffd24d", "#c44dff", "#ff8c4d",
  "#4dfff0", "#ff4dc4", "#a3ff4d", "#8c8cff", "#ffffff", "#ff9e9e"];
const POS = "#35e06b", NEG = "#ff3b3b";
const HIT = 18;          // on-screen px hit radius for points / corners
const R = 16;            // marker ring radius (on-screen px, constant at any zoom)

const parse = (s) => {
  let d = {};
  try { d = typeof s === "string" && s ? JSON.parse(s) : (s || {}); } catch { d = {}; }
  return { v: 1, space: d.space || [0, 0], objects: d.objects || [], frames: d.frames || {} };
};
const rgba = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; };

// Solid RGBA mask bitmap -> { fill canvas, w, h, paths:[{col, halo, path}] } (one Path2D of pixel-edge segments per colour).
function buildMask(bmp) {
  const w = bmp.width, h = bmp.height;
  const fill = document.createElement("canvas"); fill.width = w; fill.height = h;
  const g = fill.getContext("2d", { willReadFrequently: true }); g.drawImage(bmp, 0, 0);
  const d = g.getImageData(0, 0, w, h).data;
  const key = (i) => (d[i * 4 + 3] < 128 ? 0 : ((d[i * 4] << 16) | (d[i * 4 + 1] << 8) | d[i * 4 + 2]) + 1);
  const K = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) K[i] = key(i);
  const by = new Map();
  const P = (k) => { let p = by.get(k); if (!p) by.set(k, (p = new Path2D())); return p; };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = K[y * w + x]; if (!k) continue;
    if (x === 0 || K[y * w + x - 1] !== k) { const p = P(k); p.moveTo(x, y); p.lineTo(x, y + 1); }
    if (x === w - 1 || K[y * w + x + 1] !== k) { const p = P(k); p.moveTo(x + 1, y); p.lineTo(x + 1, y + 1); }
    if (y === 0 || K[(y - 1) * w + x] !== k) { const p = P(k); p.moveTo(x, y); p.lineTo(x + 1, y); }
    if (y === h - 1 || K[(y + 1) * w + x] !== k) { const p = P(k); p.moveTo(x, y + 1); p.lineTo(x + 1, y + 1); }
  }
  const paths = [];
  for (const [k, path] of by) {
    const c = k - 1, r = (c >> 16) & 255, gg = (c >> 8) & 255, b = c & 255;
    const lum = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
    paths.push({ path, col: `rgb(${r},${gg},${b})`, halo: lum > 140 ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.85)" });
  }
  return { fill, w, h, paths };
}

class Editor {
  constructor(node) {
    this.node = node;
    this.p = parse("");
    this.obj = 0;
    this.idx = 0;
    this.count = 0;
    this.size = [0, 0];
    this.img = null;
    this.mask = null;
    this.cache = new Map();
    this.hover = null;
    this.drag = null;
    this.showMask = true;
    this.src = null;
    this._seq = 0;
    this.fps = 24;
    this.opacity = 0.45;
    this.outline = true;
    this.playing = false;
    this.mcache = new Map();   // idx -> Promise<{im, ol}|null>

    this.host = el("div", "width:100%;height:100%;min-height:0;");
    this.root = el("div", "display:flex;flex-direction:column;width:100%;height:100%;min-height:0;gap:3px;box-sizing:border-box;font:11px sans-serif;color:#ddd;user-select:none;", this.host);
    const bar = el("div", "display:flex;gap:3px;align-items:center;flex:0 0 auto;flex-wrap:wrap;", this.root);
    this.chips = el("div", "display:flex;gap:3px;flex-wrap:wrap;", bar);
    const btn = (p, t, tip, fn) => { const b = el("button", "padding:1px 6px;font:11px sans-serif;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;cursor:pointer;", p, t); b.title = tip; b.onclick = (e) => { e.stopPropagation(); fn(b); }; return b; };
    btn(bar, "+ obj", "Add an object (new colour)", () => this.addObject());
    btn(bar, "clear frame", "Remove all prompts on this frame", () => { delete this.p.frames[this.idx]; this.commit(); });
    btn(bar, "clear obj", "Remove the selected object everywhere", () => this.removeObject(this.obj));
    this.maskBtn = btn(bar, "mask ●", "Toggle live mask preview", (b) => { this.showMask = !this.showMask; b.textContent = this.showMask ? "mask ●" : "mask ○"; this.requestMask(); this.draw(); });
    this.olBtn = btn(bar, "outline ●", "Toggle mask outline (2px object colour + contrast halo)", (b) => { this.outline = !this.outline; b.textContent = this.outline ? "outline ●" : "outline ○"; this.draw(); });
    const op = el("label", "display:flex;align-items:center;gap:2px;color:#aaa;", bar, "α");
    op.title = "Mask fill opacity";
    this.opSl = el("input", "width:60px;", op);
    Object.assign(this.opSl, { type: "range", min: 0, max: 100, value: Math.round(this.opacity * 100) });
    this.opSl.oninput = (e) => { e.stopPropagation(); this.opacity = +this.opSl.value / 100; this.draw(); };
    this.opSl.onpointerdown = (e) => e.stopPropagation();
    btn(bar, "▶ run", "Run only this node (Queue Selected Output Nodes) to load its input frames", () => this.run());
    btn(bar, "⧉", "Pop out into a floating panel", () => this.togglePopout());
    const help = el("span", "margin-left:auto;cursor:help;color:#999;padding:0 4px;", bar, "?");
    help.title = "Click: positive point  ·  Right-click empty: negative point\n" +
      "Right-click / Alt+click / middle-click on a point or box: delete\n" +
      "Drag point: move  ·  Ctrl+drag: box  ·  drag box corner: resize  ·  drag inside box: move\n" +
      "Single-keyframe mode: prompting another frame moves the keyframe there.";
    this.view = el("div", "position:relative;flex:1 1 auto;min-height:40px;background:#0c0c0c;border-radius:4px;overflow:hidden;", this.root);
    this.cv = el("canvas", "position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair;", this.view);
    this.msg = el("div", "position:absolute;left:6px;top:4px;right:6px;white-space:normal;color:#aaa;pointer-events:none;text-shadow:0 0 3px #000;", this.view);
    const sc = el("div", "display:flex;gap:4px;align-items:center;flex:0 0 auto;", this.root);
    this.playBtn = btn(sc, "⏵", "Play / pause (loops; shows tracked masks after CM Segment runs, live masks on keyframes)", () => this.togglePlay());
    btn(sc, "◀", "Previous frame", () => this.setFrame(this.idx - 1));
    this.slider = el("input", "flex:1;min-width:40px;", sc);
    Object.assign(this.slider, { type: "range", min: 0, max: 0, value: 0 });
    this.slider.oninput = () => { this.stop(); this.setFrame(+this.slider.value); };
    this.slider.onpointerdown = (e) => e.stopPropagation();
    btn(sc, "▶", "Next frame", () => this.setFrame(this.idx + 1));
    btn(sc, "⟨k", "Previous keyframe", () => this.jumpKey(-1));
    btn(sc, "k⟩", "Next keyframe", () => this.jumpKey(1));
    this.flabel = el("span", "min-width:70px;text-align:right;color:#aaa;", sc);
    this.keys = el("canvas", "width:100%;height:8px;flex:0 0 8px;cursor:pointer;", this.root);
    this.keys.onpointerdown = (e) => { e.stopPropagation(); const r = this.keys.getBoundingClientRect(); this.setFrame(Math.round(((e.clientX - r.left) / r.width) * Math.max(0, this.count - 1))); };

    for (const [ev, fn] of [["pointerdown", this.down], ["pointermove", this.move], ["pointerup", this.up], ["pointerleave", this.leave]])
      this.cv.addEventListener(ev, fn.bind(this));
    const eat = (e) => { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); };
    this.cv.addEventListener("contextmenu", eat, true);
    for (const ev of ["mousedown", "mouseup", "auxclick"]) this.cv.addEventListener(ev, (e) => { e.stopPropagation(); e.stopImmediatePropagation(); if (e.button !== 0) e.preventDefault(); }, true);
    this.view.addEventListener("wheel", (e) => { if (e.altKey) { e.preventDefault(); e.stopPropagation(); this.setFrame(this.idx + Math.sign(e.deltaY)); } }, { passive: false });
    new ResizeObserver(() => this.draw()).observe(this.view);
    this._onLoader = (e) => { if (this.src?.kind === "file" && e.detail?.id === this.src.id) this.refresh(); };
    window.addEventListener("cm-loader-changed", this._onLoader);
    // catch rewiring through reroutes and upstream widget edits that fire no callback
    this._poll = setInterval(() => { if (this.node.graph && this.sig() !== this._sig) this.refresh(); }, 1000);
  }

  // ---------- data
  get multi() { return !!wv(this.node, "multi_keyframe")?.value; }
  load() {
    this.p = parse(wv(this.node, "editor_data")?.value);
    if (!this.p.objects.length) this.addObject(false);
    this.obj = this.p.objects[0].id;
    this.renderChips();
  }
  // editor_data holds only points/boxes/objects/space, canonically ordered and rounded, and is written only
  // when that content changes, so ComfyUI's input signature (and everything downstream) stays cached.
  canon() {
    const r2 = (v) => Math.round(v * 100) / 100;
    const frames = {};
    for (const f of Object.keys(this.p.frames).map(Number).sort((a, b) => a - b)) {
      const fr = {}, src = this.p.frames[f];
      for (const o of Object.keys(src).map(Number).sort((a, b) => a - b)) {
        const v = src[o], pts = (v.points || []).map(([x, y, l]) => [r2(x), r2(y), l ? 1 : 0]);
        const box = v.box ? v.box.map(r2) : null;
        if (pts.length || box) fr[o] = { points: pts, box };
      }
      if (Object.keys(fr).length) frames[f] = fr;
    }
    const objects = this.p.objects.map((o) => ({ id: o.id, name: o.name, color: o.color }));
    return JSON.stringify({ v: 1, space: [this.p.space[0] | 0, this.p.space[1] | 0], objects, frames });
  }
  save() { const w = wv(this.node, "editor_data"); if (!w) return; const s = this.canon(); if (w.value !== s) w.value = s; }
  commit() { const before = wv(this.node, "editor_data")?.value; this.save(); if (wv(this.node, "editor_data")?.value !== before) this.mcache.delete(this.idx); this.renderChips(); this.drawKeys(); this.draw(); this.requestMask(); this.node.setDirtyCanvas(true, false); }
  color(id) { return this.p.objects.find((o) => o.id === id)?.color || PALETTE[(id - 1) % PALETTE.length]; }
  addObject(sel = true) {
    const id = Math.max(0, ...this.p.objects.map((o) => o.id)) + 1;
    this.p.objects.push({ id, name: `obj ${id}`, color: PALETTE[(id - 1) % PALETTE.length] });
    if (sel) { this.obj = id; this.save(); this.renderChips(); }
  }
  removeObject(id) {
    for (const f of Object.keys(this.p.frames)) { delete this.p.frames[f][id]; if (!Object.keys(this.p.frames[f]).length) delete this.p.frames[f]; }
    if (this.p.objects.length > 1) { this.p.objects = this.p.objects.filter((o) => o.id !== id); this.obj = this.p.objects[0].id; }
    this.commit();
  }
  renderChips() {
    this.chips.textContent = "";
    for (const o of this.p.objects) {
      const c = el("span", `padding:1px 6px;border-radius:9px;cursor:pointer;border:2px solid ${o.id === this.obj ? "#fff" : "transparent"};background:${rgba(o.color, 0.35)};color:#fff;`, this.chips, o.name);
      c.title = "Select object";
      c.onclick = (e) => { e.stopPropagation(); this.obj = o.id; this.renderChips(); this.draw(); };
    }
  }
  fp(f = this.idx, make = false) {
    let fr = this.p.frames[f];
    if (!fr && make) fr = this.p.frames[f] = {};
    return fr;
  }
  op(oid = this.obj, make = true) {
    // single-keyframe mode: prompting another frame relocates the keyframe here
    if (make && !this.multi) {
      const ks = Object.keys(this.p.frames).map(Number).filter((k) => k !== this.idx);
      if (ks.length && !this.p.frames[this.idx]) { this.p.frames[this.idx] = this.p.frames[ks[0]]; this.flash(`keyframe moved ${ks[0]} → ${this.idx}`); }
      for (const k of ks) delete this.p.frames[k];
    }
    const fr = this.fp(this.idx, make);
    if (!fr) return null;
    if (!fr[oid] && make) fr[oid] = { points: [], box: null };
    return fr[oid];
  }
  prune() {
    const fr = this.p.frames[this.idx];
    if (!fr) return;
    for (const k of Object.keys(fr)) if (!fr[k].points.length && !fr[k].box) delete fr[k];
    if (!Object.keys(fr).length) delete this.p.frames[this.idx];
  }
  flash(t) { this.msg.textContent = t; clearTimeout(this._ft); this._ft = setTimeout(() => this.status(), 1800); }
  status() {
    const k = Object.keys(this.p.frames).length;
    const kf = `${k} keyframe${k === 1 ? "" : "s"}${this.multi ? "" : " (single)"}`;
    const RUN = "run this node (select it, then the ▶ in the selection toolbox, or the ▶ run button here)";
    let t;
    if (!this.src) t = "no input: connect CM Load Video (video_info or images)";
    else if (this.src.kind === "file") {
      const f = loaderSettings(this.src.node).file;
      t = !f ? "live: CM Load Video has no file selected" : this.count ? `live: ${f} · ${this.count}f · ${this.size[0]}×${this.size[1]} · ${kf}` : (this.err ? `live: ${f} · ${this.err}` : `live: ${f} · loading…`);
    } else t = this.count ? `frames from last run · ${this.count}f · ${this.size[0]}×${this.size[1]} · ${kf}` : `no live source: ${RUN}`;
    this.msg.textContent = t;
    this.msg.title = t;
  }
  run() {
    try {
      app.canvas.deselectAll?.();
      app.canvas.selectNode?.(this.node, false);
      const cmd = app.extensionManager?.command;
      if (cmd?.execute) return cmd.execute("Comfy.QueueSelectedOutputNodes");
    } catch (e) { console.warn("[CM] run failed", e); }
    this.flash("select this node and use ▶ in the selection toolbox");
  }

  // ---------- source
  // Walk a link back through reroutes to its origin; returns {node, slot} or null.
  origin(node, slotIdx) {
    const g = node.graph || app.graph;
    let n = node, i = slotIdx;
    for (let hop = 0; hop < 16; hop++) {
      const lid = n.inputs?.[i]?.link;
      if (lid == null) return null;
      const L = g.links?.get?.(lid) ?? g.links?.[lid] ?? g._links?.get?.(lid);
      if (!L) return null;
      const o = g.getNodeById(L.origin_id);
      if (!o) return null;
      const t = o.comfyClass || o.type || "";
      if (/reroute/i.test(t) && o.inputs?.length) { n = o; i = 0; continue; }
      return { node: o, slot: L.origin_slot };
    }
    return null;
  }
  upstream() {
    const n = this.node, slot = (nm) => n.inputs?.findIndex((s) => s.name === nm) ?? -1;
    const linked = [];
    for (const nm of ["video_info", "images"]) {
      const i = slot(nm);
      if (i < 0 || n.inputs[i].link == null) continue;
      linked.push(nm);
      const o = this.origin(n, i);
      const t = o && (o.node.comfyClass || o.node.type);
      if (t === LOADER || t === LOADER_CF) return { kind: "file", id: o.node.id, node: o.node, via: nm };
    }
    return linked.length ? { kind: "node", id: n.id, via: linked.join("+") } : null;
  }
  sig() { const u = this.upstream(); return u ? `${u.kind}:${u.id}:${u.kind === "file" ? JSON.stringify(loaderSettings(u.node)) : ""}` : "none"; }
  query(extra = {}) {
    if (!this.src) return null;
    if (this.src.kind === "file") { const q = loaderSettings(this.src.node); return q.file ? { ...q, ...extra } : null; }
    return { node: String(this.node.id), ...extra };
  }
  refresh() {
    clearTimeout(this._rt);
    this._rt = setTimeout(() => this._refresh(), 200);
  }
  async _refresh() {
    this.src = this.upstream();
    this.cache.clear(); this.mcache.clear(); this.stop();
    const q = this.query();
    this._sig = this.sig();
    if (!q) { this.count = 0; this.img = null; this.status(); this.draw(); return; }
    try {
      this.err = "";
      const r = await api.fetchApi(`/coordmatrix/plan?${qs(q)}`);
      if (!r.ok) { let m = `HTTP ${r.status}`; try { m = (await r.json()).error || m; } catch {} this.err = m; throw 0; }
      const pl = await r.json();
      const [ow, oh] = this.p.space;
      if (ow && oh && (ow !== pl.width || oh !== pl.height)) this.rescale(pl.width / ow, pl.height / oh);
      this.p.space = [pl.width, pl.height];
      this.size = [pl.width, pl.height];
      this.count = pl.length;
      let lf = 0; try { lf = this.src.node ? +loaderSettings(this.src.node).fps : 0; } catch {}
      this.fps = +pl.fps || lf || 24;
    } catch { this.count = 0; }
    this.slider.max = Math.max(0, this.count - 1);
    this.status();
    this.setFrame(Math.min(this.idx, Math.max(0, this.count - 1)), true);
  }
  rescale(sx, sy) {
    for (const fr of Object.values(this.p.frames)) for (const o of Object.values(fr)) {
      o.points = o.points.map(([x, y, l]) => [x * sx, y * sy, l]);
      if (o.box) o.box = [o.box[0] * sx, o.box[1] * sy, o.box[2] * sx, o.box[3] * sy];
    }
    this.save();
  }
  frameImg(i) {
    if (this.cache.has(i)) return this.cache.get(i);
    const q = this.query({ idx: i });
    const url = api.apiURL(`/coordmatrix/frame?${qs(q)}&_ts=${Date.now()}`);
    const pr = new Promise((res) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = async () => {
        this.cache.delete(i);
        let m = "frame load failed";
        try { const r = await fetch(url); m = `frame ${i}: HTTP ${r.status} ${(await r.json()).error || ""}`; } catch {}
        this.err = m; this.flash(m); console.warn("[CM]", m, url); res(null);
      };
      im.src = url;
    });
    this.cache.set(i, pr);
    if (this.cache.size > 48) this.cache.delete(this.cache.keys().next().value);
    return pr;
  }
  async setFrame(i, force = false) {
    if (!this.count) { this.draw(); return; }
    i = Math.max(0, Math.min(this.count - 1, i | 0));
    if (i === this.idx && this.img && !force) return;
    this.idx = i; this.slider.value = i;
    this.flabel.textContent = `${i} / ${this.count - 1}`;
    if (!this.playing) this.mask = null;
    const im = await this.frameImg(i);
    if (this.idx !== i) return;
    if (im) this.img = im;
    this.drawKeys();
    if (this.playing) { this.mask = this.showMask ? await this.maskFor(i) : null; if (this.idx !== i) return; this.draw(); }
    else { this.draw(); this.requestMask(); }
    this.frameImg(i + 1);
    if (this.playing && this.showMask) this.maskFor(i + 1);
  }
  jumpKey(d) {
    const ks = Object.keys(this.p.frames).map(Number).sort((a, b) => a - b);
    const k = d > 0 ? ks.find((x) => x > this.idx) : ks.reverse().find((x) => x < this.idx);
    if (k != null) this.setFrame(k);
  }
  // ---------- playback
  togglePlay() { this.playing ? this.stop() : this.play(); }
  play() {
    if (!this.count) return;
    this.playing = true; this.playBtn.textContent = "⏸";
    const step = async () => {
      if (!this.playing) return;
      const t0 = performance.now();
      await this.setFrame((this.idx + 1) % this.count, true);
      if (!this.playing) return;
      this._pt = setTimeout(step, Math.max(0, 1000 / Math.max(1, this.fps) - (performance.now() - t0)));
    };
    step();
  }
  stop() { if (!this.playing) return; this.playing = false; clearTimeout(this._pt); this.playBtn.textContent = "⏵"; this.requestMask(); }

  requestMask() {
    clearTimeout(this._mt);
    if (!this.showMask || !this.count) { this.mask = null; this.draw(); return; }
    this._mt = setTimeout(() => this._mask(), 220);
  }
  async _mask() {
    const seq = ++this._seq, idx = this.idx;
    const m = await this.maskFor(idx);
    if (seq !== this._seq) return;
    this.mask = m; this.draw();
  }
  maskFor(idx) {
    if (this.mcache.has(idx)) return this.mcache.get(idx);
    const pr = this._fetchMask(idx);
    this.mcache.set(idx, pr);
    if (this.mcache.size > 160) this.mcache.delete(this.mcache.keys().next().value);
    pr.then((v) => { if (!v) this.mcache.delete(idx); });
    return pr;
  }
  async _fetchMask(idx) {
    let r = null;
    try {
      if (this.p.frames[idx]) {
        const body = { ...this.query({ idx }), prompts: this.p };
        r = await api.fetchApi("/coordmatrix/preview_mask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      } else {
        r = await api.fetchApi(`/coordmatrix/mask?${qs({ node: String(this.node.id), idx })}`);
      }
    } catch { r = null; }
    if (!r?.ok) return null;
    try { return buildMask(await createImageBitmap(await r.blob())); } catch { return null; }
  }

  // ---------- geometry
  xf() {
    const W = this.view.clientWidth, H = this.view.clientHeight, [w, h] = this.size;
    if (!w || !h) return { s: 1, ox: 0, oy: 0, W, H };
    const s = Math.min(W / w, H / h);
    return { s, ox: (W - w * s) / 2, oy: (H - h * s) / 2, W, H };
  }
  // The DOM widget is CSS-transformed by the graph zoom, so the bounding rect is in on-screen px while
  // clientWidth / xf() are in un-zoomed layout px. z = on-screen px per layout px (from the rect itself; ds.scale is not applied again).
  zoom() { const r = this.cv.getBoundingClientRect(); return { r, zx: r.width / (this.cv.clientWidth || 1) || 1, zy: r.height / (this.cv.clientHeight || 1) || 1 }; }
  toImg(e) {
    const { r, zx, zy } = this.zoom(), t = this.xf();
    return [((e.clientX - r.left) / zx - t.ox) / t.s, ((e.clientY - r.top) / zy - t.oy) / t.s];
  }
  clampPt([x, y]) { return [Math.max(0, Math.min(this.size[0] - 1, x)), Math.max(0, Math.min(this.size[1] - 1, y))]; }
  hitTest(x, y) {
    const fr = this.p.frames[this.idx];
    if (!fr) return null;
    const rad = Math.max(HIT, R + 2) / this.xf().s;   // layout px, same space the markers are drawn in
    let best = null, bd = rad;
    // points (selected object first, then others)
    const order = Object.keys(fr).map(Number).sort((a, b) => (b === this.obj) - (a === this.obj));
    for (const oid of order) fr[oid].points.forEach((p, i) => { const d = Math.hypot(p[0] - x, p[1] - y); if (d < bd) { bd = d; best = { t: "pt", oid, i }; } });
    if (best) return best;
    for (const oid of order) {
      const b = fr[oid].box; if (!b) continue;
      const cs = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
      for (let c = 0; c < 4; c++) if (Math.hypot(cs[c][0] - x, cs[c][1] - y) < rad) return { t: "corner", oid, c };
    }
    for (const oid of order) { const b = fr[oid].box; if (b && x > b[0] && x < b[2] && y > b[1] && y < b[3]) return { t: "box", oid }; }
    return null;
  }

  // ---------- interaction
  down(e) {
    e.stopPropagation(); e.stopImmediatePropagation(); e.preventDefault();
    if (!this.count || !this.img) return;
    const [x, y] = this.toImg(e);
    if (x < 0 || y < 0 || x >= this.size[0] || y >= this.size[1]) return;
    const h = this.hitTest(x, y);
    if ((e.button === 1 || (e.button === 0 && e.altKey)) && h) {
      if (h.t === "pt") this.p.frames[this.idx][h.oid].points.splice(h.i, 1); else this.p.frames[this.idx][h.oid].box = null;
      this.prune(); this.commit(); return;
    }
    if (e.button === 2) {
      if (h?.t === "pt") { this.p.frames[this.idx][h.oid].points.splice(h.i, 1); this.prune(); this.commit(); }
      else if (h) { this.p.frames[this.idx][h.oid].box = null; this.prune(); this.commit(); }
      else { this.op().points.push([x, y, 0]); this.commit(); }
      return;
    }
    if (e.button !== 0) return;
    this.cv.setPointerCapture(e.pointerId);
    if (e.ctrlKey || e.metaKey) { this.op(); this.drag = { t: "newbox", x0: x, y0: y, oid: this.obj }; return; }
    if (h?.t === "pt") { this.obj = h.oid; this.renderChips(); this.drag = { t: "pt", ...h }; return; }
    if (h?.t === "corner") { this.drag = { t: "corner", ...h }; return; }
    if (h?.t === "box") { const b = this.p.frames[this.idx][h.oid].box; this.drag = { t: "pend-move", oid: h.oid, x0: x, y0: y, b0: [...b], sx: e.clientX, sy: e.clientY }; return; }
    this.drag = { t: "click", x, y, sx: e.clientX, sy: e.clientY };
  }
  move(e) {
    const [x, y] = this.toImg(e);
    const d = this.drag;
    if (!d) {
      const h = this.hitTest(x, y);
      const k = h ? `${h.t}${h.oid}${h.i ?? h.c ?? ""}` : "";
      if (k !== this._hk) { this._hk = k; this.hover = h; this.draw(); }
      this.cv.style.cursor = !h ? "crosshair" : h.t === "pt" ? "grab" : h.t === "corner" ? (h.c % 2 ? "nesw-resize" : "nwse-resize") : "move";
      return;
    }
    e.stopPropagation();
    const [cx, cy] = this.clampPt([x, y]);
    const o = () => this.p.frames[this.idx]?.[d.oid];
    if (d.t === "pt") { const p = o().points[d.i]; p[0] = cx; p[1] = cy; this.cv.style.cursor = "grabbing"; }
    else if (d.t === "newbox") { const op = this.op(d.oid); op.box = [d.x0, d.y0, cx, cy]; }
    else if (d.t === "corner") {
      const b = o().box;
      if (d.c === 0) { b[0] = cx; b[1] = cy; } else if (d.c === 1) { b[2] = cx; b[1] = cy; } else if (d.c === 2) { b[2] = cx; b[3] = cy; } else { b[0] = cx; b[3] = cy; }
    } else if (d.t === "pend-move" || d.t === "move") {
      if (d.t === "pend-move" && Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < 4) return;
      d.t = "move";
      const [w, h] = this.size, b0 = d.b0;
      const dx = Math.max(-b0[0], Math.min(w - 1 - b0[2], x - d.x0)), dy = Math.max(-b0[1], Math.min(h - 1 - b0[3], y - d.y0));
      o().box = [b0[0] + dx, b0[1] + dy, b0[2] + dx, b0[3] + dy];
    } else return;
    this.draw();
  }
  up(e) {
    const d = this.drag;
    if (!d) return;
    e.stopPropagation();
    this.drag = null;
    try { this.cv.releasePointerCapture(e.pointerId); } catch {}
    if (d.t === "click" || (d.t === "pend-move" && Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < 4)) {
      const [x, y] = this.clampPt(d.t === "click" ? [d.x, d.y] : [d.x0, d.y0]);
      this.op().points.push([x, y, 1]);
    }
    if (d.t === "newbox" || d.t === "corner" || d.t === "move") {
      const op = this.p.frames[this.idx]?.[d.oid];
      if (op?.box) {
        const b = op.box; op.box = [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
        if (op.box[2] - op.box[0] < 3 || op.box[3] - op.box[1] < 3) op.box = null;
      }
      this.prune();
    }
    this.commit();
  }
  leave() { if (!this.drag && this.hover) { this.hover = null; this._hk = ""; this.draw(); } }

  // ---------- render
  drawKeys() {
    const c = this.keys, W = c.clientWidth, H = c.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!W) return;
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext("2d"); g.scale(dpr, dpr);
    g.fillStyle = "#222"; g.fillRect(0, 0, W, H);
    const n = Math.max(1, this.count - 1);
    for (const k of Object.keys(this.p.frames)) { g.fillStyle = "#ffcc33"; g.fillRect((k / n) * (W - 3), 0, 3, H); }
    g.fillStyle = "#fff"; g.fillRect((this.idx / n) * (W - 2), 0, 2, H);
  }
  marker(g, x, y, label, col, hov, sel) {
    const c = label ? POS : NEG, r = hov ? R + 1.5 : R;
    g.save();
    g.beginPath(); g.arc(x, y, r, 0, 7);
    g.fillStyle = rgba(c, hov ? 0.38 : 0.24); g.fill();
    g.lineWidth = 2.5; g.strokeStyle = "rgba(0,0,0,0.55)"; g.stroke();
    g.lineWidth = hov ? 1.8 : 1.3; g.strokeStyle = hov ? "#fff" : c; g.stroke();
    g.beginPath(); g.arc(x, y, 3.2, 0, 7); g.fillStyle = c; g.fill(); g.lineWidth = 1; g.strokeStyle = "rgba(0,0,0,0.8)"; g.stroke();
    if (this.p.objects.length > 1 || sel) { g.beginPath(); g.arc(x, y, r + 3, 0, 7); g.lineWidth = sel ? 1.6 : 1; g.strokeStyle = rgba(col, sel ? 0.95 : 0.6); g.stroke(); }
    g.restore();
  }
  // outline = mask edge contour, stroked in screen space: 1px halo each side of a 2px object-colour line
  drawOutline(g, t, dw, dh) {
    const sx = dw / this.mask.w, sy = dh / this.mask.h;
    g.save();
    g.lineCap = "round"; g.lineJoin = "round";
    for (const o of this.mask.paths) {
      const dpr = window.devicePixelRatio || 1;
      g.setTransform(dpr * sx, 0, 0, dpr * sy, dpr * t.ox, dpr * t.oy);
      const k = 1 / Math.min(sx, sy);
      g.lineWidth = 4 * k; g.strokeStyle = o.halo; g.stroke(o.path);
      g.lineWidth = 2 * k; g.strokeStyle = o.col; g.stroke(o.path);
    }
    g.restore();
  }
  draw() {
    const c = this.cv, W = this.view.clientWidth, H = this.view.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!this.img) return;
    const t = this.xf(), [w, h] = this.size, S = (x, y) => [t.ox + x * t.s, t.oy + y * t.s];
    g.imageSmoothingQuality = "high";
    g.drawImage(this.img, t.ox, t.oy, w * t.s, h * t.s);
    if (this.mask && this.showMask) {
      const dw = w * t.s, dh = h * t.s;
      g.save();
      g.imageSmoothingEnabled = false;
      if (this.opacity > 0) { g.globalAlpha = this.opacity; g.drawImage(this.mask.fill, t.ox, t.oy, dw, dh); }
      g.restore();
      if (this.outline) this.drawOutline(g, t, dw, dh);
    }
    const fr = this.p.frames[this.idx] || {};
    const hv = this.hover;
    for (const [ok, o] of Object.entries(fr)) {
      const oid = +ok, col = this.color(oid), sel = oid === this.obj;
      if (o.box) {
        const [x0, y0] = S(o.box[0], o.box[1]), [x1, y1] = S(o.box[2], o.box[3]);
        const hb = hv && hv.oid === oid && hv.t !== "pt";
        g.save();
        g.fillStyle = rgba(col, hb ? 0.14 : 0.06); g.fillRect(x0, y0, x1 - x0, y1 - y0);
        g.lineWidth = 3; g.strokeStyle = "rgba(0,0,0,0.6)"; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
        g.setLineDash(sel ? [] : [6, 4]); g.lineWidth = hb ? 2.4 : 1.8; g.strokeStyle = col; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
        g.setLineDash([]);
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].forEach(([cx, cy], i) => {
          const big = hv && hv.t === "corner" && hv.oid === oid && hv.c === i, s = big ? 5 : 3.5;
          g.fillStyle = "#fff"; g.fillRect(cx - s, cy - s, 2 * s, 2 * s); g.lineWidth = 1.5; g.strokeStyle = col; g.strokeRect(cx - s, cy - s, 2 * s, 2 * s);
        });
        g.restore();
      }
    }
    for (const [ok, o] of Object.entries(fr)) {
      const oid = +ok, col = this.color(oid);
      o.points.forEach((p, i) => { const [x, y] = S(p[0], p[1]); this.marker(g, x, y, p[2], col, hv && hv.t === "pt" && hv.oid === oid && hv.i === i, oid === this.obj); });
    }
  }

  // ---------- pop-out
  togglePopout() {
    if (this.float) {
      this.host.appendChild(this.root); this.float.remove(); this.float = null; this.ph?.remove();
      this.draw(); this.drawKeys(); return;
    }
    const f = this.float = el("div", "position:fixed;left:80px;top:80px;width:760px;height:600px;z-index:10000;background:#181818;border:1px solid #555;border-radius:6px;box-shadow:0 8px 30px #000a;display:flex;flex-direction:column;resize:both;overflow:hidden;min-width:300px;min-height:240px;", document.body);
    const hd = el("div", "flex:0 0 auto;padding:4px 8px;background:#262626;cursor:move;font:12px sans-serif;color:#ddd;display:flex;", f, `CM Point Editor #${this.node.id}`);
    const x = el("span", "margin-left:auto;cursor:pointer;padding:0 4px;", hd, "✕"); x.onclick = () => this.togglePopout();
    const body = el("div", "flex:1;min-height:0;padding:4px;", f);
    body.appendChild(this.root);
    this.ph = el("div", "width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#888;cursor:pointer;font:11px sans-serif;", this.host, "editor popped out — click to return");
    this.ph.onclick = () => this.togglePopout();
    hd.onpointerdown = (e) => {
      const ox = e.clientX - f.offsetLeft, oy = e.clientY - f.offsetTop;
      const mv = (ev) => { f.style.left = `${ev.clientX - ox}px`; f.style.top = `${Math.max(0, ev.clientY - oy)}px`; };
      const upf = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", upf); };
      window.addEventListener("pointermove", mv); window.addEventListener("pointerup", upf);
    };
    for (const ev of ["pointerdown", "wheel", "keydown"]) f.addEventListener(ev, (e) => e.stopPropagation());
    setTimeout(() => { this.draw(); this.drawKeys(); }, 0);
  }
  destroy() { this.stop(); clearInterval(this._poll); window.removeEventListener("cm-loader-changed", this._onLoader); this.float?.remove(); }
}

app.registerExtension({
  name: "CoordinationMatrix.PointEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      hideWidget(wv(this, "editor_data"));
      const ed = this.cmEditor = new Editor(this);
      this.addDOMWidget("cm_editor", "cm_editor", ed.host, { serialize: false, hideOnZoom: false, getMinHeight: () => 220 });
      const mk = wv(this, "multi_keyframe");
      if (mk) { const cb = mk.callback; mk.callback = function () { const o = cb?.apply(this, arguments); if (!mk.value) { const ks = Object.keys(ed.p.frames).map(Number).sort((a, b) => a - b); for (const k of ks.slice(1)) delete ed.p.frames[k]; ed.commit(); } ed.status(); ed.drawKeys(); return o; }; }
      ed.load();
      if (this.size[0] < 420) this.setSize([420, Math.max(this.size[1], 480)]);
      setTimeout(() => ed.refresh(), 0);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (this.cmEditor) { this.cmEditor.load(); setTimeout(() => this.cmEditor.refresh(), 50); }
      return r;
    };
    const onConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () { const r = onConn?.apply(this, arguments); this.cmEditor?.refresh(); return r; };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function () { const r = onExecuted?.apply(this, arguments); if (this.cmEditor?.src?.kind !== "file") this.cmEditor?.refresh(); return r; };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { this.cmEditor?.destroy(); return onRemoved?.apply(this, arguments); };
  },
});

// After CM Segment runs, tracked masks become available for non-keyframes.
api.addEventListener("executed", () => {
  for (const n of app.graph?._nodes || []) if (n.cmEditor) { const e = n.cmEditor; for (const k of [...e.mcache.keys()]) if (!e.p.frames[k]) e.mcache.delete(k); e.requestMask(); }
});
