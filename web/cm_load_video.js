// CM Load Video: one preview + visual trim timeline (in/out handles over a scrubber).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LOADER_IDS, LOADER_KEYS, wv, hideWidget, suppressCorePreview, snapTarget, loaderSettings, qs, fmtT, el } from "./cm_common.js";

const HANDLE = 7;

// Core video_upload only offers video/* (file-dialog filter + drop/paste mime
// check), so GIFs / animated WebP / APNG / stills picked via "All files",
// dropped, or pasted were rejected there. The timeline below accepts those
// itself and uploads them through /upload/image like the core widget does.
const ACCEPT_RE = /\.(gif|webp|apng|png|avif|jpe?g|bmp|tiff?|qoi|mp4|mov|mkv|webm|avi|m4v|mpg|mpeg|wmv|flv|ts|m2ts|mts|ogv|3gp|asf|vob|mxf)$/i;
const ACCEPT_ATTR = "video/*,image/gif,image/webp,image/apng,image/png,image/jpeg,image/avif,image/bmp,image/tiff,.gif,.webp,.apng,.png,.jpg,.jpeg,.avif,.bmp,.tif,.tiff,.qoi,.mp4,.mov,.mkv,.webm,.avi";

class Timeline {
  constructor(node) {
    this.node = node;
    this.info = null;          // /cm/plan result
    this.dur = 0;
    this.drag = null;
    this.root = el("div", "display:flex;flex-direction:column;width:100%;height:100%;min-height:0;gap:4px;box-sizing:border-box;font:11px sans-serif;color:#ddd;outline:none;");
    this.root.tabIndex = 0;
    this.root.addEventListener("keydown", (e) => this.key(e));
    const view = el("div", "position:relative;flex:1 1 auto;min-height:40px;background:#0c0c0c;border-radius:4px;overflow:hidden;", this.root);
    this.video = el("video", "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;", view);
    Object.assign(this.video, { muted: false, loop: false, playsInline: true, preload: "auto" });
    // <video> cannot play GIF / animated WebP / APNG / stills: show those through <img>.
    this.img = el("img", "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none;", view);
    this.imgMode = false;
    this.t = 0;            // virtual playhead (s) while in image mode
    this.playing = false;  // image-mode playback flag
    this._last = 0;
    this.img.addEventListener("load", () => { this.empty.style.display = "none"; this.seek(this.win()[0]); this.draw(); });
    this.img.addEventListener("pointerdown", () => { try { this.root.focus({ preventScroll: true }); } catch {} });
    this.empty = el("div", "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#777;", view, "upload, drop, or choose a video / gif");
    this.tl = el("canvas", "width:100%;height:44px;flex:0 0 44px;cursor:pointer;border-radius:3px;touch-action:none;", this.root);
    const row = el("div", "display:flex;gap:4px;align-items:center;flex:0 0 auto;", this.root);
    const btn = (t, tip, fn) => { const b = el("button", "padding:1px 6px;font:11px sans-serif;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;cursor:pointer;", row, t); b.title = tip; b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
    this.playBtn = btn("▶", "Play / pause the trimmed window", () => this.toggle());
    btn("[ in", "Set window start at the playhead", () => this.setIn(this.cur()));
    btn("out ]", "Set window end at the playhead", () => this.setOut(this.cur()));
    btn("reset", "Full clip", () => { this.set("start", 0); this.set("trim", 0); this.changed(); });
    this.upBtn = btn("⇪ gif/img", "Upload a GIF / animated image / still (or video) into input and select it", () => this.fileInput.click());
    this.fileInput = el("input", "display:none;", this.root);
    Object.assign(this.fileInput, { type: "file", multiple: true, accept: ACCEPT_ATTR });
    this.fileInput.addEventListener("change", () => { const fs = [...(this.fileInput.files || [])]; this.fileInput.value = ""; this.uploadFiles(fs); });
    this.root.addEventListener("dragover", (e) => { if ([...(e.dataTransfer?.types || [])].includes("Files")) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
    this.root.addEventListener("drop", (e) => {
      const fs = [...(e.dataTransfer?.files || [])].filter((f) => ACCEPT_RE.test(f.name || ""));
      if (!fs.length) return;
      e.preventDefault(); e.stopPropagation();
      this.uploadFiles(fs);
    });
    this.root.addEventListener("paste", (e) => {
      const fs = [...(e.clipboardData?.files || [])].filter((f) => ACCEPT_RE.test(f.name || ""));
      if (!fs.length) return;
      e.preventDefault(); e.stopPropagation();
      this.uploadFiles(fs);
    });
    this.label = el("div", "flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right;color:#aaa;", row);
    this.tl.addEventListener("pointerdown", (e) => { try { this.root.focus({ preventScroll: true }); } catch {} this.down(e); });
    this.video.addEventListener("pointerdown", () => { try { this.root.focus({ preventScroll: true }); } catch {} });
    this.tl.addEventListener("pointermove", (e) => this.move(e));
    this.tl.addEventListener("pointerup", (e) => this.up(e));
    this.tl.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
    this.video.addEventListener("loadedmetadata", () => { this.empty.style.display = "none"; this.seek(this.win()[0]); this.draw(); });
    this.video.addEventListener("timeupdate", () => this.loopGuard());
    this.video.addEventListener("play", () => { this.playBtn.textContent = "❚❚"; this.anim(); });
    this.video.addEventListener("pause", () => { this.playBtn.textContent = "▶"; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(this.tl);
  }

  // --- model
  // Browsers play real video in <video> but GIF / animated WebP / APNG / stills need <img>.
  isImageFile(f) { return /\.(gif|webp|apng|png|avif|jpe?g|bmp|tiff?|qoi)$/i.test(f || ""); }
  cur() { return this.imgMode ? (this.t || 0) : this.video.currentTime; }
  get fps() { const f = Number(wv(this.node, "fps")?.value) || 0; return f > 0 ? f : (this.info?.src_fps || 24); }
  win() {
    const s = Math.max(0, Number(wv(this.node, "start")?.value) || 0);
    const tr = Number(wv(this.node, "trim")?.value) || 0;
    const end = this.dur || s + tr;
    return [Math.min(s, end), tr > 0 ? Math.min(end, s + tr) : end];
  }
  set(name, v) {
    const w = wv(this.node, name);
    if (!w) return;
    w.value = Math.round(v * 1000) / 1000;
    w.callback?.(w.value);
  }
  setIn(t) { const [, b] = this.win(); t = Math.max(0, Math.min(t, b - 1 / this.fps)); this.set("start", t); this.set("trim", Math.max(0, b - t)); this.changed(); }
  setOut(t) { const [a] = this.win(); t = Math.max(a + 1 / this.fps, Math.min(t, this.dur || t)); this.set("trim", t >= this.dur - 1e-3 ? 0 : t - a); this.changed(); }
  changed() { this.node.setDirtyCanvas(true, true); this.refreshPlan(); window.dispatchEvent(new CustomEvent("cm-loader-changed", { detail: { id: this.node.id } })); }

  // --- media
  fileChanged() {
    const f = wv(this.node, "file")?.value;
    this.info = null; this.dur = 0;
    this.playing = false; this.playBtn.textContent = "▶";
    if (!f) {
      this.video.removeAttribute("src"); this.img.removeAttribute("src");
      this.video.style.display = ""; this.img.style.display = "none"; this.imgMode = false;
      this.empty.style.display = "flex"; this.draw(); return;
    }
    const i = f.lastIndexOf("/");
    const p = { filename: i >= 0 ? f.slice(i + 1) : f, subfolder: i >= 0 ? f.slice(0, i) : "", type: "input" };
    this.imgMode = this.isImageFile(f);
    this.video.style.display = this.imgMode ? "none" : "";
    this.img.style.display = this.imgMode ? "" : "none";
    if (this.imgMode) {
      try { this.video.pause(); } catch {}
      this.video.removeAttribute("src");
      this.img.src = api.apiURL(`/view?${qs(p)}`);
      this.empty.style.display = "flex";
    } else {
      this.img.removeAttribute("src");
      this.video.src = api.apiURL(`/view?${qs(p)}`);
    }
    this.refreshPlan();
  }
  // --- gif/animated-image upload (bypasses the core video/*-only widget)
  async uploadFiles(list) {
    const files = [...(list || [])].filter((f) => f && ACCEPT_RE.test(f.name || ""));
    if (!files.length) { console.log("[CM] uploadFiles: no matching files"); return; }
    if (this._uploading) return;
    this._uploading = true; this.upBtn.disabled = true;
    this.label.textContent = `uploading ${files[0].name}…`;
    console.log("[CM] uploading", files.map((f) => f.name));
    try {
      const post = (fd) => api.fetchApi
        ? api.fetchApi("/upload/image", { method: "POST", body: fd })
        : fetch("/upload/image", { method: "POST", body: fd });
      let first = null;
      for (const f of files) {
        const fd = new FormData();
        fd.append("image", f);
        fd.append("type", "input");
        let r = null;
        try { r = await post(fd); } catch (err) { console.warn("[CM] upload request failed", f.name, err); r = null; }
        if (!r || !r.ok) { console.warn("[CM] upload failed", f.name, r?.status); continue; }
        const j = await r.json().catch(() => null);
        console.log("[CM] upload response", j?.name, j?.subfolder);
        if (!j?.name) continue;
        if (!first) first = j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
      }
      if (first) this.chooseFile(first);
      else this.label.textContent = "upload failed (see console)";
    } finally { this._uploading = false; this.upBtn.disabled = false; }
  }
  chooseFile(p) {
    console.log("[CM] chooseFile", p);
    const w = wv(this.node, "file");
    if (!w) { console.warn("[CM] no file widget"); return; }
    if (Array.isArray(w.options)) { if (!w.options.includes(p)) w.options.push(p); }
    else if (w.options && Array.isArray(w.options.values)) { if (!w.options.values.includes(p)) w.options.values.push(p); }
    else console.log("[CM] widget options shape unknown, setting value anyway");
    w.value = p;
    try { w.callback?.(p); } catch (err) { console.warn("[CM] file callback failed", err); }
    this.fileChanged();
  }
  refreshPlan() {
    clearTimeout(this._pt);
    this._pt = setTimeout(async () => {
      const q = loaderSettings(this.node);
      if (!q.file) return;
      try {
        const r = await api.fetchApi(`/coordmatrix/plan?${qs(q)}`);
        this.info = r.ok ? await r.json() : null;
        if (this.info) this.dur = this.info.src_duration || this.video.duration || 0;
      } catch { this.info = null; }
      this.draw();
    }, 150);
  }
  toggle() {
    if (this.imgMode) {
      if (this.playing) { this.playing = false; this.playBtn.textContent = "▶"; this.draw(); }
      else { this.seekIfOutside(); this.playing = true; this.playBtn.textContent = "❚❚"; this._last = performance.now(); this.tick(); }
      return;
    }
    if (this.video.paused) { this.seekIfOutside(); this.video.play().catch(() => {}); } else this.video.pause();
  }
  tick() {
    if (!this.playing || !this.imgMode) return;
    const now = performance.now(), dt = (now - this._last) / 1000; this._last = now;
    const [a, b] = this.win();
    let t = this.cur() + dt;
    if (t >= b || t < a) t = a;
    this.t = t; this.draw();
    requestAnimationFrame(() => this.tick());
  }
  // Arrow keys skip at the locked H3 distance (17 frames) while the trimmer has focus.
  // LiteGraph never sees the keystroke, so the selected node is not nudged.
  key(e) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const t = e.target;
    if (t instanceof Element && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return; // typing in a widget
    if (!this.dur) return;
    e.stopPropagation(); e.preventDefault();
    const step = 17 / Math.max(this.fps, 1e-6);
    this.video.pause();
    const nt = Math.max(0, Math.min(this.dur, this.cur() + (e.key === "ArrowRight" ? step : -step)));
    this.seek(nt);
  }
  seek(t) { if (isFinite(t)) { if (this.imgMode) this.t = t; else try { this.video.currentTime = t; } catch {} } this.draw(); }
  seekIfOutside() { const [a, b] = this.win(); const t = this.cur(); if (t < a || t >= b - 1e-3) this.seek(a); }
  loopGuard() { if (this.video.paused) return; const [a, b] = this.win(); if (this.video.currentTime >= b || this.video.currentTime < a - 0.05) this.video.currentTime = a; }
  anim() { this.draw(); if (!this.video.paused) requestAnimationFrame(() => this.anim()); }

  // --- timeline geometry
  x2t(x) { const r = this.tl.getBoundingClientRect(); const u = (x - r.left - HANDLE) / Math.max(1, r.width - 2 * HANDLE); return Math.max(0, Math.min(1, u)) * (this.dur || 0); }
  t2x(t, W) { return HANDLE + (this.dur ? t / this.dur : 0) * (W - 2 * HANDLE); }
  hit(e) {
    const r = this.tl.getBoundingClientRect(), x = e.clientX - r.left, [a, b] = this.win();
    const xa = this.t2x(a, r.width), xb = this.t2x(b, r.width);
    if (Math.abs(x - xa) <= HANDLE + 2) return "in";
    if (Math.abs(x - xb) <= HANDLE + 2) return "out";
    if (x > xa && x < xb && e.clientY - r.top < 16) return "window";
    return "seek";
  }
  down(e) {
    e.stopPropagation(); e.preventDefault();
    if (!this.dur) return;
    this.tl.setPointerCapture(e.pointerId);
    const kind = this.hit(e), [a, b] = this.win();
    this.drag = { kind, t0: this.x2t(e.clientX), a, b };
    if (kind === "seek") { this.video.pause(); this.seek(this.x2t(e.clientX)); }
  }
  move(e) {
    if (!this.drag) { this.tl.style.cursor = { in: "ew-resize", out: "ew-resize", window: "grab", seek: "pointer" }[this.hit(e)]; return; }
    e.stopPropagation();
    const t = this.x2t(e.clientX), d = this.drag, step = 1 / this.fps;
    const q = (v) => Math.round(v / step) * step;
    if (d.kind === "in") { const s = Math.max(0, Math.min(q(t), d.b - step)); this.set("start", s); this.set("trim", d.b >= this.dur - 1e-3 ? 0 : d.b - s); this.seek(s); }
    else if (d.kind === "out") { const o = Math.max(d.a + step, Math.min(q(t), this.dur)); this.set("trim", o >= this.dur - 1e-3 ? 0 : o - d.a); this.seek(Math.max(d.a, o - step)); }
    else if (d.kind === "window") { const len = d.b - d.a; const s = Math.max(0, Math.min(q(d.a + t - d.t0), this.dur - len)); this.set("start", s); if (Number(wv(this.node, "trim")?.value) > 0) this.set("trim", len); this.seek(s); }
    else this.seek(t);
    this.draw();
  }
  up(e) { if (!this.drag) return; e.stopPropagation(); const k = this.drag.kind; this.drag = null; if (k !== "seek") this.changed(); }

  // --- render
  frames() {
    const [a, b] = this.win();
    const n = Math.max(1, Math.ceil((b - a) * this.fps - 1e-6));
    const ext = Number(wv(this.node, "extend")?.value) || 0;
    const orig = ext > 0 ? Math.max(n + 1, Math.round((n / this.fps + ext) * this.fps)) : n;
    const mode = wv(this.node, "h3_snap_mode")?.value || "down";
    return { n, orig, len: this.info?.length ?? snapTarget(orig, mode) };
  }
  draw() {
    const c = this.tl, dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.fillStyle = "#1b1b1b"; g.fillRect(0, 0, W, H);
    const tw = W - 2 * HANDLE;
    g.fillStyle = "#2a2a2a"; g.fillRect(HANDLE, 14, tw, H - 22);
    if (!this.dur) { this.label.textContent = wv(this.node, "file")?.value ? "probing…" : ""; return; }
    // seconds ruler
    g.fillStyle = "#666"; g.font = "9px sans-serif";
    const stepS = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => (s / this.dur) * tw > 38) || 600;
    for (let s = 0; s <= this.dur; s += stepS) { const x = this.t2x(s, W); g.fillRect(x, 10, 1, 4); g.fillText(fmtT(s).replace(/^0:/, ""), x + 2, 9); }
    const [a, b] = this.win(), f = this.frames();
    const xa = this.t2x(a, W), xb = this.t2x(b, W);
    // kept window after the 17k+5 snap: green = kept, red = trimmed tail, amber = padded
    const keepEnd = Math.min(b, a + Math.min(f.len, f.n) / this.fps);
    g.fillStyle = "rgba(80,200,120,0.45)"; g.fillRect(xa, 14, this.t2x(keepEnd, W) - xa, H - 22);
    if (keepEnd < b) { g.fillStyle = "rgba(230,80,80,0.45)"; g.fillRect(this.t2x(keepEnd, W), 14, xb - this.t2x(keepEnd, W), H - 22); }
    if (f.len > f.n) { const px = ((f.len - f.n) / this.fps / this.dur) * tw; g.fillStyle = "rgba(240,180,60,0.5)"; g.fillRect(xb, 14, Math.min(px, W - HANDLE - xb), H - 22); }
    g.strokeStyle = "#9fe3b4"; g.lineWidth = 1; g.strokeRect(xa + 0.5, 14.5, xb - xa - 1, H - 23);
    for (const [x, lbl] of [[xa, "in"], [xb, "out"]]) {
      g.fillStyle = "#e8e8e8"; g.beginPath(); g.roundRect?.(x - HANDLE / 2, 12, HANDLE, H - 18, 2) ?? g.rect(x - HANDLE / 2, 12, HANDLE, H - 18); g.fill();
      g.fillStyle = "#333"; g.fillRect(x - 0.5, 18, 1, H - 30);
    }
    const t = this.cur() || 0, xp = this.t2x(t, W);
    g.fillStyle = "#ffcc33"; g.fillRect(xp - 1, 8, 2, H - 10);
    g.beginPath(); g.moveTo(xp - 4, 6); g.lineTo(xp + 4, 6); g.lineTo(xp, 11); g.fill();
    const wh = this.info ? `${this.info.width}×${this.info.height}` : "";
    const snap = f.len === f.orig ? `${f.len}f` : `${f.orig}→${f.len}f`;
    this.label.textContent = `${fmtT(a)} – ${fmtT(b)} · ${snap} @${+this.fps.toFixed(3)} (${(f.len / this.fps).toFixed(2)}s) · ${wh}`;
  }
}

app.registerExtension({
  name: "CoordinationMatrix.LoadVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!LOADER_IDS.has(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const sweep = suppressCorePreview(this);
      const tl = new Timeline(this);
      this.cmTimeline = tl;
      // Start/Trim live on the visual timeline only (no numeric widgets).
      hideWidget(wv(this, "start")); hideWidget(wv(this, "trim"));
      this.addDOMWidget("cm_timeline", "cm_timeline", tl.root, { serialize: false, hideOnZoom: false, getMinHeight: () => 120 });
      for (const k of LOADER_KEYS) {
        const w = wv(this, k);
        if (!w) continue;
        const cb = w.callback;
        w.callback = function () {
          const o = cb?.apply(this, arguments);
          if (k === "file") { tl.fileChanged(); sweep(); }
          else { tl.refreshPlan(); tl.draw(); }
          window.dispatchEvent(new CustomEvent("cm-loader-changed", { detail: { id: tl.node.id } }));
          return o;
        };
      }
      if (this.size[0] < 360) this.setSize([360, Math.max(this.size[1], 460)]);
      setTimeout(() => tl.fileChanged(), 0);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () { const r = onConfigure?.apply(this, arguments); hideWidget(wv(this, "start")); hideWidget(wv(this, "trim")); setTimeout(() => this.cmTimeline?.fileChanged(), 0); return r; };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { try { this.cmTimeline.playing = false; this.cmTimeline?.video.pause(); this.cmTimeline?.video.removeAttribute("src"); this.cmTimeline?.img.removeAttribute("src"); } catch {} return onRemoved?.apply(this, arguments); };
  },
});
