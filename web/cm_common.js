// Shared helpers for CoordinationMatrix widgets (LiteGraph + addDOMWidget, no Vue).
export const LOADER = "CMLoadVideo";
// CuteAndFunny's Load Video (Versatile) now runs the same megapixels engine,
// so the point editor can read either loader's timeline.
export const LOADER_CF = "CuteAndFunnyLoadVideo";
export const LOADER_IDS = new Set([LOADER, LOADER_CF]);
export const LOADER_KEYS = ["file", "fps", "start", "trim", "extend", "megapixels",
  "resize_filter", "frame_interp", "eof_mode", "h3_snap_mode", "h3_pad"];

export const wv = (node, name) => node.widgets?.find((w) => w.name === name);

export function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
  w.type = "hidden";
}

// --- the core video upload widget (io.UploadType.video) adds a "video-preview" DOM widget whose
// computeLayoutSize follows the video's size. We draw our own single view, so neuter it.
const CORE_PREVIEW = "video-preview";
function neuter(w) {
  if (!w || w.__cmNeutered) return w;
  w.__cmNeutered = true;
  const zero = () => ({ minHeight: 0, minWidth: 0 });
  Object.defineProperty(w, "computeLayoutSize", { get: () => zero, set: () => {}, configurable: true });
  w.computeSize = () => [0, -4];
  w.hidden = true;
  if (w.element) { w.element.style.display = "none"; w.element.querySelectorAll?.("video").forEach((v) => v.pause()); }
  return w;
}
export function suppressCorePreview(node) {
  const add = node.addDOMWidget;
  node.addDOMWidget = function (name, type, element) {
    if (name === CORE_PREVIEW && element) element.style.display = "none";
    const w = add.apply(this, arguments);
    return name === CORE_PREVIEW ? neuter(w) : w;
  };
  const sweep = () => { for (const w of node.widgets || []) if (w.name === CORE_PREVIEW) neuter(w); };
  sweep();
  const odb = node.onDrawBackground;
  node.onDrawBackground = function () {
    this.imgs = undefined;
    if (this.videoContainer) this.videoContainer.style.display = "none";
    sweep();
    return odb?.apply(this, arguments);
  };
  return sweep;
}

// --- H3 17k+5 grid (mirrors cm/h3.py)
export const isGrid = (n) => n >= 5 && n % 17 === 5;
export function snapTarget(n, mode) {
  n = Math.max(1, n | 0);
  if (n <= 5) return 5;
  const down = 5 + Math.floor((n - 5) / 17) * 17;
  if (down === n) return n;
  const up = down + 17;
  if (mode === "up") return up;
  if (mode === "nearest") return (n - down) <= (up - n) ? down : up;
  return down;
}

export function loaderSettings(node) {
  const q = {};
  for (const k of LOADER_KEYS) { const w = wv(node, k); if (w) q[k] = w.value; }
  return q;
}
export const qs = (o) => Object.entries(o).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v ?? "")}`).join("&");
export const fmtT = (t) => { t = Math.max(0, t || 0); const m = Math.floor(t / 60); return `${m}:${(t - m * 60).toFixed(2).padStart(5, "0")}`; };

export function el(tag, css, parent, text) {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text != null) e.textContent = text;
  parent?.appendChild(e);
  return e;
}

// Letterboxed content rect of a canvas drawn with objectFit: contain.
export function contentRect(canvas, w, h) {
  const r = canvas.getBoundingClientRect();
  if (!w || !h || !r.width || !r.height) return { x: r.left, y: r.top, w: r.width, h: r.height, s: 1 };
  const s = Math.min(r.width / w, r.height / h);
  return { x: r.left + (r.width - w * s) / 2, y: r.top + (r.height - h * s) / 2, w: w * s, h: h * s, s };
}
