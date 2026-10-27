"""PromptServer routes backing the CM widget (frame fetch + live single-frame mask preview)."""
from __future__ import annotations

import io
import logging
import os

import numpy as np

from . import prompts as P
from . import store, video_io

log = logging.getLogger("CoordinationMatrix")
_REGISTERED = [False]
DEFAULT_PREVIEW_MODEL = ("sam2.1_hiera_base_plus", "bf16")


def _resolve_file(name: str) -> str | None:
    import folder_paths

    if not name:
        return None
    try:
        p = folder_paths.get_annotated_filepath(name)
    except Exception:
        p = None
    if not p or not os.path.isfile(p):
        return None
    # only serve inside ComfyUI's input/output/temp dirs
    real = os.path.realpath(p)
    roots = [folder_paths.get_input_directory(), folder_paths.get_output_directory(),
             folder_paths.get_temp_directory()]
    if not any(real.startswith(os.path.realpath(r) + os.sep) for r in roots):
        return None
    return p


def _jpeg(arr: np.ndarray, q: int = 88) -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=q)
    return b.getvalue()


def _png_rgba(arr: np.ndarray) -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(arr, mode="RGBA").save(b, format="PNG", compress_level=1)
    return b.getvalue()


def _int(q, k, d=0):
    try:
        return int(float(q.get(k, d)))
    except (TypeError, ValueError):
        return d


def _float(q, k, d=0.0):
    try:
        return float(q.get(k, d))
    except (TypeError, ValueError):
        return d


_PLANS: dict = {}


def _settings(q) -> dict:
    return {"file": q.get("file", ""), "fps": _float(q, "fps", 24.0), "start": _float(q, "start", 0.0),
            "trim": _float(q, "trim", 0.0), "extend": _float(q, "extend", 0.0),
            "megapixels": _float(q, "megapixels", 0.4),
            "resize_filter": q.get("resize_filter", "bicubic") or "bicubic",
            "h3_snap_mode": q.get("h3_snap_mode", "down") or "down", "h3_pad": q.get("h3_pad", "reflect") or "reflect",
            "frame_interp": q.get("frame_interp", "blend") or "blend",
            "eof_mode": q.get("eof_mode", "hold") or "hold"}


def plan_for(q):
    """Same plan CM Load Video executes with -> (path, plan, meta, out size)."""
    st = _settings(q)
    path = _resolve_file(st["file"])
    if path is None:
        return None
    key = (path, os.path.getmtime(path), st["fps"], st["start"], st["trim"], st["extend"], st["h3_snap_mode"],
           st["h3_pad"], st.get("eof_mode", "hold"))
    hit = _PLANS.get(key)
    if hit is None:
        meta = video_io.probe(path)
        plan = video_io.plan_file(meta, st["fps"], st["start"], st["trim"], st["extend"], st["h3_snap_mode"],
                                  st["h3_pad"], st.get("eof_mode", "hold"))
        hit = (meta, plan)
        if len(_PLANS) > 32:
            _PLANS.clear()
        _PLANS[key] = hit
    meta, plan = hit
    w, h = video_io.scale_megapixels(meta["width"], meta["height"], st["megapixels"], 32)
    return path, plan, meta, (w, h), st


def _frame_from_request(q) -> np.ndarray | None:
    """By node id (frames stored on execute) or by file + CM Load Video settings + output index."""
    idx = _int(q, "idx", 0)
    if q.get("file"):
        r = plan_for(q)
        if r is None:
            return None
        path, plan, _, _, st = r
        idx = max(0, min(idx, plan["length"] - 1))
        t = plan["times"][plan["index"][idx]]
        return video_io.decode_single_frame(path, t, st["megapixels"], st["resize_filter"],
                                            st.get("frame_interp", "blend"), st.get("eof_mode", "hold"))
    node = q.get("node")
    if node:
        s = store.get_frames(node)
        if s is not None:
            fr = s["frames"]
            return fr[max(0, min(idx, fr.shape[0] - 1))].numpy()
    return None


def register():
    if _REGISTERED[0]:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = getattr(PromptServer, "instance", None)
    if inst is None:
        return
    r = inst.routes
    _REGISTERED[0] = True

    @r.get("/coordmatrix/info")
    async def cm_info(request):
        path = _resolve_file(request.rel_url.query.get("file", ""))
        if path is None:
            return web.json_response({"error": "file not found"}, status=404)
        try:
            info = video_io.probe(path)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(info)

    @r.get("/coordmatrix/plan")
    async def cm_plan(request):
        q = request.rel_url.query
        try:
            if q.get("file"):
                r = plan_for(q)
                if r is None:
                    return web.json_response({"error": "file not found"}, status=404)
                _, plan, meta, (w, h), _ = r
                return web.json_response({"length": plan["length"], "original_frames": plan["original_frames"],
                                          "window_frames": plan["window_frames"], "fps": plan["fps"],
                                          "width": w, "height": h, "src_fps": meta["src_fps"],
                                          "src_duration": meta["duration"], "padded": plan["padded"]})
            s = store.get_frames(q.get("node", ""))
            if s is None:
                return web.json_response({"error": "no frames yet"}, status=404)
            fr = s["frames"]
            return web.json_response({"length": int(fr.shape[0]), "width": int(fr.shape[2]),
                                      "height": int(fr.shape[1])})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @r.get("/coordmatrix/frame")
    async def cm_frame(request):
        q = request.rel_url.query
        try:
            arr = _frame_from_request(q)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
        if arr is None:
            return web.json_response({"error": "no frame"}, status=404)
        return web.Response(body=_jpeg(arr), content_type="image/jpeg",
                            headers={"Cache-Control": "no-store", "X-CM-Width": str(arr.shape[1]),
                                     "X-CM-Height": str(arr.shape[0])})

    @r.get("/coordmatrix/mask")
    async def cm_mask(request):
        """Tracked mask for a frame after CM Propagate ran (node = source node id)."""
        q = request.rel_url.query
        tr = store.get_track(q.get("node", ""))
        if tr is None:
            return web.json_response({"error": "no track"}, status=404)
        m = tr["masks"]
        idx = max(0, min(_int(q, "idx", 0), m.shape[0] - 1))
        return web.Response(body=_png_rgba(_rgba(m[idx].numpy(), tr["colors"])), content_type="image/png",
                            headers={"Cache-Control": "no-store"})

    @r.post("/coordmatrix/preview_mask")
    async def cm_preview(request):
        import asyncio

        body = await request.json()
        try:
            arr = _frame_from_request(body)
            if arr is None:
                return web.json_response({"error": "no frame"}, status=404)
            p = P.parse(body.get("prompts") or {})
            idx = _int(body, "idx", 0)
            h, w = arr.shape[:2]
            fp = P.scaled(p, w, h).get(idx, {})
            if not fp:
                return web.Response(body=_png_rgba(np.zeros((h, w, 4), np.uint8)), content_type="image/png")
            ids = sorted(fp)
            cols = P.colors(p, ids)
            # Content-based key shared with engine.predict_frame so preview and
            # Segment keyframe masks hit the same store._PREV cache.
            pk = store.kf_cache_key(arr, fp, _preview_model_id())
            res = store.preview_get(pk)
            if res is None:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(None, _predict_blocking, arr, fp)
                # _predict_blocking (engine.predict_frame) already caches under the
                # same key; keep this put for older engine results.
                if store.preview_get(pk) is None:
                    store.preview_put(pk, res)
            stack = np.stack([res[i][0] for i in ids]) if ids else np.zeros((0, h, w), bool)
            return web.Response(body=_png_rgba(_rgba(stack, cols)), content_type="image/png",
                                headers={"Cache-Control": "no-store"})
        except Exception as e:
            log.exception("[CoordinationMatrix] preview failed")
            return web.json_response({"error": str(e)}, status=500)


def _frame_id(q, arr) -> str:
    """Identity of the preview frame: file + mtime + loader settings + idx, or a content hash."""
    import hashlib

    if q.get("file"):
        st = _settings(q)
        path = _resolve_file(st["file"]) or st["file"]
        try:
            mt = os.path.getmtime(path)
        except OSError:
            mt = 0
        return f"{path}:{mt}:{sorted(st.items())}:{_int(q, 'idx', 0)}"
    return hashlib.sha1(np.ascontiguousarray(arr[::4, ::4]).tobytes() + str(arr.shape).encode()).hexdigest()


def _preview_model_id() -> str:
    from . import model

    m = model.last_model()
    return f"{getattr(m, 'name', '')}:{getattr(m, 'precision', '')}" if m else f"default:{DEFAULT_PREVIEW_MODEL}"


def _predict_blocking(arr, fp):
    from . import engine, model

    with model.lock():
        cm = model.last_model() or model.get_model(*DEFAULT_PREVIEW_MODEL)
        return engine.predict_frame(cm, arr, fp)


def _rgba(masks: np.ndarray, colors: list[str]) -> np.ndarray:
    """masks [O,H,W] bool -> RGBA uint8 solid object colour (alpha 255); opacity/outline are drawn client-side."""
    masks = np.asarray(masks).astype(bool)
    h, w = masks.shape[-2:]
    out = np.zeros((h, w, 4), np.uint8)
    for k in range(masks.shape[0]):
        r, g, b = (int(c * 255) for c in P.hex_to_rgb(colors[k] if k < len(colors) else P.color_for(k + 1)))
        out[masks[k]] = (r, g, b, 255)
    return out
