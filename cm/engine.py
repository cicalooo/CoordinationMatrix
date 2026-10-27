"""SAM 2.1 inference: single-frame prediction + chunked, bidirectional propagation."""
from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from . import prompts as P

log = logging.getLogger("CoordinationMatrix")

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _interrupt():
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _progress(total):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total)
    except Exception:
        class _N:
            def update(self, *_):
                pass

        return _N()


# ------------------------------------------------------------ single frame

def _to_u8(image):
    if isinstance(image, torch.Tensor):
        return (image.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return image


def predict_frame(cm, image, frame_prompts: dict, multimask: bool = True):
    """image HWC (float tensor or uint8 array); frame_prompts {obj_id: {points, box}}.

    multimask=True always asks SAM for 3 candidates and keeps the best predicted IoU.
    Content-cached in store._PREV so live preview and propagation share keyframes.
    Returns {obj_id: (bool mask HxW np, iou score float)}.
    """
    from .vendor.sam2.sam2_image_predictor import SAM2ImagePredictor

    from . import store as _store

    arr = _to_u8(image)
    h, w = arr.shape[:2]
    frame_prompts = _clamp_prompt_dict(frame_prompts or {}, w, h)
    if not frame_prompts:
        return {}
    key = None
    try:
        key = _store.kf_cache_key(arr, frame_prompts, _model_id(cm))
        hit = _store.preview_get(key)
        if hit is not None:
            ok = all(getattr(v[0], "shape", None) == (h, w) for v in hit.values())
            if ok and set(int(k) for k in hit) == set(int(k) for k in frame_prompts):
                return {int(k): (np.asarray(v[0]).astype(bool).copy(), float(v[1])) for k, v in hit.items()}
    except Exception:
        key = None
    net = cm.load()
    pred = SAM2ImagePredictor(net)
    out = {}
    with torch.inference_mode(), cm.autocast():
        pred.set_image(arr)
        for oid, p in frame_prompts.items():
            pts = np.array([[x, y] for x, y, _ in p["points"]], dtype=np.float32) if p["points"] else None
            lbl = np.array([l for _, _, l in p["points"]], dtype=np.int32) if p["points"] else None
            box = np.array(p["box"], dtype=np.float32) if p["box"] is not None else None
            masks, scores, _ = pred.predict(point_coords=pts, point_labels=lbl, box=box,
                                            multimask_output=bool(multimask))
            k = int(np.argmax(scores))
            out[int(oid)] = (masks[k] > 0.5, float(scores[k]))
    pred.reset_predictor()
    try:
        if key is not None:
            _store.preview_put(key, {k: (v[0].copy(), float(v[1])) for k, v in out.items()})
    except Exception:
        pass
    return out


def _nearest_key(keys, i):
    prev = [k for k in keys if k <= i]
    return prev[-1] if prev else None


def _prompts_for_frame(frames: dict, i: int) -> dict:
    """Per-object latest keyframe <= i (image mode). Objects with no past key stay empty."""
    out: dict = {}
    for g in sorted(frames):
        if g > i:
            break
        for oid, pr in frames[g].items():
            out[int(oid)] = pr
    return out


def _model_id(cm) -> str:
    return f"{getattr(cm, 'name', '')}:{getattr(cm, 'precision', '')}"


def _clamp_prompt_dict(frame_prompts: dict, w: int, h: int) -> dict:
    out = {}
    for oid, p in (frame_prompts or {}).items():
        pts = []
        for x, y, lab in p.get("points") or []:
            try:
                fx, fy = float(x), float(y)
            except (TypeError, ValueError):
                continue
            import math
            if not math.isfinite(fx) or not math.isfinite(fy):
                continue
            fx = min(max(fx, 0.0), float(w - 1) if w > 1 else 0.0)
            fy = min(max(fy, 0.0), float(h - 1) if h > 1 else 0.0)
            try:
                li = int(lab)
            except (TypeError, ValueError):
                li = 1
            pts.append([fx, fy, 1 if li else 0])
        box = p.get("box")
        if box is not None:
            try:
                x0, y0, x1, y1 = [float(b) for b in box]
                lo_x, hi_x = min(x0, x1), max(x0, x1)
                lo_y, hi_y = min(y0, y1), max(y0, y1)
                lo_x = min(max(lo_x, 0.0), float(w - 1) if w > 1 else 0.0)
                lo_y = min(max(lo_y, 0.0), float(h - 1) if h > 1 else 0.0)
                hi_x = min(max(hi_x, 0.0), float(w))
                hi_y = min(max(hi_y, 0.0), float(h))
                if hi_x - lo_x < 1 or hi_y - lo_y < 1:
                    box = None
                else:
                    box = [lo_x, lo_y, hi_x, hi_y]
            except (TypeError, ValueError):
                box = None
        if pts or box is not None:
            out[int(oid)] = {"points": [[float(x), float(y), int(v)] for x, y, v in pts], "box": box}
    return out


def segment_images(cm, images: torch.Tensor, prompt: dict, return_info: bool = False):
    """Per-frame image segmentation (no tracking). Frame i uses the latest keyframe
    at or before i, per object; frames before an object's first key stay empty.
    Returns (masks bool [N,O,H,W], ids) or (masks, ids, info) with return_info=True."""
    n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    frames, warnings = P.scale_and_clamp(prompt, w, h, n)
    ids = P.object_ids({"objects": prompt.get("objects", []), "frames": frames})
    if not ids:
        raise ValueError("[CM Segment] no points or boxes placed")
    col = {oid: i for i, oid in enumerate(ids)}
    masks = torch.zeros((n, len(ids), h, w), dtype=torch.bool)
    scores = torch.zeros((n, len(ids)), dtype=torch.float32)
    pbar = _progress(n)
    for i in range(n):
        _interrupt()
        fp = _prompts_for_frame(frames, i)
        if not fp:
            pbar.update(1)
            continue
        for oid, (m, s) in predict_frame(cm, images[i], fp).items():
            masks[i, col[oid]] = torch.from_numpy(m)
            scores[i, col[oid]] = float(s)
        pbar.update(1)
    info = {"warnings": list(warnings), "scores": {int(oid): float(scores[:, col[oid]].max()) for oid in ids}}
    if return_info:
        return masks, ids, info
    return masks, ids


# ------------------------------------------------------------- propagation

def _init_state(predictor, frames: torch.Tensor, offload_video: bool, offload_state: bool):
    """Same dict as SAM2VideoPredictor.init_state, but from an in-memory BHWC batch."""
    device = predictor.device
    size = predictor.image_size
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    store = torch.device("cpu") if offload_video else device
    images = torch.empty((n, 3, size, size), dtype=torch.float16, device=store)
    mean, std = _MEAN.to(device), _STD.to(device)
    for i in range(0, n, 16):
        _interrupt()
        x = frames[i:i + 16].to(device, non_blocking=True).movedim(-1, 1).float()
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        images[i:i + 16] = ((x - mean) / std).to(store, torch.float16)
    st = {
        "images": images,
        "num_frames": n,
        "offload_video_to_cpu": offload_video,
        "offload_state_to_cpu": offload_state,
        "video_height": h,
        "video_width": w,
        "device": device,
        "storage_device": torch.device("cpu") if offload_state else device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    return st


def _auto_chunk(n: int, h: int, w: int, chunk: int) -> tuple[int, bool]:
    """Decide the effective window size. chunk<=0 means whole clip, but very long
    clips OOM in one SAM2 state, so auto-split with a warning."""
    chunk = int(chunk or 0)
    if chunk > 0:
        return chunk, False
    # Rough state cost ~ N * H * W of fp16 features; 128 frames @ 0.4MP is safe on most GPUs.
    if n > 160 and h * w >= 320 * 320:
        log.warning("[CoordinationMatrix] clip is %d frames @ %dx%d; auto-chunking to 128 frames "
                    "(set chunk_size explicitly to override)", n, w, h)
        return 128, True
    return 0, False


def _windows(n, chunk):
    if chunk <= 0 or n <= chunk:
        return [(0, n)]
    chunk = max(2, chunk)
    out, s = [], 0
    while True:
        e = min(n, s + chunk)
        out.append((s, e))
        if e >= n:
            return out
        s = e - 1  # 1-frame overlap: the mask is re-prompted into the next window


def _run_window(cm, frames, s, e, prompts, kf_masks, seeds, masks, dist, col,
                offload_video, offload_state, pbar):
    """Track frames[s:e] forward from the earliest and backward from the latest conditioning
    frame. Conditioning = keyframe prompts (as best-IoU masks or raw points) + seed masks
    carried from the neighbouring window. Each frame keeps the result whose conditioning
    frame is nearest (dist = frames away from a real keyframe)."""
    predictor = cm.net
    st = _init_state(predictor, frames[s:e], offload_video, offload_state)
    cond: dict = {}
    for g, objs in seeds.items():
        for oid, (m, d) in objs.items():
            if prompts.get(g, {}).get(oid) is None and bool(m.any()):
                predictor.add_new_mask(st, frame_idx=g - s, obj_id=int(oid), mask=m)
                cond.setdefault(oid, []).append((g - s, d))
    for g, objs in prompts.items():
        if not s <= g < e:
            continue
        for oid, p in objs.items():
            km = kf_masks.get(g, {}).get(oid)
            if km is not None:
                predictor.add_new_mask(st, frame_idx=g - s, obj_id=int(oid), mask=km)
            else:
                pts = lbl = None
                if p["points"]:
                    pts = np.array([[x, y] for x, y, _ in p["points"]], dtype=np.float32)
                    lbl = np.array([l for _, _, l in p["points"]], dtype=np.int32)
                box = np.array(p["box"], dtype=np.float32) if p["box"] is not None else None
                predictor.add_new_points_or_box(st, frame_idx=g - s, obj_id=int(oid), points=pts,
                                                labels=lbl, box=box, clear_old_points=True,
                                                normalize_coords=True)
            cond.setdefault(oid, []).append((g - s, 0))
    if not st["obj_ids"]:
        return
    all_c = [c for v in cond.values() for c, _ in v]
    for reverse, start in ((False, min(all_c)), (True, max(all_c))):
        for fi, obj_ids, logits in predictor.propagate_in_video(st, start_frame_idx=start, reverse=reverse):
            _interrupt()
            fi = int(fi)
            g = s + fi
            m = (logits[:, 0] > 0.0).to("cpu")
            for k, oid in enumerate(obj_ids):
                oid = int(oid)
                cands = [abs(fi - c) + d for c, d in cond.get(oid, [])
                         if (c >= fi if reverse else c <= fi)]
                if not cands:
                    continue  # this object has no conditioning on this side yet
                dd = min(cands)
                j = col[oid]
                if dd < dist[g, j]:
                    masks[g, j] = m[k]
                    dist[g, j] = dd
            pbar.update(1)
    del st


def propagate(cm, frames: torch.Tensor, prompt: dict, chunk_size: int = 0, keyframe_mask: str = "best_iou",
              offload_video: bool = True, offload_state: bool = False, return_info: bool = False):
    """Bidirectional video propagation from every keyframe.

    Returns (masks bool [N, O, H, W] on CPU, obj_ids), or (masks, ids, info) with
    return_info=True where info = {"warnings": [...], "scores": {oid: keyframe IoU},
    "keyframe_empty": [...], "chunk": effective_chunk, "auto_chunk": bool}."""
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    pf, warnings = P.scale_and_clamp(prompt, w, h, n)
    warnings = list(warnings)
    ids = P.object_ids({"objects": prompt.get("objects", []), "frames": pf})
    if not ids:
        raise ValueError("[CM Segment] no points or boxes placed on any frame")
    col = {oid: i for i, oid in enumerate(ids)}
    masks = torch.zeros((n, len(ids), h, w), dtype=torch.bool)
    dist = torch.full((n, len(ids)), float("inf"))
    cm.load()
    kf_masks: dict = {}
    kf_scores: dict = {}
    kf_empty: list = []
    if keyframe_mask == "best_iou":
        for f, objs in pf.items():
            res = predict_frame(cm, frames[f], objs, multimask=True)
            kf_masks[f] = {}
            kf_scores[f] = {}
            for oid, (m, s) in res.items():
                kf_scores[f][int(oid)] = float(s)
                if bool(m.any()):
                    kf_masks[f][int(oid)] = m
                else:
                    kf_empty.append(f"frame {f} obj {oid}: empty keyframe mask (IoU {s:.2f}); tracker seeded from raw points")
        if kf_empty:
            msg = "; ".join(kf_empty)
            log.warning("[CoordinationMatrix] %s", msg)
            warnings.extend(kf_empty)
    eff_chunk, auto = _auto_chunk(n, h, w, int(chunk_size or 0))
    wins = _windows(n, eff_chunk)
    keys = sorted(pf)
    first_w = next(i for i, (_s, e) in enumerate(wins) if e > keys[0])
    last_w = max(i for i, (_s, e) in enumerate(wins) if _s <= keys[-1])
    # Actual propagated frames: forward covers wins[first_w:], backward covers wins[:last_w].
    fwd = sum(e - _s for _s, e in wins[first_w:])
    bwd = sum(e - _s for _s, e in wins[:last_w])
    # Each window propagates both directions from its conditioning frame.
    pbar = _progress(2 * (fwd + bwd))
    with torch.inference_mode(), cm.autocast():
        # forward sweep: windows from the first keyframe on, carrying the mask at the overlap
        for i in range(first_w, len(wins)):
            s, e = wins[i]
            seeds = {}
            if i > first_w:
                seeds = {s: {oid: (masks[s, col[oid]], float(dist[s, col[oid]])) for oid in ids
                             if dist[s, col[oid]] < float("inf")}}
            _run_window(cm, frames, s, e, pf, kf_masks, seeds, masks, dist, col,
                        offload_video, offload_state, pbar)
            _soft_empty()
        # backward sweep: windows before the last keyframe, carrying masks the other way
        for i in range(min(last_w, len(wins) - 1) - 1, -1, -1):
            s, e = wins[i]
            g = e - 1
            seeds = {g: {oid: (masks[g, col[oid]], float(dist[g, col[oid]])) for oid in ids
                         if dist[g, col[oid]] < float("inf")}}
            _run_window(cm, frames, s, e, pf, kf_masks, seeds, masks, dist, col,
                        offload_video, offload_state, pbar)
            _soft_empty()
    _soft_empty()
    # Mean keyframe IoU per object as a rough confidence signal.
    scores: dict = {}
    for oid in ids:
        vals = [kf_scores.get(f, {}).get(oid) for f in pf if oid in pf.get(f, {})]
        vals = [v for v in vals if v is not None]
        scores[int(oid)] = float(sum(vals) / len(vals)) if vals else 0.0
    empty_frames = int((masks.amax(1) == 0).sum()) if n else 0
    if empty_frames:
        warnings.append(f"{empty_frames}/{n} frames have an empty union mask")
    info = {"warnings": warnings, "scores": scores, "keyframe_empty": list(kf_empty),
            "chunk": eff_chunk, "auto_chunk": bool(auto)}
    if return_info:
        return masks, ids, info
    return masks, ids


def _soft_empty():
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except ImportError:
        pass
