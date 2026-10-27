"""Small in-process stores the widget routes read from, and the on-disk mask cache."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict

import numpy as np
import torch

_LOCK = threading.Lock()
_FRAMES: "OrderedDict[str, dict]" = OrderedDict()   # node id -> {"frames": uint8 [N,H,W,3], "fmap": ...}
_TRACKS: "OrderedDict[str, dict]" = OrderedDict()   # source node id -> track
_MAX = 4


def put_frames(node_id, images: torch.Tensor, fmap: dict | None):
    if node_id is None:
        return
    u8 = (images.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()
    with _LOCK:
        _FRAMES[str(node_id)] = {"frames": u8, "fmap": fmap or {}}
        _FRAMES.move_to_end(str(node_id))
        while len(_FRAMES) > _MAX:
            _FRAMES.popitem(last=False)


def get_frames(node_id):
    with _LOCK:
        return _FRAMES.get(str(node_id))


def put_track(node_id, track: dict):
    if node_id is None:
        return
    with _LOCK:
        _TRACKS[str(node_id)] = track
        _TRACKS.move_to_end(str(node_id))
        while len(_TRACKS) > _MAX:
            _TRACKS.popitem(last=False)


def get_track(node_id):
    with _LOCK:
        return _TRACKS.get(str(node_id))


# ------------------------------------------------------------------ cache

def _cache_dir() -> str:
    import folder_paths

    d = os.path.join(folder_paths.get_temp_directory(), "cm_mask_cache")
    os.makedirs(d, exist_ok=True)
    return d


def frames_fingerprint(images: torch.Tensor) -> str:
    h = hashlib.sha1()
    h.update(str(tuple(images.shape)).encode())
    step = max(1, min(images.shape[1], images.shape[2]) // 24)
    sub = (images[:, ::step, ::step].float() * 255).to(torch.uint8).cpu().numpy()
    h.update(sub.tobytes())
    # Coarse thumbnails collide (e.g. solid colours); mix in a full-res centre
    # crop and basic stats so near-duplicates hash differently.
    try:
        n, hh, ww = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        ch, cw = min(64, hh), min(64, ww)
        y0, x0 = max(0, (hh - ch) // 2), max(0, (ww - cw) // 2)
        crop = (images[:, y0:y0 + ch, x0:x0 + cw].float() * 255).to(torch.uint8).cpu().numpy()
        h.update(crop.tobytes())
        h.update(str([float(images.float().mean()), float(images.float().std())]).encode())
    except Exception:
        pass
    return h.hexdigest()


_MEM: "OrderedDict[str, tuple]" = OrderedDict()   # in-memory LRU over the disk cache
_MEM_MAX = 6


def cache_key(images: torch.Tensor, prompt: dict, **settings) -> str:
    """SAM key: frame content + canonical prompts in image pixel space + masking settings only."""
    from . import prompts as P

    h, w = int(images.shape[1]), int(images.shape[2])
    return P.prompt_hash(prompt, w, h, frames_fp=frames_fingerprint(images), **settings)


def _mem_put(key, val):
    with _LOCK:
        _MEM[key] = val
        _MEM.move_to_end(key)
        while len(_MEM) > _MEM_MAX:
            _MEM.popitem(last=False)


def cache_load(key: str):
    with _LOCK:
        if key in _MEM:
            _MEM.move_to_end(key)
            m, ids = _MEM[key][:2]
            return m.clone(), list(ids)
    p = os.path.join(_cache_dir(), key + ".npz")
    if not os.path.isfile(p):
        return None
    try:
        z = np.load(p, allow_pickle=True)
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"])[: int(np.prod(shape))].reshape(shape).astype(bool)
        val = (torch.from_numpy(bits), [int(x) for x in z["ids"]])
    except Exception:
        return None
    _mem_put(key, val)
    return val[0].clone(), list(val[1])


def cache_load_full(key: str):
    """Like cache_load but also returns the cached info dict (scores/warnings), or {}."""
    with _LOCK:
        if key in _MEM:
            _MEM.move_to_end(key)
            m, ids = _MEM[key][:2]
            info = _MEM[key][2] if len(_MEM[key]) > 2 else {}
            return m.clone(), list(ids), dict(info)
    p = os.path.join(_cache_dir(), key + ".npz")
    if not os.path.isfile(p):
        return None
    try:
        z = np.load(p, allow_pickle=True)
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"])[: int(np.prod(shape))].reshape(shape).astype(bool)
        ids = [int(x) for x in z["ids"]]
        info = {}
        if "info" in z:
            try:
                info = json.loads(str(z["info"]))
            except Exception:
                info = {}
        val = (torch.from_numpy(bits), list(ids), dict(info))
    except Exception:
        return None
    _mem_put(key, val)
    return val[0].clone(), list(val[1]), dict(val[2])


def cache_save(key: str, masks: torch.Tensor, ids: list[int], info: dict | None = None):
    m = masks.cpu().bool()
    _mem_put(key, (m.clone(), list(ids), dict(info or {})))
    p = os.path.join(_cache_dir(), key + ".npz")
    arr = m.numpy()
    try:
        np.savez(p, bits=np.packbits(arr.ravel()), shape=np.array(arr.shape), ids=np.array(ids),
                 info=np.array(json.dumps(info or {})))
    except Exception:
        np.savez(p, bits=np.packbits(arr.ravel()), shape=np.array(arr.shape), ids=np.array(ids))
    _prune_disk_cache()


def _prune_disk_cache(max_files: int = 128, max_bytes: int = 2 << 30):
    """Bound temp-dir growth: drop oldest .npz files past count/size limits."""
    try:
        d = _cache_dir()
        files = [(os.path.getmtime(os.path.join(d, f)), os.path.join(d, f))
                 for f in os.listdir(d) if f.endswith(".npz")]
        files.sort()
        total = sum(os.path.getsize(p) for _, p in files if os.path.isfile(p))
        while (len(files) > max_files or total > max_bytes) and files:
            _, p = files.pop(0)
            try:
                total -= os.path.getsize(p)
                os.remove(p)
            except OSError:
                pass
    except Exception:
        pass


# preview (keyframe) masks: (frame id, canonical frame prompts, model) -> {oid: mask}
_PREV: "OrderedDict[str, dict]" = OrderedDict()


def frame_content_hash(arr) -> str:
    """Downsampled SHA1 of a single HWC uint8/float frame."""
    import hashlib

    import numpy as np
    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(np.asarray(a, dtype=np.float64) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return hashlib.sha1(a[::4, ::4].tobytes() + str(a.shape).encode()).hexdigest()


def kf_cache_key(arr, fp: dict, model_id: str) -> str:
    """Content-based key shared by the live preview route and engine.propagate."""
    from . import prompts as P

    h, w = (int(arr.shape[0]), int(arr.shape[1])) if hasattr(arr, "shape") else (0, 0)
    return P.prompt_hash({"frames": {0: fp}, "space": [w, h]}, w or None, h or None,
                         frame=frame_content_hash(arr), model=str(model_id))


def preview_get(key):
    with _LOCK:
        v = _PREV.get(key)
        if v is not None:
            _PREV.move_to_end(key)
        return v


def preview_put(key, val):
    with _LOCK:
        _PREV[key] = val
        _PREV.move_to_end(key)
        while len(_PREV) > 256:
            _PREV.popitem(last=False)
