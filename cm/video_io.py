"""Video decoding + index bookkeeping for CoordinationMatrix.

Every output frame carries an entry in a ``frame_map`` so prompt frame indices,
mask indices and the IMAGE batch can never drift apart, whatever fps / trim /
frame-skip / H3 snap is applied.

frame_map = {
    "source": "file" | "video" | "images",
    "file": str | None,
    "src_fps": float, "fps": float,
    "src_size": [w, h], "size": [w, h],
    "times": [seconds in source for each output frame],
    "src_index": [nearest source frame number for each output frame],
    "padded": [bool per output frame]  (True for extend/H3 pad frames),
}
"""
from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
import torch

from . import h3

RESIZE_FILTERS = ["bicubic", "lanczos", "bilinear", "nearest-exact", "area"]

# Output resampling to fps: "nearest" holds/duplicates source frames (legacy),
# "blend" linearly crossfades neighbouring source frames (smoother motion when
# upsampling low-fps sources such as GIFs to the 24 fps H3 grid).
FRAME_INTERPS = ["nearest", "blend"]
# What to do when the window runs past the end of a short source (short GIF /
# still): "hold" repeats the last frame, "loop" wraps to the window start.
EOF_MODES = ["hold", "loop"]

# Pillow-decodable image formats: animated (GIF / animated WebP / APNG / AVIF)
# plus single stills, which become 1-frame sources grown by extend/loop/H3 snap.
IMAGE_EXTS = frozenset({".gif", ".webp", ".apng", ".png", ".avif", ".jpg", ".jpeg",
                        ".bmp", ".tiff", ".tif", ".qoi"})
# Extra containers ffmpeg/av usually decodes but file pickers tend to hide.
EXTRA_VIDEO_EXTS = (".ts", ".m2ts", ".mts", ".ogv", ".3gp", ".3g2", ".asf", ".rm",
                    ".rmvb", ".vob", ".dv", ".f4v", ".mxf", ".wtv", ".y4m")

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".mpg", ".mpeg",
             ".wmv", ".flv", ".webp", ".apng", ".png", ".jpg", ".jpeg", ".bmp", ".avif",
             ".tiff", ".tif", ".qoi",
             ".ts", ".m2ts", ".mts", ".ogv", ".3gp", ".asf", ".vob", ".mxf")

_AV_INTERP = {
    "nearest-exact": "POINT",
    "bilinear": "BILINEAR",
    "bicubic": "BICUBIC",
    "area": "AREA",
    "lanczos": "LANCZOS",
}


def snap_multiple(n: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    n = max(1, int(n))
    snapped = ((n + multiple // 2) // multiple) * multiple
    return max(snapped, multiple)


def scale_megapixels(width: int, height: int, megapixels: float, multiple: int = 32) -> tuple[int, int]:
    """Target total size in MP, aspect locked to source, then snapped (H3 = 32)."""
    width, height = max(1, int(width)), max(1, int(height))
    mp = max(0.0, float(megapixels or 0.0))
    if mp <= 0:
        tw, th = width, height
    else:
        sc = math.sqrt(mp * 1_000_000 / (width * height))
        tw, th = max(1, round(width * sc)), max(1, round(height * sc))
    return snap_multiple(tw, multiple), snap_multiple(th, multiple)


def _interp(name: str):
    import av

    key = _AV_INTERP.get(name, "BICUBIC")
    return getattr(av.video.reformatter.Interpolation, key, av.video.reformatter.Interpolation.BICUBIC)


def _frame_time(frame, stream) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return 0.0


def _rotation(frame) -> int:
    rot = getattr(frame, "rotation", 0) or 0
    return int(round(rot // 90)) % 4


def _convert(frame, out_w: int, out_h: int, interp) -> np.ndarray:
    import av

    rot = _rotation(frame)
    if rot:
        arr = frame.to_ndarray(format="rgb24")
        arr = np.ascontiguousarray(np.rot90(arr, k=rot, axes=(0, 1)))
        if arr.shape[1] != out_w or arr.shape[0] != out_h:
            vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
            arr = vf.reformat(width=out_w, height=out_h, format="rgb24", interpolation=interp).to_ndarray()
        return arr
    if out_w != frame.width or out_h != frame.height:
        frame = frame.reformat(width=out_w, height=out_h, format="rgb24", interpolation=interp)
    else:
        frame = frame.reformat(format="rgb24")
    return frame.to_ndarray()


def probe(path: str) -> dict:
    """Cheap metadata read used by the widget timeline.

    Tries av first (accurate for real video), then falls back to Pillow for
    animated images (GIF / animated WebP / APNG -- av cannot decode some of
    these and reports no duration for others) and single stills.
    """
    import av

    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        try:
            return _pil_probe(path)
        except Exception:
            pass  # fall through to av
    try:
        with av.open(path, mode="r") as c:
            vs = c.streams.video[0]
            fps = float(vs.average_rate or vs.guessed_rate or 0) or 30.0
            if vs.duration is not None and vs.time_base is not None:
                dur = float(vs.duration * vs.time_base)
            elif c.duration is not None:
                dur = float(c.duration / av.time_base)
            elif vs.frames:
                dur = vs.frames / fps
            else:
                dur = 0.0
            w, h = int(vs.codec_context.width), int(vs.codec_context.height)
            # rotation metadata only known per frame; peek the first one
            try:
                for fr in c.decode(video=0):
                    if _rotation(fr) % 2:
                        w, h = h, w
                    break
            except Exception:
                pass
            meta = {"src_fps": fps, "duration": dur, "width": w, "height": h,
                    "has_audio": len(c.streams.audio) > 0, "pil": False}
    except Exception:
        meta = None
    if meta is None or meta["duration"] <= 0:
        # No usable av duration (APNG / stills / obscure containers): Pillow knows better.
        try:
            pil = _pil_probe(path)
            if meta is not None:
                pil["has_audio"] = bool(meta.get("has_audio"))
            return pil
        except Exception:
            pass
    if meta is None:
        raise ValueError(f"[CM Video Loader] cannot probe {path}")
    return meta


def _pil_probe(path: str) -> dict:
    """Frame-accurate metadata for Pillow-readable images.

    Uses real per-frame delays, so VFR GIFs get an exact duration and mean fps.
    Single stills become a 1-frame source at 24 fps (extend / loop / H3 snap
    then grow the clip from there).
    """
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1) or 1
        durs: list[float] = []
        for i in range(n):
            try:
                im.seek(i)
            except EOFError:
                break
            ms = im.info.get("duration", 100)
            try:
                ms = float(ms)
            except (TypeError, ValueError):
                ms = 100.0
            if not ms or ms <= 0:
                ms = 100.0  # GIF/WebP blank delay = default frame delay
            durs.append(ms / 1000.0)
        if not durs:
            raise ValueError(f"[CM Video Loader] no frames in {path}")
        try:
            w, h = ImageOps.exif_transpose(im.copy()).size
        except Exception:
            w, h = im.size
        animated = bool(getattr(im, "is_animated", len(durs) > 1))
    if not animated:
        return {"src_fps": 24.0, "duration": 1.0 / 24.0, "width": int(w), "height": int(h),
                "has_audio": False, "pil": True, "pil_frames": 1, "pil_durations": [1.0 / 24.0]}
    total = sum(durs)
    fps = len(durs) / total if total > 0 else 24.0
    return {"src_fps": float(fps), "duration": float(total), "width": int(w), "height": int(h),
            "has_audio": False, "pil": True, "pil_frames": len(durs), "pil_durations": durs}


def plan_times(src_fps: float, src_duration: float, fps: float, start: float, trim: float,
               frame_skip: int) -> tuple[list[float], float]:
    """Output sample times (source seconds) for the window, before extend/H3."""
    out_fps = float(fps) if fps and fps > 0 else float(src_fps or 30.0)
    start = max(0.0, float(start or 0.0))
    end = src_duration if src_duration > 0 else math.inf
    if trim and trim > 0:
        end = min(end, start + float(trim))
    step = 1.0 / out_fps
    times, t = [], start
    # guard: unknown duration -> caller stops at EOF
    limit = 10_000_000
    while t + 1e-6 < end and len(times) < limit:
        times.append(t)
        t += step
        if end is math.inf and len(times) > 2_000_000:
            break
    skip = max(1, int(frame_skip or 1))
    if skip > 1:
        times = times[::skip]
    return times, out_fps / skip


def decode_file(path: str, times: list[float] | None = None, fps: float = 24.0,
                megapixels: float = 0.4, resize_filter: str = "bicubic",
                include_audio: bool = True, max_frames: int = 0,
                frame_interp: str = "nearest", eof_mode: str = "hold"):
    """Decode the frames whose timestamps cover each requested output time.

    Returns (images float32 BHWC 0..1, out_fps float, audio | None, frame_map).
    Animated images av cannot decode (e.g. animated WebP) transparently fall
    back to Pillow; ``frame_interp="blend"`` crossfades neighbours when
    upsampling (low-fps GIF -> 24 fps); ``eof_mode="loop"`` wraps short
    sources instead of holding the last frame.
    """
    info = probe(path)
    if times is None:
        times, out_fps = plan_times(info["src_fps"], info["duration"], fps, 0.0, 0.0, 1)
    else:
        out_fps = float(fps)
    if max_frames > 0:
        times = times[:max_frames]
    if not times:
        raise ValueError(f"[CM Video Loader] window is empty for {path}")
    out_w, out_h = scale_megapixels(info["width"], info["height"], float(megapixels or 0.0), 32)
    n = len(times)
    if info.get("pil"):
        buf, src_index, starts, ends, eff_times = _decode_pil_frames(
            path, list(times), out_fps, out_w, out_h, resize_filter, info, eof_mode)
    else:
        try:
            interp = _interp(resize_filter)
            buf, src_index, starts, ends, eff_times = _decode_av_frames(
                path, list(times), out_fps, out_w, out_h, interp, info, eof_mode)
        except Exception:
            info = _pil_probe(path)  # last resort: animated WebP, odd containers
            buf, src_index, starts, ends, eff_times = _decode_pil_frames(
                path, list(times), out_fps, out_w, out_h, resize_filter, info, eof_mode)
    if str(frame_interp).lower() == "blend":
        # blend on effective (wrapped) times so looped tails weight correctly
        _blend_in_place(buf, eff_times, starts, ends)
    images = buf.float().div_(255.0)
    audio = None
    if include_audio and not info.get("pil") and info.get("has_audio"):
        first_t, last_t = times[0], times[-1]
        chunks, rate = _decode_audio(path, first_t, last_t + 1.0 / max(out_fps, 1e-6))
        if chunks:
            wav = torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0)
            want = max(1, int(round(n / out_fps * rate)))
            audio = {"waveform": wav[..., :want].contiguous(), "sample_rate": rate}
    fmap = {
        "source": "file", "file": path,
        "src_fps": float(info["src_fps"]), "fps": float(out_fps),
        "src_size": [info["width"], info["height"]], "size": [out_w, out_h],
        "times": [float(x) for x in times], "src_index": src_index, "padded": [False] * n,
    }
    return images, float(out_fps), audio, fmap


def _resolve_loop(times: list[float], duration: float, eof_mode: str):
    """Split a uniform sample grid into a monotonic head + looping tail.

    Returns (head_times, tail_index) where tail_index[i] is the head position
    output i >= len(head) loops from. ``None`` tail = hold (legacy).
    """
    n = len(times)
    if str(eof_mode).lower() != "loop" or duration <= 0:
        return list(times), None
    step = float(times[1] - times[0]) if n > 1 else 0.0
    if step <= 0:
        return list(times), None
    s = float(times[0])
    if s >= duration:
        s = s % duration
    span = duration - s
    if span <= 0:
        return list(times), None
    head = [s + k * step for k in range(n) if s + k * step < duration - 1e-9]
    if not head:
        head = [s]
    m = len(head)
    if m >= n:
        return head[:n], None
    tail = []
    for k in range(m, n):
        p = (s + k * step - s) % span
        tail.append(min(m - 1, int(round(p / step))))
    return head, tail


def _decode_av_frames(path, times, out_fps, out_w, out_h, interp, info, eof_mode):
    """Stream-decode with one-frame lookahead so VFR sources (GIF) get exact
    per-output intervals. Returns (buf uint8, src_index, starts, ends, eff_times)."""
    import av

    head, tail = _resolve_loop(times, float(info["duration"] or 0.0), eof_mode)
    n, m = len(times), len(head)
    buf = torch.empty((n, out_h, out_w, 3), dtype=torch.uint8)
    src_index = [0] * n
    starts = [0.0] * n
    ends = [0.0] * n
    src_fps = float(info["src_fps"] or 0.0) or 30.0
    first_t = head[0]
    filled = 0
    with av.open(path, mode="r") as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        if first_t > 1.0:
            try:
                c.seek(int(max(0.0, first_t - 1.0) / vs.time_base), stream=vs, backward=True, any_frame=False)
            except Exception:
                c.seek(0)
        prev_t, prev_rgb = None, None
        for frame in c.decode(video=0):
            t = _frame_time(frame, vs)
            if prev_t is not None and filled < m and head[filled] < t - 1e-4:
                # head times inside [prev_t, t): previous frame (VFR-gap hold)
                rgb = prev_rgb
                while filled < m and head[filled] < t - 1e-4:
                    buf[filled].copy_(rgb)
                    src_index[filled] = max(0, int(round(prev_t * src_fps)))
                    starts[filled], ends[filled] = prev_t, t
                    filled += 1
            prev_t = t
            prev_rgb = torch.from_numpy(np.ascontiguousarray(_convert(frame, out_w, out_h, interp)))
            if filled >= m:
                break
        if prev_rgb is not None:
            while filled < m:  # EOF early: hold last
                buf[filled].copy_(prev_rgb)
                src_index[filled] = max(0, int(round(prev_t * src_fps)))
                starts[filled], ends[filled] = prev_t, prev_t
                filled += 1
    if filled == 0:
        raise ValueError(f"[CM Video Loader] no frames decoded from {path}")
    if filled < m:  # source shorter than the head window: hold last (probe overestimated)
        for i in range(filled, m):
            buf[i].copy_(buf[filled - 1])
            src_index[i] = src_index[filled - 1]
            starts[i], ends[i] = starts[filled - 1], starts[filled - 1]
        filled = m
    if tail is not None:  # loop the head content for the tail
        for i, j in enumerate(tail, start=m):
            buf[i].copy_(buf[j])
            src_index[i] = src_index[j]
            starts[i], ends[i] = starts[j], ends[j]
    eff_times = list(head) + ([head[j] for j in tail] if tail is not None else [times[k] for k in range(m, n)])
    return buf, src_index, starts, ends, eff_times


def _writable(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _pil_filter(name: str):
    from PIL import Image

    return {"bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR,
            "nearest-exact": Image.NEAREST, "area": Image.BOX}.get(str(name), Image.BICUBIC)


def _load_pil_native(path, out_w: int, out_h: int, filt, info):
    """Native RGB frames at output size + per-frame start times + total duration.

    Shared by the batch grid resample and the exact single-frame blend below.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        n_src = int(info.get("pil_frames") or getattr(im, "n_frames", 1) or 1)
        durs = list(info.get("pil_durations") or [0.1] * n_src)
        native, src_starts = [], []
        acc = 0.0
        for i in range(n_src):
            try:
                im.seek(i)
            except EOFError:
                break
            fr = ImageOps.exif_transpose(im).convert("RGBA")
            if fr.size != (out_w, out_h):
                fr = fr.resize((out_w, out_h), filt)
            bg = Image.new("RGB", fr.size, (0, 0, 0))
            bg.paste(fr, mask=fr.split()[3])
            native.append(torch.from_numpy(_writable(np.asarray(bg))))
            src_starts.append(acc)
            acc += durs[i] if i < len(durs) else durs[-1]
    if not native:
        raise ValueError(f"[CM Video Loader] no frames decoded from {path}")
    return native, src_starts, acc


def _decode_pil_frames(path, times, out_fps, out_w, out_h, resize_filter, info, eof_mode):
    """Decode Pillow-readable sources (GIF / animated WebP / APNG / stills).

    Loads the (small) native frame list with true per-frame delays, then
    resamples to the output grid. Returns (buf uint8, src_index, starts, ends, eff_times).
    """
    filt = _pil_filter(resize_filter)
    native, src_starts, dur = _load_pil_native(path, out_w, out_h, filt, info)
    src_fps = len(native) / dur if dur > 0 else 24.0
    head, tail = _resolve_loop(times, dur, eof_mode)
    m, n = len(head), len(times)
    buf = torch.empty((len(times), out_h, out_w, 3), dtype=torch.uint8)
    src_index = [0] * len(times)
    starts = [0.0] * len(times)
    ends = [0.0] * len(times)
    st = np.asarray(src_starts, dtype=float)
    for k, t in enumerate(head):
        j = int(np.searchsorted(st, t + 1e-9, side="right") - 1)
        j = min(max(j, 0), len(native) - 1)
        buf[k].copy_(native[j])
        src_index[k] = j
        s0 = float(st[j])
        s1 = float(st[j + 1]) if j + 1 < len(st) else dur
        starts[k], ends[k] = s0, (s1 if s1 > s0 else s0)
    if tail is not None:
        for i, j in enumerate(tail, start=m):
            buf[i].copy_(buf[j])
            src_index[i] = src_index[j]
            starts[i], ends[i] = starts[j], ends[j]
    elif m < len(times):  # hold last
        for i in range(m, len(times)):
            buf[i].copy_(buf[m - 1])
            src_index[i] = src_index[m - 1]
            starts[i], ends[i] = starts[m - 1], starts[m - 1]
    eff_times = list(head) + ([head[j] for j in tail] if tail is not None else [times[k] for k in range(m, n)])
    return buf, src_index, starts, ends, eff_times


def _blend_in_place(buf: torch.Tensor, times: list[float], starts: list[float], ends: list[float]):
    """Linear crossfade between neighbouring source frames, in place (uint8).

    For each output time inside a source interval [start, end), mixes the held
    frame with the next distinct source frame by the fractional phase. No-op
    when the output grid already matches the source grid (alpha = 0), so it is
    bit-identical to nearest for fps-preserving loads.
    """
    import numpy as np

    n = int(buf.shape[0])
    if n < 2:
        return buf
    t = np.asarray(times, dtype=float)
    st = np.asarray(starts, dtype=float)
    en = np.asarray(ends, dtype=float)
    span = en - st
    alpha = np.zeros(n, dtype=np.float64)
    ok = span > 1e-9
    alpha[ok] = np.clip((t[ok] - st[ok]) / span[ok], 0.0, 1.0)
    if not np.any(alpha > 1e-6):
        return buf
    sid = np.round(st * 1000.0).astype(np.int64)  # interval id from start (ms)
    chg = np.nonzero(np.diff(sid) != 0)[0] + 1  # first index of each new run
    if chg.size == 0:
        return buf
    pos = np.searchsorted(chg, np.arange(n), side="right")
    has = pos < chg.size
    use = has & (alpha > 1e-6)
    if not np.any(use):
        return buf
    nxt = np.arange(n)
    nxt[use] = chg[pos[use]]
    a = torch.from_numpy(alpha.astype(np.float32)).to(buf.device)
    idx = torch.from_numpy(nxt.astype(np.int64)).to(buf.device)
    for s in range(0, n, 512):  # chunked: advanced indexing transiently ~2x buf
        e = min(n, s + 512)
        w = a[s:e].view(-1, 1, 1, 1)
        mixed = buf[s:e].float() * (1.0 - w) + buf[idx[s:e]].float() * w
        buf[s:e].copy_(mixed.round().clamp_(0, 255).to(torch.uint8))
    return buf


def _decode_audio(path: str, t0: float, t1: float):
    import av

    chunks, rate = [], 0
    with av.open(path, mode="r") as c:
        a = c.streams.audio[0]
        res = av.audio.resampler.AudioResampler(format="fltp")
        if t0 > 1.0:
            try:
                c.seek(int(max(0.0, t0 - 1.0) / a.time_base), stream=a, backward=True)
            except Exception:
                c.seek(0)
        lead_skip = None
        for af in c.decode(audio=0):
            at = af.time if af.time is not None else 0.0
            sr = int(af.sample_rate or a.rate or 0)
            dur = af.samples / sr if sr else 0.0
            if at + dur <= t0:
                continue
            if at >= t1:
                break
            if lead_skip is None:
                lead_skip = max(0, int(round((t0 - at) * sr))) if sr else 0
            for rf in res.resample(af):
                chunks.append(rf.to_ndarray())
                rate = int(rf.sample_rate or sr or 1)
        if chunks and lead_skip:
            chunks[0] = chunks[0][:, lead_skip:]
    return chunks, rate


def decode_single_frame(path: str, t: float, megapixels: float, resize_filter: str,
                        frame_interp: str = "nearest", eof_mode: str = "hold") -> np.ndarray:
    """One RGB uint8 frame at source time t (same selection rule as decode_file).

    With ``frame_interp="blend"`` this is the exact source-level crossfade the
    batch path emits for that output time -- blending is a no-op on a 1-frame
    batch, so it is computed here from the bracketing native frames instead of
    going through decode_file. Pillow sources only (GIF / animated WebP / APNG /
    stills); AV sources keep the held frame.
    """
    if str(frame_interp).lower() == "blend":
        info = probe(path)
        if info.get("pil"):
            out_w, out_h = scale_megapixels(info["width"], info["height"],
                                            float(megapixels or 0.0), 32)
            native, src_starts, dur = _load_pil_native(path, out_w, out_h,
                                                      _pil_filter(resize_filter), info)
            tt = float(t)
            if str(eof_mode).lower() == "loop" and dur > 0:
                tt = tt % dur
            st = np.asarray(src_starts, dtype=float)
            j = int(np.searchsorted(st, tt + 1e-9, side="right") - 1)
            j = min(max(j, 0), len(native) - 1)
            s0 = float(st[j])
            s1 = float(st[j + 1]) if j + 1 < len(st) else dur
            a = 0.0 if s1 <= s0 else min(1.0, max(0.0, (tt - s0) / (s1 - s0)))
            w = float(np.float32(a))  # same float32 weight the batch blend uses
            if w > 1e-6 and j + 1 < len(native):
                mixed = native[j].float() * (1.0 - w) + native[j + 1].float() * w
                return mixed.round().clamp_(0, 255).to(torch.uint8).numpy()
            return native[j].clone().numpy()
    imgs, _, _, _ = decode_file(path, [float(t)], 1.0, megapixels, resize_filter, include_audio=False,
                                frame_interp=frame_interp, eof_mode=eof_mode)
    return (imgs[0].numpy() * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------- tensors

def resize_images(images: torch.Tensor, megapixels: float, method: str) -> torch.Tensor:
    import comfy.utils

    h, w = int(images.shape[1]), int(images.shape[2])
    tw, th = scale_megapixels(w, h, float(megapixels or 0.0), 32)
    if (tw, th) == (w, h):
        return images
    out = []
    for i in range(0, images.shape[0], 16):
        chunk = images[i:i + 16].movedim(-1, 1)
        out.append(comfy.utils.common_upscale(chunk, tw, th, method, "disabled").movedim(1, -1).clamp(0, 1))
    return torch.cat(out, 0)


def select_from_tensor(images: torch.Tensor, src_fps: float, fps: float, start: float, trim: float,
                       frame_skip: int, src_map: dict | None = None):
    """Timestamp selection on an in-memory batch. Returns (images, out_fps, fmap, idx)."""
    n = int(images.shape[0])
    src_fps = float(src_fps or 0) or 24.0
    times, out_fps = plan_times(src_fps, n / src_fps, fps, start, trim, frame_skip)
    idx = [min(n - 1, max(0, int(math.floor(t * src_fps + 1e-4)))) for t in times]
    if not idx:
        raise ValueError("[CM Video Loader] window is empty for the given images")
    sel = images[idx]
    h, w = int(sel.shape[1]), int(sel.shape[2])
    if src_map:
        st = src_map.get("times") or []
        si = src_map.get("src_index") or []
        sp = src_map.get("padded") or []
        fmap = dict(src_map)
        fmap["times"] = [float(st[i]) if i < len(st) else float(times[k]) for k, i in enumerate(idx)]
        fmap["src_index"] = [int(si[i]) if i < len(si) else int(i) for i in idx]
        fmap["padded"] = [bool(sp[i]) if i < len(sp) else False for i in idx]
        fmap["fps"] = float(out_fps)
    else:
        fmap = {
            "source": "images", "file": None, "src_fps": src_fps, "fps": float(out_fps),
            "src_size": [w, h], "times": [float(t) for t in times], "src_index": idx,
            "padded": [False] * len(idx),
        }
    fmap["size"] = [w, h]
    return sel, float(out_fps), fmap, idx


def apply_index(images: torch.Tensor, fmap: dict, index: torch.Tensor, padded_from: int) -> tuple[torch.Tensor, dict]:
    """Re-index frames and frame_map together (extend / H3 pad / trim)."""
    idx = [int(i) for i in index.tolist()]
    out = images[index]
    m = dict(fmap)
    for k in ("times", "src_index"):
        src = fmap.get(k) or []
        m[k] = [src[i] for i in idx] if src else []
    pad = fmap.get("padded") or [False] * images.shape[0]
    m["padded"] = [bool(pad[i]) or (j >= padded_from) for j, i in enumerate(idx)]
    return out, m


def extend_hold(images, fmap, fps: float, extend: float):
    n = int(images.shape[0])
    if extend <= 0 or n < 1 or fps <= 0:
        return images, fmap
    target = max(n + 1, int(round((n / fps + extend) * fps)))
    index = torch.cat([torch.arange(n), torch.full((target - n,), n - 1, dtype=torch.long)])
    return apply_index(images, fmap, index, n)


def h3_snap(images, fmap, mode: str, pad_mode: str):
    n = int(images.shape[0])
    target = h3.h3_snap_target(n, mode)
    if target == n:
        return images, fmap
    h3.h3_snap_note("CM Video Loader", n, target, mode, pad_mode)
    return apply_index(images, fmap, h3.h3_pad_index(n, target, pad_mode), n)


def frac(fps: float) -> Fraction:
    return Fraction(round(float(fps) * 1000), 1000)


# ------------------------------------------------------------- full plan

def plan_file(info: dict, fps: float, start: float, trim: float, extend: float,
              snap_mode: str = "down", pad_mode: str = "reflect", eof_mode: str = "hold") -> dict:
    """Window -> fps resample -> extend (hold) -> H3 17k+5 snap (always on).

    With ``eof_mode="loop"`` the window is NOT clamped to the source duration:
    the extra time wraps to the window start in decode (short GIFs/stills can
    fill an arbitrary trim). ``"hold"`` keeps the legacy clamp.

    Returns {"times": unique decode times, "index": output frame -> times index,
    "padded": [bool], "fps", "window_frames", "original_frames", "length"}.
    Shared by the node and the live editor route so frame indices always agree.
    """
    dur = float(info.get("duration") or 0.0)
    if str(eof_mode).lower() == "loop" and dur > 0 and trim and float(trim) > 0 \
            and float(start) + float(trim) > dur:
        times, out_fps = plan_times(info["src_fps"], float(start) + float(trim), fps, start, trim, 1)
    else:
        times, out_fps = plan_times(info["src_fps"], info["duration"], fps, start, trim, 1)
    if not times:
        raise ValueError(f"[CM Load Video] window is empty (start={start}, trim={trim})")
    n = len(times)
    index = list(range(n))
    if extend and extend > 0:
        target = max(n + 1, int(round((n / out_fps + float(extend)) * out_fps)))
        index += [n - 1] * (target - n)
    original = len(index)
    target = h3.h3_snap_target(original, snap_mode)
    if target != original:
        h3.h3_snap_note("CM Load Video", original, target, snap_mode, pad_mode)
        index = [index[i] for i in h3.h3_pad_index(original, target, pad_mode).tolist()]
    padded = [j >= n for j in range(len(index))]
    return {"times": times, "index": index, "padded": padded, "fps": float(out_fps),
            "window_frames": n, "original_frames": original, "length": len(index)}
