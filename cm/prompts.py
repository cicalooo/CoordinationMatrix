"""CoordinationMatrix prompt format (what the DOM widget writes, CM_PROMPTS carries).

Widget / JSON form::

    {
      "v": 1,
      "space": [W, H],                       # pixel space the coords were placed in
      "objects": [{"id": 1, "name": "obj 1", "color": "#ff4d4d"}],
      "frames": {                             # output-frame index -> object id -> prompt
        "0":  {"1": {"points": [[x, y, 1], [x, y, 0]], "box": [x0, y0, x1, y1]}},
        "42": {"1": {"points": [[x, y, 1]], "box": null}}
      }
    }

label 1 = positive, 0 = negative. Frame keys are output frame indices (after
fps / trim / frame-skip), so they line up with the IMAGE batch and the masks.
"""
from __future__ import annotations

import json

PALETTE = ["#ff4d4d", "#4dabff", "#4dff88", "#ffd24d", "#c44dff", "#ff8c4d",
           "#4dfff0", "#ff4dc4", "#a3ff4d", "#8c8cff", "#ffffff", "#ff9e9e"]


def color_for(obj_id: int) -> str:
    return PALETTE[(int(obj_id) - 1) % len(PALETTE)]


def hex_to_rgb(c: str) -> tuple[float, float, float]:
    c = (c or "#ff4d4d").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (1.0, 0.3, 0.3)


def empty() -> dict:
    return {"v": 1, "space": [0, 0], "objects": [], "frames": {}}


def parse(raw) -> dict:
    """Accept a JSON string or dict; return a normalised prompt dict."""
    if raw is None or raw == "":
        return empty()
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"[CoordinationMatrix] prompt JSON is invalid: {e}") from e
    else:
        data = raw
    if not isinstance(data, dict):
        raise ValueError("[CoordinationMatrix] prompt JSON must be an object")
    out = empty()
    sp = data.get("space") or [0, 0]
    out["space"] = [int(sp[0] or 0), int(sp[1] or 0)]
    seen = {}
    for o in data.get("objects") or []:
        oid = int(o.get("id"))
        seen[oid] = {"id": oid, "name": str(o.get("name") or f"obj {oid}"),
                     "color": str(o.get("color") or color_for(oid))}
    frames = {}
    for fk, objs in (data.get("frames") or {}).items():
        f = int(fk)
        if f < 0:
            continue
        for ok, p in (objs or {}).items():
            oid = int(ok)
            pts = []
            for pt in (p or {}).get("points") or []:
                if isinstance(pt, dict):
                    pts.append([float(pt["x"]), float(pt["y"]), int(pt.get("label", 1))])
                else:
                    pts.append([float(pt[0]), float(pt[1]), int(pt[2]) if len(pt) > 2 else 1])
            box = (p or {}).get("box")
            if box is not None:
                x0, y0, x1, y1 = [float(v) for v in box]
                box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    box = None
            if not pts and box is None:
                continue
            frames.setdefault(f, {})[oid] = {"points": pts, "box": box}
            if oid not in seen:
                seen[oid] = {"id": oid, "name": f"obj {oid}", "color": color_for(oid)}
    out["frames"] = dict(sorted(frames.items()))
    out["objects"] = [seen[k] for k in sorted(seen)]
    for k in ("source_node", "frame_map"):
        if k in data:
            out[k] = data[k]
    return out


def to_json(p: dict, indent: int | None = None) -> str:
    d = {"v": 1, "space": p.get("space", [0, 0]), "objects": p.get("objects", []),
         "frames": {str(f): {str(o): v for o, v in objs.items()} for f, objs in p.get("frames", {}).items()}}
    return json.dumps(d, indent=indent)


def scaled(p: dict, width: int, height: int) -> dict:
    """Return frames rescaled from p['space'] to (width, height)."""
    sw, sh = p.get("space") or [0, 0]
    if not sw or not sh or (sw == width and sh == height):
        # Return a copy so callers can clamp/mutate without touching the prompt.
        return {f: {oid: {"points": [list(pt) for pt in v["points"]],
                          "box": list(v["box"]) if v["box"] is not None else None}
                     for oid, v in objs.items()}
                for f, objs in p["frames"].items()}
    fx, fy = width / float(sw), height / float(sh)
    out = {}
    for f, objs in p["frames"].items():
        out[f] = {}
        for oid, v in objs.items():
            pts = [[x * fx, y * fy, l] for x, y, l in v["points"]]
            box = None if v["box"] is None else [v["box"][0] * fx, v["box"][1] * fy, v["box"][2] * fx, v["box"][3] * fy]
            out[f][oid] = {"points": pts, "box": box}
    return out


def clamp_frames(frames: dict, n: int) -> dict:
    dropped = [f for f in frames if f >= n]
    if dropped:
        print(f"[CoordinationMatrix] WARNING prompts on frames {dropped} are past the clip end ({n} frames); ignored")
    return {f: v for f, v in frames.items() if f < n}


def scale_and_clamp(p: dict, width: int, height: int, n: int) -> tuple[dict, list[str]]:
    """Scaled prompts with coords clamped into [0,W-1]x[0,H-1] + out-of-range frames dropped.

    Returns (frames, warnings). Warnings are plain strings so nodes can surface them in the UI.
    """
    warnings: list[str] = []
    frames = scaled(p, int(width), int(height))
    dropped = sorted(f for f in frames if f >= n or f < 0)
    if dropped:
        warnings.append(f"prompts on frames {dropped} are past the clip end ({n} frames); ignored")
        print(f"[CoordinationMatrix] WARNING {warnings[-1]}")
    frames = {f: v for f, v in frames.items() if 0 <= f < n}
    for f, objs in frames.items():
        for oid, v in objs.items():
            fixed = []
            for x, y, lab in v.get("points") or []:
                try:
                    fx, fy = float(x), float(y)
                except (TypeError, ValueError):
                    continue
                # NaN/inf or out-of-bounds points confuse normalize_coords; clamp them.
                import math
                if not math.isfinite(fx) or not math.isfinite(fy):
                    continue
                cx = min(max(fx, 0.0), float(width - 1) if width > 1 else 0.0)
                cy = min(max(fy, 0.0), float(height - 1) if height > 1 else 0.0)
                if abs(cx - fx) > 1e-6 or abs(cy - fy) > 1e-6:
                    warnings.append(f"frame {f} obj {oid}: point clamped into frame")
                try:
                    li = int(lab)
                except (TypeError, ValueError):
                    li = 1
                fixed.append([cx, cy, 1 if li else 0])
            v["points"] = fixed
            box = v.get("box")
            if box is not None:
                try:
                    x0, y0, x1, y1 = [float(b) for b in box]
                    import math
                    if not all(math.isfinite(b) for b in (x0, y0, x1, y1)):
                        raise ValueError("non-finite box")
                    x0 = min(max(min(x0, x1), 0.0), float(width - 1) if width > 1 else 0.0)
                    x1 = min(max(max(x0, x1), 0.0), float(width) if width else 0.0)
                    y0 = min(max(min(y0, y1), 0.0), float(height - 1) if height > 1 else 0.0)
                    y1 = min(max(max(y0, y1), 0.0), float(height) if height else 0.0)
                    if x1 - x0 < 1 or y1 - y0 < 1:
                        v["box"] = None
                        warnings.append(f"frame {f} obj {oid}: degenerate box dropped")
                    else:
                        v["box"] = [x0, y0, x1, y1]
                except (TypeError, ValueError):
                    v["box"] = None
    return frames, warnings


def object_ids(p: dict) -> list[int]:
    ids = {o["id"] for o in p.get("objects", [])}
    for objs in p.get("frames", {}).values():
        ids.update(objs.keys())
    # only objects that actually have a prompt somewhere can be tracked
    used = set()
    for objs in p.get("frames", {}).values():
        used.update(objs.keys())
    return sorted(i for i in ids if i in used)


def colors(p: dict, ids: list[int]) -> list[str]:
    m = {o["id"]: o["color"] for o in p.get("objects", [])}
    return [m.get(i, color_for(i)) for i in ids]


def sam3_style(p: dict, frame: int, obj: int) -> tuple[str, str, list]:
    """positive_coords / negative_coords JSON + BoundingBox list like stock SAM3 Detect."""
    frames = p.get("frames", {})
    if not frames:
        return "[]", "[]", []
    if frame < 0 or frame not in frames:
        frame = next(iter(frames))
    pos, neg, boxes = [], [], []
    for oid, v in frames.get(frame, {}).items():
        if obj and oid != obj:
            continue
        for x, y, l in v["points"]:
            (pos if l else neg).append({"x": int(round(x)), "y": int(round(y))})
        if v["box"] is not None:
            x0, y0, x1, y1 = v["box"]
            boxes.append({"x": int(round(x0)), "y": int(round(y0)),
                          "width": int(round(x1 - x0)), "height": int(round(y1 - y0))})
    return json.dumps(pos), json.dumps(neg), boxes


def canonical(p: dict, width: int | None = None, height: int | None = None, frames=None) -> dict:
    """Masking-relevant content only: per frame, per object, points (0.01 px) + labels + box, in the
    target pixel space when width/height are given. No names, colours, source_node, view state."""
    fr = scaled(p, width, height) if (width and height) else p.get("frames", {})
    out = {}
    for f in sorted(fr, key=int):
        if frames is not None and int(f) not in frames:
            continue
        objs = {}
        for oid in sorted(fr[f], key=int):
            v = fr[f][oid]
            pts = [[round(float(x), 2), round(float(y), 2), int(l)] for x, y, l in v.get("points") or []]
            box = None if v.get("box") is None else [round(float(b), 2) for b in v["box"]]
            if pts or box:
                objs[str(int(oid))] = {"p": pts, "b": box}
        if objs:
            out[str(int(f))] = objs
    return out


def prompt_hash(p: dict, width: int | None = None, height: int | None = None, frames=None, **extra) -> str:
    import hashlib

    blob = json.dumps({"c": canonical(p, width, height, frames), "x": extra}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()
