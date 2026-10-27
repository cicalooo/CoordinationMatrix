"""Mask post-processing and overlay rendering (CPU/GPU torch, scipy for labeling)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _dev():
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return torch.device("cpu")


def _disk_kernel(r: int, device, dtype):
    import math
    r = int(r)
    y, x = torch.meshgrid(torch.arange(-r, r + 1, device=device),
                          torch.arange(-r, r + 1, device=device), indexing="ij")
    return ((x * x + y * y) <= r * r).to(dtype).view(1, 1, 2 * r + 1, 2 * r + 1)


def dilate(m: torch.Tensor, px: int, shape: str = "square") -> torch.Tensor:
    """m float [B,H,W]; px>0 grows, px<0 shrinks.

    Large radii are applied iteratively (32px steps) so grow=256 does not
    allocate a 513px kernel. shape="circle" uses a disk kernel (binary).
    """
    if px == 0:
        return m
    px = int(px)
    if abs(px) > 32:
        out = m
        rem = px
        while rem != 0:
            step = max(-32, min(32, rem))
            out = dilate(out, step, shape=shape)
            rem -= step
        return out
    if str(shape).lower().startswith("circ"):
        r = abs(px)
        k = _disk_kernel(r, m.device, m.dtype)
        x = (m > 0.5).float().unsqueeze(1)
        y = F.conv2d(F.pad(x, (r, r, r, r), mode="constant", value=0), k)
        if px > 0:
            return (y > 0).to(m.dtype).squeeze(1)
        return (y >= (k.sum() - 0.5)).to(m.dtype).squeeze(1)
    k = 2 * abs(px) + 1
    x = m.unsqueeze(1)
    if px > 0:
        x = F.max_pool2d(x, k, stride=1, padding=abs(px))
    else:
        x = -F.max_pool2d(-x, k, stride=1, padding=abs(px))
    return x.squeeze(1)


def feather(m: torch.Tensor, px: float) -> torch.Tensor:
    if px <= 0:
        return m
    sigma = float(px) / 2.0
    r = max(1, int(round(sigma * 3)))
    if r > 64:
        # Bound the kernel: blur at reduced resolution, then upsample.
        s = max(2, (r + 63) // 64)
        n, h, w = m.shape
        dh, dw = max(1, h // s), max(1, w // s)
        small = F.interpolate(m.unsqueeze(1), size=(dh, dw), mode="bilinear", align_corners=False)
        small = feather(small.squeeze(1), px / s)
        return F.interpolate(small.unsqueeze(1), size=(h, w), mode="bilinear",
                             align_corners=False).squeeze(1).clamp(0, 1)
    t = torch.arange(-r, r + 1, device=m.device, dtype=m.dtype)
    g = torch.exp(-(t * t) / (2 * sigma * sigma))
    g = g / g.sum()
    x = m.unsqueeze(1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
    return x.squeeze(1).clamp(0, 1)


def _area_filter(m: np.ndarray, max_area: int, holes: bool) -> np.ndarray:
    from scipy import ndimage

    target = ~m if holes else m
    lab, n = ndimage.label(target)
    if n == 0:
        return m
    areas = np.bincount(lab.ravel())
    small = areas <= max_area
    small[0] = False
    if holes:
        # do not treat background touching the border as a hole
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        small[border] = False
        return m | small[lab]
    return m & ~small[lab]


def fill_holes(m: torch.Tensor, max_area: int) -> torch.Tensor:
    if max_area <= 0:
        return m
    out = [torch.from_numpy(_area_filter(x.cpu().numpy() > 0.5, max_area, True)) for x in m]
    return torch.stack(out).to(m.device, m.dtype)


def remove_islands(m: torch.Tensor, max_area: int) -> torch.Tensor:
    if max_area <= 0:
        return m
    out = [torch.from_numpy(_area_filter(x.cpu().numpy() > 0.5, max_area, False)) for x in m]
    return torch.stack(out).to(m.device, m.dtype)


def temporal_smooth(m: torch.Tensor, window: int, hw_chunk: int = 65536) -> torch.Tensor:
    """Majority vote over +-window frames (m [N,H,W] binary). HW-chunked to bound memory."""
    if window <= 0 or m.shape[0] < 2:
        return m
    n, h, w = m.shape
    flat = m.reshape(n, -1).t().contiguous()  # [HW,N]
    out = torch.empty_like(flat)
    k = 2 * int(window) + 1
    for s in range(0, flat.shape[0], hw_chunk):
        x = flat[s:s + hw_chunk].unsqueeze(1).to(m.dtype)  # [C,1,N]
        x = F.avg_pool1d(F.pad(x, (window, window), mode="replicate"), k, stride=1)
        out[s:s + hw_chunk] = (x.squeeze(1) > 0.5).to(flat.dtype)
    return out.t().reshape(n, h, w).to(m.dtype)


def process(m: torch.Tensor, grow: int, feather_px: float, holes: int, islands: int, smooth: int,
            chunk: int = 64) -> torch.Tensor:
    """m float [N,H,W] -> float [N,H,W] 0..1.

    Order: clean per-frame holes/islands first (so the temporal vote does not
    smear them across frames), then temporal smooth, then grow/feather.
    """
    m = (m > 0.5).float()
    if holes > 0:
        m = fill_holes(m, holes)
    if islands > 0:
        m = remove_islands(m, islands)
    if smooth > 0:
        m = temporal_smooth(m, smooth)
    if grow == 0 and feather_px <= 0:
        return m
    dev = _dev()
    out = []
    for i in range(0, m.shape[0], chunk):
        x = m[i:i + chunk].to(dev)
        x = feather(dilate(x, grow), feather_px)
        out.append(x.cpu())
    return torch.cat(out, 0)


def outline(m: torch.Tensor, px: int, shape: str = "square") -> torch.Tensor:
    """Inner border of a binary mask batch [B,H,W]."""
    if px <= 0:
        return torch.zeros_like(m)
    return (m - dilate(m, -px, shape=shape)).clamp(0, 1)


def resolve_exclusive(m: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Winner-takes-all over objects: m [O,N,H,W] -> same shape, losers zeroed.

    Pixels where every object is <= thresh stay background.
    """
    if m.shape[0] <= 1:
        return m
    mx, am = m.max(dim=0, keepdim=True)
    keep = (mx > float(thresh)) & (m == mx)
    # Ties (two objects with identical soft values) go to the lowest index.
    first = torch.zeros_like(m, dtype=torch.bool)
    acc = torch.zeros_like(mx, dtype=torch.bool)
    for k in range(m.shape[0]):
        take = keep[k:k + 1] & ~acc
        first[k:k + 1] = take
        acc = acc | take
    return (m * first.to(m.dtype))


def overlay(images: torch.Tensor, masks: torch.Tensor, colors: list, opacity: float, border_px: int,
            mode: str = "overlay", chunk: int = 32) -> torch.Tensor:
    """images [N,H,W,3]; masks [N,O,H,W] float/bool. Returns [N,H,W,3]."""
    dev = _dev()
    cols = torch.tensor(colors, dtype=torch.float32, device=dev).view(1, -1, 1, 1, 3)
    out = []
    for i in range(0, images.shape[0], chunk):
        img = images[i:i + chunk].to(dev).float()
        m = masks[i:i + chunk].to(dev).float()
        b, o, h, w = m.shape
        if mode == "mask":
            base = torch.zeros_like(img)
        elif mode == "matte":
            union = m.amax(1, keepdim=False).unsqueeze(-1) if o else torch.zeros_like(img[..., :1])
            out.append((img * union).cpu())
            continue
        else:
            base = img
        res = base
        for k in range(o):
            mk = m[:, k].unsqueeze(-1)
            a = (opacity if mode == "overlay" else 1.0) * mk
            res = res * (1 - a) + cols[:, k] * a
            if border_px > 0:
                ed = outline(m[:, k], border_px).unsqueeze(-1)
                res = res * (1 - ed) + cols[:, k] * ed
        out.append(res.clamp(0, 1).cpu())
    return torch.cat(out, 0)
