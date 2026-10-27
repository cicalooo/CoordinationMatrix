"""MiniMax H3 17k+5 length helpers.

Copied (with light edits) from CuteAndFunnyNodes `Load Video (Versatile)` so this
pack stays standalone and produces identical lengths.
"""
from __future__ import annotations

import math

import torch

_H3_FPS = 24
H3_SNAP_MODES = ["down", "nearest", "up"]
H3_PAD_MODES = ["reflect", "hold"]


def _h3_align_frame_count(n: int) -> int:
    """Smallest 17k+5 value >= n (min 5)."""
    n = max(1, int(n))
    if n <= 5:
        return 5
    return n + ((5 - n) % 17)


def duration_from_seconds(seconds: float, as_float: bool):
    seconds = float(seconds or 0.0)
    if as_float:
        return seconds
    return int(math.floor(seconds + 0.5))


def duration_seconds(total_frames: int, frame_rate, as_float: bool):
    fps = float(frame_rate) if frame_rate else 0.0
    seconds = (int(total_frames) / fps) if fps > 0 else 0.0
    return duration_from_seconds(seconds, as_float)


def h3_is_grid(n: int) -> bool:
    n = int(n)
    return n >= 5 and n % 17 == 5


def h3_snap_target(n: int, mode: str = "down") -> int:
    n = max(1, int(n))
    if h3_is_grid(n):
        return n
    up = int(_h3_align_frame_count(n))
    down = n - ((n - 5) % 17) if n >= 5 else 5
    mode = str(mode or "down").strip().lower()
    if mode == "up":
        return up
    if mode == "nearest":
        return down if (n - down) <= (up - n) else up
    return down


def h3_pad_index(n: int, target: int, pad_mode: str = "reflect") -> torch.Tensor:
    """Indices (len target) into a length-n sequence: identity, then reflect
    (ping-pong) or hold-last; trimming keeps the head."""
    n, target = int(n), int(target)
    if target <= n:
        return torch.arange(target, dtype=torch.long)
    idx = list(range(n))
    if str(pad_mode) == "reflect" and n >= 2:
        period = 2 * (n - 1)
        for j in range(n, target):
            k = j % period
            idx.append(k if k < n else period - k)
    else:
        idx.extend([n - 1] * (target - n))
    return torch.tensor(idx, dtype=torch.long)


def h3_fit_audio(audio, n_frames: int, fps: float = 24.0):
    if not audio:
        return audio
    wav = audio.get("waveform")
    sr = int(audio.get("sample_rate") or 0)
    if wav is None or sr <= 0 or not fps or float(fps) <= 0:
        return audio
    need = max(1, int(round(int(n_frames) * sr / float(fps))))
    have = int(wav.shape[-1])
    if have == need:
        return audio
    out = dict(audio)
    if have > need:
        out["waveform"] = wav[..., :need].contiguous()
    elif have > 0:
        idx = h3_pad_index(have, need, "reflect").to(wav.device)
        out["waveform"] = wav.index_select(-1, idx).contiguous()
    return out


def h3_snap_note(label: str, n: int, target: int, mode: str, pad_mode: str) -> str:
    if n == target:
        return ""
    if target < n:
        msg = f"{n}f -> {target}f (trimmed {n - target} tail frames; snap {mode})"
    else:
        msg = f"{n}f -> {target}f (padded {target - n} frames by {pad_mode}; snap {mode})"
    print(f"[{label}] WARNING length not on the 17k+5 grid: {msg}")
    return msg
