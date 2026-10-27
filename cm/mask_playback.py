"""CM Mask Playback: fast animation of a mask and/or already-highlighted frames.

Downscales to a short preview, writes one small mp4, and shows it on the node.
No VAE, no full-size overlay tensor. Wire CM Segment ``mask`` (and optionally
its images or the source video) or a highlighted IMAGE batch.

Cloned from CuteAndFunnyNodes ``CF Masking Playback`` and retitled for the CM
pack (new node id, no dependency on the CF pack).
"""
from __future__ import annotations

import math
import os
import time
import uuid
from fractions import Fraction

import torch
import torch.nn.functional as F
from comfy_api.latest import InputImpl, io, ui

_CM_PLAYBACK_COLOR = (1.0, 0.2, 0.2)
_CM_PLAYBACK_FPS = 24.0
_CM_FPS_NOTICE_SHOWN: list = []


def cm_playback_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    """Longest side <= max_side, both even (yuv420). Never upscales."""
    height, width = max(1, int(height)), max(1, int(width))
    max_side = int(max_side)
    if max_side > 0 and max(height, width) > max_side:
        scale = max_side / max(height, width)
        height = max(2, int(round(height * scale)))
        width = max(2, int(round(width * scale)))
    if height % 2:
        height -= 1
    if width % 2:
        width -= 1
    return max(2, height), max(2, width)


def cm_fit_batch(x: torch.Tensor, n: int) -> torch.Tensor:
    """Trim, or repeat the last item, to n along dim 0. A length-1 batch expands."""
    n = int(n)
    have = int(x.shape[0])
    if have == n:
        return x
    if have == 1:
        return x.expand(n, *x.shape[1:])
    if have > n:
        return x[:n]
    return torch.cat([x, x[-1:].expand(n - have, *x.shape[1:])], 0)


def cm_chunk_images(frames: torch.Tensor, a: int, b: int, ph: int, pw: int) -> torch.Tensor:
    """[N,H,W,C] slice -> float [b-a, ph, pw, 3], area-downscaled. No upscale."""
    x = frames[a:b, ..., :3].float()
    if int(x.shape[1]) == ph and int(x.shape[2]) == pw:
        return x
    return F.interpolate(x.movedim(-1, 1), size=(ph, pw), mode="area").movedim(1, -1)


def cm_chunk_mask(mask: torch.Tensor, a: int, b: int, ph: int, pw: int) -> torch.Tensor:
    """[N,H,W] slice -> float [b-a, ph, pw] in 0..1. Bilinear keeps a soft edge."""
    m = mask[a:b].float().clamp(0, 1)
    if int(m.shape[-2]) == ph and int(m.shape[-1]) == pw:
        return m
    return F.interpolate(m[:, None], size=(ph, pw), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)


def cm_highlight(images: torch.Tensor | None, mask: torch.Tensor | None, opacity: float) -> torch.Tensor:
    """images [k,ph,pw,3] and/or mask [k,ph,pw] -> highlighted frames. Mask alone is red on black."""
    opacity = float(max(0.0, min(1.0, opacity)))
    color = torch.tensor(_CM_PLAYBACK_COLOR, dtype=torch.float32).view(1, 1, 1, 3)
    if mask is None:
        return images.float().clamp(0, 1)
    m = mask.float().clamp(0, 1)[..., None]
    tint = color * (opacity * m)
    if images is None:
        return tint.clamp(0, 1)
    return (images.float() * (1.0 - opacity * m) + tint).clamp(0, 1)


def _as_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        return mask[None]
    if mask.ndim == 3:
        return mask
    return mask.reshape(-1, mask.shape[-2], mask.shape[-1])


def _write_mp4(path: str, n: int, ph: int, pw: int, fps: float, frame_fn) -> None:
    import av
    rate = Fraction(round(float(fps) * 1000), 1000)
    with av.open(path, mode="w", format="mp4") as output:
        stream = output.add_stream("libx264", rate=rate)
        stream.width = pw
        stream.height = ph
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "28"}
        for a in range(0, n, 8):
            comp = frame_fn(a, min(n, a + 8))
            u8 = (comp.clamp(0, 1) * 255.0).to(torch.uint8).numpy()
            for i in range(u8.shape[0]):
                frame = av.VideoFrame.from_ndarray(u8[i], format="rgb24")
                output.mux(stream.encode(frame.reformat(format="yuv420p")))
        output.mux(stream.encode(None))


def _cm_resample_indices(n: int, src_fps: float, dst_fps: float = _CM_PLAYBACK_FPS) -> list[int]:
    """Nearest source frame for each output time t = k / dst over the source duration."""
    n = int(n)
    if n <= 0:
        return []
    src, dst = float(src_fps), float(dst_fps)
    if src <= 0 or dst <= 0 or abs(src - dst) < 1e-6:
        return list(range(n))
    duration = n / src
    count = max(1, int(math.ceil(duration * dst - 1e-6)))
    return [min(n - 1, int(math.floor((k / dst) * src + 0.5 + 1e-9))) for k in range(count)]


def _cm_resample_to_24(frames: torch.Tensor, src_fps: float) -> torch.Tensor:
    idx = _cm_resample_indices(int(frames.shape[0]), src_fps, _CM_PLAYBACK_FPS)
    if len(idx) == int(frames.shape[0]) and idx == list(range(len(idx))):
        return frames
    return frames[torch.tensor(idx, dtype=torch.long, device=frames.device)]


def _cm_video_source(value, video_fps, label="CM Mask Playback"):
    """IMAGE tensor or VIDEO input -> (frames, source fps)."""
    detected = None
    if hasattr(value, "get_components"):
        comps = value.get_components()
        frames = comps.images
        rate = getattr(comps, "frame_rate", None)
        detected = float(rate) if rate else None
    else:
        frames = value
    forced = float(video_fps or 0.0)
    if forced > 0:
        fps = forced
    elif detected and detected > 0:
        fps = detected
        if abs(fps - _CM_PLAYBACK_FPS) > 1e-3:
            print(f"[{label}] images: source {fps:.3f} fps -> resampled to {_CM_PLAYBACK_FPS:g} fps by timestamp.")
    else:
        fps = float(_CM_PLAYBACK_FPS)
        if not _CM_FPS_NOTICE_SHOWN:
            _CM_FPS_NOTICE_SHOWN.append(True)
            print(f"[{label}] IMAGE frames carry no fps; assuming {_CM_PLAYBACK_FPS:g} (shown once). Wire a VIDEO or set video_fps.")
    return frames, fps


class CMMaskPlayback(io.ComfyNode):
    """Downscale a mask or highlighted frames and play them on the node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CMMaskPlayback",
            display_name="CM Mask Playback",
            category="CoordinationMatrix",
            is_output_node=True,
            search_aliases=["cm mask playback", "mask preview", "mask video", "cm segment preview", "coordinationmatrix"],
            description=(
                "Fast player for a mask or for frames that are already highlighted. "
                "Downscales to max_side, writes one small mp4, and plays it on the node. "
                "Wire CM Segment `mask` here (optionally with its images or the source video), "
                "or the highlighted IMAGE from a preview. Mask alone is red on black. "
                "Images alone play as they are. Both paints the mask onto the images. No VAE."
            ),
            inputs=[
                io.MultiType.Input(
                    io.Image.Input("images", tooltip="Highlighted frames, source frames, or a VIDEO. Optional if mask is connected. A VIDEO is resampled to 24 fps only when a mask is also connected, so it lines up with CM Segment."),
                    types=[io.Image, io.Video],
                    optional=True,
                ),
                io.Mask.Input("mask", optional=True, tooltip="MASK batch, 1 = highlighted. One mask per frame, or a single mask for every frame. Optional if images is connected."),
                io.Int.Input("max_side", default=512, min=64, max=1280, step=16,
                    tooltip="Longest side of the player. Lower is faster. Never upscales."),
                io.Float.Input("fps", default=24.0, min=0.0, max=60.0, step=1.0,
                    tooltip="Player rate. Does not drop or add frames. 0 = the VIDEO's own rate, or 24 if the input is an IMAGE batch."),
                io.Float.Input("opacity", default=0.55, min=0.05, max=1.0, step=0.05,
                    tooltip="How strong the red highlight is when a mask is connected. Ignored for images-only playback."),
                io.Float.Input("video_fps", default=0.0, min=0.0, max=240.0, step=0.001, advanced=True,
                    tooltip="Source fps override when images is a VIDEO and a mask is connected (the 24 fps resample). 0 = auto."),
            ],
            outputs=[
                io.Video.Output(display_name="video", tooltip="The small preview mp4."),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, images=None, mask=None, max_side=512, fps=24.0, opacity=0.55, video_fps=0.0) -> io.NodeOutput:
        import folder_paths
        if images is None and mask is None:
            raise ValueError("CM Mask Playback: connect a mask, highlighted frames, or both.")
        t0 = time.perf_counter()
        frames = None
        src_fps = None
        if images is not None:
            if hasattr(images, "get_components"):
                frames, src_fps = _cm_video_source(images, video_fps)
                if mask is not None:
                    frames = _cm_resample_to_24(frames, src_fps)
                    src_fps = 24.0
            else:
                frames = images
        mk = _as_mask(mask) if mask is not None else None
        note = ""
        if frames is not None and mk is not None and int(frames.shape[0]) != int(mk.shape[0]):
            if int(mk.shape[0]) == 1:
                mk = cm_fit_batch(mk, int(frames.shape[0]))
            elif int(frames.shape[0]) == 1:
                frames = cm_fit_batch(frames, int(mk.shape[0]))
            else:
                n_keep = min(int(frames.shape[0]), int(mk.shape[0]))
                note = f" length trimmed to {n_keep} (images {int(frames.shape[0])}, mask {int(mk.shape[0])})"
                frames = frames[:n_keep]
                mk = mk[:n_keep]
        if frames is not None:
            n, src_h, src_w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        else:
            n, src_h, src_w = int(mk.shape[0]), int(mk.shape[-2]), int(mk.shape[-1])
        if n < 1:
            raise ValueError("CM Mask Playback: no frames.")
        ph, pw = cm_playback_size(src_h, src_w, max_side)
        play_fps = float(fps) if float(fps or 0.0) > 0 else float(src_fps or 24.0)
        if play_fps <= 0:
            play_fps = 24.0

        def frame_fn(a, b):
            imgs = cm_chunk_images(frames, a, b, ph, pw) if frames is not None else None
            part = cm_chunk_mask(mk, a, b, ph, pw) if mk is not None else None
            return cm_highlight(imgs, part, opacity)

        temp_dir = folder_paths.get_temp_directory()
        filename = f"cm_mask_playback_{uuid.uuid4().hex[:8]}.mp4"
        path = os.path.join(temp_dir, filename)
        t_enc = time.perf_counter()
        _write_mp4(path, n, ph, pw, play_fps, frame_fn)
        enc_s = time.perf_counter() - t_enc
        kind = "mask on images" if frames is not None and mk is not None else ("images" if mk is None else "mask")
        info = (f"{kind}; {n}f {pw}x{ph} @ {play_fps:g} fps (source {src_w}x{src_h})"
                f"{note}; encode {enc_s:.2f}s, total {time.perf_counter() - t0:.2f}s")
        print(f"[CM Mask Playback] {info}")
        video = InputImpl.VideoFromFile(path)
        preview = ui.PreviewVideo([ui.SavedResult(filename, "", io.FolderType.temp)])
        return io.NodeOutput(video, info, ui=preview)
