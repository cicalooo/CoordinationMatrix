"""CoordinationMatrix V3 nodes: Model Loader, Load Video, Video Info, Point Editor, Segment."""
from __future__ import annotations

import logging
import os

import torch
from comfy_api.latest import ComfyExtension, InputImpl, Types, io

from . import h3, model, postproc, prompts as P, store, video_io

log = logging.getLogger("CoordinationMatrix")
CAT = "CoordinationMatrix"

SAM2Model = io.Custom("CM_SAM2MODEL")
Prompts = io.Custom("CM_PROMPTS")
VideoInfo = io.Custom("CM_VIDEO_INFO")

VIDEO_EXT = video_io.VIDEO_EXT


def _video_files():
    import folder_paths

    d = folder_paths.get_input_directory()
    out = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(VIDEO_EXT):
                out.append(os.path.relpath(os.path.join(root, f), d).replace(os.sep, "/"))
    return sorted(out) or [""]


# ------------------------------------------------------------------ model

class CMModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMModelLoader", display_name="CM Model Loader", category=CAT,
            description="Loads a SAM 2.1 model (auto-downloads Kijai/sam2-safetensors into models/sam2). "
                        "Managed by comfy.model_management.",
            inputs=[
                io.Combo.Input("model", options=list(model.MODELS), default="sam2.1_hiera_base_plus"),
                io.Combo.Input("precision", options=model.PRECISIONS, default="bf16"),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto", advanced=True),
            ],
            outputs=[SAM2Model.Output(display_name="sam2_model")],
        )

    @classmethod
    def execute(cls, model, precision, device="auto") -> io.NodeOutput:
        from . import model as M

        return io.NodeOutput(M.get_model(model, precision, device))


# -------------------------------------------------------------- load video

def load_video(file, fps, start, trim, extend, megapixels, resize_filter, h3_snap_mode, h3_pad,
               frame_interp="blend", eof_mode="hold"):
    """Shared by the node and tests. Returns (images, audio|None, info dict)."""
    import folder_paths

    if not file:
        raise ValueError("[CM Load Video] choose or upload a video file")
    path = folder_paths.get_annotated_filepath(file)
    meta = video_io.probe(path)
    plan = video_io.plan_file(meta, fps, start, trim, extend, h3_snap_mode, h3_pad, eof_mode)
    base, out_fps, aud, fmap = video_io.decode_file(path, plan["times"], plan["fps"], megapixels,
                                                     resize_filter, include_audio=True,
                                                     frame_interp=frame_interp, eof_mode=eof_mode)
    index = torch.tensor(plan["index"], dtype=torch.long)
    imgs = base[index]
    fmap = dict(fmap, file=file,
                times=[plan["times"][i] for i in plan["index"]],
                src_index=[fmap["src_index"][i] for i in plan["index"]],
                padded=list(plan["padded"]))
    n, h, w = int(imgs.shape[0]), int(imgs.shape[1]), int(imgs.shape[2])
    if aud is not None:
        aud = h3.h3_fit_audio(aud, n, out_fps)
    src_frames = int(round(meta["duration"] * meta["src_fps"])) if meta["duration"] > 0 else 0
    info = {
        "file": file, "path": path,
        "start": float(start), "trim": float(trim), "extend": float(extend),
        "window_seconds": plan["window_frames"] / out_fps,
        "src_fps": float(meta["src_fps"]), "src_frames": src_frames, "src_duration": float(meta["duration"]),
        "src_width": int(meta["width"]), "src_height": int(meta["height"]),
        "fps": float(out_fps), "width": w, "height": h, "length": n,
        "original_frames": int(plan["original_frames"]), "duration": n / float(out_fps),
        "first_frame": imgs[:1].clone(), "last_frame": imgs[-1:].clone(),
        "frame_map": fmap,
        "settings": {"file": file, "fps": fps, "start": start, "trim": trim, "extend": extend,
                     "megapixels": megapixels, "resize_filter": resize_filter,
                     "h3_snap_mode": h3_snap_mode, "h3_pad": h3_pad,
                     "frame_interp": frame_interp, "eof_mode": eof_mode},
    }
    return imgs, aud, info


class CMLoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMLoadVideo", display_name="CM Load Video", category=CAT,
            description="Load + trim a video on a visual timeline. Window = start .. start+trim, resampled to "
                        "fps, optional extend (hold last frame), then ALWAYS snapped to the MiniMax H3 17k+5 "
                        "frame grid. Output size targets megapixels (aspect locked, snapped to 32px for H3); "
                        "audio always passes through. GIF / animated WebP / APNG / stills are decoded via "
                        "Pillow with true frame delays; frame_interp=blend smooths low-fps sources up to 24 fps; "
                        "eof_mode=loop wraps short sources instead of holding the last frame. "
                        "Wire video_info into CM Video Info.",
            inputs=[
                io.Combo.Input("file", options=_video_files(), upload=io.UploadType.video),
                io.Float.Input("fps", default=24.0, min=0.0, max=240.0, step=1.0,
                               tooltip="Output fps (0 = source fps). MiniMax H3 expects 24."),
                io.Float.Input("start", default=0.0, min=0.0, max=86400.0, step=0.01,
                               tooltip="Window start in seconds (drag the in-handle on the timeline)."),
                io.Float.Input("trim", default=0.0, min=0.0, max=86400.0, step=0.01,
                               tooltip="Window length in seconds, 0 = to the end (drag the out-handle)."),
                io.Float.Input("extend", default=0.0, min=0.0, max=600.0, step=0.01,
                               tooltip="Hold the last frame for this many extra seconds (before the H3 snap)."),
                io.Float.Input("megapixels", default=0.4, min=0.05, max=16.0, step=0.05,
                               tooltip="Target total size in megapixels (aspect locked to source, "
                                       "then snapped to 32px so H3 always accepts it)."),
                io.Combo.Input("resize_filter", options=video_io.RESIZE_FILTERS, default="bicubic"),
                io.Combo.Input("frame_interp", options=video_io.FRAME_INTERPS, default="blend",
                               tooltip="Resampling to fps: nearest holds/duplicates frames (steppy when "
                                       "upsampling low-fps GIFs); blend crossfades neighbours, giving the H3 "
                                       "motion model smoother motion. Bit-identical to nearest when fps matches."),
                io.Combo.Input("eof_mode", options=video_io.EOF_MODES, default="hold",
                               tooltip="Window past the source end (short GIF / still): hold repeats the last "
                                       "frame, loop wraps to the window start."),
                io.Combo.Input("h3_snap_mode", options=h3.H3_SNAP_MODES, default="down",
                               tooltip="17k+5 snap (always on). down trims the tail, nearest picks the "
                                       "closest, up pads."),
                io.Combo.Input("h3_pad", options=h3.H3_PAD_MODES, default="reflect",
                               tooltip="When the snap pads: reflect (ping-pong) or hold (repeat last)."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Video.Output(display_name="video"),
                VideoInfo.Output(display_name="video_info"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, file="", **kw):
        import folder_paths

        try:
            p = folder_paths.get_annotated_filepath(file)
            return f"{file}:{os.path.getmtime(p)}:{os.path.getsize(p)}:{sorted(kw.items())}"
        except Exception:
            return f"{file}:{sorted(kw.items())}"

    @classmethod
    def validate_inputs(cls, file="", **kw):
        return True

    @classmethod
    def execute(cls, file, fps, start, trim, extend, megapixels, resize_filter,
                h3_snap_mode, h3_pad, frame_interp="blend", eof_mode="hold") -> io.NodeOutput:
        imgs, aud, info = load_video(file, fps, start, trim, extend, megapixels, resize_filter,
                                    h3_snap_mode, h3_pad, frame_interp, eof_mode)
        uid = cls.hidden.unique_id
        info["node"] = str(uid)
        store.put_frames(uid, imgs, info["frame_map"])
        vid = InputImpl.VideoFromComponents(Types.VideoComponents(images=imgs, audio=aud,
                                                                  frame_rate=video_io.frac(info["fps"])))
        summary = {k: info[k] for k in ("width", "height", "length", "original_frames", "fps", "duration",
                                        "src_fps", "src_frames", "src_duration")}
        return io.NodeOutput(imgs, aud, vid, info, ui={"cm_info": [summary]})


# -------------------------------------------------------------- video info

class CMVideoInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMVideoInfo", display_name="CM Video Info", category=CAT,
            description="Unpacks CM Load Video's video_info. width/height/length/fps/duration wire straight into "
                        "MiniMax H3 (Empty AV Latent, Image to Video, Reference to Video) and CuteAndFunny H3 "
                        "nodes; first_frame/last_frame into first_frame/last_frame inputs.",
            inputs=[VideoInfo.Input("video_info")],
            outputs=[
                io.Int.Output(display_name="width", tooltip="Output frame width (H3 width)."),
                io.Int.Output(display_name="height", tooltip="Output frame height (H3 height)."),
                io.Int.Output(display_name="length", tooltip="Frame count after trim/fps/extend + 17k+5 snap "
                                                            "(H3 length)."),
                io.Float.Output(display_name="fps", tooltip="Output fps (video_fps / frame_rate / fps)."),
                io.Float.Output(display_name="duration", tooltip="length / fps in seconds (duration_seconds)."),
                io.Image.Output(display_name="first_frame", tooltip="Single still: first output frame."),
                io.Image.Output(display_name="last_frame", tooltip="Single still: last output frame."),
                io.Int.Output(display_name="original_frames",
                              tooltip="Frames before the 17k+5 snap (wire to CF H3 Unpad original_frames)."),
                io.Float.Output(display_name="start", tooltip="Window start (s) in the source."),
                io.Float.Output(display_name="trim", tooltip="Window length setting (s), 0 = to end."),
                io.Float.Output(display_name="source_fps"),
                io.Int.Output(display_name="source_frames"),
                io.String.Output(display_name="source_path"),
            ],
        )

    @classmethod
    def execute(cls, video_info) -> io.NodeOutput:
        i = video_info
        return io.NodeOutput(int(i["width"]), int(i["height"]), int(i["length"]), float(i["fps"]),
                             float(i["duration"]), i["first_frame"], i["last_frame"], int(i["original_frames"]),
                             float(i["start"]), float(i["trim"]), float(i["src_fps"]), int(i["src_frames"]),
                             str(i["path"]))


# ------------------------------------------------------------ point editor

class CMPointEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMPointEditor", display_name="CM Point Editor", category=CAT, is_output_node=True,
            description="Interactive point / box editor. Reads frames live from the connected CM Load Video "
                        "(no run needed); with only images connected it uses the frames from the last run. "
                        "Click = positive, right-click empty = negative, right-click a point/box = delete, "
                        "Ctrl+drag = box. Re-runs only when the points/boxes actually change.",
            inputs=[
                VideoInfo.Input("video_info", optional=True, tooltip="From CM Load Video: live frames."),
                io.Image.Input("images", optional=True, tooltip="Any IMAGE batch (frames shown after a run)."),
                io.Boolean.Input("multi_keyframe", default=False,
                                 tooltip="Off: one keyframe (default frame 0). On: prompts on any number of "
                                         "frames, all used as keyframes."),
                io.String.Input("editor_data", default="", socketless=True),
            ],
            outputs=[Prompts.Output(display_name="prompts")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, multi_keyframe=False, editor_data="", **kw):
        # Points/boxes + object colours only; view state never reaches here (the editor keeps it out of editor_data).
        try:
            p = P.parse(editor_data)
            return P.prompt_hash(p, multi=bool(multi_keyframe), space=p["space"],
                                 colors=[o["color"] for o in p["objects"]])
        except Exception:
            return editor_data

    @classmethod
    def execute(cls, multi_keyframe=False, editor_data="", video_info=None, images=None) -> io.NodeOutput:
        uid = str(cls.hidden.unique_id)
        pr = P.parse(editor_data)
        if not multi_keyframe and len(pr["frames"]) > 1:
            k = next(iter(pr["frames"]))
            pr["frames"] = {k: pr["frames"][k]}
        size = None
        if images is not None:
            store.put_frames(uid, images, None)
            size = [int(images.shape[2]), int(images.shape[1])]
        elif video_info is not None:
            size = [int(video_info["width"]), int(video_info["height"])]
        if not pr["space"][0] and size:
            pr["space"] = size
        pr["source_node"] = uid
        n_pts = sum(len(v["points"]) + (v["box"] is not None) for o in pr["frames"].values() for v in o.values())
        return io.NodeOutput(pr, ui={"cm_editor": [{"keyframes": list(pr["frames"]), "prompts": n_pts,
                                                    "has_images": images is not None,
                                                    "size": size or [0, 0]}]})


# ----------------------------------------------------------------- segment

class CMSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMSegment", display_name="CM Segment", category=CAT,
            description="SAM 2.1 segmentation. One image -> best-IoU multimask. Video -> bidirectional "
                        "propagation from every keyframe (chunked with mask re-prompting), then optional "
                        "refinement. Outputs union mask, per-object masks and an overlay preview.",
            inputs=[
                SAM2Model.Input("sam2_model"),
                io.Image.Input("images"),
                Prompts.Input("prompts"),
                io.Combo.Input("mode", options=["auto", "video", "image"], default="auto",
                               tooltip="auto: 1 frame = image, more = video tracking. image: per-frame, no "
                                       "tracking (each frame uses the nearest earlier keyframe's prompts)."),
                io.Combo.Input("keyframe_mask", options=["best_iou", "points"], default="best_iou",
                               tooltip="best_iou: keyframes are first segmented with 3 candidates and the best "
                                       "predicted-IoU mask seeds the tracker. points: feed raw clicks."),
                io.Int.Input("chunk_size", default=0, min=0, max=100000, step=1,
                             tooltip="0 = whole clip in one state. N = windows of N frames (less memory)."),
                io.Boolean.Input("offload_video", default=True, advanced=True),
                io.Boolean.Input("offload_state", default=False, advanced=True),
                io.Int.Input("fill_holes", default=0, min=0, max=1 << 20, tooltip="Fill holes up to N px area."),
                io.Int.Input("remove_islands", default=0, min=0, max=1 << 20,
                             tooltip="Remove blobs up to N px area."),
                io.Int.Input("grow", default=1, min=-256, max=256, tooltip="Grow (+) / shrink (-) in px. 1 seals 1px SAM gaps."),
                io.Float.Input("feather", default=1.0, min=0.0, max=256.0, step=0.5,
                               tooltip="Soft edge in px (sigma = px/2). 1.0 removes stair-steps without bleed."),
                io.Int.Input("temporal_smooth", default=0, min=0, max=15,
                             tooltip="Majority vote over +/- N frames (removes flicker)."),
                io.Float.Input("overlay_opacity", default=0.45, min=0.0, max=1.0, step=0.05),
                io.Combo.Input("resolve", options=["union", "exclusive"], default="union",
                               tooltip="union: overlapping objects merge. exclusive: each pixel goes to the "
                                       "strongest object (background stays background)."),
                io.Int.Input("border_px", default=2, min=0, max=32, step=1,
                             tooltip="Overlay outline width in px (0 = off)."),
                io.Boolean.Input("use_cache", default=True, advanced=True),
            ],
            outputs=[
                io.Mask.Output(display_name="mask", tooltip="Union of all objects, one per frame (1 = object)."),
                io.Mask.Output(display_name="object_masks",
                               tooltip="Per object, object-major: [obj1 frames..., obj2 frames...]."),
                io.Image.Output(display_name="overlay"),
                io.Int.Output(display_name="object_count"),
                io.Float.Output(display_name="confidence",
                                tooltip="Mean keyframe predicted-IoU across objects (rough confidence)."),
                io.String.Output(display_name="warnings",
                                 tooltip="Clamped prompts, empty keyframe masks, empty frames, auto-chunk notes."),
            ],
        )

    @classmethod
    def execute(cls, sam2_model, images, prompts, mode="auto", keyframe_mask="best_iou", chunk_size=0,
                offload_video=True, offload_state=False, fill_holes=0, remove_islands=0, grow=1, feather=1.0,
                temporal_smooth=0, overlay_opacity=0.45, resolve="union", border_px=2,
                use_cache=True) -> io.NodeOutput:
        from . import engine

        n = int(images.shape[0])
        use_video = mode == "video" or (mode == "auto" and n > 1)
        key = store.cache_key(images, prompts, model=sam2_model.name, precision=sam2_model.precision,
                              video=use_video, kf=keyframe_mask, chunk=int(chunk_size))
        info: dict = {}
        hit = store.cache_load_full(key) if use_cache else None
        if hit is not None:
            masks, ids, info = hit
        else:
            with model.lock():
                if use_video:
                    res = engine.propagate(sam2_model, images, prompts, chunk_size, keyframe_mask,
                                           offload_video, offload_state, return_info=True)
                else:
                    res = engine.segment_images(sam2_model, images, prompts, return_info=True)
                if isinstance(res, tuple) and len(res) == 3:
                    masks, ids, info = res
                else:  # mocked engine in tests returns (masks, ids)
                    masks, ids = res
                    info = {}
            if use_cache:
                store.cache_save(key, masks, ids, info)
        cols = P.colors(prompts, ids)
        o = len(ids)
        per = masks.float().permute(1, 0, 2, 3)  # [O,N,H,W]
        refined = torch.stack([postproc.process(per[k], grow, feather, fill_holes, remove_islands,
                                                temporal_smooth) for k in range(o)]) if o else per
        if str(resolve).lower().startswith("excl") and o > 1:
            refined = postproc.resolve_exclusive(refined)
        union = refined.amax(0)
        obj = refined.reshape(o * n, *refined.shape[2:])
        ov = postproc.overlay(images, refined.permute(1, 0, 2, 3), [P.hex_to_rgb(c) for c in cols],
                              overlay_opacity, int(border_px))
        src = prompts.get("source_node")
        if src is not None:
            store.put_track(src, {"masks": (refined.permute(1, 0, 2, 3) > 0.5), "colors": cols, "ids": ids})
        scores = info.get("scores") if isinstance(info, dict) else None
        conf = float(sum(scores.values()) / len(scores)) if scores else 0.0
        warns = info.get("warnings") if isinstance(info, dict) else None
        warn_str = "\n".join(str(x) for x in warns) if warns else ""
        return io.NodeOutput(union, obj, ov, o, conf, warn_str)


class CoordinationMatrixExtension(ComfyExtension):
    async def get_node_list(self):
        from . import routes
        from .mask_playback import CMMaskPlayback

        routes.register()
        return [CMModelLoader, CMLoadVideo, CMVideoInfo, CMPointEditor, CMSegment, CMMaskPlayback]


async def comfy_entrypoint() -> CoordinationMatrixExtension:
    return CoordinationMatrixExtension()
