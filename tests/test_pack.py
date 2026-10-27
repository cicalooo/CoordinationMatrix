"""CPU-only checks: imports, schemas, prompt/mask helpers, and a tiny random-weight SAM2 run.

Run from the directory that CONTAINS the pack:  python -m pytest CoordinationMatrix/tests -q
or:  python CoordinationMatrix/tests/test_pack.py
"""
import asyncio
import importlib
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import stubs  # noqa: E402

TMP = stubs.install()
PKG = os.path.basename(os.path.dirname(HERE))
pack = importlib.import_module(PKG)
nodes = importlib.import_module(PKG + ".cm.nodes")
P = importlib.import_module(PKG + ".cm.prompts")
post = importlib.import_module(PKG + ".cm.postproc")
store = importlib.import_module(PKG + ".cm.store")
engine = importlib.import_module(PKG + ".cm.engine")
model = importlib.import_module(PKG + ".cm.model")
vio = importlib.import_module(PKG + ".cm.video_io")
h3 = importlib.import_module(PKG + ".cm.h3")


def test_exports():
    names = {n for n in dir(pack) if not n.startswith("_")}
    assert "NODE_CLASS_MAPPINGS" not in names
    assert pack.WEB_DIRECTORY == "./web" and callable(pack.comfy_entrypoint)
    ext = asyncio.run(pack.comfy_entrypoint())
    lst = asyncio.run(ext.get_node_list())
    titles = {c.define_schema()["display_name"] for c in lst}
    assert titles == {"CM Model Loader", "CM Load Video", "CM Video Info", "CM Point Editor", "CM Segment", "CM Mask Playback"}
    ids = [c.define_schema()["node_id"] for c in lst]
    assert len(set(ids)) == len(ids) and "CuteAndFunnyLoadVideo" not in ids and "CMMaskPlayback" in ids
    assert all(c.define_schema()["category"] in ("CoordinationMatrix", "CuteAndFunny/video") for c in lst)
    assert {c.define_schema()["display_name"] for c in lst if c.define_schema().get("is_output_node")} == {"CM Point Editor", "CM Mask Playback"}


def test_prompts_roundtrip():
    raw = {"space": [200, 100], "frames": {"3": {"1": {"points": [[10, 20, 1], {"x": 5, "y": 6, "label": 0}],
                                                       "box": [50, 60, 10, 20]}}}}
    p = P.parse(raw)
    assert p["frames"][3][1]["box"] == [10, 20, 50, 60]
    assert P.object_ids(p) == [1]
    s = P.scaled(p, 400, 200)
    assert s[3][1]["points"][0][:2] == [20, 40]
    q = P.parse(P.to_json(p))
    assert q["frames"] == p["frames"]
    pos, neg, boxes = P.sam3_style(p, -1, 0)
    assert '"x": 10' in pos and '"x": 5' in neg and boxes[0]["width"] == 40


def test_postproc():
    m = torch.zeros(3, 32, 32)
    m[:, 8:24, 8:24] = 1
    m[:, 15, 15] = 0          # hole
    m[:, 1, 1] = 1            # island
    out = post.process(m, grow=2, feather_px=0, holes=4, islands=4, smooth=1)
    assert out[0, 15, 15] == 1 and out[0, 1, 1] == 0 and out[0, 6, 6] == 1
    img = torch.rand(3, 32, 32, 3)
    ov = post.overlay(img, (m > 0).unsqueeze(1), [(1, 0, 0)], 0.5, 1)
    assert ov.shape == img.shape


def test_cache_roundtrip():
    m = torch.rand(4, 2, 9, 7) > 0.5
    store.cache_save("k", m, [1, 2])
    m2, ids = store.cache_load("k")
    assert torch.equal(m, m2) and ids == [1, 2]


def test_video_helpers():
    t, f = vio.plan_times(30, 2.0, 24, 0.5, 1.0, 1)
    assert len(t) == 24 and f == 24
    assert vio.scale_megapixels(1920, 1080, 0.4, 32) == (832, 480)
    meta = {"src_fps": 30.0, "duration": 4.0, "frames": 120, "width": 64, "height": 48}
    for mode, want in (("down", 22), ("nearest", 22), ("up", 39)):
        pl = vio.plan_file(meta, 24, 0.5, 1.2, 0.0, mode, "reflect")
        assert pl["length"] == want and h3.h3_is_grid(pl["length"]) and pl["original_frames"] == 29, (mode, pl["length"], pl["original_frames"])
        assert len(pl["index"]) == want
    pl = vio.plan_file(meta, 24, 0.0, 1.0, 1.0, "up", "hold")
    assert pl["original_frames"] == 48 and pl["length"] == 56 and pl["padded"][-1]


def _tiny_predictor():
    torch.manual_seed(0)
    net = model.build_predictor("sam2.1_hiera_tiny").float().eval()
    net.cm_maskmem_dtype = torch.float32

    class CM:
        name, precision = "tiny", "fp32"
        image_size = int(net.image_size)

        def __init__(self):
            self.net = net

        def load(self):
            return net

        def autocast(self):
            import contextlib
            return contextlib.nullcontext()

    return CM()


def test_engine_random_weights():
    """Exercises the state dict / propagation code on CPU with random weights (shape check only)."""
    if os.environ.get("CM_SKIP_ENGINE"):
        return
    cm = _tiny_predictor()
    frames = torch.rand(5, 64, 96, 3)
    p = P.parse({"space": [96, 64], "frames": {"1": {"1": {"points": [[40, 30, 1]], "box": None}},
                                               "3": {"2": {"points": [], "box": [10, 10, 50, 40]}}}})
    for kfm in ("best_iou", "points"):
        for chunk in (0, 3):
            m, ids = engine.propagate(cm, frames, p, chunk, kfm)
            assert m.shape == (5, 2, 64, 96) and ids == [1, 2] and m.dtype == torch.bool
    r = engine.segment_images(cm, frames[:2], p)
    assert r[0].shape[0] == 2


def test_video_loader_file():
    try:
        import av
    except ImportError:
        return
    import numpy as np
    path = os.path.join(TMP, "clip.mp4")
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = 128, 96, "yuv420p"
        for i in range(60):
            a = np.full((96, 128, 3), i * 4, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    imgs, aud, info = nodes.load_video("clip.mp4", 24.0, 0.5, 1.0, 0.0, 0.05, "bicubic", "down", "reflect")
    assert imgs.shape[1:] == (192, 256, 3) and imgs.shape[0] == 22 and info["length"] == 22
    assert info["original_frames"] == 24 and abs(info["duration"] - 22 / 24) < 1e-6
    vi = nodes.CMVideoInfo.execute(info).result
    assert vi[0] == 256 and vi[1] == 192 and vi[2] == 22 and vi[3] == 24.0 and vi[7] == 24
    pe = nodes.CMPointEditor.execute(False, '{"frames":{"0":{"1":{"points":[[3,4,1]]}},"5":{"1":{"points":[[1,1,0]]}}}}', info)
    assert list(pe.result[0]["frames"]) == [0] and pe.result[0]["space"] == [256, 192]


def _gif_frames(path, durations):
    import numpy as np
    from PIL import Image

    frames = [Image.fromarray(np.full((48, 64, 3), i * 30, np.uint8)) for i in range(6)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return frames


def test_gif_probe_and_blend_noop():
    import numpy as np

    _gif_frames(os.path.join(TMP, "anim.gif"), 100)
    meta = vio.probe(os.path.join(TMP, "anim.gif"))
    assert abs(meta["src_fps"] - 10.0) < 0.01 and abs(meta["duration"] - 0.6) < 1e-6
    assert meta["has_audio"] is False
    kw = dict(fps=10.0, start=0.0, trim=0.0, extend=0.0, megapixels=0.05, resize_filter="bicubic",
              h3_snap_mode="up", h3_pad="hold")
    a, _, ia = nodes.load_video("anim.gif", eof_mode="hold", frame_interp="nearest", **kw)
    b, _, ib = nodes.load_video("anim.gif", eof_mode="hold", frame_interp="blend", **kw)
    assert ia["length"] == 22 and ib["length"] == 22  # 6f snapped up to the H3 grid
    assert torch.equal(a, b)  # aligned grids: blend is bit-identical to nearest


def test_gif_upsample_blend_smooths():
    _gif_frames(os.path.join(TMP, "anim.gif"), 100)
    kw = dict(fps=24.0, start=0.0, trim=0.0, extend=0.0, megapixels=0.05, resize_filter="bicubic",
              h3_snap_mode="up", h3_pad="hold")
    n, _, _ = nodes.load_video("anim.gif", eof_mode="hold", frame_interp="nearest", **kw)
    b, _, info = nodes.load_video("anim.gif", eof_mode="hold", frame_interp="blend", **kw)
    assert n.shape == b.shape and h3.h3_is_grid(info["length"])
    assert not torch.equal(n, b)  # 10 -> 24 fps: blend crossfades the held frames
    assert bool(((b[1:] - b[:-1]).abs().sum() > 0)) and b.min() >= 0.0 and b.max() <= 1.0


def test_gif_vfr_coverage():
    _gif_frames(os.path.join(TMP, "vfr.gif"), [50, 200, 50, 200, 50, 200])
    meta = vio.probe(os.path.join(TMP, "vfr.gif"))
    assert abs(meta["duration"] - 0.75) < 1e-6 and abs(meta["src_fps"] - 8.0) < 0.01
    imgs, _, _ = nodes.load_video("vfr.gif", 8.0, 0.0, 0.0, 0.0, 0.05, "bicubic", "up", "hold",
                                  "nearest", "hold")
    px = (imgs[:, 0, 0, 0] * 255.0 + 0.5).round().int().tolist()
    # t=0.25 falls in source frame 2 ([0.25, 0.30)); hold-last would give frame 1
    assert px[2] == 60, px


def test_gif_loop_vs_hold():
    _gif_frames(os.path.join(TMP, "anim.gif"), 100)
    kw = dict(fps=10.0, start=0.0, trim=2.0, extend=0.0, megapixels=0.05, resize_filter="bicubic",
              h3_snap_mode="up", h3_pad="hold")
    h, _, ih = nodes.load_video("anim.gif", eof_mode="hold", frame_interp="nearest", **kw)
    lo, _, il = nodes.load_video("anim.gif", eof_mode="loop", frame_interp="nearest", **kw)
    assert ih["original_frames"] == 6 and il["original_frames"] == 20
    assert torch.equal(h[6], h[5])  # hold: clamp + snap pad repeat the last frame
    tail = lo[:20]
    assert torch.equal(tail[6], tail[0]) and torch.equal(tail[13], tail[1])  # loop wraps


def test_animated_webp_falls_back_to_pil():
    import numpy as np
    from PIL import Image

    frames = [Image.fromarray(np.full((48, 64, 3), i * 40, np.uint8)) for i in range(4)]
    frames[0].save(os.path.join(TMP, "anim.webp"), save_all=True, append_images=frames[1:],
                    duration=100, loop=0)
    meta = vio.probe(os.path.join(TMP, "anim.webp"))
    assert abs(meta["src_fps"] - 10.0) < 0.01 and abs(meta["duration"] - 0.4) < 1e-6
    imgs, aud, info = nodes.load_video("anim.webp", 10.0, 0.0, 0.0, 0.0, 0.05, "bicubic",
                                       "up", "hold", "nearest", "hold")
    assert aud is None and info["original_frames"] == 4 and h3.h3_is_grid(info["length"])
    assert imgs.shape[0] == info["length"]


def test_apng_and_still():
    import numpy as np
    from PIL import Image

    frames = [Image.fromarray(np.full((48, 64, 3), i * 40, np.uint8)) for i in range(6)]
    frames[0].save(os.path.join(TMP, "anim_apng.png"), save_all=True, append_images=frames[1:],
                    duration=100, loop=0)
    meta = vio.probe(os.path.join(TMP, "anim_apng.png"))
    assert abs(meta["duration"] - 0.6) < 1e-6  # av reports no duration; Pillow knows
    Image.fromarray(np.full((48, 64, 3), 128, np.uint8)).save(os.path.join(TMP, "still.jpg"))
    smeta = vio.probe(os.path.join(TMP, "still.jpg"))
    assert smeta["src_fps"] == 24.0 and abs(smeta["duration"] - 1 / 24.0) < 1e-9
    imgs, _, info = nodes.load_video("still.jpg", 24.0, 0.0, 0.0, 0.0, 0.05, "bicubic",
                                     "up", "hold", "nearest", "hold")
    assert info["original_frames"] == 1 and info["length"] == 5
    assert torch.equal(imgs[0], imgs[-1])  # single still held across the snapped clip


def test_loop_blend_uses_wrapped_times():
    import numpy as np
    from PIL import Image

    frames = [Image.fromarray(np.full((48, 64, 3), i * 40, np.uint8)) for i in range(5)]
    frames[0].save(os.path.join(TMP, "anim5.gif"), save_all=True, append_images=frames[1:],
                    duration=100, loop=0)
    kw = dict(fps=10.0, start=0.0, trim=1.5, extend=0.0, megapixels=0.05, resize_filter="bicubic",
              h3_snap_mode="up", h3_pad="hold")
    lo, _, il = nodes.load_video("anim5.gif", eof_mode="loop", frame_interp="blend", **kw)
    assert il["original_frames"] == 15  # 1.5 s @ 10 fps from a 0.5 s GIF
    for k in (5, 9, 14):
        assert torch.equal(lo[k], lo[k - 5])  # 0.5 s loop == 5 steps: same wrapped blend
    assert lo.min() >= 0.0 and lo.max() <= 1.0


def test_resolve_loop_math():
    head, tail = vio._resolve_loop([0.0, 0.1, 0.2, 0.3], 0.25, "loop")
    assert head == [0.0, 0.1, 0.2] and tail == [1]  # 0.05 / 0.1 rounds to head 1
    assert vio._resolve_loop([0.0, 0.1], 0.5, "hold") == ([0.0, 0.1], None)
    assert vio._resolve_loop([0.0, 0.1, 0.2], 0.0, "loop") == ([0.0, 0.1, 0.2], None)


def test_single_frame_blend_matches_batch():
    """decode_single_frame accounts for the blend smoothing: at an upsampled
    output time it returns the crossfade the batch path emits, not the held frame."""
    import numpy as np

    _gif_frames(os.path.join(TMP, "blend.gif"), 100)  # 6f @ 10 fps, 0.6 s
    path = os.path.join(TMP, "blend.gif")
    meta = vio.probe(path)
    mp, rf = 0.05, "bicubic"
    times, _ = vio.plan_times(meta["src_fps"], meta["duration"], 24.0, 0.0, 0.0, 1)
    imgs, _, _, _ = vio.decode_file(path, times, 24.0, mp, rf, include_audio=False,
                                    frame_interp="blend", eof_mode="hold")
    for k in (0, 1, 2, 5, 7, len(times) - 1):
        s = vio.decode_single_frame(path, times[k], mp, rf, "blend", "hold")
        b = (imgs[k].numpy() * 255.0 + 0.5).astype(np.uint8)
        assert np.abs(s.astype(int) - b.astype(int)).max() <= 1, (k, times[k])
    # mid-interval: blended, not the held nearest frame (0.05 is halfway 0 -> 30)
    mid = vio.decode_single_frame(path, 0.05, mp, rf, "blend", "hold")
    near = vio.decode_single_frame(path, 0.05, mp, rf, "nearest", "hold")
    assert not np.array_equal(mid, near) and abs(int(mid[0, 0, 0]) - int(near[0, 0, 0])) > 5
    # loop wraps past the end, hold clamps
    lo = vio.decode_single_frame(path, meta["duration"] + 0.05, mp, rf, "blend", "loop")
    ref = vio.decode_single_frame(path, 0.05, mp, rf, "blend", "loop")
    assert np.array_equal(lo, ref)


def test_video_ext_lists_stills():
    assert ".gif" in vio.VIDEO_EXT
    for ext in (".tif", ".tiff", ".qoi"):
        assert ext in vio.VIDEO_EXT and ext in vio.IMAGE_EXTS


def test_prompt_hash_ignores_view_state():
    base = {"space": [64, 48], "objects": [{"id": 1, "name": "a", "color": "#ff0000"}],
            "frames": {"0": {"1": {"points": [[10.0001, 20, 1], [5, 5, 0]], "box": None}}}}
    other = dict(base, objects=[{"id": 1, "name": "renamed", "color": "#00ff00"}], source_node="77",
                 view={"idx": 9, "opacity": 0.2, "playing": True, "popout": True})
    other["frames"] = {"0": {"1": {"points": [[10.0, 20.0, 1], [5.0, 5.0, 0]], "box": None}}, "3": {}}
    a, b = P.parse(base), P.parse(other)
    assert P.prompt_hash(a, 64, 48) == P.prompt_hash(b, 64, 48)
    c = P.parse(dict(base, space=[128, 96], frames={"0": {"1": {"points": [[20.0002, 40, 1], [10, 10, 0]], "box": None}}}))
    assert P.prompt_hash(a, 64, 48) == P.prompt_hash(c, 64, 48)
    d = P.parse(dict(base, frames={"0": {"1": {"points": [[11, 20, 1], [5, 5, 0]], "box": None}}}))
    assert P.prompt_hash(a, 64, 48) != P.prompt_hash(d, 64, 48)
    f = nodes.CMPointEditor.fingerprint_inputs
    assert f(False, P.to_json(a)) == f(False, P.to_json(dict(a, source_node="x")))


def test_segment_reuses_sam_cache_on_postproc_change():
    calls = []
    imgs = torch.rand(3, 32, 32, 3)

    def fake_prop(m, images, prompts, *a, **k):
        calls.append(1)
        mk = torch.zeros(images.shape[0], 1, 32, 32, dtype=torch.bool)
        mk[:, :, 8:24, 8:24] = True
        return mk, [1]
    old = engine.propagate
    engine.propagate = fake_prop
    try:
        class M:
            name = "stub-%f" % torch.rand(1).item()
            precision = "fp32"
        pr = P.parse({"frames": {"0": {"1": {"points": [[16, 16, 1]], "box": None}}}})
        S = nodes.CMSegment.execute
        r1 = S(M, imgs, pr, grow=0)
        r2 = S(M, imgs, pr, grow=4, feather=2.0, fill_holes=5, remove_islands=3, temporal_smooth=1,
               overlay_opacity=0.9)
        assert len(calls) == 1, calls
        store._MEM.clear()
        S(M, imgs, dict(pr, source_node="zz"), grow=-2)
        assert len(calls) == 1, calls
        S(M, imgs, P.parse({"frames": {"0": {"1": {"points": [[17, 16, 1]], "box": None}}}}))
        assert len(calls) == 2, calls
        g = lambda r: getattr(r, "args", None) or getattr(r, "result", None)
        assert float(g(r2)[0].sum()) > float(g(r1)[0].sum())
    finally:
        engine.propagate = old


def test_segment_defaults_grow_feather():
    s = nodes.CMSegment.define_schema()
    # schema inputs carry the tuned grow/feather defaults (dict in stubs, objects live)
    if isinstance(s, dict):
        def _name(inp):
            if isinstance(inp, tuple) and len(inp) >= 4:
                a = inp[2]
                if a:
                    return a[0]
            return None
        def _kw(inp, key):
            if isinstance(inp, tuple) and len(inp) >= 4:
                k = inp[3]
                if isinstance(k, dict):
                    return k.get(key)
            return getattr(inp, key, None)
        by_name = {_name(i): i for i in s["inputs"]}
        assert _kw(by_name["grow"], "default") == 1
        assert _kw(by_name["feather"], "default") == 1.0
    else:
        inputs = {i.id: i for i in s.inputs}
        assert inputs["grow"].default == 1
        assert inputs["feather"].default == 1.0
    import inspect
    sig = inspect.signature(nodes.CMSegment.execute)
    assert sig.parameters["grow"].default == 1
    assert sig.parameters["feather"].default == 1.0
    # grow=1 + feather=1 seals 1px gaps: single-pixel hole filled vs grow=0
    m = torch.zeros(1, 16, 16)
    m[:, 4:12, 4:12] = 1
    m[:, 7, 7] = 0
    assert post.process(m, 0, 0.0, 0, 0, 0)[0, 7, 7] == 0
    assert float(post.process(m, 1, 0.0, 0, 0, 0)[0, 7, 7]) == 1.0
    assert float(post.process(m, 1, 1.0, 0, 0, 0)[0, 7, 7]) > 0.99


def test_cm_mask_playback_schema_and_execute():
    mp = importlib.import_module(PKG + ".cm.mask_playback")
    s = mp.CMMaskPlayback.define_schema()
    assert s["node_id"] == "CMMaskPlayback"
    assert s["display_name"] == "CM Mask Playback"
    assert s["category"] == "CoordinationMatrix"
    assert s.get("is_output_node") is True

    def _inp_name(inp):
        # real api: .id ; stubs: ("in-multi", name, ...) or ("in", name, args, kwargs)
        if isinstance(inp, tuple):
            if inp[0] == "in-multi":
                return inp[1]
            if len(inp) >= 4:
                a, k = inp[2], inp[3]
                if isinstance(k, dict) and "name" in k:
                    # _Type.Input(*a, **k) called as Input("images", ...) -> a[0] is name
                    pass
                if a:
                    return a[0]
            return None
        return getattr(inp, "id", None)

    ids = [_inp_name(i) for i in s["inputs"]]
    assert ids == ["images", "mask", "max_side", "fps", "opacity", "video_fps"]
    assert mp.cm_playback_size(100, 200, 64) == (32, 64)
    assert mp.cm_playback_size(33, 33, 512) == (32, 32)
    assert torch.equal(mp.cm_fit_batch(torch.ones(1, 2), 3), torch.ones(3, 2))
    try:
        mp.CMMaskPlayback.execute()
        assert False, "expected ValueError"
    except ValueError:
        pass
    out = mp.CMMaskPlayback.execute(images=torch.rand(4, 32, 48, 3),
                                    mask=(torch.rand(4, 32, 48) > 0.5))
    res = getattr(out, "args", None) or getattr(out, "result", None)
    assert res[1].startswith("mask on images; 4f ")
    vid = res[0]
    assert hasattr(vid, "get_components") or hasattr(vid, "get_stream_source") or hasattr(vid, "save_to")


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
