"""Minimal stand-ins for ComfyUI modules so the pack imports outside ComfyUI (CPU only)."""
import contextlib
import sys
import tempfile
import types

import torch


class _Any:
    """Permissive object: any attribute/call returns another _Any."""

    def __init__(self, *a, **k):
        self.args, self.kwargs = a, k

    def __getattr__(self, n):
        return _Any()

    def __call__(self, *a, **k):
        return _Any(*a, **k)

    def __iter__(self):
        return iter(())


def install():
    tmp = tempfile.mkdtemp(prefix="cm_test_")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("cpu")
    mm.unet_offload_device = lambda: torch.device("cpu")
    mm.soft_empty_cache = lambda *a, **k: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.load_models_gpu = lambda *a, **k: None
    patcher = types.ModuleType("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, model, load_device, offload_device):
            self.model, self.load_device, self.offload_device = model, load_device, offload_device

    patcher.ModelPatcher = ModelPatcher
    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total, self.n = total, 0

        def update(self, k):
            self.n += k

    utils.ProgressBar = ProgressBar
    utils.load_torch_file = lambda p: {}

    def common_upscale(x, w, h, method, crop):
        return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    utils.common_upscale = common_upscale
    comfy = types.ModuleType("comfy")
    comfy.model_management, comfy.model_patcher, comfy.utils = mm, patcher, utils
    fp = types.ModuleType("folder_paths")
    fp.models_dir = tmp
    fp.folder_names_and_paths = {}
    fp.get_temp_directory = lambda: tmp
    fp.get_input_directory = lambda: tmp
    fp.get_output_directory = lambda: tmp
    fp.get_annotated_filepath = lambda n: f"{tmp}/{n}"
    fp.add_model_folder_path = lambda *a, **k: None
    fp.get_folder_paths = lambda n: [f"{tmp}/{n}"]

    io = types.ModuleType("io_stub")

    class ComfyNode:
        hidden = types.SimpleNamespace(unique_id="7")

    class NodeOutput:
        def __init__(self, *args, ui=None, **k):
            self.result, self.ui = args, ui

    class _Type:
        def __init__(self, name):
            self.name = name

        def Input(self, *a, **k):
            return ("in", self.name, a, k)

        def Output(self, *a, **k):
            return ("out", self.name, a, k)

    io.ComfyNode, io.NodeOutput = ComfyNode, NodeOutput
    io.Custom = _Type
    for n in ["Combo", "Float", "Int", "String", "Boolean", "Image", "Video", "Audio", "Mask", "BoundingBox"]:
        setattr(io, n, _Type(n))

    class _MultiType:
        @staticmethod
        def Input(inner, types=None, optional=False):
            # inner is ("in", name, args, kwargs); preserve the socket name
            name = None
            try:
                if isinstance(inner, tuple) and len(inner) == 4:
                    a, k = inner[2], inner[3]
                    if a:
                        name = a[0]
                    else:
                        name = k.get("name", None)
            except Exception:
                name = None
            return ("in-multi", name, inner, types, optional)

    io.MultiType = _MultiType
    io.FolderType = types.SimpleNamespace(temp="temp", input="input", output="output")
    io.Schema = lambda **k: k
    io.Hidden = types.SimpleNamespace(unique_id="UNIQUE_ID")
    io.UploadType = types.SimpleNamespace(video="video")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = _Any()

    class ComfyExtension:
        pass

    latest.ComfyExtension = ComfyExtension
    latest.InputImpl = _Any()
    latest.Types = _Any()
    capi = types.ModuleType("comfy_api")
    capi.latest = latest
    sys.modules.update({
        "comfy": comfy, "comfy.model_management": mm, "comfy.model_patcher": patcher, "comfy.utils": utils,
        "folder_paths": fp, "comfy_api": capi, "comfy_api.latest": latest,
    })
    return tmp
