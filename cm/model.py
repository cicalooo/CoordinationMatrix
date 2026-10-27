"""SAM 2.1 model loading (kijai safetensors -> vendored official SAM2VideoPredictor)."""
from __future__ import annotations

import importlib
import logging
import os
import re
import threading
import urllib.request

import torch
import yaml

log = logging.getLogger("CoordinationMatrix")

HF_REPO = "Kijai/sam2-safetensors"
MODELS = {
    "sam2.1_hiera_tiny": "sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1_hiera_l.yaml",
}
PRECISIONS = ["bf16", "fp16", "fp32"]
_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG_DIR = os.path.join(_HERE, "vendor", "sam2", "configs")
_VENDOR_PKG = __package__ + ".vendor"

_LOCK = threading.RLock()
_LOADED: dict[tuple, "CMSam2Model"] = {}
_LAST: list = [None]


def models_dir() -> str:
    import folder_paths

    d = os.path.join(folder_paths.models_dir, "sam2")
    try:
        if "sam2" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("sam2", d)
    except Exception:
        pass
    return d


def checkpoint_name(model: str, precision: str) -> str:
    return f"{model}-fp16.safetensors" if precision in ("fp16", "bf16") else f"{model}.safetensors"


def ensure_checkpoint(model: str, precision: str) -> str:
    import folder_paths

    fn = checkpoint_name(model, precision)
    try:
        p = folder_paths.get_full_path("sam2", fn)
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    d = models_dir()
    path = os.path.join(d, fn)
    if os.path.isfile(path):
        return path
    os.makedirs(d, exist_ok=True)
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{fn}"
    log.info("[CM Model Loader] downloading %s -> %s", url, path)
    print(f"[CM Model Loader] downloading {url}")
    tmp = path + ".part"
    try:
        from huggingface_hub import hf_hub_download  # resumable when available

        got = hf_hub_download(repo_id=HF_REPO, filename=fn, local_dir=d)
        if os.path.abspath(got) != os.path.abspath(path):
            os.replace(got, path)
    except ImportError:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, path)
    return path


_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _fix_scalars(v):
    if isinstance(v, dict):
        return {k: _fix_scalars(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_fix_scalars(x) for x in v]
    if isinstance(v, str) and _FLOAT_RE.match(v):  # PyYAML 1.1 reads 1e-6 as str
        return float(v)
    return v


def _resolve(target: str):
    if target == "sam2.modeling.sam2_base.SAM2Base":
        target = "sam2.sam2_video_predictor.SAM2VideoPredictor"
    mod, _, cls = target.rpartition(".")
    m = importlib.import_module("." + mod, package=_VENDOR_PKG)
    return getattr(m, cls)


def _instantiate(cfg):
    """Minimal hydra.utils.instantiate replacement for the SAM 2.1 yaml files."""
    if isinstance(cfg, dict):
        if "_target_" in cfg:
            kwargs = {k: _instantiate(v) for k, v in cfg.items() if k != "_target_"}
            return _resolve(cfg["_target_"])(**kwargs)
        return {k: _instantiate(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_instantiate(v) for v in cfg]
    return cfg


def build_predictor(model: str):
    with open(os.path.join(_CFG_DIR, MODELS[model]), "r") as f:
        cfg = _fix_scalars(yaml.safe_load(f))["model"]
    # same overrides as official build_sam2_video_predictor (vos_optimized=False)
    cfg["sam_mask_decoder_extra_args"] = {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }
    cfg["binarize_mask_from_pts_for_mem_enc"] = True
    cfg["fill_hole_area"] = 0  # official uses 8 but needs the CUDA ext; CM Mask Post fills holes
    cfg["compile_image_encoder"] = False
    return _instantiate(cfg)


class CMSam2Model:
    """Holds the predictor in a ModelPatcher so ComfyUI can load/unload it."""

    def __init__(self, name: str, precision: str, device: str):
        import comfy.model_management as mm
        import comfy.model_patcher
        import comfy.utils

        self.name = name
        self.precision = precision
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        ckpt = ensure_checkpoint(name, precision)
        net = build_predictor(name)
        sd = comfy.utils.load_torch_file(ckpt)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            log.warning("[CM Model Loader] state_dict mismatch: missing=%s unexpected=%s",
                        missing[:8], unexpected[:8])
        # Weights stay fp32 (as upstream); bf16/fp16 runs through autocast.
        net = net.float().eval()
        self.load_device = torch.device("cpu") if device == "cpu" else mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        if self.load_device.type == "cpu" and self.dtype != torch.float32:
            log.warning("[CM Model Loader] %s on CPU is slow/unsupported; using fp32", precision)
            self.dtype = torch.float32
        # upstream stores memory features in bf16 and relies on CUDA autocast; fp32/CPU needs fp32
        net.cm_maskmem_dtype = torch.float32 if self.dtype == torch.float32 else torch.bfloat16
        self.patcher = comfy.model_patcher.ModelPatcher(net, load_device=self.load_device,
                                                        offload_device=self.offload_device)
        self.image_size = int(net.image_size)

    @property
    def net(self):
        return self.patcher.model

    def load(self):
        import comfy.model_management as mm

        if self.load_device.type == "cpu":
            self.net.to("cpu")
            return self.net
        try:
            mm.load_models_gpu([self.patcher], force_full_load=True)
        except Exception as e:  # odd module -> fall back to a plain move
            log.warning("[CoordinationMatrix] load_models_gpu failed (%s); moving model manually", e)
            self.net.to(self.load_device)
        if next(self.net.parameters()).device.type != self.load_device.type:
            self.net.to(self.load_device)
        if self.load_device.type == "cuda" and torch.cuda.get_device_properties(self.load_device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return self.net

    def autocast(self):
        if self.load_device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast("cuda", dtype=self.dtype)
        import contextlib

        return contextlib.nullcontext()


def get_model(name: str, precision: str, device: str = "auto") -> CMSam2Model:
    key = (name, precision, device)
    with _LOCK:
        m = _LOADED.get(key)
        if m is None:
            m = CMSam2Model(name, precision, device)
            _LOADED.clear()  # keep one SAM2 resident; ComfyUI handles VRAM
            _LOADED[key] = m
        _LAST[0] = m
        return m


def last_model() -> CMSam2Model | None:
    return _LAST[0]


def lock():
    return _LOCK
