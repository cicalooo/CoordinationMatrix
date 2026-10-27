"""CoordinationMatrix: SAM 2.1 point/box video segmentation for ComfyUI (V3 nodes)."""
from .cm.nodes import comfy_entrypoint

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
