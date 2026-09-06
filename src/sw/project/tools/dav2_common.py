# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: 2026 RVLab Student Project
"""Shared helpers: load Depth-Anything V2 Small, preprocess an image, and run
the PyTorch float reference used as ground truth for the quantised engine.

Requires a host python environment with torch (CPU is fine), numpy and pillow,
plus a checkout of the upstream Depth-Anything-V2 repository for the model
definition. Set DAV2_REF_DIR and DAV2_CKPT to override the default locations.
"""
import sys, types, math
from pathlib import Path
import numpy as np
import torch

import os

# Upstream model definition (git clone https://github.com/DepthAnything/Depth-Anything-V2)
REF = Path(os.environ.get("DAV2_REF_DIR",
                          Path(__file__).parent / "ref/Depth-Anything-V2"))


def _default_ckpt():
    """Fetch the checkpoint from the Hugging Face hub unless one is given."""
    env = os.environ.get("DAV2_CKPT")
    if env:
        return env
    from huggingface_hub import hf_hub_download
    return hf_hub_download("depth-anything/Depth-Anything-V2-Small",
                           "depth_anything_v2_vits.pth")

# dpt.py imports cv2 and torchvision only for the image-loading helpers we do not
# use; stub them so the model classes can be imported without those deps.
if "cv2" not in sys.modules:
    cv2 = types.ModuleType("cv2")
    cv2.INTER_AREA = 3
    cv2.INTER_CUBIC = 2
    cv2.INTER_LINEAR = 1
    sys.modules["cv2"] = cv2
if "torchvision" not in sys.modules:
    tv = types.ModuleType("torchvision")
    tvt = types.ModuleType("torchvision.transforms")
    class Compose:
        def __init__(self, *a, **k): pass
    tvt.Compose = Compose
    tv.transforms = tvt
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tvt

sys.path.insert(0, str(REF))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(ckpt=None):
    m = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    sd = torch.load(ckpt or _default_ckpt(), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


def load_image(path, size):
    """Resize to size x size (bilinear, like torch's antialias=False path) and
    apply ImageNet normalisation. Returns float32 CHW and the uint8 RGB image."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)), rgb


@torch.no_grad()
def float_reference(model, chw):
    x = torch.from_numpy(chw)[None]
    depth = model(x)  # (1, H, W)
    return depth[0].numpy()


@torch.no_grad()
def interpolated_pos_embed(model, size):
    """Run DINOv2's interpolate_pos_encoding for our fixed resolution so the C
    engine can use a precomputed table (bicubic + the 0.1 offset trick)."""
    n_patch = (size // 14) ** 2
    dummy = torch.zeros(1, n_patch + 1, model.pretrained.embed_dim)
    return model.pretrained.interpolate_pos_encoding(dummy, size, size)[0].numpy()
