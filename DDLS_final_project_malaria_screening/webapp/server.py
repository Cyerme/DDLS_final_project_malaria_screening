# webapp/server.py
from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from starlette.staticfiles import StaticFiles

# -----------------------------------------------------------------------------
# Paths & artifacts
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RESULTS_DIR = PROJECT_ROOT / "results"
WEIGHTS_FT = RESULTS_DIR / "mobilenetv2_finetune_tail.pt"
OP_JSON = RESULTS_DIR / "operating_point_val.json"       # temperature + decision threshold
ABSTAIN_JSON = RESULTS_DIR / "abstention_rule.json"      # δ and σ_thr
OOD_JSON = RESULTS_DIR / "ood_threshold.json"            # MSP τ

if not RESULTS_DIR.exists():
    raise RuntimeError(f"Missing results folder: {RESULTS_DIR}")

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

# -----------------------------------------------------------------------------
# Load calibration & thresholds (robust fallbacks)
# -----------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

calib_payload = _load_json(OP_JSON)
T_CAL = float(calib_payload.get("temperature_T", 1.0))
OP_THRESHOLD = float(calib_payload.get("operating_point", {}).get("threshold", 0.5))

abstain_payload = _load_json(ABSTAIN_JSON)
DELTA = float(abstain_payload.get("delta", 0.05))      # |p-0.5| band
SIGMA_THR = float(abstain_payload.get("sigma_thr", 0.06))

ood_payload = _load_json(OOD_JSON)
MSP_TAU = float(ood_payload.get("threshold", 0.85))

# -----------------------------------------------------------------------------
# Model (MobileNetV2 single-logit) + weights
# -----------------------------------------------------------------------------
def build_mobilenetv2_single_logit(pretrained: bool = False) -> nn.Module:
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v2(weights=weights)
    except Exception:
        net = models.mobilenet_v2(pretrained=pretrained)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, 1)
    return net

model = build_mobilenetv2_single_logit(pretrained=False)

if not WEIGHTS_FT.exists():
    raise RuntimeError(f"Missing fine-tuned weights: {WEIGHTS_FT}")

# Safe load across PyTorch versions
try:
    state = torch.load(WEIGHTS_FT, map_location="cpu", weights_only=True)  # PyTorch ≥2.4
except TypeError:
    state = torch.load(WEIGHTS_FT, map_location="cpu")
model.load_state_dict(state)
model = model.to(device).eval()

# -----------------------------------------------------------------------------
# Preprocessing (pad-to-square by border ring color + bicubic 128² + ImageNet norm)
# -----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
to_tensor = T.ToTensor()
normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Pillow resampling shim (new vs old API)
try:
    from PIL import Image as _PILImage
    RESAMPLE_BICUBIC = _PILImage.Resampling.BICUBIC  # Pillow ≥ 9.1
except Exception:  # pragma: no cover
    RESAMPLE_BICUBIC = Image.BICUBIC                  # older Pillow

def ring_mode_rgb_from_image(arr: np.ndarray, ring: int = 2, max_pixels: int = 20000) -> Tuple[int, int, int]:
    """Return the modal RGB triplet along the image border ring."""
    h, w, _ = arr.shape
    top = arr[0:ring, :, :]
    bottom = arr[h-ring:h, :, :]
    left = arr[:, 0:ring, :]
    right = arr[:, w-ring:w, :]
    rp = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    if rp.shape[0] > max_pixels:
        idx = np.random.choice(rp.shape[0], size=max_pixels, replace=False)
        rp = rp[idx]
    # Make the arity explicit for the type checker
    tuples: List[Tuple[int, int, int]] = [(int(v[0]), int(v[1]), int(v[2])) for v in rp.reshape(-1, 3)]
    return max(tuples, key=tuples.count)

NEAR_BLACK_THR = 10
MIN_MODE_FRAC  = 0.60

def ring_mode_and_frac(arr, ring=2, max_pixels=20000):
    h, w, _ = arr.shape
    top, bottom = arr[0:ring], arr[h-ring:h]
    left, right = arr[:, 0:ring], arr[:, w-ring:w]
    rp = np.concatenate([top.reshape(-1,3), bottom.reshape(-1,3),
                         left.reshape(-1,3), right.reshape(-1,3)], axis=0)
    if rp.shape[0] > max_pixels:
        idx = np.random.choice(rp.shape[0], size=max_pixels, replace=False)
        rp = rp[idx]
    vals, counts = np.unique(rp, axis=0, return_counts=True)
    i = int(np.argmax(counts))
    mode = tuple(int(x) for x in vals[i])
    frac = float(counts[i] / rp.shape[0])
    return mode, frac

class PadResizeBicubic:
    def __init__(self, size=128, ring=2):
        self.size = size; self.ring = ring
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB"))
        h, w, _ = arr.shape
        mode_rgb, frac = ring_mode_and_frac(arr, ring=self.ring)
        near_black = all(c <= NEAR_BLACK_THR for c in mode_rgb)
        use_constant = near_black and (frac >= MIN_MODE_FRAC)

        if w != h:
            side = max(w, h)
            pl = (side - w)//2; pr = side - w - pl
            pt = (side - h)//2; pb = side - h - pt
            if use_constant:
                padded = ImageOps.expand(Image.fromarray(arr),
                                         border=(pl, pt, pr, pb), fill=mode_rgb)
            else:
                pad_width = ((pt, pb), (pl, pr), (0, 0))
                padded = Image.fromarray(np.pad(arr, pad_width, mode="reflect"))
        else:
            padded = Image.fromarray(arr)
        return padded.resize((self.size, self.size), resample=RESAMPLE_BICUBIC)

pad_resize = PadResizeBicubic(size=128, ring=2)

def preprocess_both(pil_img: Image.Image) -> Tuple[Image.Image, torch.Tensor]:
    """Return (PIL_128x128 for overlays, normalized tensor for the model)."""
    img128 = pad_resize(pil_img)
    x = to_tensor(img128)
    x = normalize(x)
    return img128, x

# -----------------------------------------------------------------------------
# TTA (mild, as in training) + abstention rule
# -----------------------------------------------------------------------------
tta_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=10, fill=0),
    T.ColorJitter(brightness=0.2, contrast=0.2),  # ~[0.8, 1.2]
])

def tta_probs(x_pil: Image.Image, N: int = 8) -> Tuple[float, float]:
    """Return (mean_p, std_p) across N mild TTA draws (calibrated)."""
    xs = []
    for _ in range(N):
        xs.append(normalize(to_tensor(tta_aug(x_pil))))
    xb = torch.stack(xs, 0).to(device)
    with torch.no_grad():
        logits = model(xb).view(-1) / float(T_CAL)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return float(probs.mean()), float(probs.std())

def abstain_decision(mean_p: float, std_p: float, delta: float = DELTA, sigma_thr: float = SIGMA_THR) -> bool:
    """Abstain if probability is near 0.5 or TTA variability is high."""
    return (abs(mean_p - 0.5) < float(delta)) or (std_p > float(sigma_thr))

# -----------------------------------------------------------------------------
# OOD via MSP
# -----------------------------------------------------------------------------
def msp_from_prob(p: float) -> float:
    return float(max(p, 1.0 - p))

def is_ood_from_msp(p: float, tau: float = MSP_TAU) -> Tuple[bool, float]:
    msp = msp_from_prob(p)
    return (msp < float(tau)), msp

# -----------------------------------------------------------------------------
# Grad-CAM (last block)
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target = target_layer
        self.activ: Optional[torch.Tensor] = None
        self.grads: Optional[torch.Tensor] = None
        self.hf = target_layer.register_forward_hook(self._fh)
        try:
            self.hb = target_layer.register_full_backward_hook(self._bh)
        except Exception:  # pragma: no cover (older torch)
            self.hb = target_layer.register_backward_hook(self._bh)

    def _fh(self, m, i, o):  # forward hook
        self.activ = o.detach()

    def _bh(self, m, gi, go):  # backward hook
        self.grads = go[0].detach()

    def compute(self, x1: torch.Tensor) -> np.ndarray:
        """
        x1: [1,3,128,128], normalized.
        Returns a [128,128] float32 CAM in [0,1].
        """
        # Ensure autograd is enabled even if caller used no_grad
        with torch.enable_grad():
            x1 = x1.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x1).view(-1)

            # Backprop from the calibrated logit for consistency with our decision path
            (logits / float(T_CAL)).backward(retain_graph=True)

        assert self.activ is not None and self.grads is not None, "GradCAM hooks not populated"
        A = self.activ           # [1,C,h,w]
        dA = self.grads          # [1,C,h,w]

        wts = dA.mean(dim=(2, 3), keepdim=True)                 # [1,C,1,1]
        cam = torch.relu((wts * A).sum(dim=1, keepdim=True))    # [1,1,h,w]
        cam = F.interpolate(cam, size=(128, 128), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam = (cam / (cam.max() + 1e-8)).clamp(0, 1).cpu().numpy().astype(np.float32)
        return cam

    def close(self):
        self.hf.remove()
        self.hb.remove()

target_layer = model.features[-1]  # last conv block before GAP
gcam = GradCAM(model, target_layer)

def overlay_cam(pil_128: Image.Image, cam_128: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Blend a CAM heatmap over the input (RGBA-free)."""
    # Matplotlib new API, with fallback to classic cm
    try:
        import matplotlib.colormaps as cmaps
        cmap = cmaps.get_cmap("jet")
    except Exception:  # pragma: no cover
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")

    base = np.asarray(pil_128).astype(np.float32) / 255.0
    heat = cmap(cam_128)[..., :3]
    mix = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.clip(mix * 255, 0, 255).astype(np.uint8))

def pil_to_data_url(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

# -----------------------------------------------------------------------------
# Inference for a single PIL image
# -----------------------------------------------------------------------------
def infer_one(pil_img: Image.Image) -> Dict:
    t0 = time.time()

    # Preprocess
    img128, x0 = preprocess_both(pil_img)              # PIL and normalized tensor
    x0 = x0.unsqueeze(0).to(device)

    # Calibrated probability (no_grad OK)
    with torch.no_grad():
        logit = model(x0).view(-1) / float(T_CAL)
        p_cal = torch.sigmoid(logit)[0].item()

    # TTA mean/std & abstention
    mean_p, std_p = tta_probs(img128, N=8)
    will_abstain = abstain_decision(mean_p, std_p, DELTA, SIGMA_THR)

    # OOD via MSP
    ood, msp = is_ood_from_msp(p_cal, MSP_TAU)

    # Grad-CAM overlay (always compute; cheap for 128²)
    cam = gcam.compute(x0)
    overlay = overlay_cam(img128, cam, alpha=0.35)
    overlay_url = pil_to_data_url(overlay, fmt="PNG")
    input_url = pil_to_data_url(img128, fmt="PNG")

    # Decision text
    label = "Parasitized" if p_cal >= OP_THRESHOLD else "Uninfected"
    decision = "ABSTAIN" if will_abstain or ood else label

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "probability": round(p_cal, 6),
        "label_at_threshold": label,
        "threshold": OP_THRESHOLD,
        "temperature_T": T_CAL,
        "tta_mean": round(mean_p, 6),
        "tta_std": round(std_p, 6),
        "abstain": bool(will_abstain),
        "abstention_rule": {
            "delta": DELTA,
            "sigma_thr": SIGMA_THR,
            "text": "abstain if |mean_p - 0.5| < delta OR std_p > sigma_thr"
        },
        "ood": {
            "is_ood": bool(ood),
            "msp": round(msp, 6),
            "tau": MSP_TAU
        },
        "final_decision": decision,
        "cam_overlay_png": overlay_url,
        "input_png": input_url,
        "latency_ms": latency_ms,
        "device": DEVICE_NAME,
    }

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Malaria Thin-Smear Assistant", version="1.0")

# CORS: allow local dev UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static: frontend files and results assets
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/assets", StaticFiles(directory=RESULTS_DIR), name="assets")

@app.get("/", response_class=HTMLResponse)
def index():
    index_path = HERE / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h2>index.html not found</h2>", status_code=404)
    return FileResponse(index_path)

@app.get("/api/config")
def config():
    return {
        "device": DEVICE_NAME,
        "temperature_T": T_CAL,
        "threshold": OP_THRESHOLD,
        "abstention": {"delta": DELTA, "sigma_thr": SIGMA_THR},
        "ood": {"tau": MSP_TAU},
        "assets": {
            "roc": "/assets/roc_test.png",
            "reliability": "/assets/reliability_diagram_test.png",
            "gradcam_panel": "/assets/gradcam_panel.png",
            "robust_brightness": "/assets/robustness_brightness_curve.png",
            "robust_contrast": "/assets/robustness_contrast_curve.png",
        },
        "notes": "For decision support only; not a medical device.",
    }

@app.post("/api/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    for f in files:
        try:
            content = await f.read()
            pil = Image.open(io.BytesIO(content)).convert("RGB")
            res = infer_one(pil)
            res["filename"] = f.filename
            results.append(res)
        except Exception as e:  # pragma: no cover
            results.append({"filename": f.filename, "error": str(e)})

    return JSONResponse({"results": results})

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    root_print = f"Serving from {PROJECT_ROOT}  (results → {RESULTS_DIR})"
    print(root_print)
    uvicorn.run(app, host="127.0.0.1", port=8000)