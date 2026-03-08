# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**照妖鏡 (Deepfake Detection System)** — A two-service forensics platform combining:
1. **IID-Net Detection Service** (`DDD/`, port 5000) — Image forgery/tampering localization using PyTorch
2. **Robust Watermark Service** (`rubust_matermark/`, port 5001) — Steganographic watermark embedding/reveal using TensorFlow 2.15

## Running the Services

### IID-Net Detection Service (port 5000)
```bash
# Set weight path (or place weights at DDD/IID_weight/IID_weights.pth)
set IID_WEIGHTS=C:\path\to\IID_weights.pth
cd DDD
python app.py
```

### Robust Watermark Service (port 5001)
```bash
# Activate the venv first (see cd_robust.txt)
cd rubust_matermark
venv\Scripts\activate
python app_wm.py serve --host 0.0.0.0 --port 5001 --root ./outputs
```

### Training

**Watermark model:**
```bash
python robust_watermark.py train --cover_dir <dir> --secret_dir <dir> --save_root <dir>
```

**IID-Net (from DDD/):**
```bash
python main.py train
```

**Watermark inference/reveal:**
```bash
python robust_watermark.py infer --model_dir <dir> --cover_img <img> --secret_img <img>
python robust_watermark.py external_reveal --reveal_h5 <h5> --attack_img <img>
```

## Architecture

### DDD/ (Detection)
- `app.py` — Flask API server. Loads `IID_Net` from `main.py`. On `POST /result`, runs inference, produces tamper mask overlays, saves a forensics delivery package (`forensics_deliveries/<run_id>/`), then optionally forwards to the WM service via `_forward_to_watermark()`.
- `main.py` — IID-Net model definition + training harness. Architecture: Enhancement Block (normal/Bayar/PF convolutions) → 10 Extraction cells (NAS-selected depthwise separable convs with dilation) → Decision Block (Global-Local Attention + 3× bilinear upsampling → 1-ch sigmoid mask).
- `detect_27042.py` — Standalone forensics script: loads `InpaintingForensics`, runs batch inference, generates `report.json`, `report.md`, `SHA256SUMS.txt`, and optional PDF via reportlab.

### rubust_matermark/ (Watermark)
- `app_wm.py` — Flask API server. Loads TF models (combined + reveal). `POST /wm/embed` embeds a secret into a cover image. `POST /external_reveal` (alias: `/wm/reveal_masked`) reveals watermark from an attacked image using bottom-row tile scoring + optional Real-ESRGAN upscale + Reinhard color transfer.
- `robust_watermark.py` — Training CLI: builds `Hiding Network` + `Reveal Network` (both encoder-decoder, residual blocks, LeakyReLU) + `RobustAttackLayer` (noise/blur/pooling during training). Losses: perceptual (VGG19) + MSE for cover; MSE + DCT + color consistency for secret.

### project_root/frontend/
Static HTML pages (`index.html`, `detect.html`, `watermark.html`, `restore.html`) + `main.js`. JavaScript communicates with IID service at port 5000 (`API_BASE`) and WM service at port 5001 (`WM_API_BASE`).

### Service Integration
The IID service automatically forwards to the WM service after detection. The WM service receives `source.png` + `mask_bin.png` and returns `patched_masked.png` (watermark revealed on the tampered region). Controlled by environment variables.

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `IID_WEIGHTS` | — | Path to `IID_weights.pth` |
| `WM_FORWARD` | `1` | Forward to WM service after detection |
| `WM_URL` | `http://127.0.0.1:5001/wm/reveal_masked` | WM service endpoint |
| `WM_TIMEOUT_SEC` | `900` | HTTP timeout for WM forwarding |
| `WM_CONTAINER_MODE` | `auto` | `auto`/`direct`/`residual` — container blending strategy |
| `WM_PSNR_TARGET` | `40.0` | Target PSNR for residual embedding |
| `WM_OPT_REVEAL` | `1` | Optimize scale via reveal SSIM |
| `REALESRGAN_REPO` | `Real-ESRGAN` | Path to Real-ESRGAN repo for tile upscaling |

## Weight Files

- IID-Net: `DDD/IID_weight/IID_weights.pth` (or set `IID_WEIGHTS`)
- WM combined model: `rubust_matermark/weights/combined_model.h5`
- WM reveal model: `rubust_matermark/weights/reveal_network.h5`

## Key API Endpoints

**IID Service (port 5000):**
- `POST /result` — Upload image, get overlay PNG back + save forensics run
- `GET /27042/view/<run_id>` — Browser view of forensics report
- `GET /27042/download/<run_id>` — Download ZIP of run artifacts
- `GET /27042/latest` — Latest run metadata
- `GET /health` — Service health + weight path

**WM Service (port 5001):**
- `POST /wm/embed` — Embed watermark (multipart: `cover`, optional `secret`)
- `POST /external_reveal` or `/wm/reveal_masked` — Reveal watermark from attacked image (multipart: `image`/`attack`, optional `mask`)
- `GET /27037/latest?kind=embed|reveal` — Latest job metadata
- `GET /health` — Service health + weight presence

## Forensics Output Structure

Each detection run creates `DDD/forensics_deliveries/<run_id>/`:
```
inputs/source.png
artifacts/overlay.png          # red semi-transparent overlay on tampered areas
artifacts/mask_gray.png        # continuous confidence map
artifacts/mask_bin.png         # binary mask (threshold 0.5)
artifacts/patched_masked.png   # watermark revealed on masked region (from WM service)
report.json                    # structured machine-readable report
report.html                    # offline-viewable HTML report
sha256sums.txt
```

## Dependencies

- **DDD**: Python, PyTorch, OpenCV (`cv2`), Flask, flask-cors, requests, numpy, torchvision
- **rubust_matermark**: Python 3.10/3.11, TensorFlow 2.15 / Keras 2.15, Pillow, OpenCV, Flask, h5py, scikit-image (optional), reportlab (optional for PDF), Real-ESRGAN (optional for upscaling)
