# -*- coding: utf-8 -*-
import os, io, json, hashlib, time, datetime, math, zipfile
from pathlib import Path
import numpy as np
import cv2
import torch

# ========= 你的模型入口（沿用現有 InpaintingForensics / giid_model）=========
from main import InpaintingForensics  # 與你原本一致

# ========= 參數 =========
IMG_SIZE = (256, 256)
THRESHOLD = 64          # 二值化門檻
OVERLAY_ALPHA = 0.6     # 疊色透明度
RUN_ID = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

BASE = Path.cwd()
DELIV_DIR = BASE / "forensics_deliveries" / RUN_ID
IN_DIR  = DELIV_DIR / "inputs"
ART_DIR = DELIV_DIR / "artifacts"
DELIV_DIR.mkdir(parents=True, exist_ok=True); IN_DIR.mkdir(exist_ok=True); ART_DIR.mkdir(exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_audit(event: str, payload: dict):
    rec = {"ts": datetime.datetime.utcnow().isoformat()+"Z", "event": event, **payload}
    with open(DELIV_DIR / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")

def quadrant_of_point(cx, cy, w, h):
    # 以九宮格/四象限給人類可讀描述
    v = "upper" if cy < h/3 else ("middle" if cy < 2*h/3 else "lower")
    hpos = "left" if cx < w/3 else ("center" if cx < 2*w/3 else "right")
    return f"{v}-{hpos}"

# ========= 1) 載入模型與紀錄環境 =========
append_audit("init", {"run_id": RUN_ID})
model = InpaintingForensics()
model.giid_model.load()
model.giid_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
append_audit("model_loaded", {"device": device})

# 嘗試記錄權重檔資訊（如果可取得）
weight_path = None
try:
    # 依你專案情況填入權重檔實際路徑；或在 load() 中回傳
    # weight_path = Path("/content/weights/IID_weights.pth")
    pass
except Exception:
    pass

# ========= 2) 讀入影像（指定路徑或串列跑批次）=========
# 範例：單張
src_path = Path("/content/demo_input/attack.png")  # 依你需求調整
orig_bgr = cv2.imread(str(src_path))
if orig_bgr is None:
    raise FileNotFoundError(f"Image not found: {src_path}")
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
img = cv2.resize(orig_rgb, IMG_SIZE)
img_norm = img.astype(np.float32) / 255.0

# 保存輸入副本 + SHA256
in_copy_path = IN_DIR / src_path.name
cv2.imwrite(str(in_copy_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
input_sha256 = sha256_file(in_copy_path)
append_audit("ingest", {"input_path": str(in_copy_path), "sha256": input_sha256, "shape": list(img.shape)})

# ========= 3) 前處理 & 推論 =========
t0 = time.time()
tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
tensor = (tensor - 0.5) / 0.5
tensor = tensor.to(device)
with torch.no_grad():
    pred_mask = model.giid_model(tensor)
pred_mask = pred_mask.squeeze().detach().cpu().numpy()  # [H, W], 0~1 或未歸一化
t1 = time.time()
append_audit("inference_done", {"dt_sec": round(t1 - t0, 4)})

# ========= 4) 後處理與統計 =========
# 規範化到 [0,255] 以產出視覺化；同時保留 raw npy（float）
raw_mask_path = ART_DIR / "mask_raw.npy"
np.save(raw_mask_path, pred_mask.astype(np.float32))
mask_raw_sha256 = sha256_file(raw_mask_path)

mask_uint8 = np.clip(pred_mask * 255.0, 0, 255).astype(np.uint8)
mask_png_path = ART_DIR / "mask_gray.png"
cv2.imwrite(str(mask_png_path), mask_uint8)
mask_png_sha256 = sha256_file(mask_png_path)

mask_bin = (mask_uint8 > THRESHOLD).astype(np.uint8) * 255
mask_bin_path = ART_DIR / "mask_bin.png"
cv2.imwrite(str(mask_bin_path), mask_bin)
mask_bin_sha256 = sha256_file(mask_bin_path)

# 疊色
red = np.zeros_like(img); red[:, :] = [150, 0, 0]
overlay = cv2.addWeighted(red.astype(np.uint8), OVERLAY_ALPHA, img, 1.0, 0, dtype=cv2.CV_32F)
overlay = np.clip(overlay, 0, 255).astype(np.uint8)
overlay = np.where(mask_bin[..., None] == 255, overlay, img)
overlay_path = ART_DIR / "overlay.png"
cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
overlay_sha256 = sha256_file(overlay_path)

# 覆蓋率與區塊摘要（連通元件）
H, W = mask_bin.shape
coverage = float((mask_bin > 0).sum()) / float(H * W)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_bin > 0).astype(np.uint8), connectivity=8)
regions = []
for lab in range(1, num_labels):
    x, y, w, h, area = stats[lab]
    cx, cy = centroids[lab]
    regions.append({
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "area_px": int(area),
        "centroid_xy": [float(cx), float(cy)],
        "position": quadrant_of_point(cx, cy, W, H),
    })
regions = sorted(regions, key=lambda r: r["area_px"], reverse=True)
conf_stats = {
    "mean": float(pred_mask.mean()),
    "p95": float(np.percentile(pred_mask, 95)),
    "p99": float(np.percentile(pred_mask, 99)),
}

append_audit("postprocess_done", {
    "coverage": round(coverage, 6),
    "regions": len(regions),
    "threshold": THRESHOLD,
})

# ========= 5) 結構化報告（機器可讀）=========
report = {
    "run_id": RUN_ID,
    "tool": {
        "name": "IID-Net Inpainting Forensics",
        "version": "giid_model",  # 可自填 commit / 內部版本
        "framework": {
            "python": f"{os.sys.version.split()[0]}",
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "parameters": {
            "img_size": IMG_SIZE,
            "normalize": "[0,1] -> [-1,1]",
            "threshold_bin": THRESHOLD,
            "overlay_alpha": OVERLAY_ALPHA,
        },
        "weights": {
            "path": str(weight_path) if weight_path else None,
            "sha256": sha256_file(weight_path) if weight_path and weight_path.exists() else None
        }
    },
    "input": {
        "path": str(in_copy_path),
        "sha256": input_sha256,
        "shape_hw3": list(img.shape),
    },
    "outputs": {
        "mask_raw_npy": {"path": str(raw_mask_path), "sha256": mask_raw_sha256},
        "mask_gray_png": {"path": str(mask_png_path), "sha256": mask_png_sha256},
        "mask_bin_png": {"path": str(mask_bin_path), "sha256": mask_bin_sha256},
        "overlay_png":   {"path": str(overlay_path), "sha256": overlay_sha256},
    },
    "analysis": {
        "tamper_coverage_ratio": coverage,       # 0~1
        "confidence_stats": conf_stats,
        "region_summaries": regions[:10],        # 只列前 10 大可讀
        "human_summary": f"Estimated tamper coverage ≈ {coverage*100:.2f}% ; major regions near: " +
                         (", ".join([r['position'] for r in regions[:3]]) if regions else "none"),
    },
    "timing": {
        "inference_sec": round(t1 - t0, 4),
        "end_utc": datetime.datetime.utcnow().isoformat()+"Z"
    }
}
write_json(DELIV_DIR / "report.json", report)

# ========= 6) 人可讀 Markdown 報告 =========
md = []
md.append(f"# IID-Net Evidence Report (RUN {RUN_ID})")
md.append(f"- **Input SHA256**: `{input_sha256}`")
md.append(f"- **Tamper Coverage**: **{coverage*100:.2f}%**")
if regions:
    md.append(f"- **Top Regions**: {', '.join([r['position'] for r in regions[:3]])}")
md.append(f"- **Threshold**: {THRESHOLD} | **Overlay α**: {OVERLAY_ALPHA}")
md.append(f"\n## Visuals\n- mask_gray.png\n- mask_bin.png\n- overlay.png")
md.append(f"\n## Confidence Stats\n```json\n{json.dumps(conf_stats, indent=2)}\n```")
md.append(f"\n## Parameters\n```json\n{json.dumps(report['tool']['parameters'], indent=2)}\n```")
with open(DELIV_DIR / "report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
append_audit("report_written", {"paths": ["report.json", "report.md"]})

# ========= 7) SHA256SUMS（全檔案）=========
all_files = []
for p in [in_copy_path, raw_mask_path, mask_png_path, mask_bin_path, overlay_path, (DELIV_DIR / "report.json"), (DELIV_DIR / "report.md"), (DELIV_DIR / "audit.jsonl")]:
    if p and Path(p).exists():
        all_files.append(Path(p))
with open(DELIV_DIR / "SHA256SUMS.txt", "w", encoding="utf-8") as f:
    for p in all_files:
        f.write(f"{sha256_file(p)}  {p.name}\n")
append_audit("sha256_sums_done", {"count": len(all_files)})

print(f"[OK] Evidence package at: {DELIV_DIR}")

# ========= 8) 產出 PDF（純 Python；無系統依賴）=========
# 需要：pip install reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import cm
    from textwrap import wrap

    def _draw_wrapped_text(c, text, x, y, max_width_px, line_height=14, font_name="Helvetica", font_size=11):
        c.setFont(font_name, font_size)
        # 粗略依字寬估算每行可容納字數
        max_chars = max(1, int(max_width_px / (font_size * 0.55)))
        cur_y = y
        for line in text.split("\n"):
            for seg in wrap(line, max_chars):
                c.drawString(x, cur_y, seg)
                cur_y -= line_height
        return cur_y

    def _draw_image_block(c, title, img_path, x, y, max_w, max_h):
        if not Path(img_path).exists(): 
            return y
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, title); y -= 12+6
        img = ImageReader(str(img_path))
        iw, ih = img.getSize()
        scale = min(max_w/iw, max_h/ih)
        w, h = iw*scale, ih*scale
        c.drawImage(img, x, y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
        return y - h - 10

    pdf_path = DELIV_DIR / "forensics_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4
    margin = 2*cm
    x0, y0 = margin, H - margin

    # 封面 / 摘要
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x0, y0, f"IID-Net Evidence Report — RUN {RUN_ID}")
    y = y0 - 22

    # 摘要欄
    summary = [
        f"Input: {in_copy_path.name}",
        f"Input SHA256: {input_sha256}",
        f"Tamper Coverage: {coverage*100:.2f}%",
        f"Top Regions: {', '.join([r['position'] for r in regions[:3]]) if regions else 'none'}",
        f"Threshold: {THRESHOLD} | Overlay α: {OVERLAY_ALPHA}",
        f"Inference: {report['timing']['inference_sec']}s | Device: {device}",
        f"Torch: {report['tool']['framework']['torch']} | Python: {report['tool']['framework']['python']}",
    ]
    y = _draw_wrapped_text(c, "\n".join(summary), x0, y, W - 2*margin, font_size=11) - 6

    # 視覺化（第一頁放 overlay）
    y = _draw_image_block(c, "Overlay (Suspected tampering areas are displayed in red semi-transparency)", overlay_path, x0, y, W-2*margin, (H/2)-margin)
    c.showPage()

    # 第二頁：灰階與二值遮罩
    x0, y0 = margin, H - margin
    c.setFont("Helvetica-Bold", 14); c.drawString(x0, y0, "IID-Net Visual Outputs")
    y = y0 - 18
    half_w = (W - 3*margin) / 2
    block_h = H - 3*margin - 40

    # 左：mask_gray
    y_left = _draw_image_block(c, "Mask (Gray)", mask_png_path, x0, y, half_w, block_h)
    # 右：mask_bin
    y_right = _draw_image_block(c, "Mask (Binary)", mask_bin_path, x0 + half_w + margin, y, half_w, block_h)

    c.showPage()

    # 第三頁：技術細節 + SHA256SUMS 摘要
    x0, y0 = margin, H - margin
    c.setFont("Helvetica-Bold", 14); c.drawString(x0, y0, "Technical Details & Integrity")
    y = y0 - 18

    # 參數 / 權重 / 信心統計
    details = {
        "Parameters": report["tool"]["parameters"],
        "Weights": report["tool"]["weights"],
        "Confidence Stats": report["analysis"]["confidence_stats"],
    }
    y = _draw_wrapped_text(c, json.dumps(details, ensure_ascii=False, indent=2), x0, y, W-2*margin, font_size=10) - 6

    # SHA256SUMS（只顯示前 20 行；完整檔在目錄）
    sums_path = DELIV_DIR / "SHA256SUMS.txt"
    if sums_path.exists():
        c.setFont("Helvetica-Bold", 12); c.drawString(x0, y, "SHA256SUMS (partial preview)")
        y -= 14
        with open(sums_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        preview = "".join(lines[:20]) + ("" if len(lines) <= 20 else f"...(+{len(lines)-20} more)")
        y = _draw_wrapped_text(c, preview, x0, y, W-2*margin, font_size=9)

    c.save()
    append_audit("pdf_written", {"path": str(pdf_path)})
    print(f"[OK] PDF written: {pdf_path}")

except Exception as e:
    append_audit("pdf_error", {"error": str(e)})
    print("[WARN] PDF generation failed:", e)