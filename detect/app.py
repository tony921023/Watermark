# -*- coding: utf-8 -*-
import os, io, json, zipfile, hashlib, traceback, html
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2, numpy as np, torch
from torchvision import transforms
from flask import Flask, request, send_file, jsonify, make_response, url_for, redirect
from flask_cors import CORS

# ==== NEW: 轉送到 Watermark 服務 ====
import requests

# ---------- 行為開關 ----------
SANITIZE_WEIGHTS = True  # True: report.json 只寫出檔名與 SHA-256，不暴露本機路徑
# ==== NEW: WM 相關環境變數（可改）====
WM_FORWARD      = (os.getenv("WM_FORWARD", "1").lower() in ("1", "true", "yes"))
WM_URL          = os.getenv("WM_URL", "http://127.0.0.1:5001/wm/reveal_masked").strip()
WM_TIMEOUT_SEC  = int(os.getenv("WM_TIMEOUT_SEC", "900"))
WM_TILES_HINT   = os.getenv("WM_TILES_HINT", "auto").strip()   # 2/3/4/auto
WM_SECRET_REF   = os.getenv("WM_SECRET_REF", "auto").strip()   # auto 或指定路徑/URL

# ---------- cuDNN 固定 ----------
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ---------- 基底路徑 ----------
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_ROOT = BASE_DIR / "forensics_deliveries"
EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- 權重路徑 ----------
_env_w = os.getenv("IID_WEIGHTS", "").strip()
_candidates = []
if _env_w: _candidates.append(Path(_env_w))
_candidates.append(Path(r"C:\Users\test\Desktop\DDD\IID_weight\IID_weights.pth"))
_candidates.append(BASE_DIR / "IID_weight" / "IID_weights.pth")

def _pick_weights(cands):
    for p in cands:
        if p and Path(p).exists(): return Path(p)
    return None

WEIGHTS_PATH = _pick_weights(_candidates)

# ---------- 參數 ----------
PRINT_STATS = True
PIXEL_THR = 1e-3
BIN_THR = 0.5
OVERLAY_ALPHA = 0.6
SUPPRESS_HIGHCOVER_THR = 0.90
SUPPRESS_HIGHCOVER_RATIO = 0.70
SUPPRESS_STD_MAX = 0.08
SUPPRESS_QGAP_MIN = 0.15
SUPPRESS_MIN_BLOB_RATIO = 0.001
SUPPRESS_MAX_BLOB_RATIO = 0.50
SUPPRESS_DELTA_MEAN_MIN = 0.06

# ---------- Flask ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- 匯入模型 ----------
from main import IID_Model, IID_Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def _print_stats(name, arr: np.ndarray):
    if not PRINT_STATS: return
    arr = np.asarray(arr, np.float32)
    print(f"[STATS] {name}: {{'min':{float(np.nanmin(arr)) if arr.size else 0.0}, 'max':{float(np.nanmax(arr)) if arr.size else 0.0}, 'nan':{int(np.isnan(arr).sum())}, 'shape':{arr.shape}, 'dtype':str(arr.dtype)}}")

def build_infer_net():
    if WEIGHTS_PATH is None or not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError("找不到 IID 權重檔，請設定 IID_WEIGHTS 或放在 IID_weight/IID_weights.pth")
    if torch.cuda.is_available():
        mdl = IID_Model()
        state = torch.load(str(WEIGHTS_PATH), map_location="cuda")
        mdl.gen.load_state_dict(state, strict=True)
        net = mdl.gen
    else:
        net = IID_Net()
        state = torch.load(str(WEIGHTS_PATH), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k
            for pref in ("module.", "gen.module.", "gen."):
                if nk.startswith(pref): nk = nk[len(pref):]
            cleaned[nk] = v
        net.load_state_dict(cleaned, strict=False)
        net = net.to(device)
    net.eval()
    for p in net.parameters(): p.requires_grad = False
    return net

def should_suppress(mask: np.ndarray) -> bool:
    H, W = mask.shape
    m_mean, m_std = float(mask.mean()), float(mask.std())
    q50, q99 = float(np.quantile(mask, 0.50)), float(np.quantile(mask, 0.99))
    high = (mask >= SUPPRESS_HIGHCOVER_THR).astype(np.uint8)
    if float(high.mean()) >= SUPPRESS_HIGHCOVER_RATIO: return True
    if (m_std <= SUPPRESS_STD_MAX) and ((q99 - q50) <= SUPPRESS_QGAP_MIN): return True
    n, lbl = cv2.connectedComponents(high)
    max_area, max_mask = 0, None
    for i in range(1, n):
        comp = (lbl == i); a = int(comp.sum())
        if a > max_area: max_area, max_mask = a, comp
    if max_area == 0: return True
    max_ratio = max_area / float(H * W)
    if (max_ratio < SUPPRESS_MIN_BLOB_RATIO) or (max_ratio > SUPPRESS_MAX_BLOB_RATIO): return True
    if (float(mask[max_mask].mean()) - m_mean) < SUPPRESS_DELTA_MEAN_MIN: return True
    return False

# ---------- run 建置 / 壓縮 ----------
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _new_run_dir() -> Tuple[str, Path]:
    run_id = _now_utc_str()
    run_dir = EVIDENCE_ROOT / run_id
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir

def _zip_run_dir(run_dir: Path) -> Path:
    z = run_dir.parent / f"{run_dir.name}.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(run_dir)))
    return z

def _latest_run_id() -> Optional[str]:
    items = []
    for p in EVIDENCE_ROOT.iterdir():
        try:
            if p.is_dir(): items.append((p.name, p.stat().st_mtime))
            elif p.suffix.lower() == ".zip": items.append((p.stem, p.stat().st_mtime))
        except Exception: pass
    if not items: return None
    items.sort(key=lambda t: t[1], reverse=True)
    return items[0][0]

def _safe_run_id(s: str) -> bool:
    return bool(s) and all(c.isalnum() or c in "-_" for c in s)

def _dir_of_run(run_id: str) -> Optional[Path]:
    p = EVIDENCE_ROOT / run_id
    return p if p.exists() and p.is_dir() else None

def _zip_of_run(run_id: str) -> Optional[Path]:
    z = EVIDENCE_ROOT / f"{run_id}.zip"
    return z if z.exists() else None

def _find_in_dir(root: Path, names) -> Optional[str]:
    names_lower = tuple(n.lower() for n in names)
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(names_lower):
                return str((Path(dp) / f).relative_to(root)).replace("\\", "/")
    return None

def _find_in_zip(zip_path: Path, names) -> Optional[str]:
    names_lower = tuple(n.lower() for n in names)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            if n.lower().endswith(names_lower): return n
    return None

def _read_report_json(run_id: str) -> Optional[dict]:
    d = _dir_of_run(run_id)
    if d:
        p = d / "report.json"
        if p.exists():
            try: return json.loads(p.read_text(encoding="utf-8"))
            except Exception: pass
    z = _zip_of_run(run_id)
    if z:
        try:
            with zipfile.ZipFile(z, "r") as zf:
                for n in zf.namelist():
                    if n.lower().endswith("/report.json") or n.lower() == "report.json":
                        with zf.open(n) as f:
                            return json.loads(f.read().decode("utf-8", "ignore"))
        except Exception: pass
    return None

# ---------- 雜湊 / 離線報告 ----------
def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _write_sha256sums_file(run_dir: Path) -> Path:
    files = [p for p in sorted(run_dir.rglob("*")) if p.is_file()]
    lines = []
    for ap in files:
        rp = str(ap.relative_to(run_dir)).replace("\\", "/")
        try:
            sha = _sha256_path(ap)
            sz = ap.stat().st_size
            mt = _utc_iso(ap.stat().st_mtime)
            lines.append(f"{sha}  {rp}  ({sz} bytes)  {mt}")
        except Exception:
            pass
    out = run_dir / "sha256sums.txt"
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out

def _build_offline_report_html(run_id: str, run_dir: Path, report: dict) -> Path:
    cov = report.get("analysis", {}).get("tamper_coverage_ratio")
    cov_txt = f"{cov*100:.2f}%" if isinstance(cov, (int, float)) else "—"
    rep_txt = json.dumps(report, ensure_ascii=False, indent=2)
    sha_path = run_dir / "sha256sums.txt"
    sha_txt = sha_path.read_text(encoding="utf-8") if sha_path.exists() else "尚未產生 sha256sums.txt"

    def esc(s: str) -> str: return html.escape(s, quote=False)

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>27042 取證報告（{run_id}，離線版）</title>
<style>
  :root{{--line:#e6eefc;--muted:#64748b}}
  body{{background:#f5f8ff;color:#0f172a;font:14px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans TC",Arial}}
  .wrap{{max-width:1100px;margin:22px auto;padding:0 16px}}
  header{{display:flex;justify-content:space-between;align-items:center}}
  h1{{font-size:20px;margin:0 0 10px 0}}
  .card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0;box-shadow:0 8px 22px rgba(25,118,210,.08)}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}
  img{{max-width:100%;background:#fff;border:1px solid var(--line);border-radius:10px}}
  pre{{background:#fbfdff;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto}}
  .muted{{color:var(--muted)}}
  .actions a{{display:inline-block;margin-left:8px;padding:6px 12px;border-radius:999px;text-decoration:none;font-weight:700;border:1px solid #cbd5e1}}
</style>
<div class="wrap">
  <header>
    <h1>27042 取證報告（{run_id}）</h1>
    <div class="actions">
      <a href="sha256sums.txt" download>下載 SHA256SUMS</a>
    </div>
  </header>
  <div class="card">Tamper Coverage：<strong>{cov_txt}</strong></div>
  <div class="grid">
    <div class="card"><h3>Overlay</h3><img src="artifacts/overlay.png" alt="overlay"/></div>
    <div class="card"><h3>Mask (Gray)</h3><img src="artifacts/mask_gray.png" alt="mask_gray"/></div>
    <div class="card"><h3>Mask (Binary)</h3><img src="artifacts/mask_bin.png" alt="mask_bin"/></div>
    <div class="card"><h3>Patched (Reveal on Mask)</h3><img src="artifacts/patched_masked.png" alt="patched"/></div>
  </div>
  <div class="card"><h3>report.json</h3><pre>{esc(rep_txt)}</pre></div>
  <div class="card"><h3>SHA256SUMS</h3><pre>{esc(sha_txt)}</pre></div>
</div>
</html>"""
    out = run_dir / "report.html"
    out.write_text(html_doc, encoding="utf-8")
    return out

def _ensure_offline_and_sums(run_id: str, run_dir: Path, report: Optional[dict] = None):
    if report is None and (run_dir / "report.json").exists():
        try:
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        except Exception:
            report = {}
    if not (run_dir / "report.html").exists():
        _build_offline_report_html(run_id, run_dir, report or {})
    _write_sha256sums_file(run_dir)
    _build_offline_report_html(run_id, run_dir, report or {})

# ============================== Watermark Forward ==============================
def _forward_to_watermark(run_id: str, run_dir: Path, tiles_hint: str, secret_ref: str) -> dict:
    result = {"ok": False, "error": None, "response": None, "saved_image": None}
    if not WM_FORWARD:
        result["error"] = "WM_FORWARD=0"
        return result
    src = run_dir / "inputs" / "source.png"
    msk = run_dir / "artifacts" / "mask_bin.png"
    if not (src.exists() and msk.exists()):
        result["error"] = "source/mask not found"
        return result

    try:
        files = {
            "attack_img": ("source.png", open(src, "rb"), "image/png"),
            "mask":       ("mask_bin.png", open(msk, "rb"), "image/png"),
        }
        data = {
            "tiles_hint": tiles_hint or WM_TILES_HINT,
            "secret_ref": secret_ref or WM_SECRET_REF,
            "iid_run_id": run_id,
        }
        headers = {"Accept": "application/json, image/png;q=0.9"}
        resp = requests.post(WM_URL, files=files, data=data,
                             headers=headers, timeout=WM_TIMEOUT_SEC)
        ct = (resp.headers.get("content-type") or "").lower()

        if resp.ok and "image/png" in ct:
            out_img = run_dir / "artifacts" / "patched_masked.png"
            out_img.write_bytes(resp.content)
            result.update({"ok": True, "saved_image": str(out_img)})
            return result

        j = {}
        try: j = resp.json()
        except Exception: pass
        result["response"] = j
        (run_dir / "artifacts" / "wm_result.json").write_text(
            json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        patched_url = (j.get("images", {}) or {}).get("patched_masked_http") \
                      or (j.get("images", {}) or {}).get("patched_masked_url") \
                      or j.get("patched_masked_url")
        if isinstance(patched_url, str) and patched_url.lower().startswith(("http://","https://")):
            try:
                r2 = requests.get(patched_url, timeout=WM_TIMEOUT_SEC)
                if r2.ok and "image" in (r2.headers.get("content-type","").lower()):
                    out_img = run_dir / "artifacts" / "patched_masked.png"
                    out_img.write_bytes(r2.content)
                    result.update({"ok": True, "saved_image": str(out_img)})
                    return result
            except Exception:
                pass

        result["ok"] = bool(j.get("ok"))
        if not result["ok"] and not result.get("error"):
            result["error"] = f"WM HTTP {resp.status_code}"
        return result

    except Exception as e:
        result["error"] = repr(e)
        return result

# ============================== Routes ==============================

@app.route("/health", methods=["GET"])
def health():
    try:
        msg = {"ok": True, "base_dir": str(BASE_DIR),
               "weights_path": str(WEIGHTS_PATH) if WEIGHTS_PATH else None,
               "device": str(device),
               "wm_forward": WM_FORWARD, "wm_url": WM_URL}
        return msg, 200
    except Exception as e:
        return {"ok": False, "error": repr(e)}, 500

@app.route("/", methods=["GET"])
def index():
    return (
        "<h1>IID-Net 竄改偵測</h1>"
        "<p>上傳圖片，系統會用 IID-Net 產生遮罩並以紅色半透明疊加在原圖上。</p>"
        '<form method="POST" action="/result" enctype="multipart/form-data">'
        '<input type="file" name="image" accept="image/*" required /> '
        '<label>遮罩透明度 (0~1)：</label>'
        f'<input type="number" name="alpha" min="0" max="1" step="0.05" value="{OVERLAY_ALPHA}" /> '
        '<label style="margin-left:12px">tiles_hint：</label>'
        '<select name="wm_tiles_hint"><option value="auto" selected>auto</option>'
        '<option value="2">2x2</option><option value="3">3x3</option><option value="4">4x4</option></select> '
        '<button type="submit">開始偵測</button>'
        "</form>"
    )

@app.route("/result", methods=["POST"])
def result():
    if "image" not in request.files or request.files["image"].filename == "":
        return "沒有選擇圖片", 400
    file = request.files["image"]
    if not allowed_file(file.filename): return "檔案格式不支援", 400

    try:
        alpha = float(request.form.get("alpha", str(OVERLAY_ALPHA)))
    except Exception:
        alpha = OVERLAY_ALPHA
    if not np.isfinite(alpha): alpha = OVERLAY_ALPHA
    alpha = float(np.clip(alpha, 0.0, 1.0))

    form_tiles_hint = (request.form.get("wm_tiles_hint") or "").strip() or WM_TILES_HINT
    form_secret_ref = (request.form.get("wm_secret_ref") or "").strip() or WM_SECRET_REF

    try:
        arr = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None: return "無法讀取圖片", 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        tensor = transform(img_rgb).unsqueeze(0).to(device)

        net = build_infer_net()
        with torch.inference_mode():
            out = net(tensor)
            o_min, o_max = float(out.min().item()), float(out.max().item())
            prob = out if (0.0 <= o_min <= 1.0 and 0.0 <= o_max <= 1.0) else torch.sigmoid(out)

        mask = prob.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        mask = np.clip(mask, 0.0, 1.0)
        _print_stats("mask", mask)

        H0, W0 = img_bgr.shape[:2]
        if mask.shape != (H0, W0):
            mask = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_LINEAR)

        base = img_bgr.astype(np.float32)
        blended = base.copy()
        if not should_suppress(mask):
            idx = mask > PIXEL_THR
            if np.any(idx):
                red = np.zeros_like(base, dtype=np.float32); red[..., 2] = 255.0
                a = (alpha * mask[idx]).astype(np.float32)[:, None]
                blended[idx] = base[idx] * (1.0 - a) + red[idx] * a
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        _print_stats("blended_uint8", blended)

        run_id, run_dir = _new_run_dir()
        (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(run_dir / "inputs" / "source.png"), img_bgr)
        cv2.imwrite(str(run_dir / "artifacts" / "overlay.png"), blended)
        cv2.imwrite(str(run_dir / "artifacts" / "mask_gray.png"), (mask * 255.0).astype(np.uint8))
        mask_bin = (mask >= BIN_THR).astype(np.uint8) * 255
        cv2.imwrite(str(run_dir / "artifacts" / "mask_bin.png"), mask_bin)

        coverage = float((mask_bin > 0).mean())

        weights_rec = None
        if WEIGHTS_PATH and Path(WEIGHTS_PATH).exists():
            if SANITIZE_WEIGHTS:
                weights_rec = {
                    "name": Path(WEIGHTS_PATH).name,
                    "sha256": _sha256_path(Path(WEIGHTS_PATH))
                }
            else:
                weights_rec = str(WEIGHTS_PATH)

        report = {
            "schema": "com.mirrorlab.27042/v1",
            "run": {
                "id": run_id,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "engine": "IID-Net",
                "device": str(device),
                "weights": weights_rec,
                "overlay_alpha": alpha,
                "binary_threshold": BIN_THR
            },
            "inputs": {"original": "inputs/source.png"},
            "artifacts": {
                "overlay":  "artifacts/overlay.png",
                "mask_gray":"artifacts/mask_gray.png",
                "mask_bin": "artifacts/mask_bin.png"
            },
            "analysis": {"tamper_coverage_ratio": coverage}
        }

        wm_rec = _forward_to_watermark(run_id, run_dir,
                                       tiles_hint=form_tiles_hint,
                                       secret_ref=form_secret_ref)
        report["watermark"] = {
            "forward": WM_FORWARD,
            "wm_url": WM_URL,
            "result": wm_rec
        }
        if wm_rec.get("saved_image"):
            report["artifacts"]["patched_masked"] = "artifacts/patched_masked.png"

        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        _build_offline_report_html(run_id, run_dir, report)
        _write_sha256sums_file(run_dir)
        _build_offline_report_html(run_id, run_dir, report)

        _zip_run_dir(run_dir)

        ok, buf = cv2.imencode(".png", blended)
        if not ok: return "編碼輸出失敗", 500
        bio = io.BytesIO(buf.tobytes()); bio.seek(0)
        return send_file(bio, mimetype="image/png")

    except FileNotFoundError as e:
        traceback.print_exc()
        return (f"<h3>伺服器錯誤：找不到必要檔案</h3><pre>{str(e)}</pre>"
                f"<p>BASE_DIR: {BASE_DIR}</p><p>WEIGHTS_PATH: {WEIGHTS_PATH}</p>"), 500
    except Exception as e:
        traceback.print_exc()
        return ("<h3>伺服器錯誤</h3><pre>{}</pre>".format(repr(e))), 500

@app.route("/27042/latest")
def latest_27042():
    rid = _latest_run_id()
    if not rid: return jsonify({"ok": False, "error": "尚未產生任何 27042 報告"}), 404
    return jsonify({"ok": True,
                    "run_id": rid,
                    "view_url": url_for("view_27042", run_id=rid, _external=True),
                    "download_url": url_for("download_27042", run_id=rid, _external=True)})

# ---------- 檢視頁 ----------
@app.route("/27042/view/<run_id>")
def view_27042(run_id):
    if not _safe_run_id(run_id): return "Invalid run_id", 400
    html_tpl = r"""<!doctype html>
<html lang="zh-Hant">
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>27042 取證報告（__RUN_ID__）</title>
<style>
  :root{--blue-600:#0b5ed7;--blue-700:#0a53be;--blue-500:#1976d2;--ink:#0f172a;--muted:#64748b;--card:#ffffff;--page:#f5f8ff;--line:#e6eefc}
  *{box-sizing:border-box}
  body{background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans TC",Arial,sans-serif;margin:0}
  header{background:linear-gradient(135deg,var(--blue-600) 0%,var(--blue-500) 100%);color:#fff;padding:16px 20px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 6px 18px rgba(11,62,145,.25)}
  h1{font-size:18px;margin:0;letter-spacing:.5px}
  .actions a{display:inline-block;margin-left:12px;padding:8px 16px;border-radius:999px;text-decoration:none;font-weight:700;transition:transform .15s, background-color .2s, border-color .2s}
  .actions a.dl{background:#fff0;border:2px solid #fff;color:#fff}
  .actions a.dl:hover{background:#fff;color:var(--blue-700)}
  .actions a.sha{background:#fff0;border:2px solid #fff;color:#fff}
  .actions a.sha:hover{background:#fff;color:var(--blue-700)}
  .container{max-width:1200px;margin:20px auto;padding:0 16px}
  .kv{margin:6px 0 18px 0;font-size:15px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 26px rgba(11,62,145,.08);padding:14px}
  .card h3{margin:0 0 10px 0;color:var(--blue-700)}
  .muted{color:var(--muted)}
  img{max-width:100%;height:auto;border-radius:10px;background:#fff;border:1px solid var(--line)}
  pre.code{margin:0;background:#fbfdff;border:1px solid var(--line);border-radius:10px;padding:12px;max-height:380px;overflow:auto;font-size:13px;line-height:1.45}
  ul.meta{margin:0;padding-left:16px;color:#111}
  ul.meta li{margin:4px 0}
</style>
</head>
<body>
  <header>
    <h1>27042 取證報告（__RUN_ID__）</h1>
    <div class="actions">
      <a class="sha" href="__SHA_FILE_URL__" download>下載 SHA256SUMS</a>
      <a class="dl"  href="/27042/download/__RUN_ID__">下載 ZIP</a>
    </div>
  </header>
  <div class="container">
    <div class="kv">Tamper Coverage：<strong id="cov">—</strong></div>
    <div class="grid">
      <div class="card"><h3>Overlay</h3><div id="ovr" class="muted">尚未找到 overlay.png</div></div>
      <div class="card"><h3>Mask (Gray)</h3><div id="mg" class="muted">尚未找到 mask_gray.png</div></div>
      <div class="card"><h3>Mask (Binary)</h3><div id="mb" class="muted">尚未找到 mask_bin.png</div></div>
      <div class="card"><h3>Patched (Reveal on Mask)</h3><div id="pm" class="muted">尚未找到 patched_masked.png</div></div>
      <div class="card" style="grid-column:1/-1;"><h3>report.json</h3><pre id="jsonPre" class="code muted">尚未產生 report.json</pre></div>
      <div class="card"><h3>Technical Details &amp; Integrity</h3><ul id="techList" class="meta"><li class="muted">載入中…</li></ul></div>
      <div class="card"><h3>SHA256SUMS (partial preview)</h3><pre id="shaPre" class="code muted">計算中…</pre></div>
    </div>
  </div>
  <script>
  (function(){
    const cov=document.getElementById('cov'),ovr=document.getElementById('ovr'),mg=document.getElementById('mg'),mb=document.getElementById('mb'),pm=document.getElementById('pm'),pre=document.getElementById('jsonPre'),tech=document.getElementById('techList'),sha=document.getElementById('shaPre');
    const checkUrl=location.origin+"__CHECK_URL__", reportUrl=location.origin+"__REPORT_URL__", techUrl=location.origin+"__TECH_URL__", shaUrl=location.origin+"__SHA_URL__";
    let tries=0, MAX=150;
    const fetchOnce=async (url, handler)=>{try{const r=await fetch(url+"?t="+Date.now()); if(r.ok){handler(await r.json()); return true;}}catch(e){} return false;};
    const renderTech=(d)=>{tech.innerHTML=""; const add=(k,v)=>{const li=document.createElement("li"); li.innerHTML="<strong>"+k+"</strong>： "+v; tech.appendChild(li);}; add("Run ID", d.run_id||"—"); add("Source", d.source||"—"); add("Created At", d.created_at||"—"); add("Files Total", d.files_total??"—"); add("overlay.png", d.has_overlay?"✔︎":"✘"); add("mask_gray.png", d.has_mask_gray?"✔︎":"✘"); add("mask_bin.png", d.has_mask_bin?"✔︎":"✘"); add("patched_masked.png", d.has_patched?"✔︎":"✘"); add("sha256sums.txt", d.has_sha256sums?"✔︎":"✘"); if(d.notes&&d.notes.length){add("Notes", d.notes.join("；"));}};
    const renderSha=(d)=>{ if(d&&d.ok&&d.lines&&d.lines.length){ sha.textContent=d.lines.join("\n"); sha.classList.remove("muted"); } else { sha.textContent="尚無可供預覽的雜湊。"; }};
    const put=(div,src)=>{ if(!src) return; const img=new Image(); img.onload=()=>{div.innerHTML=''; div.classList.remove('muted'); div.appendChild(img);}; img.src=src;};
    const tick=async()=>{ try{const r=await fetch(checkUrl+'?t='+Date.now()); if(r.ok){const d=await r.json(); if(typeof d.coverage==="number"){cov.textContent=(d.coverage*100).toFixed(2)+'%';} put(ovr,d.overlay); put(mg,d.mask_gray); put(mb,d.mask_bin); put(pm,d.patched_masked);} }catch(e){} if(pre&&pre.classList.contains('muted')){ fetchOnce(reportUrl, j=>{pre.textContent=JSON.stringify(j,null,2); pre.classList.remove('muted');}); } if(tech&&tech.querySelector('.muted')){ fetchOnce(techUrl, renderTech); } if(sha&&sha.classList.contains('muted')){ fetchOnce(shaUrl, renderSha); } if(++tries<MAX) setTimeout(tick,1200); };
    tick();
  })();
  </script>
</body>
</html>
"""
    html_page = (html_tpl
                 .replace("__RUN_ID__", run_id)
                 .replace("__CHECK_URL__", url_for("check_27042", run_id=run_id))
                 .replace("__REPORT_URL__", url_for("report_27042", run_id=run_id))
                 .replace("__TECH_URL__", url_for("tech_27042", run_id=run_id))
                 .replace("__SHA_URL__", url_for("sha256sums_27042", run_id=run_id))
                 .replace("__SHA_FILE_URL__", url_for("sha256sums_file_27042", run_id=run_id)))
    resp = make_response(html_page)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

# ---------- check / report / tech / sha ----------
@app.route("/27042/check/<run_id>")
def check_27042(run_id):
    if not _safe_run_id(run_id): return jsonify({"ok": False, "error": "bad run_id"}), 400
    overlay = maskg = maskb = patched = None; cov = None
    d = _dir_of_run(run_id)
    if d:
        ro = _find_in_dir(d, ("overlay.png",)); rg = _find_in_dir(d, ("mask_gray.png",)); rb = _find_in_dir(d, ("mask_bin.png",)); rp = _find_in_dir(d, ("patched_masked.png",))
        if ro: overlay = url_for("raw_27042", run_id=run_id, relpath=ro)
        if rg: maskg   = url_for("raw_27042", run_id=run_id, relpath=rg)
        if rb: maskb   = url_for("raw_27042", run_id=run_id, relpath=rb)
        if rp: patched = url_for("raw_27042", run_id=run_id, relpath=rp)
    else:
        z = _zip_of_run(run_id)
        if z:
            ro = _find_in_zip(z, ("overlay.png",)); rg = _find_in_zip(z, ("mask_gray.png",)); rb = _find_in_zip(z, ("mask_bin.png",)); rp = _find_in_zip(z, ("patched_masked.png",))
            if ro: overlay = url_for("zip_asset_27042", run_id=run_id, inner=ro)
            if rg: maskg   = url_for("zip_asset_27042", run_id=run_id, inner=rg)
            if rb: maskb   = url_for("zip_asset_27042", run_id=run_id, inner=rb)
            if rp: patched = url_for("zip_asset_27042", run_id=run_id, inner=rp)
    rep = _read_report_json(run_id)
    if rep:
        try: cov = float(rep.get("analysis", {}).get("tamper_coverage_ratio"))
        except Exception: cov = None
    return jsonify({"ok": True, "run_id": run_id, "coverage": cov,
                    "overlay": overlay, "mask_gray": maskg, "mask_bin": maskb,
                    "patched_masked": patched})

@app.route("/27042/report/<run_id>")
def report_27042(run_id):
    if not _safe_run_id(run_id): return jsonify({"ok": False, "error": "bad run_id"}), 400
    rep = _read_report_json(run_id)
    if not rep: return jsonify({"ok": False, "error": "report.json not found"}), 404
    return jsonify(rep)

@app.route("/27042/tech/<run_id>")
def tech_27042(run_id):
    if not _safe_run_id(run_id): return jsonify({"ok": False, "error": "bad run_id"}), 400
    d = _dir_of_run(run_id); z = _zip_of_run(run_id)
    result = {"ok": True, "run_id": run_id, "source": None, "files_total": 0, "created_at": None,
              "has_overlay": False, "has_mask_gray": False, "has_mask_bin": False,
              "has_patched": False, "has_sha256sums": False, "notes": []}
    if d and d.exists():
        result["source"] = "dir"
        try: result["created_at"] = _utc_iso(d.stat().st_mtime)
        except Exception: pass
        files = []
        for dp, _, fs in os.walk(d):
            for f in fs: files.append(1)
        result["files_total"] = len(files)
        result["has_overlay"] = bool(_find_in_dir(d, ("overlay.png",)))
        result["has_mask_gray"] = bool(_find_in_dir(d, ("mask_gray.png",)))
        result["has_mask_bin"] = bool(_find_in_dir(d, ("mask_bin.png",)))
        result["has_patched"] = bool(_find_in_dir(d, ("patched_masked.png",)))
        result["has_sha256sums"] = (d / "sha256sums.txt").exists()
    elif z and z.exists():
        result["source"] = "zip"
        try: result["created_at"] = _utc_iso(z.stat().st_mtime)
        except Exception: pass
        try:
            with zipfile.ZipFile(z, "r") as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                result["files_total"] = len(names)
                result["has_overlay"] = bool(_find_in_zip(z, ("overlay.png",)))
                result["has_mask_gray"] = bool(_find_in_zip(z, ("mask_gray.png",)))
                result["has_mask_bin"] = bool(_find_in_zip(z, ("mask_bin.png",)))
                result["has_patched"] = bool(_find_in_zip(z, ("patched_masked.png",)))
                result["has_sha256sums"] = any(n.lower().endswith("sha256sums.txt") for n in names)
        except Exception as e:
            result["notes"].append(f"zip read error: {e!r}")
    else:
        return jsonify({"ok": False, "error": "run not found"}), 404
    return jsonify(result)

def _sha256_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

@app.route("/27042/sha256sums/<run_id>")
def sha256sums_27042(run_id):
    if not _safe_run_id(run_id): return jsonify({"ok": False, "error": "bad run_id"}), 400
    max_lines = int(request.args.get("limit", "30"))
    d = _dir_of_run(run_id)
    if d and (d / "sha256sums.txt").exists():
        txt = (d / "sha256sums.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
        return jsonify({"ok": True, "run_id": run_id, "lines": txt[:max_lines]})
    lines = []
    if d:
        files = [p for p in sorted(d.rglob("*")) if p.is_file()]
        for ap in files:
            if len(lines) >= max_lines: break
            rp = str(ap.relative_to(d)).replace("\\", "/")
            try:
                sha, sz, mt = _sha256_path(ap), ap.stat().st_size, _utc_iso(ap.stat().st_mtime)
                lines.append(f"{sha}  {rp}  ({sz} bytes)  {mt}")
            except Exception: pass
        return jsonify({"ok": True, "run_id": run_id, "lines": lines})
    z = _zip_of_run(run_id)
    if z:
        try:
            with zipfile.ZipFile(z, "r") as zf:
                if "sha256sums.txt" in [n.lower() for n in zf.namelist()]:
                    with zf.open("sha256sums.txt") as fh:
                        txt = fh.read().decode("utf-8", "ignore").splitlines()
                        return jsonify({"ok": True, "run_id": run_id, "lines": txt[:max_lines]})
                names = [n for n in zf.namelist() if not n.endswith("/")]
                for n in names:
                    if len(lines) >= max_lines: break
                    try:
                        data = zf.read(n)
                        mt = "N/A"
                        lines.append(f"{_sha256_bytes(data)}  {n}  ({len(data)} bytes)  {mt}")
                    except Exception: pass
        except Exception as e:
            return jsonify({"ok": False, "error": f"zip read error: {e!r}"}), 500
        return jsonify({"ok": True, "run_id": run_id, "lines": lines})
    return jsonify({"ok": False, "error": "run not found"}), 404

@app.route("/27042/sha256sums_file/<run_id>")
def sha256sums_file_27042(run_id):
    if not _safe_run_id(run_id): return "bad run_id", 400
    d = _dir_of_run(run_id)
    if d:
        _ensure_offline_and_sums(run_id, d)
        sums = d / "sha256sums.txt"
        if sums.exists():
            return send_file(str(sums), as_attachment=True, download_name="sha256sums.txt", mimetype="text/plain; charset=utf-8")
    return "not found", 404

# ---------- 資產與下載 ----------
@app.route("/27042/raw/<run_id>/<path:relpath>")
def raw_27042(run_id, relpath):
    if not _safe_run_id(run_id): return "bad run_id", 400
    run_dir = _dir_of_run(run_id)
    if not run_dir: return "not found", 404
    full = (run_dir / relpath).resolve()
    if not str(full).startswith(str(run_dir.resolve())): return "forbidden", 403
    if not full.exists(): return "not found", 404
    return send_file(str(full))

@app.route("/27042/zip_asset/<run_id>/<path:inner>")
def zip_asset_27042(run_id, inner):
    if not _safe_run_id(run_id): return "bad run_id", 400
    z = _zip_of_run(run_id)
    if not z: return "not found", 404
    try:
        with zipfile.ZipFile(z, "r") as zf:
            data = zf.read(inner)
    except Exception: return "not found", 404
    bio = io.BytesIO(data); bio.seek(0)
    return send_file(bio)

@app.route("/27042/download/<run_id>")
def download_27042(run_id):
    if not _safe_run_id(run_id): return "bad run_id", 400
    d = _dir_of_run(run_id)
    if d:
        _ensure_offline_and_sums(run_id, d)
        z = _zip_run_dir(d)
    else:
        z = _zip_of_run(run_id)
        if not z: return "not found", 404
    return send_file(str(z), as_attachment=True, download_name=f"{run_id}.zip", mimetype="application/zip")

# ============================== 相容別名（重點補強） ==============================
# 1) /files/<run_id>/artifacts/* 與 /files/<run_id>/inputs/* 直接對應到實體檔
@app.route("/files/<run_id>/artifacts/<path:name>")
def files_artifacts(run_id, name):
    return raw_27042(run_id, f"artifacts/{name}")

@app.route("/files/<run_id>/inputs/<path:name>")
def files_inputs(run_id, name):
    return raw_27042(run_id, f"inputs/{name}")

# 2) /files/<run_id>/<name>：聰明對映常見檔名到 artifacts/ 或 inputs/
@app.route("/files/<run_id>/<path:name>")
def files_smart(run_id, name):
    # 先嘗試原樣（相對於 run 根目錄）
    r = raw_27042(run_id, name)
    if getattr(r, "status_code", 200) == 200:
        return r
    # 聰明對映
    tail = Path(name).name.lower()
    if tail in ("overlay.png", "mask_gray.png", "mask_bin.png", "patched_masked.png", "wm_result.json"):
        return raw_27042(run_id, f"artifacts/{tail}")
    if tail in ("source.png", "original.png"):
        return raw_27042(run_id, f"inputs/{tail}")
    # 否則回 404
    return "not found", 404

# 3) 舊前端可能請求 /report/<run_id> 或 /dl/<run_id>
@app.route("/report/<run_id>")
def alias_report(run_id):
    return redirect(url_for("report_27042", run_id=run_id), code=307)

@app.route("/dl/<run_id>")
def alias_download(run_id):
    return redirect(url_for("download_27042", run_id=run_id), code=307)

# ============================== main ==============================
if __name__ == "__main__":
    print(f"[INFO] Flask on http://0.0.0.0:5000  (cwd={os.getcwd()})")
    if WEIGHTS_PATH: print(f"[DEBUG] Using weights: {WEIGHTS_PATH}")
    else: print("[WARN] 未設定權重，推論會失敗；但後端可啟動以便測試頁面/打包流程。")
    print(f"[INFO] WM_FORWARD={WM_FORWARD} WM_URL={WM_URL}")
    app.run(host="0.0.0.0", port=5000, debug=True)
