#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app_wm.py — Robust WM (TF2.15 / Keras2.15)

重點：
- 報告外觀：Forensics Report — {case_id}（Session/Started/Operator/Base、Environment JSON、Audit Log 預覽、Images、Artifacts）
- 新增：Download Report 按鈕（/dl/report/<job_id>）
- /27037/latest：<root>/_latest.json 持久化，回報建立中/就緒
- 產出：cover.png / secret_in.png / container.png / residual.png / secret.png / report.html / logs/audit.jsonl / environment.json
- Real-ESRGAN（可選）放大最佳小圖
"""

import os, io, time, uuid, json, threading, subprocess, glob, re, math, zipfile, datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from blockchain import init_chain, get_chain, sha256_bytes

import numpy as np
from PIL import Image, PngImagePlugin
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory, send_file, abort
import cv2

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False

# ---- 基礎 ----
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("GPU devices:", gpus)
    except RuntimeError as e:
        print("GPU init error:", e)
else:
    print("GPU devices: []")

IMG_SIZE = (256, 256)
_resample = getattr(Image, "Resampling", Image)
LANCZOS  = getattr(_resample, "LANCZOS", Image.LANCZOS)

# ---- 權重路徑 ----
from pathlib import Path as _P
def _resolve_weight_path(primary: str, fallbacks: list) -> str:
    p = _P(primary)
    if p.is_file():
        return str(p)
    for alt in fallbacks:
        if _P(alt).is_file():
            return str(alt)
    return primary

WEIGHTS_DIR = _P(r"C:\Users\admin\Desktop\114屆照妖鏡\rubust_matermark\weights")
COMBINED_H5 = _resolve_weight_path(str(WEIGHTS_DIR / "combined_model.h5"), [])
REVEAL_H5   = _resolve_weight_path(
    str(WEIGHTS_DIR / "reveal_network.h5"),
    [str(WEIGHTS_DIR / "reveal_network .h5")]
)

# ---- 小工具 ----
def env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return default

def env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except Exception: return default

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, None)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def now_ms() -> int:
    return int(time.time() * 1000)

def gen_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

# ---- 影像 I/O ----
def load_from_bytes(b: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("RGB").resize(IMG_SIZE, LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return (arr / 127.5) - 1.0

def to_uint8_image(arr_m11: np.ndarray) -> Image.Image:
    arr = ((np.asarray(arr_m11, np.float32) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def compute_residual_img(a_u8: Image.Image, b_u8: Image.Image) -> Image.Image:
    a = np.asarray(a_u8, np.int16)
    b = np.asarray(b_u8, np.int16)
    diff = np.abs(a - b).clip(0, 255).astype(np.uint8)
    return Image.fromarray(diff)

def psnr_m11(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12: return 99.0
    return 20.0 * np.log10(2.0) - 10.0 * np.log10(mse)

def ssim_m11(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(a, b, channel_axis=2, data_range=2.0))
    except Exception:
        return -float(np.mean((a - b) ** 2))

# ---- 模型 ----
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs); self.epsilon = epsilon
    def build(self, input_shape):
        ch = int(input_shape[-1])
        self.gamma = self.add_weight("gamma", shape=(ch,), initializer="ones", trainable=True)
        self.beta  = self.add_weight("beta",  shape=(ch,), initializer="zeros", trainable=True)
        super().build(input_shape)
    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        var  = tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True)
        xhat = (x - mean) / tf.sqrt(var + self.epsilon)
        return self.gamma * xhat + self.beta

class AttackLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(), initializer="ones", trainable=False)
        self.beta  = self.add_weight(name="beta",  shape=(), initializer="zeros", trainable=False)
        super().build(input_shape)
    def call(self, x):
        g = tf.cast(self.gamma, x.dtype); b = tf.cast(self.beta,  x.dtype)
        return x * g + b

from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Add, UpSampling2D, Concatenate, Lambda
from tensorflow.keras.models import Model

def _norm(norm: str, name=None):
    return InstanceNormalization(name=name) if norm=="in" else tf.keras.layers.BatchNormalization(name=name)

def robust_residual_block(x, filters, norm="bn", stem="rb"):
    sc = x
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal', name=f"{stem}_conv1")(x)
    x = _norm(norm, name=f"{stem}_norm1")(x)
    x = LeakyReLU(alpha=0.2, name=f"{stem}_lrelu1")(x)
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal', name=f"{stem}_conv2")(x)
    x = _norm(norm, name=f"{stem}_norm2")(x)
    return Add(name=f"{stem}_add")([sc, x])

def build_hiding_network(input_shape=(256, 256, 3), norm="bn"):
    S1_input = Input(shape=input_shape, name='S1_Input')
    S2_input = Input(shape=input_shape, name='S2_Input')
    x = Concatenate(name="concat_S1S2")([S1_input, S2_input])
    x = Conv2D(64, 3, padding='same', name="h_conv0")(x); x = _norm(norm, "h_norm0")(x); x = LeakyReLU(alpha=0.2, name="h_lrelu0")(x)
    for i in range(6): x = robust_residual_block(x, 64, norm=norm, stem=f"h_rb{i}")
    x = Conv2D(128, 3, strides=2, padding='same', name="h_down")(x); x = _norm(norm, "h_norm_down")(x); x = LeakyReLU(alpha=0.2, name="h_lrelu_down")(x)
    for i in range(6): x = robust_residual_block(x, 128, norm=norm, stem=f"h_rb2_{i}")
    x = UpSampling2D(name="h_up")(x)
    x = Conv2D(64, 3, padding='same', name="h_conv_out")(x); x = _norm(norm, "h_norm_out")(x); x = LeakyReLU(alpha=0.2, name="h_lrelu_out")(x)
    S1_prime = Conv2D(3, 3, padding='same', activation='tanh', name='S1_Prime')(x)
    return Model([S1_input, S2_input], S1_prime, name='Robust_Hiding_Network')

def build_reveal_network(input_shape=(256, 256, 3), norm="bn"):
    S1p_in = Input(shape=input_shape, name='S1_Prime_Input')
    x = Conv2D(64, 3, padding='same', name="r_conv0")(S1p_in); x = _norm(norm, "r_norm0")(x); x = LeakyReLU(alpha=0.2, name="r_lrelu0")(x)
    for i in range(6): x = robust_residual_block(x, 64, norm=norm, stem=f"r_rb{i}")
    x = Conv2D(128, 3, strides=2, padding='same', name="r_down")(x); x = _norm(norm, "r_norm_down")(x); x = LeakyReLU(alpha=0.2, name="r_lrelu_down")(x)
    for i in range(6): x = robust_residual_block(x, 128, norm=norm, stem=f"r_rb2_{i}")
    x = UpSampling2D(name="r_up")(x)
    x = Conv2D(64, 3, padding='same', name="r_conv_out")(x); x = _norm(norm, "r_norm_out")(x); x = LeakyReLU(alpha=0.2, name="r_lrelu_out")(x)
    S2_Prime = Conv2D(3, 3, padding='same', activation='tanh', name='S2_Prime')(x)
    return Model(S1p_in, S2_Prime, name='Robust_Reveal_Network')

def build_combined(norm="bn"):
    S1 = Input(shape=(256, 256, 3), name="Cover_Image")
    S2 = Input(shape=(256, 256, 3), name="Secret_Image")
    hiding = build_hiding_network(norm=norm)
    reveal = build_reveal_network(norm=norm)
    S1p_clean = hiding([S1, S2])
    S1p_att = AttackLayer(name="S1_Prime_Attack")(S1p_clean)
    S2p     = reveal(S1p_att)
    S1_out = Lambda(lambda x: x, name="S1_Prime_out")(S1p_clean)
    S2_out = Lambda(lambda x: x, name="S2_Prime_out")(S2p)
    return Model([S1, S2], [S1_out, S2_out], name="Stego_2out"), hiding, reveal

# ---- 權重載入 ----
import h5py
def _detect_norm_in_h5(h5_path: str) -> str:
    try:
        with h5py.File(h5_path, "r") as f:
            text = ",".join(list(f.keys()))
            if "model_weights" in f:
                paths = []
                f["model_weights"].visit(lambda k: paths.append(k))
                text += " " + " ".join(paths)
            text = text.lower()
            return "in" if ("instance" in text or "instancenorm" in text or "in_norm" in text) else "bn"
    except Exception:
        return "bn"

def _strict_load_weights(model: tf.keras.Model, path: str) -> Tuple[bool, str]:
    try:
        model.load_weights(path)
        return True, "ok"
    except Exception as e:
        return False, str(e)

def _byname_load_weights(model: tf.keras.Model, path: str) -> Tuple[bool, str]:
    try:
        model.load_weights(path, by_name=True, skip_mismatch=False)
        return True, "ok"
    except Exception as e:
        try:
            model.load_weights(path, by_name=True, skip_mismatch=True)
            return True, f"by_name+skip_mismatch: {e}"
        except Exception as e2:
            return False, f"{e} | {e2}"

_combined_model = None
_reveal_model   = None
_load_lock = threading.Lock()

def get_models() -> Tuple[tf.keras.Model, tf.keras.Model]:
    global _combined_model, _reveal_model
    with _load_lock:
        norm_guess = _detect_norm_in_h5(REVEAL_H5)
        ng2 = _detect_norm_in_h5(COMBINED_H5)
        if ng2 != norm_guess: norm_guess = ng2
        print(f"[wm] 掃描權重判定 norm = {norm_guess.upper()}")

        if _combined_model is None:
            cm, _, _ = build_combined(norm=norm_guess)
            ok, msg = _strict_load_weights(cm, COMBINED_H5)
            if not ok:
                print(f"[wm] combined 嚴格載入失敗：{msg} -> 改 by_name")
                ok2, msg2 = _byname_load_weights(cm, COMBINED_H5)
                if not ok2: raise RuntimeError(f"Combined 權重載入失敗：{msg2}")
            _combined_model = cm
            print(f"[wm] Combined 權重載入完成：{COMBINED_H5}")

        if _reveal_model is None:
            rv = build_reveal_network(norm=norm_guess)
            ok, msg = _strict_load_weights(rv, REVEAL_H5)
            if not ok:
                print(f"[wm] reveal 嚴格載入失敗：{msg} -> 改 by_name")
                ok2, msg2 = _byname_load_weights(rv, REVEAL_H5)
                if not ok2: raise RuntimeError(f"Reveal 權重載入失敗：{msg2}")
            _reveal_model = rv
            print(f"[wm] Reveal 權重載入完成：{REVEAL_H5}")

    return _combined_model, _reveal_model

# ---- Secret 拼貼（無上傳 secret 時）----
def make_tiled_secret_from_cover(cover_u8: Image.Image, grid: int) -> Image.Image:
    grid = int(grid)
    if grid not in (2, 3, 4): grid = 2
    edge = IMG_SIZE[0] // grid
    w, h = cover_u8.width, cover_u8.height
    s = min(w, h)
    cx, cy = w // 2, h // 2
    left = max(0, cx - s // 2); top = max(0, cy - s // 2)
    patch = cover_u8.crop((left, top, left + s, top + s)).resize((edge, edge), LANCZOS)
    canvas = Image.new("RGB", IMG_SIZE)
    for yy in range(0, IMG_SIZE[1], edge):
        for xx in range(0, IMG_SIZE[0], edge):
            canvas.paste(patch, (xx, yy))
    return canvas

# ---- Real-ESRGAN ----
REALESRGAN_REPO = Path(os.getenv("REALESRGAN_REPO", "Real-ESRGAN"))
REALESRGAN_MODEL = os.getenv("REALESRGAN_MODEL", "RealESRGAN_x2plus")

def _have_realesrgan() -> bool:
    return (REALESRGAN_REPO / "inference_realesrgan.py").is_file()

def _run_realesrgan_on_file(in_path: Path, out_dir: Path, outscale: float, face_enhance: bool=False) -> Optional[Path]:
    try:
        py = Path(os.getenv("PYTHON_EXE", ""))  # 子環境
        if not py.is_file(): py = Path(os.sys.executable)
        script = REALESRGAN_REPO / "inference_realesrgan.py"
        if not script.is_file(): return None

        cmd = [
            str(py), str(script),
            "-n", REALESRGAN_MODEL,
            "-i", str(in_path),
            "-o", str(out_dir),
            "--outscale", str(float(outscale)),
            "--tile_pad", "10"
        ]

        tile_env = os.getenv("REALESRGAN_TILE", "").strip()
        cmd += ["--tile", tile_env if tile_env.isdigit() and int(tile_env)>0 else "0"]

        if env_bool("REALESRGAN_FP32", True):
            cmd.append("--fp32")

        model_path = os.getenv("REALESRGAN_MODEL_PATH", "").strip()
        if model_path and Path(model_path).is_file():
            cmd += ["--model_path", model_path]

        if face_enhance: cmd.append("--face_enhance")

        print("[Real-ESRGAN] run:", " ".join(cmd))
        r = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("[Real-ESRGAN] log:\n", r.stdout[:2000])
        if r.returncode != 0: return None

        stem = in_path.stem
        outs = []
        for p in glob.glob(str(out_dir / f"{stem}_out.*")):
            if re.search(r"_out\.(png|jpg|jpeg|webp)$", p, re.I):
                outs.append(Path(p))
        return outs[0] if outs else None
    except Exception as e:
        print("[Real-ESRGAN] failed:", e)
        return None

def _upscale_best_tile_with_realesrgan(tile_u8: Image.Image, work_dir: Path, target_edge: int=256) -> Image.Image:
    if not _have_realesrgan():
        print("[Real-ESRGAN] repo not found → fallback to LANCZOS.")
        return tile_u8.resize((target_edge, target_edge), LANCZOS)

    in_path = work_dir / "tile_best_src.png"
    tile_u8.save(in_path)

    w, h = tile_u8.size
    edge = max(w, h) if max(w,h)>0 else 1
    outscale = max(1.0, float(target_edge) / float(edge))

    out_path = _run_realesrgan_on_file(in_path, work_dir, outscale=outscale, face_enhance=False)
    if out_path is None or (not out_path.is_file()):
        print("[Real-ESRGAN] inference failed → fallback to LANCZOS.")
        return tile_u8.resize((target_edge, target_edge), LANCZOS)

    try:
        with Image.open(out_path).convert("RGB") as up:
            up = up.copy()
        return up.resize((target_edge, target_edge), LANCZOS) if up.size != (target_edge, target_edge) else up
    except Exception as e:
        print("[Real-ESRGAN] open result failed:", e)
        return tile_u8.resize((target_edge, target_edge), LANCZOS)

# ---- 評分 / 量測 ----
def _clip_m11(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)

def _find_scale_for_psnr(cover: np.ndarray, delta: np.ndarray, psnr_target: float,
                         s_min: float, s_max: float, iters: int = 16) -> float:
    lo, hi = s_min, s_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = _clip_m11(cover + mid * delta)
        p = psnr_m11(cand, cover)
        if p >= psnr_target: lo = mid
        else: hi = mid
    return lo

def _refine_by_reveal(rv: tf.keras.Model, cover: np.ndarray, delta: np.ndarray,
                      secret: np.ndarray, base_s: float, psnr_target: float) -> Tuple[float, float]:
    span  = 0.25 * base_s + 1e-6
    steps = 6
    best_s, best_score = base_s, -1e9
    for i in range(steps):
        t = (i / (steps - 1))
        s = max(0.0, base_s - span/2 + t * span)
        cand = _clip_m11(cover + s * delta)
        if psnr_m11(cand, cover) < psnr_target: continue
        sec_hat = rv.predict(cand[None, ...], verbose=0)[0]
        score = ssim_m11(sec_hat, secret)
        if score > best_score:
            best_score, best_s = score, s
    return best_s, best_score

# ---- Environment & Audit ----
def _env_snapshot() -> dict:
    return {
        "python": os.sys.version.split()[0],
        "tensorflow": tf.__version__,
        "cuda_visible": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "torch": None,
        "torch_cuda": None,
        "have_skimage": _HAVE_SKIMAGE,
        "realesrgan_repo": str(REALESRGAN_REPO),
        "realesrgan_model": REALESRGAN_MODEL,
        "time_utc": datetime.datetime.utcnow().isoformat() + "Z"
    }

def _audit_append(job_dir: Path, obj: dict):
    log_dir = ensure_dir(job_dir / "logs")
    p = log_dir / "audit.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---- Forensics Report 樣式 ----
MONO_CSS = """
body{font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,'Noto Sans','Apple Color Emoji','Segoe UI Emoji'; margin:24px;}
h1,h2,h3{margin:0 0 8px 0}
h1{font-size:22px; font-weight:700}
h2{font-size:18px; margin-top:18px}
h3{font-size:16px; margin-top:14px}
.small{font-size:12px; color:#666}
.kv{display:grid; grid-template-columns:180px 1fr; gap:6px 12px; margin:8px 0 12px}
pre.code{background:#fafafa; border:1px solid #eee; padding:10px; overflow:auto; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size:12px; line-height:1.45}
.grid-img{display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top:8px}
.card{border:1px solid #eee; border-radius:6px; padding:10px; background:#fff}
.card h3{margin:0 0 6px 0}
.row{display:flex; gap:8px; align-items:center}
a{color:#0b57d0; text-decoration:none}
a:hover{text-decoration:underline}
hr{border:0;border-top:1px solid #eee; margin:16px 0}
.code-inline{font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px}
.btn{display:inline-block; padding:8px 14px; border-radius:8px; border:1px solid #e5e7eb; text-decoration:none; color:#111; background:#fff}
.btn:hover{background:#f9fafb; text-decoration:none}
"""

REPORT_HTML = """
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Forensics Report — {case_title}</title>
<style>{css}</style>
<body>
  <h1>Forensics Report — {case_title}</h1>

  <div class="row" style="margin:8px 0 6px">
    <a class="btn" href="/dl/report/{case_title}">Download Report</a>
  </div>

  <div class="kv">
    <div><strong>Session:</strong></div><div>{session_id}</div>
    <div><strong>Started (UTC):</strong></div><div>{started_utc}</div>
    <div><strong>Operator:</strong></div><div>{operator}</div>
    <div><strong>Base:</strong></div><div class="code-inline">{base}</div>
  </div>

  <h2>Environment</h2>
  <pre class="code">{environment_json}</pre>

  <h2>Audit Log (Preview)</h2>
  <div class="small">Full log: <span class="code-inline">{audit_path}</span></div>
  <pre class="code">{audit_preview}</pre>

  <h2>Images</h2>
  <div class="grid-img">
    {image_cards}
  </div>

  <hr>
  <div class="row small">
    <div>Artifacts:</div>
    <div><a href="{zip_href}">Download ZIP</a></div>
    <div>•</div>
    <div><a href="{manifest_href}">manifest.json</a></div>
    <div>•</div>
    <div><a href="{sha_href}">sha256sums.txt</a></div>
  </div>
</body>
</html>
"""

def _img_card(title, url):
    if not url: return ""
    return f'''
    <div class="card">
      <h3>{title}</h3>
      <img src="{url}" alt="{title}" style="max-width:100%;height:auto;border-radius:4px;border:1px solid #eee">
    </div>'''

def _read_jsonl_preview(p: Path, max_lines=120) -> str:
    if not p.exists(): return ""
    out = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                out.append(line.rstrip("\n"))
    except Exception as e:
        out.append(f'{{"error":"{str(e)}"}}')
    return "\n".join(out)

# ---- 嵌入核心 ----
def run_embed_core(cover_m11: np.ndarray, secret_m11: np.ndarray, out_dir: Path) -> Dict[str, str]:
    cm, rv = get_models()

    ensure_dir(out_dir)
    (out_dir / "environment.json").write_text(json.dumps(_env_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    _audit_append(out_dir, {"stage":"embed_start","ts":int(time.time())})

    t0 = now_ms()
    c_out, _ = cm.predict([cover_m11[None, ...], secret_m11[None, ...]], verbose=0)
    t1 = now_ms()
    c_out = c_out[0]

    mode      = os.getenv("WM_CONTAINER_MODE", "auto").lower()
    res_init  = env_float("WM_RES_SCALE", 1.0)
    psnr_tgt  = env_float("WM_PSNR_TARGET", 40.0)
    s_min     = env_float("WM_RES_SCALE_MIN", 0.0)
    s_max     = env_float("WM_RES_SCALE_MAX", 1.0)
    opt_rev   = env_bool("WM_OPT_REVEAL", True)

    cand_direct = _clip_m11(c_out)
    used = ""
    if mode == "direct":
        container_m11 = cand_direct
        used = "direct"
    else:
        base_s = _find_scale_for_psnr(cover_m11, c_out, psnr_tgt, s_min, s_max)
        cont_res = _clip_m11(cover_m11 + base_s * c_out)
        psnr_res = psnr_m11(cont_res, cover_m11)
        psnr_dir = psnr_m11(cand_direct, cover_m11)
        if mode == "auto":
            if psnr_dir >= psnr_res:
                container_m11 = cand_direct
                used = f"auto->direct (PSNR_dir={psnr_dir:.2f}dB >= PSNR_res={psnr_res:.2f}dB)"
            else:
                container_m11 = cont_res
                used = f"auto->residual s={base_s:.3f} (PSNR_res={psnr_res:.2f}dB > PSNR_dir={psnr_dir:.2f}dB)"
        else:
            container_m11 = cont_res
            used = f"residual s={base_s:.3f}"

        if opt_rev:
            s_best, _ = _refine_by_reveal(rv, cover_m11, c_out, secret_m11, base_s, psnr_tgt)
            container_m11 = _clip_m11(cover_m11 + s_best * c_out)
            used += f"; refine_reveal s*={s_best:.3f}"

    secret_rec = rv.predict(container_m11[None, ...], verbose=0)[0]

    cover_u8      = to_uint8_image(cover_m11)
    secret_in_u8  = to_uint8_image(secret_m11)
    container_u8  = to_uint8_image(container_m11)
    secret_rec_u8 = to_uint8_image(secret_rec)
    residual_u8   = compute_residual_img(container_u8, cover_u8)

    p_cover     = out_dir / "cover.png"
    p_container = out_dir / "container.png"
    p_secret_in = out_dir / "secret_in.png"
    p_secret    = out_dir / "secret.png"
    p_residual  = out_dir / "residual.png"
    p_reportimg = out_dir / "report.png"

    cover_u8.save(p_cover)
    container_u8.save(p_container)
    secret_rec_u8.save(p_secret)
    secret_in_u8.save(p_secret_in)
    residual_u8.save(p_residual)

    # 五聯圖（保留給舊前端）
    w, h = cover_u8.width, cover_u8.height
    report = Image.new("RGB", (w * 5, h))
    report.paste(cover_u8,      (0 * w, 0))
    report.paste(secret_in_u8,  (1 * w, 0))
    report.paste(container_u8,  (2 * w, 0))
    report.paste(residual_u8,   (3 * w, 0))
    report.paste(secret_rec_u8, (4 * w, 0))
    report.save(p_reportimg)

    psnr_final = psnr_m11(container_m11, cover_m11)
    _audit_append(out_dir, {"stage":"embed_done","ts":int(time.time()),"psnr_final_db":psnr_final,"mode_used":used})

    return {
        "cover":     str(p_cover),
        "container": str(p_container),
        "secret_in": str(p_secret_in),
        "secret":    str(p_secret),
        "residual":  str(p_residual),
        "report_img":str(p_reportimg),
        "latency_ms": str(t1 - t0),
        "mode_used": used,
        "psnr_final_db": f"{psnr_final:.2f}"
    }

# ---- 評分與色彩對齊 ----
def _to_square_center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side)//2
    top  = (h - side)//2
    return img.crop((left, top, left+side, top+side))

def _laplacian_variance(img_rgb: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _psnr_rgb(imgA: Image.Image, imgB: Image.Image) -> float:
    a = np.asarray(imgA, dtype=np.float32); b = np.asarray(imgB, dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12: return 100.0
    PIXEL_MAX = 255.0
    return 20.0 * math.log10(PIXEL_MAX / np.sqrt(mse))

def _ssim_gray(imgA: Image.Image, imgB: Image.Image) -> float:
    a = cv2.cvtColor(np.asarray(imgA), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.asarray(imgB), cv2.COLOR_RGB2GRAY)
    if _HAVE_SKIMAGE:
        return float(skimage_ssim(a, b, data_range=255))
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    ncc = float(np.mean(a * b))
    return max(0.0, min(1.0, 0.5*(ncc + 1.0)))

# ===== Bottom-Row Proto-Consensus：只在最下排擇優，避開上半部竄改 =====
def _score_tiles_vs_attackref(reveal_u8: Image.Image, attack_ref128: Image.Image, grid: int) -> Tuple[Image.Image, Tuple[int,int], float]:
    """
    Bottom-Row Proto-Consensus：
      - 僅在最下排 tiles 中挑選最佳（grid=2 時即「下面兩張」）
      - 以最下排 tiles 的「逐像素中位數影像」作為 proto（標準）
      - 主分：SSIM(tile, proto)；次分：清晰度；參考圖極低權重避免帶偏
    score = 0.78*ssim_proto + 0.20*sharp + 0.02*ssim_ref
    回傳：最佳 tile、其 (x,y)、分數
    """
    grid = int(grid)
    if grid not in (2,3,4): grid = 2
    if reveal_u8.size != (256,256):
        reveal_u8 = reveal_u8.resize((256,256), LANCZOS)

    tile_w = 256 // grid
    tile_h = 256 // grid
    ref = attack_ref128.resize((tile_w, tile_h), LANCZOS)

    # 1) 切格
    tiles: List[Tuple[Tuple[int,int], Image.Image]] = []
    for yy in range(grid):
        for xx in range(grid):
            x0, y0 = xx*tile_w, yy*tile_h
            x1, y1 = (xx+1)*tile_w, (yy+1)*tile_h
            t = reveal_u8.crop((x0,y0,x1,y1))
            tiles.append(((xx,yy), t))

    # 2) 只取最下排作為候選 + 建立原型
    bottom_indices = [i for i, ((xx,yy), _) in enumerate(tiles) if yy == grid-1]
    if not bottom_indices:  # 理論上不會發生，保底處理
        bottom_indices = list(range(len(tiles)))

    tiles_arr_bottom = [np.asarray(tiles[i][1], dtype=np.uint8) for i in bottom_indices]
    stack = np.stack(tiles_arr_bottom, axis=0)                 # (Nb, H, W, 3)
    proto_arr = np.median(stack, axis=0).astype(np.uint8)
    proto = Image.fromarray(proto_arr, mode="RGB")

    # 3) 指標（全域清晰度正規化；相似度僅算候選）
    sharp_vals_all = [_laplacian_variance(t) for _, t in tiles]
    smin, smax = float(min(sharp_vals_all)), float(max(sharp_vals_all))
    def norm_sharp(x: float) -> float:
        return 0.5 if smax == smin else (x - smin) / (smax - smin)

    ssim_proto = {}
    ssim_ref   = {}
    for i in bottom_indices:
        t = tiles[i][1]
        ssim_proto[i] = _ssim_gray(t, proto)
        ssim_ref[i]   = _ssim_gray(t, ref)

    # 4) 僅在最下排候選中打分挑選
    best_img, best_xy, best_score = None, (0,0), -1e9
    for i in bottom_indices:
        (xy, t) = tiles[i]
        sharp = norm_sharp(sharp_vals_all[i])
        score = 0.78*ssim_proto[i] + 0.20*sharp + 0.02*ssim_ref[i]
        if score > best_score:
            best_score, best_img, best_xy = score, t, xy

    return best_img, best_xy, float(best_score)

def _color_transfer_reinhard(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    s_mean, s_std = src_lab.mean(axis=(0,1), keepdims=True), src_lab.std(axis=(0,1), keepdims=True) + 1e-6
    r_mean, r_std = ref_lab.mean(axis=(0,1), keepdims=True), ref_lab.std(axis=(0,1), keepdims=True) + 1e-6
    out = (src_lab - s_mean) * (r_std / s_std) + r_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

def _color_align_to_attack(tile: Image.Image, ref128: Image.Image) -> Image.Image:
    t = np.asarray(tile.convert("RGB"))
    r = np.asarray(ref128.convert("RGB").resize(tile.size, LANCZOS))
    aligned = _color_transfer_reinhard(t, r)
    return Image.fromarray(aligned)

# ---- 還原核心 ----
def run_reveal_core(image_m11: np.ndarray, out_dir: Path, tiles_hint: int, attack_ref128: Optional[Image.Image]) -> Dict[str, str]:
    _, rv = get_models()

    ensure_dir(out_dir)
    (out_dir / "environment.json").write_text(json.dumps(_env_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    _audit_append(out_dir, {"stage":"reveal_start","ts":int(time.time()),"grid":tiles_hint})

    t0 = now_ms()
    y = rv.predict(image_m11[None, ...], verbose=0)[0]
    t1 = now_ms()

    y_u8 = to_uint8_image(y)
    y_u8.save(out_dir / "reveal.png")

    if attack_ref128 is None:
        attack_ref128 = y_u8.crop((64,64,192,192)).resize((128,128), LANCZOS)

    best_tile_u8, (bx, by), score = _score_tiles_vs_attackref(y_u8, attack_ref128, int(tiles_hint))
    best_tile_aligned = _color_align_to_attack(best_tile_u8, attack_ref128)

    best_tile_u8.save(out_dir / "tile_best.png")
    best_tile_aligned.save(out_dir / "tile_best_src.png")

    best_256 = _upscale_best_tile_with_realesrgan(best_tile_aligned, out_dir, target_edge=256)
    best_256.save(out_dir / "tile_best_256.png")

    sheet = Image.new("RGB", (128*5, 128))
    sheet.paste(attack_ref128,      (0, 0))
    sheet.paste(best_tile_u8,       (128, 0))
    sheet.paste(best_tile_aligned,  (256, 0))
    sheet.paste(best_256.resize((128,128), LANCZOS), (384, 0))
    sheet.paste(y_u8.resize((128,128), LANCZOS),     (512, 0))
    sheet.save(out_dir/"reveal_preview.png")

    _audit_append(out_dir, {"stage":"reveal_done","ts":int(time.time()),"best_xy":[bx,by],"score":score})

    return {
        "tile_best":      str(out_dir / "tile_best.png"),
        "tile_best_src":  str(out_dir / "tile_best_src.png"),
        "tile_best_256":  str(out_dir / "tile_best_256.png"),
        "reveal_full":    str(out_dir / "reveal.png"),
        "reveal_preview": str(out_dir / "reveal_preview.png"),
        "tile_best_xy": f"{bx},{by}",
        "tile_best_score": f"{score:.4f}",
        "grid": str(tiles_hint),
        "latency_ms": str(t1 - t0)
    }

# ---- 檔案工具 ----
def _write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def _zip_job(job_dir: Path, out_zip: Path):
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in job_dir.glob("*"):
            if p.is_file():
                z.write(p, arcname=p.name)
        for p in (job_dir/"logs").glob("*"):
            if p.is_file():
                z.write(p, arcname=f"logs/{p.name}")
        for name in ("manifest.json","sha256sums.txt"):
            p = job_dir / name
            if p.is_file():
                z.write(p, arcname=p.name)

# ---- latest 狀態 ----
class LatestStore:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "_latest.json"
        self.lock = threading.Lock()
        self.data = {"embed": None, "reveal": None}
        self._load()
    def _load(self):
        try:
            if self.path.is_file():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {"embed": None, "reveal": None}
    def _save(self):
        try:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print("[latest] save error:", e)
    def set_building(self, kind: str, job_id: str):
        with self.lock:
            self.data[kind] = {"ready": False, "job_id": job_id, "report_url": None, "zip_url": None, "created_ts": int(time.time()), "kind": kind}
            self._save()
    def set_ready(self, kind: str, job_id: str, report_url: str, zip_url: str):
        with self.lock:
            self.data[kind] = {"ready": True, "job_id": job_id, "report_url": report_url, "zip_url": zip_url, "created_ts": int(time.time()), "kind": kind}
            self._save()
    def get(self, kind: Optional[str] = None):
        with self.lock:
            if kind in ("embed","reveal"): return self.data.get(kind)
            return self.data.get("embed")

# ---- Flask ----
def build_app(static_root: Path) -> Flask:
    app = Flask(__name__)
    ensure_dir(static_root)
    latest = LatestStore(static_root)
    init_chain(static_root / "blockchain.json")

    @app.after_request
    def cors(r):
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Expose-Headers"]= "Content-Disposition"
        return r

    def _new_job_dir() -> Tuple[Path, str]:
        jid = gen_job_id()
        return ensure_dir(static_root / jid), jid

    # 共用報告生成
    def _generate_report_html(job_dir: Path, job_id: str, image_pairs: list):
        try: env = json.loads((job_dir/"environment.json").read_text(encoding="utf-8"))
        except Exception: env = {}
        env_str = json.dumps(env, ensure_ascii=False, indent=2)
        audit_path = str((job_dir / "logs" / "audit.jsonl").resolve())
        audit_preview = _read_jsonl_preview(job_dir / "logs" / "audit.jsonl", max_lines=120)

        def as_url(p: Path): return f"/files/{job_id}/{p.name}"
        cards = []
        for title, filename in image_pairs:
            p = job_dir / filename
            if p.is_file(): cards.append(_img_card(title, as_url(p)))
        image_cards = "\n".join(cards)

        zip_href      = f"/dl/zip/{job_id}"
        manifest_href = f"/files/{job_id}/manifest.json"
        sha_href      = f"/files/{job_id}/sha256sums.txt"

        html = REPORT_HTML.format(
            css=MONO_CSS,
            case_title=job_id,
            session_id=job_id,
            started_utc=datetime.datetime.utcnow().isoformat()+"Z",
            operator="operator",
            base=str(job_dir.resolve()),
            environment_json=env_str,
            audit_path=audit_path,
            audit_preview=audit_preview,
            image_cards=image_cards,
            zip_href=zip_href,
            manifest_href=manifest_href,
            sha_href=sha_href
        )
        return html

    # 健康檢查
    @app.get("/")
    def root():
        return jsonify({"ok": True, "service": "watermark", "health": "/health"})

    @app.get("/health")
    def health():
        return jsonify({
            "ok": True,
            "combined": Path(COMBINED_H5).exists(),
            "reveal":   Path(REVEAL_H5).exists(),
            "blockchain": get_chain().chain_info(),
            "env": {
                "WM_CONTAINER_MODE": os.getenv("WM_CONTAINER_MODE", "auto"),
                "WM_PSNR_TARGET": os.getenv("WM_PSNR_TARGET", "40.0"),
                "WM_RES_SCALE": os.getenv("WM_RES_SCALE", "1.0"),
                "WM_OPT_REVEAL": os.getenv("WM_OPT_REVEAL", "1"),
                "REALESRGAN_REPO": str(REALESRGAN_REPO),
                "REALESRGAN_MODEL": REALESRGAN_MODEL,
                "REALESRGAN_READY": _have_realesrgan(),
                "HAVE_SKIMAGE": _HAVE_SKIMAGE
            }
        })

    # 靜態檔案
    @app.get("/files/<path:sub>")
    def files(sub):
        return send_from_directory(str(static_root), sub, as_attachment=False)

    @app.get("/open/image/<job>/<name>")
    def open_image(job, name):
        p = static_root / job / name
        if not p.is_file(): abort(404)
        return send_file(str(p), mimetype="image/png")

    @app.get("/dl/image/<job>/<name>")
    def dl_image(job, name):
        p = static_root / job / name
        if not p.is_file(): abort(404)
        return send_file(str(p), as_attachment=True, download_name=name)

    @app.get("/dl/zip/<job>")
    def dl_zip(job):
        job_dir = static_root / job
        if not job_dir.is_dir(): abort(404)
        out_zip = job_dir / f"{job}.zip"
        _zip_job(job_dir, out_zip)
        return send_file(str(out_zip), as_attachment=True, download_name=out_zip.name)

    # 新增：下載報告（HTML）
    @app.get("/dl/report/<job_id>")
    def dl_report(job_id: str):
        job_dir = static_root / job_id
        if not job_dir.is_dir(): abort(404)
        html_path = job_dir / "report.html"
        if not html_path.is_file(): abort(404)
        return send_file(str(html_path), as_attachment=True,
                         download_name=f"Forensics_Report_{job_id}.html",
                         mimetype="text/html; charset=utf-8")

    # 報告頁
    @app.get("/report/<job_id>")
    def open_report(job_id: str):
        job_dir = static_root / job_id
        if not job_dir.is_dir(): abort(404)
        html_path = job_dir / "report.html"
        if not html_path.exists():
            pairs = [("Cover","cover.png"),("Secret (in)","secret_in.png"),
                     ("Container","container.png"),("Residual","residual.png"),
                     ("Secret (out)","secret.png")]
            html = _generate_report_html(job_dir, job_id, pairs)
            _write_text(html_path, html)
        return send_file(str(html_path), mimetype="text/html; charset=utf-8")

    # ---- 27037 最新 ----
    @app.get("/27037/latest")
    def latest_27037():
        kind = (request.args.get("kind") or "embed").lower()
        if kind not in ("embed","reveal"): kind = "embed"
        info = latest.get(kind)
        return jsonify({"ok": True, "kind": kind, "latest": info or {"ready": False}})

    # ---- 嵌入 ----
    @app.post("/wm/embed")
    def wm_embed():
        try:
            job_dir, jid = _new_job_dir()
            latest.set_building("embed", jid)

            f_cover  = request.files.get("cover")
            f_secret = request.files.get("secret")
            grid     = request.form.get("secret_grid") or request.form.get("hint_grid") \
                       or request.form.get("tiles_hint") or "2"
            identity_name = (request.form.get("identity_name") or "").strip()
            identity_unit = (request.form.get("identity_unit") or "").strip()
            identity_note = (request.form.get("identity_note") or "").strip()
            if not f_cover:
                return jsonify({"ok": False, "error": "need cover file"}), 400

            cover_m11 = load_from_bytes(f_cover.read())
            if f_secret:
                secret_m11 = load_from_bytes(f_secret.read())
            else:
                cover_u8 = to_uint8_image(cover_m11)
                sec_u8   = make_tiled_secret_from_cover(cover_u8, int(grid))
                secret_m11 = (np.asarray(sec_u8, np.float32) / 127.5) - 1.0

            res = run_embed_core(cover_m11, secret_m11, job_dir)

            # --- 區塊鏈：對 container.png 算 SHA256，寫入新區塊，並把區塊資訊嵌入 PNG metadata ---
            container_path = Path(res["container"])
            container_sha256 = sha256_bytes(container_path.read_bytes())
            bc_block = get_chain().record_embed(
                job_id=jid,
                image_sha256=container_sha256,
                metadata={
                    "psnr_final_db":   res.get("psnr_final_db", ""),
                    "mode_used":       res.get("mode_used", ""),
                    "identity_name":   identity_name,
                    "identity_unit":   identity_unit,
                    "identity_note":   identity_note,
                }
            )
            # 把區塊鏈資訊寫進 PNG text chunks（不影響像素，隨圖片一起傳遞）
            _png_meta = PngImagePlugin.PngInfo()
            _png_meta.add_text("wm_job_id",     jid)
            _png_meta.add_text("wm_block_index", str(bc_block["index"]))
            _png_meta.add_text("wm_block_hash",  bc_block["block_hash"])
            _png_meta.add_text("wm_timestamp",   str(bc_block["timestamp"]))
            Image.open(container_path).convert("RGB").save(container_path, pnginfo=_png_meta)

            (job_dir / "blockchain_record.json").write_text(
                json.dumps(bc_block, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # -----------------------------------------------------------------------

            pairs = [("Cover","cover.png"),("Secret (in)","secret_in.png"),
                     ("Container","container.png"),("Residual","residual.png"),
                     ("Secret (out)","secret.png")]
            html = _generate_report_html(job_dir, jid, pairs)
            _write_text(job_dir/"report.html", html)

            report_url = f"/report/{jid}"
            zip_url    = f"/dl/zip/{jid}"
            latest.set_ready("embed", jid, report_url=report_url, zip_url=zip_url)

            base = f"/files/{jid}"
            return jsonify({
                "ok": True, "job_id": jid, "ready": True,
                "latency_ms": res["latency_ms"],
                "mode_used": res.get("mode_used", ""),
                "psnr_final_db": res.get("psnr_final_db", ""),
                "blockchain": {
                    "block_index": bc_block["index"],
                    "block_hash":  bc_block["block_hash"],
                    "image_sha256": container_sha256,
                },
                "images": {
                    "cover":     f"{base}/cover.png",
                    "container": f"{base}/container.png",
                    "secret_in": f"{base}/secret_in.png",
                    "secret_out":f"{base}/secret.png",
                    "residual":  f"{base}/residual.png",
                    "report":    f"{base}/report.png"
                },
                "download_url": f"/dl/image/{jid}/container.png",
                "download_name": "container.png",
                "report_url":   report_url,
                "zip_url":      zip_url
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    # ---- 還原 ----
    @app.post("/wm/verify")
    def wm_verify():
        """只做區塊鏈驗證，不執行 reveal。回傳鏈上身分訊息供前端顯示。"""
        try:
            f = request.files.get("image")
            if not f:
                return jsonify({"ok": False, "error": "need image file"}), 400

            raw = f.read()

            try:
                tmp_img = Image.open(io.BytesIO(raw))
                wm_job_id    = tmp_img.info.get("wm_job_id", "").strip()
                wm_block_hash = tmp_img.info.get("wm_block_hash", "").strip()
            except Exception:
                wm_job_id = ""
                wm_block_hash = ""

            if not wm_job_id or not wm_block_hash:
                return jsonify({
                    "ok": False,
                    "blockchain_verified": False,
                    "reason": "此圖片不含浮水印區塊鏈資訊，請確認為系統產出的 container 圖片。",
                    "detail": "no_metadata",
                }), 403

            bc_ok, bc_block, bc_reason = get_chain().verify_by_job_id(wm_job_id, wm_block_hash)
            if not bc_ok:
                reason_map = {
                    "not_registered": "此圖片的區塊鏈記錄不存在，無法驗證。",
                    "hash_mismatch":  "圖片 metadata 與區塊鏈記錄不符，可能已被竄改。",
                }
                return jsonify({
                    "ok": False,
                    "blockchain_verified": False,
                    "reason": reason_map.get(bc_reason, "區塊鏈驗證失敗。"),
                    "detail": bc_reason,
                    "wm_job_id": wm_job_id,
                }), 403

            meta = bc_block.get("metadata", {})
            return jsonify({
                "ok": True,
                "blockchain_verified": True,
                "identity": {
                    "name": meta.get("identity_name", ""),
                    "unit": meta.get("identity_unit", ""),
                    "note": meta.get("identity_note", ""),
                },
                "blockchain": {
                    "block_index":  bc_block["index"],
                    "block_hash":   bc_block["block_hash"],
                    "embed_job_id": bc_block["job_id"],
                    "timestamp":    bc_block["timestamp"],
                    "image_sha256": bc_block["image_sha256"],
                    "psnr_final_db": meta.get("psnr_final_db", ""),
                    "mode_used":     meta.get("mode_used", ""),
                },
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/external_reveal")
    def external_reveal():
        try:
            job_dir, jid = _new_job_dir()
            latest.set_building("reveal", jid)

            f = request.files.get("image") or request.files.get("attack")
            tiles_hint = request.form.get("tiles_hint") or request.form.get("hint_grid") or "2"
            if not f:
                return jsonify({"ok": False, "error": "need image(file)"}), 400

            raw = f.read()

            # --- 區塊鏈驗證：從 PNG metadata 讀取嵌入的區塊資訊，查鏈，不通過就拒絕 ---
            try:
                _tmp_img = Image.open(io.BytesIO(raw))
                wm_job_id    = _tmp_img.info.get("wm_job_id", "").strip()
                wm_block_hash = _tmp_img.info.get("wm_block_hash", "").strip()
            except Exception:
                wm_job_id = ""
                wm_block_hash = ""

            if not wm_job_id or not wm_block_hash:
                return jsonify({
                    "ok": False,
                    "blockchain_verified": False,
                    "reason": "此圖片不含浮水印區塊鏈資訊，請確認為系統產出的 container 圖片。",
                    "detail": "no_metadata",
                }), 403

            bc_ok, bc_block, bc_reason = get_chain().verify_by_job_id(wm_job_id, wm_block_hash)
            if not bc_ok:
                _reason_map = {
                    "not_registered": "此圖片的區塊鏈記錄不存在，無法還原浮水印。",
                    "hash_mismatch":  "圖片 metadata 與區塊鏈記錄不符，可能已被竄改。",
                }
                return jsonify({
                    "ok": False,
                    "blockchain_verified": False,
                    "reason": _reason_map.get(bc_reason, "區塊鏈驗證失敗，鏈可能已被竄改。"),
                    "detail": bc_reason,
                    "wm_job_id": wm_job_id,
                }), 403
            # -----------------------------------------------------------------------

            try:
                atk_img = Image.open(io.BytesIO(raw)).convert("RGB")
                atk_ref128 = _to_square_center_crop(atk_img).resize((128,128), LANCZOS)
            except Exception:
                atk_ref128 = None

            x_m11 = load_from_bytes(raw)
            res = run_reveal_core(x_m11, job_dir, tiles_hint=int(tiles_hint), attack_ref128=atk_ref128)

            pairs = [("Best tile","tile_best.png"),
                     ("Best tile (aligned)","tile_best_src.png"),
                     ("Best 256 (ESRGAN)","tile_best_256.png"),
                     ("Reveal (full)","reveal.png"),
                     ("Preview x5 (sheet)","reveal_preview.png")]
            html = _generate_report_html(job_dir, jid, pairs)
            _write_text(job_dir/"report.html", html)

            report_url = f"/report/{jid}"
            zip_url    = f"/dl/zip/{jid}"
            latest.set_ready("reveal", jid, report_url=report_url, zip_url=zip_url)

            base = f"/files/{jid}"
            return jsonify({
                "ok": True, "job_id": jid, "ready": True,
                "latency_ms": res["latency_ms"],
                "grid": res.get("grid", str(tiles_hint)),
                "blockchain_verified": True,
                "blockchain": {
                    "block_index":  bc_block["index"],
                    "block_hash":   bc_block["block_hash"],
                    "embed_job_id": bc_block["job_id"],
                    "image_sha256": img_sha256,
                },
                "images": {
                    "tile_best":     f"{base}/tile_best.png",
                    "tile_best_src": f"{base}/tile_best_src.png",
                    "tile_best_256": f"{base}/tile_best_256.png",
                    "reveal_full":   f"{base}/reveal.png",
                    "preview":       f"{base}/reveal_preview.png"},
                "tile_best_xy": res.get("tile_best_xy", ""),
                "tile_best_score": res.get("tile_best_score", ""),
                "reveal_report_url":  report_url,
                "zip_url":            zip_url
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    # 別名
    app.add_url_rule("/infer27037", view_func=wm_embed, methods=["POST"])
    app.add_url_rule("/wm/external_reveal", view_func=external_reveal, methods=["POST"])
    return app

# ---- main ----
def run_server(host: str, port: int, root: str):
    app = build_app(ensure_dir(Path(root)))
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=5001)
    s.add_argument("--root", default="./outputs")

    a = p.parse_args()
    if a.cmd == "serve":
        run_server(a.host, a.port, a.root)
