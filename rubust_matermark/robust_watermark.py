# -*- coding: utf-8 -*-
"""
robustwm_vscode.py
Portable (VSCode/local) version of the uploaded Colab script.

- Keeps original model, losses, callbacks, and inference logic intact.
- Removes Colab-only bits (e.g., google drive mount, /content paths).
- Adds CLI to run: train / infer / external_reveal
    python robustwm_vscode.py train --cover_dir ... --secret_dir ... --save_root ...
    python robustwm_vscode.py infer --model_dir ... --cover_img ... --secret_img ...
    python robustwm_vscode.py external_reveal --reveal_h5 ... --attack_img ... [--container_img ...]

Tested with Python 3.10/3.11 + TensorFlow 2.15 on Windows.
"""

import os, math, time, glob, argparse, datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# TF / Keras imports (same as original)
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, LeakyReLU,
                                     Add, UpSampling2D, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        CSVLogger, Callback, TerminateOnNaN)
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from sklearn.model_selection import train_test_split

# =========================
# 0. 環境設置（GPU 記憶體成長、混合精度）
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow 已啟用 GPU:", gpus)
    except RuntimeError as e:
        print("❌ GPU 初始化失敗:", e)
else:
    print("⚠️ 未偵測到 GPU（仍可用 CPU 跑）")

print("TensorFlow:", tf.__version__)
mixed_precision.set_global_policy('mixed_float16')

# =========================
# 實用工具
# =========================
def make_unique_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_dir, f"final_exports_{ts}")
    os.makedirs(out, exist_ok=True)
    return out

def _atomic_save(model: tf.keras.Model, path: str):
    root, ext = os.path.splitext(path)        # .h5 / .keras
    if not ext:
        ext = ".h5"
        path = root + ext
    tmp = f"{root}.tmp{ext}"
    model.save(tmp, include_optimizer=False)
    os.replace(tmp, path)

# =========================
# 1. 資料載入 & Dataset（維持原邏輯）
# =========================
def load_image(filepath, image_size):
    image = tf.io.read_file(filepath)
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except tf.errors.InvalidArgumentError:
        image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5  # [-1,1]
    return image

def create_dataset(s1_filepaths, s2_filepaths, batch_size, image_size, shuffle=True, augment=True):
    dataset = tf.data.Dataset.from_tensor_slices((s1_filepaths, s2_filepaths))
    def _load_pair(s1_fp, s2_fp):
        s1 = load_image(s1_fp, image_size)  # 不再 add_attack（與上傳檔一致）
        s2 = load_image(s2_fp, image_size)
        return (s1, s2), (s1, s2)
    dataset = dataset.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# =========================
# 2. 模型：Block 與網路
# =========================
def robust_residual_block(x, filters, name=None):
    sc = x
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
               name=None if name is None else f'{name}_conv1')(x)
    x = BatchNormalization(name=None if name is None else f'{name}_bn1')(x)
    x = LeakyReLU(0.2, name=None if name is None else f'{name}_lrelu1')(x)
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
               name=None if name is None else f'{name}_conv2')(x)
    x = BatchNormalization(name=None if name is None else f'{name}_bn2')(x)
    return tf.keras.layers.Add(name=None if name is None else f'{name}_add')([sc, x])

def build_hiding_network(input_shape):
    S1_input = Input(shape=input_shape, name='S1_Input')
    S2_input = Input(shape=input_shape, name='S2_Input')
    x = tf.keras.layers.Concatenate(-1)([S1_input, S2_input])
    x = Conv2D(64, 3, padding='same')(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    for _ in range(6): x = robust_residual_block(x, 64)
    x = Conv2D(128, 3, strides=2, padding='same')(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    for _ in range(6): x = robust_residual_block(x, 128)
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same')(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    S1_prime = Conv2D(3, 3, padding='same', activation='tanh', name='S1_Prime')(x)
    return Model([S1_input, S2_input], S1_prime, name='Robust_Hiding_Network')

def build_reveal_network(input_shape):
    S1_prime_in = Input(shape=input_shape, name='S1_Prime_Input')
    x = Conv2D(64, 3, padding='same')(S1_prime_in); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    for _ in range(6): x = robust_residual_block(x, 64)
    x = Conv2D(128, 3, strides=2, padding='same')(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    for _ in range(6): x = robust_residual_block(x, 128)
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same')(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)
    S2_prime = Conv2D(3, 3, padding='same', activation='tanh', name='S2_Prime')(x)
    return Model(S1_prime_in, S2_prime, name='Robust_Reveal_Network')

# ========== Robust 攻擊層（與原始一致）==========
class RobustAttackLayer(tf.keras.layers.Layer):
    def __init__(self, noise_std_max=0.01, blur_prob=0.3, pool_scales=(1, 2, 3, 4), **kwargs):
        super().__init__(**kwargs)
        self.noise_std_max = self.add_weight(
            name='noise_std_max', shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(float(noise_std_max)),
            trainable=False
        )
        self.blur_prob = self.add_weight(
            name='blur_prob', shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(float(blur_prob)),
            trainable=False
        )
        self.pool_scales_py = list(pool_scales)
        self.num_scales = len(self.pool_scales_py)

    def build(self, input_shape):
        self.H = int(input_shape[1]); self.W = int(input_shape[2])
        self.target_size = [self.H, self.W]
        super().build(input_shape)

    def _gaussian_blur(self, x, sigma):
        d = x.dtype
        ks = 5
        ax = tf.range(-ks // 2 + 1, ks // 2 + 1, dtype=tf.float32)
        sigma32 = tf.cast(sigma, tf.float32)
        g = tf.exp(-(ax**2) / (tf.cast(2.0, tf.float32) * (sigma32**2)))
        k = g[:, None] * g[None, :]
        k = k / tf.reduce_sum(k)
        k = tf.reshape(k, (ks, ks, 1, 1))
        k = tf.tile(k, [1, 1, tf.shape(x)[-1], 1])
        k = tf.cast(k, d)
        x = tf.cast(x, d)
        x_blur = tf.nn.depthwise_conv2d(x, k, strides=[1,1,1,1], padding='SAME')
        return tf.cast(x_blur, d)

    def call(self, x, training=None):
        d = x.dtype
        x = tf.cast(x, d)
        noise_std_max = tf.cast(self.noise_std_max, d)
        blur_prob     = tf.cast(self.blur_prob, d)

        if training is False:
            return tf.cast(tf.clip_by_value(x, -1.0, 1.0), d)

        std = tf.random.uniform([], tf.cast(0.0, d), noise_std_max, dtype=d)
        noise = tf.random.normal(tf.shape(x), mean=tf.cast(0.0, d), stddev=std, dtype=d)
        x_noisy = tf.clip_by_value(x + noise, -1.0, 1.0)

        cands = []
        for k in self.pool_scales_py:
            if k == 1:
                xk = x_noisy
            else:
                x_small = tf.nn.avg_pool2d(x_noisy, ksize=k, strides=k, padding='SAME')
                x_small = tf.cast(x_small, d)
                xk = tf.image.resize(x_small, size=self.target_size, method='bilinear')
            xk = tf.cast(xk, d)
            cands.append(xk)

        cands = tf.stack(cands, axis=0)
        idx   = tf.random.uniform([], 0, self.num_scales, dtype=tf.int32)
        x_scaled = tf.gather(cands, idx, axis=0)

        B2 = tf.shape(x_scaled)[0]
        rnd  = tf.random.uniform([B2, 1, 1, 1], dtype=d)
        gate = tf.cast(rnd < blur_prob, d)

        sigma = tf.random.uniform([], tf.cast(0.5, d), tf.cast(1.2, d), dtype=d)
        x_blur = self._gaussian_blur(tf.cast(x_scaled, d), sigma)

        one = tf.cast(1.0, d)
        x_out = gate * x_blur + (one - gate) * x_scaled
        return tf.cast(tf.clip_by_value(x_out, -1.0, 1.0), d)

    def compute_output_shape(self, input_shape):
        return input_shape

# ---- 組裝（兩個命名輸出）----
def build_combined_model(input_shape=(256,256,3)):
    hiding_network = build_hiding_network(input_shape)
    reveal_network = build_reveal_network(input_shape)

    S1 = tf.keras.Input(shape=input_shape, name='Cover_Image')
    S2 = tf.keras.Input(shape=input_shape, name='Secret_Image')

    S1_prime_clean = hiding_network([S1, S2])
    attack_layer = RobustAttackLayer(
        noise_std_max=0.01, blur_prob=0.3, pool_scales=(1, 2, 3, 4), name='S1_Prime_Attack'
    )
    S1_prime_att  = attack_layer(S1_prime_clean)
    S2_prime      = reveal_network(S1_prime_att)

    S1_Prime_out = Lambda(lambda x: x, name='S1_Prime_out')(S1_prime_clean)
    S2_Prime_out = Lambda(lambda x: x, name='S2_Prime_out')(S2_prime)

    combined_model = Model([S1, S2], [S1_Prime_out, S2_Prime_out], name='Stego_2out')
    return combined_model, hiding_network, reveal_network

# ========== 4. 損失函數（與原始一致） ==========
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256,256,3))
vgg.trainable = False
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)

def _scalar_f32(x):
    x = tf.cast(x, tf.float32)
    return tf.reshape(tf.reduce_mean(x), [])

def _to_vgg(x):
    x = (tf.cast(x, tf.float32) + 1.0) * 127.5  # [-1,1]→[0,255]
    return vgg19_preprocess(x)

def perceptual_loss(y_true, y_pred):
    x_true = _to_vgg(y_true)
    x_pred = _to_vgg(y_pred)
    f_true = vgg_model(x_true)
    f_pred = vgg_model(x_pred)
    diff = tf.cast(f_true, tf.float32) - tf.cast(f_pred, tf.float32)
    return _scalar_f32(tf.square(diff))

def cover_loss(y_true, y_pred):
    return _scalar_f32(tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))

def dct2_tf(x):
    x = tf.cast(x, tf.float32)
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.signal.dct(x, type=2, norm='ortho')
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.signal.dct(x, type=2, norm='ortho')
    x = tf.transpose(x, [0, 3, 1, 2])
    return x

def dct_loss(y_true, y_pred):
    y_true_d = dct2_tf(y_true)
    y_pred_d = dct2_tf(y_pred)
    return _scalar_f32(tf.square(y_true_d - y_pred_d))

def color_consistency_loss(y_true, y_pred):
    return _scalar_f32(tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))

def secret_reconstruction_loss(y_true, y_pred):
    mse   = cover_loss(y_true, y_pred)
    dct_e = dct_loss(y_true, y_pred)
    col   = color_consistency_loss(y_true, y_pred)
    return _scalar_f32(0.4*mse + 0.4*dct_e + 0.2*col)

# ========== 5. Callbacks（與原始一致/合併） ==========
class Heartbeat(Callback):
    def __init__(self, every=50, steps_per_epoch=None, batch_size=None):
        super().__init__()
        self.every = max(1, int(every))
        self.steps = steps_per_epoch
        self.bs = batch_size
        self.t0 = None
        self.epoch_t0 = None
        self.dt_hist = []
        self.sum_loss = 0.0
        self.sum_s1 = 0.0
        self.sum_s2 = 0.0
        self.count = 0

    def _get_lr(self):
        lr = getattr(self.model.optimizer, "learning_rate", None)
        try:
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            try:
                return float(tf.keras.backend.get_value(self.model.optimizer.lr))
            except Exception:
                return None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_t0 = time.time()
        self.dt_hist.clear()
        self.sum_loss = self.sum_s1 = self.sum_s2 = 0.0
        self.count = 0

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.every == 0:
            self.t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        total = logs.get('loss')
        s1    = logs.get('S1_Prime_out_loss')
        s2    = logs.get('S2_Prime_out_loss')
        for v, attr in [(total, 'sum_loss'), (s1, 'sum_s1'), (s2, 'sum_s2')]:
            if v is not None and np.isfinite(v):
                setattr(self, attr, getattr(self, attr) + float(v))
        self.count += 1
        if batch % self.every == 0 and self.t0 is not None:
            dt = time.time() - self.t0
            self.dt_hist.append(dt / max(1, self.every))
            avg_dt = np.mean(self.dt_hist[-10:]) if self.dt_hist else None
            if self.steps and avg_dt is not None:
                remaining = self.steps - (batch + 1)
                eta_s = max(0, remaining) * avg_dt
                eta_txt = time.strftime('%M:%S', time.gmtime(eta_s))
            else:
                eta_txt = 'NA'
            if self.bs is not None and avg_dt is not None:
                ips = self.bs / avg_dt
            else:
                ips = None
            lr = self._get_lr()
            lr_txt = f"{lr:.2e}" if lr is not None else "NA"
            msg = (f"⏱ step {batch+1}/{self.steps or '?'} "
                   f"— loss={total:.4f} | S1={s1:.4f} | S2={s2:.4f} "
                   f"| dt/step≈{avg_dt:.3f}s"
                   f"{(f' | ips≈{ips:.1f}' if ips is not None else '')}"
                   f" | ETA {eta_txt} | lr={lr_txt}")
            print(msg)

    def on_epoch_end(self, epoch, logs=None):
        elapse = time.time() - (self.epoch_t0 or time.time())
        avg_loss = self.sum_loss / max(1, self.count)
        avg_s1   = self.sum_s1  / max(1, self.count)
        avg_s2   = self.sum_s2  / max(1, self.count)
        print(f"📊 Epoch {epoch+1} done in {elapse:.1f}s — "
              f"avg loss={avg_loss:.4f} | avg S1={avg_s1:.4f} | avg S2={avg_s2:.4f}")

class NanGuard(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        total = logs.get('loss')
        if total is not None and not np.isfinite(total):
            print(f"❌ NaN/Inf loss at train batch {batch}")
            self.model.stop_training = True
        for w in self.model.trainable_weights[:10]:
            if tf.math.reduce_any(tf.math.is_nan(w)).numpy():
                print(f"❌ NaN in weights of {w.name} at batch {batch}")
                self.model.stop_training = True
                break

class PeriodicExportCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_root, hiding_model, reveal_model, period=50):
        super().__init__()
        self.save_root = save_root
        self.hiding_model = hiding_model
        self.reveal_model = reveal_model
        self.period = max(1, int(period))

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % self.period == 0:
            export_dir = make_unique_dir(os.path.join(self.save_root, f"exports_ep{ep:04d}"))
            print(f"\n💾 [Epoch {ep}] 週期性匯出到：{export_dir}")
            _atomic_save(self.model,        os.path.join(export_dir, "combined_model.h5"))
            _atomic_save(self.hiding_model, os.path.join(export_dir, "hiding_network.h5"))
            _atomic_save(self.reveal_model, os.path.join(export_dir, "reveal_network.h5"))
            print("✅ 已完成三個模型的週期性匯出")

    def on_train_end(self, logs=None):
        final_dir = make_unique_dir(os.path.join(self.save_root, "final_exports"))
        print(f"\n🏁 訓練結束，最終匯出到：{final_dir}")
        _atomic_save(self.model,        os.path.join(final_dir, "combined_model.h5"))
        _atomic_save(self.hiding_model, os.path.join(final_dir, "hiding_network.h5"))
        _atomic_save(self.reveal_model, os.path.join(final_dir, "reveal_network.h5"))
        print("✅ 三個模型已輸出到：", final_dir)

# =========================
# 6. 編譯（與原始一致）
# =========================
def compile_combined_model(combined_model):
    PERCEPTUAL_W = 0.1
    optimizer = Adam(learning_rate=1e-4, global_clipnorm=1.0)
    combined_model.compile(
        optimizer=optimizer,
        loss={
            'S1_Prime_out': lambda yt, yp: _scalar_f32(cover_loss(yt, yp) + PERCEPTUAL_W * perceptual_loss(yt, yp)),
            'S2_Prime_out': secret_reconstruction_loss
        },
        loss_weights={'S1_Prime_out': 1.0, 'S2_Prime_out': 1.0},
        run_eagerly=False,
        steps_per_execution=16
    )
    print("✅ Model compiled with LR=1e-4, global_clipnorm=1.0")

# =========================
# 7. 便利函式：載入/儲存影像（推論用）
# =========================
from PIL import Image, ImageOps

def load_rgb_m11(path, size=(256,256)):
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return (arr - 127.5) / 127.5  # [-1,1]

def to_uint8_from_m11(x):
    return Image.fromarray(np.clip((x+1.0)*127.5,0,255).astype(np.uint8))

def m11_to_01(x):
    return np.clip((x + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

# =========================
# 8. 訓練流程
# =========================
def run_train(cover_dir: str, secret_dir: str, save_root: str, batch_size=4, epochs=1000, image_size=(256,256)):
    cover_files = sorted([str(p) for p in Path(cover_dir).glob("*.*") if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tiff")])
    secret_files= sorted([str(p) for p in Path(secret_dir).glob("*.*") if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tiff")])
    print(f"✅ Cover 圖片數量: {len(cover_files)}")
    print(f"✅ Secret 圖片數量: {len(secret_files)}")

    if len(cover_files) == len(secret_files):
        print("🔗 使用排序後的一一配對")
        paired_s1, paired_s2 = cover_files, secret_files
    else:
        print("⚠️ 數量不一致，使用檔名交集")
        s1_dict = {Path(f).name: f for f in cover_files}
        s2_dict = {Path(f).name: f for f in secret_files}
        common = sorted(set(s1_dict) & set(s2_dict))
        if not common:
            raise ValueError("❌ 無法配對，請確認檔名或數量")
        paired_s1 = [s1_dict[n] for n in common]
        paired_s2 = [s2_dict[n] for n in common]

    s1_train, s1_val, s2_train, s2_val = train_test_split(
        paired_s1, paired_s2, test_size=0.1, random_state=42
    )
    print(f"📂 訓練集: {len(s1_train)}, 驗證集: {len(s1_val)}")

    train_dataset = create_dataset(s1_train, s2_train, batch_size, image_size, shuffle=True, augment=True)
    val_dataset   = create_dataset(s1_val,   s2_val,   batch_size, image_size, shuffle=False, augment=False)

    steps_per_epoch = math.ceil(len(s1_train) / batch_size)
    val_steps       = math.ceil(len(s1_val)  / batch_size)
    print(f"✅ steps_per_epoch={steps_per_epoch}, val_steps={val_steps}")

    os.makedirs(save_root, exist_ok=True)
    print(f"💾 模型將保存至：{save_root}")

    combined_model, hiding_net, reveal_net = build_combined_model(input_shape=(image_size[0], image_size[1], 3))
    compile_combined_model(combined_model)

    best_combined_ckpt = ModelCheckpoint(
        filepath=os.path.join(save_root, 'best_combined_model.h5'),
        monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1
    )
    csv_logger = CSVLogger(os.path.join(save_root, 'train_log.csv'))

    periodic_export = PeriodicExportCallback(
        save_root=save_root, hiding_model=hiding_net, reveal_model=reveal_net, period=50
    )
    safety_callbacks = [TerminateOnNaN(), NanGuard()]
    hb = Heartbeat(every=20, steps_per_epoch=steps_per_epoch, batch_size=batch_size)

    history = combined_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[hb, periodic_export, best_combined_ckpt, ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1), csv_logger] + safety_callbacks,
        verbose=1
    )

    # plot loss
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_root, "training_curve.png"), dpi=150)
    plt.close()
    print("📈 Saved training_curve.png")

# =========================
# 9. 推論（combined 或子網）
# =========================
def run_infer(model_dir: str, cover_img: str, secret_img: str, out_dir: str = None, image_size=(256,256)):
    model_dir = Path(model_dir)
    out_dir = model_dir / "inference_outputs" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    COMBINED = model_dir / "combined_model_final.h5"
    HIDING   = model_dir / "hiding_network_final.h5"
    REVEAL   = model_dir / "reveal_network_final.h5"

    # load combined if possible
    model = None
    try:
        model = tf.keras.models.load_model(str(COMBINED), compile=False, custom_objects={"RobustAttackLayer": RobustAttackLayer})
        print("✅ 載入 combined_model_final.h5 成功")
    except Exception as e:
        print(f"ℹ️ combined_model_final.h5 載入失敗：{e}")

    if model is None:
        # fallback to subnets
        try:
            hiding_net = tf.keras.models.load_model(str(HIDING), compile=False, custom_objects={"RobustAttackLayer": RobustAttackLayer})
        except Exception:
            hiding_net = build_hiding_network((image_size[0], image_size[1], 3))
            hiding_net.load_weights(str(HIDING))
        try:
            reveal_net = tf.keras.models.load_model(str(REVEAL), compile=False, custom_objects={"RobustAttackLayer": RobustAttackLayer})
        except Exception:
            reveal_net = build_reveal_network((image_size[0], image_size[1], 3))
            reveal_net.load_weights(str(REVEAL))
        print("✅ 使用 hiding + reveal 推論")
    else:
        hiding_net = reveal_net = None

    cover  = load_rgb_m11(cover_img, size=image_size)
    from PIL import Image
    # build tiled secret as in original helper (defaults good for robustness)
    secret = load_rgb_m11(secret_img, size=image_size)

    if model is not None:
        container, revealed = model.predict([cover[None,...], secret[None,...]], verbose=0)
        container, revealed = container[0], revealed[0]
    else:
        container = hiding_net.predict([cover[None,...], secret[None,...]], verbose=0)[0]
        revealed  = reveal_net.predict(container[None,...], verbose=0)[0]

    cover01    = m11_to_01(cover)
    container01= m11_to_01(container)
    secret01   = m11_to_01(secret)
    revealed01 = m11_to_01(revealed)

    psnr_cover_container = float(tf.image.psnr(cover01[None,...], container01[None,...], max_val=1.0).numpy()[0])
    psnr_secret_revealed = float(tf.image.psnr(secret01[None,...], revealed01[None,...], max_val=1.0).numpy()[0])

    to_uint8_from_m11(cover).save(out_dir / "cover.png")
    to_uint8_from_m11(secret).save(out_dir / "secret.png")
    to_uint8_from_m11(container).save(out_dir / "container.png")
    to_uint8_from_m11(revealed).save(out_dir / "revealed_secret.png")

    print(f"✅ 推論完成，結果已儲存到 {out_dir}")
    print(f"PSNR (Cover vs Container): {psnr_cover_container:.2f} dB")
    print(f"PSNR (Secret vs Revealed): {psnr_secret_revealed:.2f} dB")

# =========================
# 10. 外部 container 單獨解碼 & 以小塊重建
# =========================
import cv2

def load_container_for_reveal(path, size=(256, 256), mode='resize'):
    img = Image.open(path).convert("RGB")
    if mode == 'resize':
        img = img.resize(size, Image.LANCZOS)
    elif mode == 'center-crop':
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top  = (h - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize(size, Image.LANCZOS)
    elif mode == 'pad-to-square':
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        canvas.paste(img, ((s - w)//2, (s - h)//2))
        img = canvas.resize(size, Image.LANCZOS)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    arr = np.asarray(img, dtype=np.float32)
    return (arr - 127.5) / 127.5  # [-1,1]

def _to_u8_from_m11_np(x):
    return np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8)

def _rotate_bound_keep_size(img_bgr, angle_deg):
    H, W = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W/2, H/2), angle_deg, 1.0)
    return cv2.warpAffine(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _match_template_color(search_bgr, templ_bgr):
    acc = None
    for ch in range(3):
        res = cv2.matchTemplate(search_bgr[..., ch], templ_bgr[..., ch], cv2.TM_CCOEFF_NORMED)
        acc = res if acc is None else acc + res
    acc = acc / 3.0
    _, maxVal, _, maxLoc = cv2.minMaxLoc(acc)
    return acc, float(maxVal), (maxLoc[1], maxLoc[0])  # yx

def _extract_block(img_bgr, yx, edge):
    y, x = yx
    y = max(0, min(y, img_bgr.shape[0] - edge))
    x = max(0, min(x, img_bgr.shape[1] - edge))
    return img_bgr[y:y+edge, x:x+edge].copy()

def _tile_block_to_256(block_bgr, canvas_size=(256, 256)):
    H, W = canvas_size
    e = block_bgr.shape[0]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for yy in range(0, H, e):
        for xx in range(0, W, e):
            if yy + e <= H and xx + e <= W:
                canvas[yy:yy+e, xx:xx+e] = block_bgr
    return canvas

def reconstruct_from_reveal(revealed_m11, multiscale=(64, 32), rot_search_deg=3, rot_step_deg=1, rec_dir: Path = None):
    rev_u8  = _to_u8_from_m11_np(revealed_m11)
    rev_bgr = cv2.cvtColor(rev_u8, cv2.COLOR_RGB2BGR)

    best = {"score": -1.0, "edge": None, "yx": (0, 0), "angle": 0, "block": None}

    for edge in multiscale:
        cy, cx = rev_bgr.shape[0] // 2, rev_bgr.shape[1] // 2
        y0, x0 = max(0, cy - edge // 2), max(0, cx - edge // 2)
        templ0 = rev_bgr[y0:y0+edge, x0:x0+edge]
        if templ0.shape[:2] != (edge, edge):
            continue

        local = {"score": -1.0, "yx": (0, 0), "angle": 0}
        for ang in range(-rot_search_deg, rot_search_deg + 1, rot_step_deg):
            templ = templ0 if ang == 0 else _rotate_bound_keep_size(templ0, ang)
            _, sc, yx = _match_template_color(rev_bgr, templ)
            if sc > local["score"]:
                local.update({"score": sc, "yx": yx, "angle": ang})

        block = _extract_block(rev_bgr, local["yx"], edge)
        if local["score"] > best["score"]:
            best.update({"score": local["score"], "edge": edge, "yx": local["yx"], "angle": local["angle"], "block": block})

    if best["block"] is None:
        raise RuntimeError("找不到合適的小塊；請確認 reveal 尺寸為 256×256、multiscale 設定正確。")

    recon_bgr = _tile_block_to_256(best["block"], canvas_size=(256, 256))
    recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)

    if rec_dir is not None:
        rec_dir.mkdir(parents=True, exist_ok=True)
        best_path  = rec_dir / f"best_block_{best['edge']}.png"
        recon_path = rec_dir / f"reconstructed_256_from_{best['edge']}.png"
        cv2.imwrite(str(best_path), best["block"])
        cv2.imwrite(str(recon_path), cv2.cvtColor(recon_rgb, cv2.COLOR_RGB2BGR))
        print(f"🗂️ saved: {best_path} , {recon_path}")

    print(f"✅ Best edge={best['edge']} score={best['score']:.4f} at yx={best['yx']} angle={best['angle']}°")
    return recon_rgb, best

def run_external_reveal(reveal_h5: str, attack_img: str, container_img: str = None, out_dir: str = None):
    out_dir = Path(out_dir) if out_dir else Path(reveal_h5).parent / "inference_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = out_dir / "recovered_from_reveal"
    rec_dir.mkdir(parents=True, exist_ok=True)

    # 1) load reveal model
    try:
        reveal_net = tf.keras.models.load_model(
            reveal_h5, compile=False, custom_objects={"RobustAttackLayer": RobustAttackLayer}
        )
        print("✅ 載入 reveal_network 成功")
    except Exception:
        reveal_net = build_reveal_network((256,256,3))
        reveal_net.load_weights(reveal_h5)
        print("✅ 以 build + load_weights 載入 Reveal Net")

    # 2) decode attack image
    x_m11 = load_container_for_reveal(attack_img, mode='resize')
    y_m11 = reveal_net.predict(x_m11[None, ...], verbose=0)[0]   # [-1,1]
    to_uint8_from_m11(y_m11).save(out_dir / "reveal.png")

    # 3) reconstruct by best block
    reconstruct_from_reveal(y_m11, multiscale=(64,32), rot_search_deg=3, rot_step_deg=1, rec_dir=rec_dir)

    # optional: copy inputs for convenience
    from shutil import copyfile
    if container_img and Path(container_img).exists():
        copyfile(container_img, out_dir / "container.png")
    if Path(attack_img).exists():
        copyfile(attack_img, out_dir / "attack_edit.png")

    print(f"✅ 完成外部解碼與重建，輸出位置：{out_dir}")

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Robust watermark (train/infer/external_reveal)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--cover_dir", default=r"C:\Users\admin\Desktop\rubust_matermark\CoverImage",
                     help="Folder of cover images")
    p_train.add_argument("--secret_dir", default=r"C:\Users\admin\Desktop\rubust_matermark\SecretImage",
                     help="Folder of secret images")
    p_train.add_argument("--save_root", default=r"C:\Users\admin\Desktop\rubust_matermark\output",
                     help="Where to save checkpoints/exports")
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=1000)
    p_train.add_argument("--img_size", type=int, default=256)

    p_infer = sub.add_parser("infer", help="Run inference (container + reveal)")
    p_infer.add_argument("--model_dir", required=True, help="Directory containing *_final.h5 models")
    p_infer.add_argument("--cover_img", required=True)
    p_infer.add_argument("--secret_img", required=True)
    p_infer.add_argument("--out_dir", default=None)
    p_infer.add_argument("--img_size", type=int, default=256)

    p_ext = sub.add_parser("external_reveal", help="Decode external attacked image with reveal model")
    p_ext.add_argument("--reveal_h5", required=True, help="Path to reveal_network.h5/.keras")
    p_ext.add_argument("--attack_img", required=True, help="Path to attacked image to decode")
    p_ext.add_argument("--container_img", default=None, help="(Optional) Original container image for reference")
    p_ext.add_argument("--out_dir", default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        run_train(args.cover_dir, args.secret_dir, args.save_root,
                  batch_size=args.batch_size, epochs=args.epochs, image_size=(args.img_size, args.img_size))
    elif args.cmd == "infer":
        run_infer(args.model_dir, args.cover_img, args.secret_img,
                  out_dir=args.out_dir, image_size=(args.img_size, args.img_size))
    elif args.cmd == "external_reveal":
        run_external_reveal(args.reveal_h5, args.attack_img, args.container_img, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
