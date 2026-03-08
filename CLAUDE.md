# CLAUDE.md

本文件提供 Claude Code 在此儲存庫工作時的操作指引。

## 專案概述

**照妖鏡（Deepfake Detection System）** — 雙服務取證平台，結合：
1. **IID-Net 偵測服務**（`detect/`，埠 5000）— 使用 PyTorch 進行影像偽造／竄改定位
2. **強韌浮水印服務**（`rubust_watermark/`，埠 5001）— 使用 TensorFlow 2.15 進行隱寫浮水印嵌入與還原

---

## 啟動服務

### IID-Net 偵測服務（埠 5000）
```bash
# 設定權重路徑（或將權重放在 detect/IID_weight/IID_weights.pth）
set IID_WEIGHTS=C:\path\to\IID_weights.pth
cd detect
python app.py
```

### 強韌浮水印服務（埠 5001）
```bash
# 先啟動虛擬環境（詳見 cd_robust.txt）
cd rubust_watermark
venv\Scripts\activate
python app_wm.py serve --host 0.0.0.0 --port 5001 --root ./outputs
```

---

## 訓練指令

**浮水印模型：**
```bash
python robust_watermark.py train --cover_dir <dir> --secret_dir <dir> --save_root <dir>
```

**IID-Net（在 detect/ 目錄下）：**
```bash
python main.py train
```

**浮水印推論／還原：**
```bash
python robust_watermark.py infer --model_dir <dir> --cover_img <img> --secret_img <img>
python robust_watermark.py external_reveal --reveal_h5 <h5> --attack_img <img>
```

---

## 架構說明

### detect/（偵測服務）
- `app.py` — Flask API 伺服器。載入 `IID_Net`，`POST /result` 時執行推論、產生竄改遮罩疊圖，儲存取證交付包（`forensics_deliveries/<run_id>/`），並可透過 `_forward_to_watermark()` 轉送至浮水印服務。
- `main.py` — IID-Net 模型定義與訓練框架。架構：增強區塊（Normal／Bayar／PF 卷積）→ 10 個提取單元（NAS 選取的深度可分離卷積＋膨脹）→ 決策區塊（全域-局部注意力 + 3× 雙線性上取樣 → 1 通道 sigmoid 遮罩）。
- `detect_27042.py` — 獨立取證腳本：載入 `InpaintingForensics`，批次推論，產生 `report.json`、`report.md`、`SHA256SUMS.txt`，以及可選的 PDF（需 reportlab）。

### rubust_watermark/（浮水印服務）
- `app_wm.py` — Flask API 伺服器。載入 TF 模型（combined + reveal）。`POST /wm/embed` 將 secret 嵌入 cover 影像；`POST /external_reveal`（別名：`/wm/external_reveal`）從受攻擊影像還原浮水印，使用底排格評分＋可選 Real-ESRGAN 放大＋Reinhard 色彩遷移。
- `robust_watermark.py` — 訓練 CLI：建構 `Hiding Network` + `Reveal Network`（皆為 Encoder-Decoder 殘差架構，LeakyReLU）+ `RobustAttackLayer`（訓練中模擬雜訊／模糊／池化）。損失函數：感知損失（VGG19）+ MSE（cover）；MSE + DCT + 色彩一致性（secret）。

### project_root/frontend/（前端）
靜態 HTML 頁面（`index.html`、`detect.html`、`watermark.html`、`restore.html`）+ `main.js` + `style.css`。
JavaScript 透過埠 5000（`API_BASE`）與偵測服務通訊，透過埠 5001（`WM_API_BASE`）與浮水印服務通訊。

### 服務整合
IID-Net 偵測後自動轉送至浮水印服務。浮水印服務接收 `source.png` + `mask_bin.png`，回傳 `patched_masked.png`（在竄改區域上還原浮水印）。可透過環境變數控制。

---

## 關鍵環境變數

| 變數 | 預設值 | 說明 |
|---|---|---|
| `IID_WEIGHTS` | — | `IID_weights.pth` 的路徑 |
| `WM_FORWARD` | `1` | 偵測後是否轉送至浮水印服務 |
| `WM_URL` | `http://127.0.0.1:5001/wm/reveal_masked` | 浮水印服務端點 |
| `WM_TIMEOUT_SEC` | `900` | 轉送 HTTP 逾時秒數 |
| `WM_CONTAINER_MODE` | `auto` | 容器混合策略：`auto`／`direct`／`residual` |
| `WM_PSNR_TARGET` | `40.0` | 殘差嵌入目標 PSNR（dB） |
| `WM_OPT_REVEAL` | `1` | 是否以還原 SSIM 最佳化縮放比 |
| `REALESRGAN_REPO` | `Real-ESRGAN` | Real-ESRGAN 儲存庫路徑（選用） |

---

## 權重檔位置

- IID-Net：`detect/IID_weight/IID_weights.pth`（或透過 `IID_WEIGHTS` 環境變數設定）
- WM combined 模型：`rubust_watermark/weights/combined_model.h5`
- WM reveal 模型：`rubust_watermark/weights/reveal_network.h5`

---

## API 端點

### 偵測服務（埠 5000）
| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/result` | 上傳影像，取得疊圖 PNG 並儲存取證執行結果 |
| `GET` | `/27042/view/<run_id>` | 瀏覽器檢視取證報告 |
| `GET` | `/27042/download/<run_id>` | 下載執行產物 ZIP |
| `GET` | `/27042/latest` | 最新執行元資料 |
| `GET` | `/health` | 服務健康狀態 + 權重路徑 |

### 浮水印服務（埠 5001）
| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/wm/embed` | 嵌入浮水印（multipart：`cover`、可選 `secret`） |
| `POST` | `/external_reveal` | 從受攻擊影像還原浮水印（multipart：`image`／`attack`、可選 `mask`） |
| `GET` | `/27037/latest?kind=embed\|reveal` | 最新工作元資料 |
| `GET` | `/health` | 服務健康狀態 + 權重是否存在 |

---

## 取證輸出結構

每次偵測執行會建立 `detect/forensics_deliveries/<run_id>/`：
```
inputs/source.png               # 原始上傳影像
artifacts/overlay.png           # 紅色半透明疊加於竄改區域
artifacts/mask_gray.png         # 連續信心值遮罩
artifacts/mask_bin.png          # 二值遮罩（門檻 0.5）
artifacts/patched_masked.png    # 浮水印還原至遮罩區域（來自 WM 服務）
report.json                     # 結構化機器可讀報告
report.html                     # 可離線檢視的 HTML 報告
sha256sums.txt                  # 所有產物的 SHA-256 校驗值
```

---

## 相依套件

- **detect/**：Python、PyTorch、OpenCV（`cv2`）、Flask、flask-cors、requests、numpy、torchvision
- **rubust_watermark/**：Python 3.10/3.11、TensorFlow 2.15 / Keras 2.15、Pillow、OpenCV、Flask、h5py、scikit-image（選用）、reportlab（選用，用於 PDF）、Real-ESRGAN（選用，用於超解析放大）

---

## 前端注意事項

- 所有頁面共用 `project_root/frontend/style.css`，請勿在 HTML 頁面內寫重複的 `<style>` 區塊
- zoom（放大）、`toggleMenu()`、`WM_API_BASE` 等共用函式由 `main.js` 全域提供；各頁面的 `DOMContentLoaded` 只需呼叫 `bindZoomToNewImages()` 初始化即可
- 前端以純靜態方式部署，不需要 Flask 樣板語法（`{{ url_for(...) }}`）
