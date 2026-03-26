# CLAUDE.md

本文件提供 Claude Code 在此儲存庫工作時的操作指引。

## 專案概述

**照妖鏡（Deepfake Defense System）** — 浮水印取證平台，目前僅剩一個後端服務：

| 服務 | 目錄 | 埠號 | 技術棧 |
|---|---|---|---|
| 強韌浮水印服務 | `rubust_watermark/` | 5001 | TensorFlow 2.15 / Keras 2.15、Flask |

> ⚠️ `detect/`（IID-Net 偵測服務，埠 5000）已於先前版本移除。前端 `main.js` 仍保留 `API_BASE = port 5000` 常數但目前未使用。

---

## 啟動服務

### 強韌浮水印服務（埠 5001）
```bash
# 先啟動虛擬環境（詳見 cd_robust.txt）
cd rubust_watermark
venv\Scripts\activate
python app_wm.py serve --host 0.0.0.0 --port 5001 --root ./outputs
```

---

## 訓練指令

```bash
# 訓練
python robust_watermark.py train --cover_dir <dir> --secret_dir <dir> --save_root <dir>

# 推論 / 還原
python robust_watermark.py infer --model_dir <dir> --cover_img <img> --secret_img <img>
python robust_watermark.py external_reveal --reveal_h5 <h5> --attack_img <img>
```

---

## 架構說明

### rubust_watermark/（浮水印服務）

- **`app_wm.py`** — Flask API 伺服器。
  - 載入 TF 模型（combined + reveal）。
  - `POST /wm/embed`：將 secret 嵌入 cover 影像，支援殘差嵌入（PSNR 目標控制）+ 還原 SSIM 優化。
  - `POST /external_reveal`：從受攻擊影像還原浮水印，使用底排格評分 + 可選 Real-ESRGAN 放大 + Reinhard 色彩遷移。
  - 每個工作會產生 Forensics Report（HTML）、Audit Log（JSONL）、環境快照（JSON）、區塊鏈存證記錄。
  - 啟動時呼叫 `init_chain()` 初始化本地區塊鏈（`blockchain.json`）。

- **`blockchain.py`** — 區塊鏈存證模組。
  - `app_wm.py` 需要 `init_chain`、`get_chain`、`sha256_bytes` 三個介面。
  - ⚠️ **注意**：目前 `blockchain.py` 內只有 `upload_evidence()`（連接 Hyperledger Besu 私有鏈 `http://127.0.0.1:8545`），**缺少** `init_chain`、`get_chain`、`sha256_bytes`，啟動時會出現 `ImportError`。需補全這三個函式才能正常運行。

- **`robust_watermark.py`** — 訓練 CLI：`Hiding Network` + `Reveal Network`（Encoder-Decoder 殘差架構，LeakyReLU）+ `RobustAttackLayer`。損失函數：感知損失（VGG19）+ MSE（cover）；MSE + DCT + 色彩一致性（secret）。

### project_root/frontend/（前端）

靜態 HTML 頁面：

| 頁面 | 說明 |
|---|---|
| `index.html` | 首頁（Hero、運作原理、鑑識報告入口） |
| `watermark.html` | 施法加印（嵌入浮水印） |
| `restore.html` | 照出真身（還原浮水印） |

共用資源：`main.js`（全域工具函式 + API 位址設定）、`style.css`（所有樣式集中於此，勿在頁面內寫 `<style>`）。

---

## 關鍵環境變數

| 變數 | 預設值 | 說明 |
|---|---|---|
| `WM_CONTAINER_MODE` | `auto` | 容器混合策略：`auto`／`direct`／`residual` |
| `WM_PSNR_TARGET` | `40.0` | 殘差嵌入目標 PSNR（dB） |
| `WM_RES_SCALE` | `1.0` | 殘差初始縮放比 |
| `WM_RES_SCALE_MIN` | `0.0` | 殘差縮放下界 |
| `WM_RES_SCALE_MAX` | `1.0` | 殘差縮放上界 |
| `WM_OPT_REVEAL` | `1` | 是否以還原 SSIM 優化縮放比 |
| `REALESRGAN_REPO` | `Real-ESRGAN` | Real-ESRGAN 儲存庫路徑（選用） |
| `REALESRGAN_MODEL` | `RealESRGAN_x2plus` | Real-ESRGAN 模型名稱 |
| `REALESRGAN_MODEL_PATH` | — | 指定模型檔案路徑（選用） |
| `REALESRGAN_TILE` | — | 分塊推論大小（選用，0=不分塊） |
| `REALESRGAN_FP32` | `1` | 使用 FP32 推論 |
| `PYTHON_EXE` | — | Real-ESRGAN 子進程用 Python 路徑（選用） |

---

## 權重檔位置

預設路徑（硬編碼在 `app_wm.py` 第 62 行）：
```
C:\Users\admin\Desktop\114屆照妖鏡\rubust_matermark\weights\combined_model.h5
C:\Users\admin\Desktop\114屆照妖鏡\rubust_matermark\weights\reveal_network.h5
```

若路徑不存在，`_resolve_weight_path()` 會嘗試 fallback 列表（目前僅 `reveal_network .h5` 有空格容錯）。

---

## API 端點（浮水印服務，埠 5001）

| 方法 | 路徑 | 說明 |
|---|---|---|
| `GET` | `/` | 服務根目錄（ping） |
| `GET` | `/health` | 服務健康狀態 + 權重是否存在 + 區塊鏈資訊 |
| `POST` | `/wm/embed` | 嵌入浮水印（multipart：`cover`、可選 `secret`） |
| `POST` | `/external_reveal` | 從受攻擊影像還原浮水印（multipart：`image`／`attack`、可選 `mask`） |
| `GET` | `/files/<job>/<name>` | 取得工作產物檔案 |
| `GET` | `/open/image/<job>/<name>` | 瀏覽器直接顯示影像 |
| `GET` | `/dl/image/<job>/<name>` | 下載單一影像 |
| `GET` | `/dl/zip/<job>` | 下載工作產物 ZIP |
| `GET` | `/dl/report/<job_id>` | 下載 HTML 報告 |
| `GET` | `/report/<job_id>` | 瀏覽器檢視 Forensics Report |
| `GET` | `/27037/latest` | 最新工作元資料（`?kind=embed\|reveal`） |

---

## 工作輸出結構

每次工作會在 `<root>/<job_id>/` 建立：
```
cover.png               # 原始 cover 影像（256×256）
secret_in.png           # 輸入 secret 影像
container.png           # 嵌入浮水印後的影像
residual.png            # container 與 cover 的差異圖
secret.png              # 從 container 還原的 secret
report.png              # 五聯圖（cover/secret_in/container/residual/secret）
report.html             # Forensics Report（含環境、Audit Log、影像）
environment.json        # Python / TF / CUDA 環境快照
blockchain_record.json  # 區塊鏈存證結果
sha256sums.txt          # 所有產物的 SHA-256 校驗值
manifest.json           # 產物清單
logs/audit.jsonl        # 操作稽核日誌
```

---

## 相依套件

- **rubust_watermark/**：Python 3.10/3.11、TensorFlow 2.15 / Keras 2.15、Pillow、OpenCV（cv2）、Flask、flask-cors、h5py、numpy、web3、scikit-image（選用）、Real-ESRGAN（選用）

---

## 前端注意事項

- 所有頁面共用 `project_root/frontend/style.css`，請勿在 HTML 頁面內寫重複的 `<style>` 區塊
- `toggleMenu()`、`bindZoomToNewImages()`、`WM_API_BASE` 等共用函式由 `main.js` 全域提供
- 前端以純靜態方式部署，不使用 Flask 樣板語法（`{{ url_for(...) }}`）
- `API_BASE`（port 5000）保留於 `main.js` 但目前無對應後端
