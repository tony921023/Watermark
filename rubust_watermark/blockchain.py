import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    from web3 import Web3
    _w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
except ImportError:
    _w3 = None

# ---- 原有函式（連接 Hyperledger Besu 私有鏈）----

def upload_evidence(video_path, result_text):
    """將偵測結果雜湊上傳至 Besu 私有鏈（需 web3 + 節點運行中）。"""
    if _w3 is None:
        print(" web3 未安裝，跳過上鏈。")
        return None
    with open(video_path, "rb") as f:
        video_hash = hashlib.sha256(f.read()).hexdigest()
    evidence = f"【真·照妖鏡存證】結果: {result_text} | 影片雜湊: {video_hash}"
    if _w3.eth.accounts:
        tx_hash = _w3.eth.send_transaction({
            'from': _w3.eth.accounts[0],
            'to':   _w3.eth.accounts[0],
            'data': _w3.to_hex(text=evidence)
        })
        print(f" 證據已永久保存！交易序號：{tx_hash.hex()}")
        return tx_hash.hex()
    else:
        print(" 節點內找不到可用帳號，請確認 Besu 設定。")
        return None

# ---- 本地 JSON 區塊鏈（供 app_wm.py 使用）----

def sha256_bytes(data: bytes) -> str:
    """計算位元組資料的 SHA-256，回傳 hex 字串。"""
    return hashlib.sha256(data).hexdigest()


def _calc_block_hash(index: int, prev_hash: str, timestamp: float,
                     job_id: str, image_sha256: str, metadata: dict) -> str:
    content = json.dumps({
        "index":        index,
        "prev_hash":    prev_hash,
        "timestamp":    timestamp,
        "job_id":       job_id,
        "image_sha256": image_sha256,
        "metadata":     metadata,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class LocalChain:
    """本地 JSON 區塊鏈，用於記錄與驗證浮水印嵌入事件。"""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._blocks: list = []
        self._load()

    # ---- 持久化 ----

    def _load(self):
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._blocks = data.get("blocks", [])
            except Exception:
                self._blocks = []
        if not self._blocks:
            genesis = self._make_genesis()
            self._blocks = [genesis]
            self._save()

    def _save(self):
        self._path.write_text(
            json.dumps({"blocks": self._blocks}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def _make_genesis(self) -> dict:
        ts = time.time()
        bh = _calc_block_hash(0, "0" * 64, ts, "__genesis__", "0" * 64, {})
        return {
            "index":        0,
            "prev_hash":    "0" * 64,
            "timestamp":    ts,
            "job_id":       "__genesis__",
            "image_sha256": "0" * 64,
            "metadata":     {},
            "block_hash":   bh,
        }

    # ---- 公開介面 ----

    def chain_info(self) -> dict:
        """回傳鏈的基本狀態，用於 /health。"""
        with self._lock:
            return {
                "length":      len(self._blocks),
                "latest_hash": self._blocks[-1]["block_hash"] if self._blocks else "",
                "path":        str(self._path),
            }

    def record_embed(self, job_id: str, image_sha256: str, metadata: dict) -> dict:
        """嵌入浮水印後呼叫，將事件寫成新區塊並持久化。

        回傳的 dict 包含：index、block_hash、timestamp、job_id。
        """
        with self._lock:
            prev = self._blocks[-1]
            index = len(self._blocks)
            timestamp = time.time()
            bh = _calc_block_hash(index, prev["block_hash"], timestamp,
                                   job_id, image_sha256, metadata)
            block = {
                "index":        index,
                "prev_hash":    prev["block_hash"],
                "timestamp":    timestamp,
                "job_id":       job_id,
                "image_sha256": image_sha256,
                "metadata":     metadata,
                "block_hash":   bh,
            }
            self._blocks.append(block)
            self._save()
            return block

    def unlock_single_embed(self, job_id: str) -> bool:
        """解除指定 job_id 的單次嵌入鎖定，將 metadata.single_embed 設為 False 並持久化。"""
        with self._lock:
            for block in self._blocks:
                if block.get("job_id") == job_id:
                    block["metadata"]["single_embed"] = False
                    self._save()
                    return True
        return False

    def find_by_cover_hash(self, cover_hash: str) -> Optional[dict]:
        """查詢鏈上是否已有相同 cover 圖的嵌入記錄（用於單次嵌入鎖定）。

        回傳第一個符合的 block，若無則回傳 None。
        只比對 metadata.cover_hash 且 metadata.single_embed == True 的記錄。
        """
        with self._lock:
            for block in self._blocks:
                meta = block.get("metadata", {})
                if meta.get("single_embed") and meta.get("cover_hash") == cover_hash:
                    return block
        return None

    def verify_by_job_id(self, job_id: str, block_hash: str) -> Tuple[bool, dict, str]:
        """還原浮水印前呼叫，驗證圖片 metadata 是否與鏈上記錄吻合。

        回傳 (ok, block, reason)：
          - ok=True：驗證通過
          - reason="not_registered"：鏈上無此 job_id
          - reason="hash_mismatch"：job_id 存在但 block_hash 不符（圖片可能被竄改）
        """
        with self._lock:
            for block in self._blocks:
                if block.get("job_id") == job_id:
                    if block["block_hash"] == block_hash:
                        return True, block, "ok"
                    else:
                        return False, block, "hash_mismatch"
            return False, {}, "not_registered"


# ---- 單例管理 ----

_chain: Optional[LocalChain] = None
_chain_lock = threading.Lock()


def init_chain(path) -> None:
    """服務啟動時呼叫一次，從 path 載入（或建立）本地區塊鏈。"""
    global _chain
    with _chain_lock:
        if _chain is None:
            _chain = LocalChain(Path(path))


def get_chain() -> LocalChain:
    """取得區塊鏈單例，必須在 init_chain() 之後呼叫。"""
    if _chain is None:
        raise RuntimeError("區塊鏈尚未初始化，請先呼叫 init_chain(path)。")
    return _chain
