import hashlib
from web3 import Web3
 
# 1. 初始化連線
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
 
def upload_evidence(video_path, result_text):
    # 2. 計算影片的數位指紋
    with open(video_path, "rb") as f:
        video_hash = hashlib.sha256(f.read()).hexdigest()
    # 3. 準備存證訊息
    evidence = f"【真·照妖鏡存證】結果: {result_text} | 影片雜湊: {video_hash}"
    # 4. 發送一筆空交易並帶入存證資料 (Data 欄位)
    # 在私有鏈測試環境，我們通常不需要私鑰，直接使用節點預設帳號
    if w3.eth.accounts:
        tx_hash = w3.eth.send_transaction({
            'from': w3.eth.accounts[0],
            'to': w3.eth.accounts[0],
            'data': w3.to_hex(text=evidence)
        })
        print(f" 證據已永久保存！交易序號：{tx_hash.hex()}")
        return tx_hash.hex()
    else:
        print(" 節點內找不到可用帳號，請確認 Besu 設定。")
 
# 呼叫範例：
# upload_evidence("sample_deepfake.mp4", "Deepfake Detected (Score: 0.98)")