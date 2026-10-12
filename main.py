import asyncio
import base64
import cv2
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
import json
import numpy as np
import os
import re
import smtplib
import sqlite3
import threading
import time
import traceback
import unicodedata
import uuid
from datetime import datetime
from queue import Queue
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import uvicorn
from starlette.middleware.sessions import SessionMiddleware

# テンプレートエンジンの準備
app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here")

# グローバル変数として追加
clients = []
last_detected_qr = "" # 直近のQRコードを保持する変数

# =========================================================================
# ⚙️ システム設定値（調整可能）
# =========================================================================
DB_PATH = "gym_security.db"
DUPLICATE_QR_WINDOW = 5.0
CO_TRAILING_WINDOW = 5.0
LIMIT_SECONDS = 10.0
FACE_TIMEOUT = 5.0            # 顔の類似度（直近の最大値）を有効とみなす秒数
FACE_MATCH_THRESHOLD = 0.55   # 顔が一致したとみなすコサイン類似度（ログを見て調整）

# 📧 メール送信設定（実際の運用に合わせて書き換えてください）
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "chocozap123456@gmail.com"  # 送信元のメールアドレス
SENDER_PASSWORD = "rfrlsheyiuzhyubg"   # アプリパスワード等

LINE_X = 320
CAMERA_ID = 0  # 内蔵カメラ1台

# 🪑 椅子・座り判定の設定
SEAT_CLASSES = [56]           # COCO: 56=chair（普通の椅子のみ。ベンチ等も拾いたい場合は 13=bench, 57=couch を追加）
ROOM_MODEL_PATH = "yolo11n.pt"  # 椅子の検出が不安定なら "yolo11s.pt" や "yolo11m.pt" に変更（精度↑・速度↓）
SEAT_RELEASE_GRACE = 5.0      # 座っていない状態がこの秒数続いたら、利用終了（マシンを空きに戻す）
ROOM_CONF = 0.20              # ルームモードの検出しきい値（低めにして椅子の取りこぼしを減らす）
SIT_OVERLAP_RATIO = 0.35      # 人の下半身ボックスが椅子ボックスと重なっている割合の下限（画面の overlap= を見て調整）
SIT_CONFIRM_SECONDS = 2.0     # 「座っている」がこの秒数続いて初めてタイマー開始（椅子にかすっただけを除外）
SIT_BREAK_TOLERANCE = 1.0     # 座り判定がこの秒数以内の途切れなら、座り続けているとみなす
POSE_MODEL_PATH = "yolo11n-pose.pt"  # 骨格推定モデル（初回起動時に自動ダウンロード）
KP_CONF = 0.4                 # 骨格の点（腰・膝など）を「見えている」とみなす信頼度
SIT_THIGH_DROP_RATIO = 0.75   # (膝−腰の縦の差)÷(胴の長さ)。立つと約1.0、座ると0〜0.5。これ以上なら立っているとみなす
REID_THRESHOLD = 0.70         # 服の色の類似度(0〜1)がこれ以上で「同じ人」とみなす（画面の color= を見て調整）
REID_MARGIN = 0.10            # 2番目に似ている会員との差がこれ以上あるときだけ採用
REID_INTERVAL = 1.0           # 顔で特定できないとき、服の色で推定する間隔（秒）
REID_ADD_INTERVAL = 5.0       # 顔で特定できている間、座った姿の特徴量を追加する間隔（秒）
REID_MAX_FEATS = 10           # 会員1人あたりに保持する特徴量の数
REID_TTL_SECONDS = 3 * 3600   # 特徴量を使える期間（服装は日が変わると変わる）
REID_MIN_W, REID_MIN_H = 40, 80   # これより小さい人からは特徴量を取らない
IDENTITY_HOLD_SECONDS = 15.0  # 顔認識が外れても、直前に特定した会員をこの秒数は保持（座っている間は保持し続ける）
SEAT_MEMORY_SECONDS = 10.0    # 椅子が検出できなくなっても、この秒数は直前の位置を覚えておく（座った人に隠れる対策）
SEAT_OCCLUDED_RATIO = 0.15    # 人が椅子ボックスの面積のこれ以上に重なっていたら「隠れている」とみなし、椅子の記憶を保持
HIP_SEAT_MARGIN = 0.10        # 腰が椅子ボックスの上端よりこの割合だけ上でも許容
STANDING_ASPECT_RATIO = 3.0   # 骨格が取れないときの予備：人ボックスの縦÷横がこれ以上なら立っているとみなす

# 🎛️ カメラの役割切り替え用フラグ ('entrance' または 'room')
active_mode = "entrance"
mode_lock = threading.Lock()

# 🏋️‍♂️ マシン・椅子の管理情報
MACHINE_LIMITS = {
    "bench_press": {"name": "ベンチプレス", "limit": 15.0},
    "squat_rack": {"name": "スクワットラック", "limit": 20.0}
}

# =========================================================================
# 💾 リクエスト用データモデル（先頭側に定義）
# =========================================================================
class RegisterRequest(BaseModel):
    name: str
    email: str
    image: str

class AdminAuthRequest(BaseModel):
    username: str
    password: str

# =========================================================================
# 💾 データベース自動初期化
# =========================================================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    
    # 会員マスタテーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS members (
            member_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            qr_token TEXT UNIQUE NOT NULL,
            face_embedding BLOB
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS passing_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            direction TEXT,
            member_id TEXT,
            is_alert INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS occupancy_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id TEXT,
            start_time TEXT,
            end_time TEXT,
            duration REAL
        )
    """)
    # 既存のDBには machine 列が無いので、足りなければ追加する
    cursor.execute("PRAGMA table_info(occupancy_logs)")
    if "machine" not in [row[1] for row in cursor.fetchall()]:
        cursor.execute("ALTER TABLE occupancy_logs ADD COLUMN machine TEXT")
    cursor.execute("PRAGMA table_info(passing_logs)")
    if "note" not in [row[1] for row in cursor.fetchall()]:
        cursor.execute("ALTER TABLE passing_logs ADD COLUMN note TEXT")
    conn.commit()
    conn.close()

init_database()
print(f">> 使用DB: {os.path.abspath(DB_PATH)}")

print(">> [1/3] YOLO（人・椅子検出）モデルを読み込み中...")
yolo_model = None
qr_detector = cv2.QRCodeDetector()

# =========================================================================
# 💾 システム状態管理（ステート）用クラス
# =========================================================================
class SystemStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.in_count = 0
        self.out_count = 0
        self.last_scanned_qr = "None"
        self.last_qr_time = 0.0
        self.accepted_token = None   # 直近に認証成功したQRトークン
        self.rejected_token = None   # 直近に拒否した（未登録/読み取り崩れ）QRの文字列
        self.rejected_time = 0.0
        self.co_trailing_alert = False
        
        # 入口の1:1照合用に直近の会員情報と顔特徴量を保持
        self.last_member_id = None
        self.last_member_embedding = None

        
        # ルーム監視用の状態管理
        self.active_users = {} 
        
        self.machine_states = {
            k: {"user": "-", "duration": 0.0, "status": "free", "start_time": None, "last_seen": None, "member_id": None} 
            for k in MACHINE_LIMITS.keys()
        }
        
        self.update_queue = Queue()

    def should_process_qr(self, qr_data: str, now: float) -> bool:
        """同じQRが映り続けている間は再認証しない。有効QRなら有効時間(last_qr_time)だけ延長する"""
        with self._lock:
            if qr_data == self.accepted_token and (now - self.last_qr_time) <= DUPLICATE_QR_WINDOW:
                self.last_qr_time = now
                return False
            if qr_data == self.rejected_token and (now - self.rejected_time) <= DUPLICATE_QR_WINDOW:
                return False
            return True

    def process_qr(self, qr_data: str, current_time: float):
        with self._lock:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT member_id, name, face_embedding FROM members WHERE qr_token = ?", (qr_data,))
            member = cursor.fetchone()
            conn.close()

            if member:
                self.accepted_token = qr_data
                self.rejected_token = None
                self.last_scanned_qr = f"{member['name']} ({member['member_id']})"
                self.last_member_id = member['member_id']
                if member['face_embedding']:
                    self.last_member_embedding = np.frombuffer(member['face_embedding'], dtype=np.float32)
                else:
                    self.last_member_embedding = None
                print(f"🔓 【QR認証成功】 歓迎: {member['name']} (ID: {member['member_id']})")
            else:
                self.rejected_token = qr_data
                self.rejected_time = current_time
                auth_still_valid = (
                    self.last_member_id is not None
                    and (current_time - self.last_qr_time) <= CO_TRAILING_WINDOW
                )
                if auth_still_valid:
                    # 有効な認証の直後に読み取りが崩れた/未登録QRが映っただけなら、認証状態を上書きしない
                    print(f"⚠️ 【QR読み取り無視】 有効な認証の直後に未登録の文字列を検出（認証は維持）: {qr_data!r}")
                    return
                self.accepted_token = None
                self.last_scanned_qr = "Invalid QR"
                self.last_member_id = None
                self.last_member_embedding = None
                print(f"❌ 【QR認証失敗】 無効なトークンです: {qr_data!r} (長さ {len(qr_data)})")

            self.last_qr_time = current_time
            self.co_trailing_alert = False
            self.push_update()

    def set_alert(self, state: bool):
        with self._lock:
            self.co_trailing_alert = state
            self.push_update()

    def register_pass(self, direction: str, member_id: str, is_alert: int, note: str = None):
        with self._lock:
            if direction == "IN":
                self.in_count += 1
            else:
                self.out_count += 1
            
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute(
                "INSERT INTO passing_logs (timestamp, direction, member_id, is_alert, note) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), direction, member_id, is_alert, note)
            )
            conn.commit()
            conn.close()
            self.push_update()

    def _log_occupancy(self, m_key, m_state):
        """利用が終わった占有を occupancy_logs に記録する（self._lock を持った状態で呼ぶ）"""
        if m_state["user"] == "-" or m_state["start_time"] is None or m_state["duration"] <= 0:
            return
        start = datetime.fromtimestamp(m_state["start_time"])
        end = datetime.fromtimestamp(m_state["start_time"] + m_state["duration"])
        who = m_state.get("member_id") or m_state["user"]   # 会員IDが不明なら名前で残す
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute(
                "INSERT INTO occupancy_logs (member_id, start_time, end_time, duration, machine) VALUES (?, ?, ?, ?, ?)",
                (who, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"),
                 float(m_state["duration"]), m_key)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ 占有ログの保存に失敗: {e}")

    def update_machine_occupancy(self, active_machine_key, user_name, current_time, started_at=None, member_id=None):
        with self._lock:
            for m_key in self.machine_states.keys():
                if m_key == active_machine_key and user_name != "Guest":
                    m_state = self.machine_states[m_key]
                    m_state["last_seen"] = current_time
                    if m_state["user"] != user_name:
                        self._log_occupancy(m_key, m_state)   # 別の人に替わったら、前の人の利用を記録
                        m_state["user"] = user_name
                        m_state["member_id"] = member_id
                        m_state["start_time"] = started_at if started_at is not None else current_time
                        m_state["duration"] = current_time - m_state["start_time"]
                    else:
                        if member_id is not None:
                            m_state["member_id"] = member_id
                        if m_state["start_time"] is not None:
                            m_state["duration"] = current_time - m_state["start_time"]

                    limit = MACHINE_LIMITS[m_key]["limit"]
                    if m_state["duration"] > limit:
                        m_state["status"] = "overtime"
                    else:
                        m_state["status"] = "using"

    def release_idle_machines(self, current_time):
        """座っている検知が SEAT_RELEASE_GRACE 秒途切れたら利用終了（記録して空きに戻す）。戻ってきたら0秒から計測し直す"""
        with self._lock:
            for m_key, m_state in self.machine_states.items():
                if m_state["user"] != "-" and m_state["last_seen"] is not None \
                        and (current_time - m_state["last_seen"]) > SEAT_RELEASE_GRACE:
                    print(f"🪑 【利用終了】 {m_state['user']} / 占有 {m_state['duration']:.0f}秒")
                    self._log_occupancy(m_key, m_state)
                    m_state.update({"user": "-", "duration": 0.0, "status": "free",
                                    "start_time": None, "last_seen": None, "member_id": None})

    def push_update(self):
        machines_payload = []
        for key, info in MACHINE_LIMITS.items():
            st = self.machine_states[key]
            machines_payload.append({
                "name": info["name"],
                "user": st["user"],
                "duration": int(st["duration"]),
                "status": st["status"]
            })

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, direction, member_id, is_alert FROM passing_logs ORDER BY id DESC LIMIT 5")
        logs_payload = [dict(row) for row in cursor.fetchall()]
        conn.close()

        with mode_lock:
            current_active_mode = active_mode

        data = {
            "active_mode": current_active_mode,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "last_qr": self.last_scanned_qr,
            "co_trailing_alert": self.co_trailing_alert,
            "user_name": "Multiple" if len(self.active_users) > 0 else "Guest",
            "accumulated_time": 0,
            "is_overtime": False,
            "machines": machines_payload,
            "logs": logs_payload
        }
        self.update_queue.put(data)

state = SystemStateManager()

current_output_frame = None
render_lock = threading.Lock()

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def judge_entry(time_since_qr, last_scanned_qr, has_embedding, best_sim):
    """入場判定。 (OKかどうか, NGの理由) を返す"""
    if last_scanned_qr == "None":
        return False, "QRが読み取られていません"
    if last_scanned_qr == "Invalid QR":
        return False, "QRが無効（未登録のトークン）"
    if time_since_qr > CO_TRAILING_WINDOW:
        return False, f"QR認証から{time_since_qr:.1f}秒経過（上限{CO_TRAILING_WINDOW}秒）"
    if not has_embedding:
        return False, "この会員は顔データが未登録"
    if best_sim is None:
        return False, "ゲート通過前に顔を検出・照合できなかった"
    if best_sim <= FACE_MATCH_THRESHOLD:
        return False, f"顔の類似度が不足（最大{best_sim:.2f} ≤ しきい値{FACE_MATCH_THRESHOLD}）"
    return True, ""

def sit_overlap_ratio(p_box, seat_boxes):
    """人の下半身（ボックスの下半分）が椅子ボックスと重なっている割合の最大値（0〜1）"""
    px1, py1, px2, py2 = p_box
    mid_y = py1 + (py2 - py1) * 0.5
    lower_area = max(1.0, (px2 - px1) * (py2 - mid_y))
    best = 0.0
    for sx1, sy1, sx2, sy2 in seat_boxes:
        ix1, iy1 = max(px1, sx1), max(mid_y, sy1)
        ix2, iy2 = min(px2, sx2), min(py2, sy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        best = max(best, inter / lower_area)
    return best

class SitConfirmer:
    """座っている状態が SIT_CONFIRM_SECONDS 続いたら確定する（かすっただけ・一瞬の誤検知を除外）"""
    def __init__(self):
        self.since = None
        self.last_seen = 0.0

    def update(self, sitting_now, now):
        """戻り値: (確定したか, 座り始めの時刻)"""
        if not sitting_now:
            return False, self.since
        if self.since is None or (now - self.last_seen) > SIT_BREAK_TOLERANCE:
            self.since = now
        self.last_seen = now
        return (now - self.since) >= SIT_CONFIRM_SECONDS, self.since

class IdentityHolder:
    """顔認識が一瞬外れても、直前に特定した会員を保持する。座っている間は保持し続け、離れて一定時間でリセット"""
    def __init__(self):
        self.name = None
        self.member_id = None
        self.time = 0.0

    def update(self, recognized_name, recognized_id, is_sitting, now):
        """recognized_name/id: 今フレームで特定できた会員（できなければ "Guest", None）。戻り値: (使う名前, 会員ID, 保持中か)"""
        if recognized_name != "Guest":
            self.name, self.member_id, self.time = recognized_name, recognized_id, now
            return recognized_name, recognized_id, False
        if self.name is not None and (now - self.time) <= IDENTITY_HOLD_SECONDS:
            if is_sitting:
                self.time = now
            return self.name, self.member_id, True
        self.name, self.member_id = None, None
        return "Guest", None, False

def _color_hist(region):
    """領域の色の分布。照明が変わっても崩れにくいよう、「色み(r,g比)」と「明るさ」を分けて数える。
    割合にしてから平方根を取る → 内積が類似度(0〜1)になる（長さ1）"""
    px = region.reshape(-1, 3).astype(np.float32)       # B, G, R
    total_rgb = px.sum(axis=1)
    bright = px.max(axis=1)
    bright_hist = np.histogram(bright, bins=4, range=(0, 256))[0].astype(np.float32)

    dark = total_rgb < 120                               # 暗すぎる画素は色みが不安定なので別枠
    chroma = np.zeros(5 * 5 + 1, np.float32)
    chroma[-1] = dark.sum()
    ok = ~dark
    if ok.any():
        r = px[ok, 2] / total_rgb[ok]
        g = px[ok, 1] / total_rgb[ok]
        ir = np.minimum((r * 5).astype(int), 4)
        ig = np.minimum((g * 5).astype(int), 4)
        chroma[:25] = np.bincount(ir * 5 + ig, minlength=25)
    if chroma.sum() <= 0:
        return None
    chroma /= chroma.sum()
    bright_hist /= max(bright_hist.sum(), 1.0)
    return np.concatenate([np.sqrt(0.75) * np.sqrt(chroma), np.sqrt(0.25) * np.sqrt(bright_hist)]).astype(np.float32)

def extract_body_feature(frame, box):
    """人の枠から、上半身(胴)と下半身の服の色の分布を特徴量にする。OpenCVだけで動き、モデルのファイルは不要
    背景が入りにくいよう、枠の左右の端は使わない。上半身60%・下半身40%の重みで、長さ1に正規化"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    w, h = x2 - x1, y2 - y1
    if w < REID_MIN_W or h < REID_MIN_H:
        return None
    cx1, cx2 = x1 + int(w * 0.2), x2 - int(w * 0.2)
    upper = frame[y1 + int(h * 0.15):y1 + int(h * 0.5), cx1:cx2]
    lower = frame[y1 + int(h * 0.5):y1 + int(h * 0.9), cx1:cx2]
    if upper.size == 0 or lower.size == 0:
        return None
    hu, hl = _color_hist(upper), _color_hist(lower)
    if hu is None or hl is None:
        return None
    return np.concatenate([np.sqrt(0.6) * hu, np.sqrt(0.4) * hl]).astype(np.float32)

class ReIDGallery:
    """会員ごとの全身特徴量を保持する（期限切れは使わない）"""
    def __init__(self):
        self.items = {}   # member_id -> {"name", "feats", "time"}

    def add(self, member_id, name, feat, now):
        e = self.items.setdefault(member_id, {"name": name, "feats": [], "time": now})
        e["name"], e["time"] = name, now
        e["feats"].append(feat)
        if len(e["feats"]) > REID_MAX_FEATS:
            e["feats"].pop(0)

    def identify(self, feat, now):
        """戻り値: (member_id, 名前, 類似度, 2位との差)。保持している会員がいなければ None"""
        scores = []
        for mid, e in self.items.items():
            if now - e["time"] > REID_TTL_SECONDS:
                continue
            scores.append((max(float(np.dot(feat, f)) for f in e["feats"]), mid, e["name"]))
        if not scores:
            return None
        scores.sort(reverse=True)
        margin = scores[0][0] - scores[1][0] if len(scores) > 1 else 1.0
        return scores[0][1], scores[0][2], scores[0][0], margin

class BodyReID:
    """顔で特定できないときに、服の色で「座っている人」が誰かを推定する"""
    def __init__(self):
        self.gallery = ReIDGallery()
        self.last_add = 0.0
        self.last_try = 0.0
        self.result = None
        self.result_time = 0.0
        self.logged_name = None

    def add_entrance(self, frame, box, member_id, name, now):
        """入口で、QRと顔が一致した人の服の色を登録"""
        if member_id is None:
            return
        feat = extract_body_feature(frame, box)
        if feat is not None:
            self.gallery.add(member_id, name, feat, now)

    def resolve(self, frame, sit_box, is_sitting, face_name, face_id, face_center, now):
        """戻り値: (使う名前, 会員ID, Re-IDで決めたか, 画面表示用の文字列)"""
        if not is_sitting or sit_box is None:
            self.result = None
            return face_name, face_id, False, ""

        # 顔の持ち主が「座っている人」と別人なら、その顔の名前は使わない
        if face_name != "Guest":
            inside = (face_center is not None
                      and sit_box[0] <= face_center[0] <= sit_box[2]
                      and sit_box[1] <= face_center[1] <= sit_box[3])
            if not inside:
                face_name, face_id = "Guest", None

        if face_name != "Guest":
            # 顔で特定できている間に、座った姿の服の色を追加（一定間隔）
            if face_id is not None and now - self.last_add >= REID_ADD_INTERVAL:
                self.last_add = now
                feat = extract_body_feature(frame, sit_box)
                if feat is not None:
                    self.gallery.add(face_id, face_name, feat, now)
            return face_name, face_id, False, ""

        # 顔で特定できない：服の色で推定（一定間隔）
        if now - self.last_try >= REID_INTERVAL:
            self.last_try = now
            feat = extract_body_feature(frame, sit_box)
            self.result = self.gallery.identify(feat, now) if feat is not None else None
            self.result_time = now
        if self.result is not None and now - self.result_time <= REID_INTERVAL * 3:
            mid, name, score, margin = self.result
            text = f"color={score:.2f}({name}) margin={margin:.2f}"
            if score >= REID_THRESHOLD and margin >= REID_MARGIN:
                if self.logged_name != name:
                    print(f"🧍 服の色で特定: {name} (score={score:.2f}, margin={margin:.2f})")
                    self.logged_name = name
                return name, mid, True, text
            return "Guest", None, False, text
        return "Guest", None, False, ""

def _box_inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

def _box_area(a):
    return max(1.0, (a[2] - a[0]) * (a[3] - a[1]))

def update_seat_memory(memory, detected, person_boxes, now):
    """椅子の位置を覚えておく。座った人に隠れて検出できなくなっても、直前の位置を使い続ける。
    memory: [{"box", "last_seen", "live"}] を更新し、使う椅子ボックスのリストを返す"""
    for m in memory:
        m["live"] = False
    used = set()
    for m in memory:
        occluded = any(_box_inter(p, m["box"]) / _box_area(m["box"]) >= SEAT_OCCLUDED_RATIO for p in person_boxes)
        best_j, best_score = None, 0.0
        for j, d in enumerate(detected):
            if j in used:
                continue
            inter = _box_inter(m["box"], d)
            iou = inter / (_box_area(m["box"]) + _box_area(d) - inter)
            contained = inter / _box_area(d)        # 検出が(隠れて)小さくなっても、記憶の枠内なら同じ椅子とみなす
            score = max(iou, contained if contained >= 0.7 else 0.0)
            if score >= 0.3 and score > best_score:
                best_j, best_score = j, score
        if best_j is not None:
            used.add(best_j)
            m["live"] = True
            m["last_seen"] = now
            if not occluded:                         # 人に隠れていない時だけ位置を更新（一部だけ映った枠で縮まないように）
                m["box"] = detected[best_j]
        elif occluded:
            m["last_seen"] = now
    for j, d in enumerate(detected):
        if j not in used:
            memory.append({"box": d, "last_seen": now, "live": True})
    memory[:] = [m for m in memory if now - m["last_seen"] <= SEAT_MEMORY_SECONDS]
    return [m["box"] for m in memory]

def _visible_mean(kxy, kconf, idxs):
    """骨格の点のうち、見えているものの平均座標。ひとつも見えなければ None"""
    pts = [kxy[i] for i in idxs if kconf[i] >= KP_CONF and (kxy[i][0] > 0 or kxy[i][1] > 0)]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

def _hip_on_seat(hip, seat):
    """腰が椅子ボックスの内側（上端にわずかな余裕）にあるか"""
    sx1, sy1, sx2, sy2 = seat
    top = sy1 - HIP_SEAT_MARGIN * (sy2 - sy1)
    return sx1 <= hip[0] <= sx2 and top <= hip[1] <= sy2

def evaluate_sitting(box, kxy, kconf, seat_boxes):
    """座っているか判定。 (座っているか, 重なり割合, 判定根拠) を返す
    骨格(COCO): 5,6=肩 / 11,12=腰 / 13,14=膝
    """
    overlap = sit_overlap_ratio(box, seat_boxes) if seat_boxes else 0.0
    if overlap < SIT_OVERLAP_RATIO:
        return False, overlap, "overlap"

    hip = _visible_mean(kxy, kconf, (11, 12))
    if hip is not None:
        # 腰が椅子の位置になければ、前に立っているだけ・かすっただけ
        if not any(_hip_on_seat(hip, s) for s in seat_boxes):
            return False, overlap, "hip-outside"
        knee = _visible_mean(kxy, kconf, (13, 14))
        if knee is not None:
            shoulder = _visible_mean(kxy, kconf, (5, 6))
            torso = (hip[1] - shoulder[1]) if shoulder is not None else (box[3] - box[1]) * 0.3
            torso = max(torso, 1.0)
            # 立っていると膝は腰のかなり下、座ると腰と同じくらいの高さ
            if (knee[1] - hip[1]) / torso >= SIT_THIGH_DROP_RATIO:
                return False, overlap, "standing-legs"
        return True, overlap, "pose"

    # 骨格が取れないときの予備：縦長すぎる人ボックスは立っているとみなす
    h, w = box[3] - box[1], max(box[2] - box[0], 1.0)
    if h / w >= STANDING_ASPECT_RATIO:
        return False, overlap, "standing-aspect"
    return True, overlap, "box"

# =========================================================================
# 🎥 カメラ処理ループ
# =========================================================================
def camera_processing_loop():
    global current_output_frame, yolo_model
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis

    yolo_model = YOLO("yolo11n.pt")
    # ルームモード専用（.track() を一度も呼ばない＝トラッカーが付かないモデル）
    room_model = YOLO(ROOM_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)   # 人の骨格（腰・膝の位置）用。トラッカーなし
    
    print("👤 顔認識モデルを読み込み中...")
    face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ エラー: カメラを開けませんでした。")
        return

    track_history = {}
    face_sim_history = {}  # track_id -> {best, time, member_id}
    sit_confirmer = SitConfirmer()
    seat_memory = []
    identity_holder = IdentityHolder()
    body_reid = BodyReID()
    sit_log = {"key": None, "time": 0.0}
    print("\n====================================================")
    print(" 📹 カメラ処理ループが稼働しました（椅子検知＋座り判定モード）。")
    print("====================================================\n")

    while cap.isOpened():
        current_time = time.time()
        success, frame = cap.read()
        if not success:
            time.sleep(0.03)
            continue

        frame = cv2.resize(frame, (640, 480))
        with mode_lock:
            mode = active_mode

        if mode == "entrance":
            qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame)
            if qr_bbox is not None and len(qr_bbox) > 0:
                pts = qr_bbox[0].astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
                if qr_data:
                    if state.should_process_qr(qr_data, current_time):
                        state.process_qr(qr_data, current_time)

            faces = face_app.get(frame)

            yolo_results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
            cv2.line(frame, (LINE_X, 0), (LINE_X, 480), (255, 0, 0), 2)
            cv2.putText(frame, "GATE LINE", (LINE_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 古い顔類似度の記録を掃除
            for tid in [t for t, r in face_sim_history.items() if current_time - r["time"] > FACE_TIMEOUT * 4]:
                del face_sim_history[tid]

            if yolo_results[0].boxes.id is not None:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x_center = int((box[0] + box[2]) / 2)
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                    # QR認証済み会員の顔との類似度を毎フレーム計算し、直近(FACE_TIMEOUT秒)の最大値を保持
                    if state.last_member_embedding is not None:
                        for face in faces:
                            fcx = (face.bbox[0] + face.bbox[2]) / 2
                            fcy = (face.bbox[1] + face.bbox[3]) / 2
                            if box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3]:
                                sim = cosine_sim(state.last_member_embedding, face.embedding)
                                rec = face_sim_history.get(track_id)
                                if (rec is None or rec["member_id"] != state.last_member_id
                                        or current_time - rec["time"] > FACE_TIMEOUT):
                                    face_sim_history[track_id] = {"best": sim, "time": current_time, "member_id": state.last_member_id}
                                elif sim > rec["best"]:
                                    rec["best"] = sim
                                    rec["time"] = current_time

                    rec = face_sim_history.get(track_id)
                    if rec is not None and current_time - rec["time"] <= FACE_TIMEOUT:
                        cv2.putText(frame, f"sim:{rec['best']:.2f}", (int(box[0]), max(15, int(box[1]) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    if track_id in track_history:
                        prev_x = track_history[track_id]
                        if prev_x < LINE_X and x_center >= LINE_X:
                            time_since_qr = current_time - state.last_qr_time
                            rec = face_sim_history.get(track_id)
                            best_sim = None
                            if (rec is not None and rec["member_id"] == state.last_member_id
                                    and current_time - rec["time"] <= FACE_TIMEOUT):
                                best_sim = rec["best"]

                            ok, reason = judge_entry(
                                time_since_qr, state.last_scanned_qr,
                                state.last_member_embedding is not None, best_sim
                            )
                            if ok:
                                print(f"✅ 【入場OK】 {state.last_scanned_qr} / 顔の類似度 {best_sim:.2f}")
                                state.register_pass("IN", state.last_scanned_qr, 0, note=f"QRと顔が一致（顔の類似度 {best_sim:.2f}）")

                                body_reid.add_entrance(frame, box, state.last_member_id,
                                                       state.last_scanned_qr.split(" (")[0], current_time)
                            else:
                                print(f"🚨 【入場警告】 {reason}")
                                state.set_alert(True)
                                state.register_pass("IN", "Unknown / Face Mismatch", 1, note=reason)
                        elif prev_x > LINE_X and x_center <= LINE_X:
                            state.register_pass("OUT", "Unknown", 0)
                            state.set_alert(False)

                    track_history[track_id] = x_center

            cv2.putText(frame, "MODE: [ENTRANCE (QR + 1:1 Face)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"IN: {state.in_count} | OUT: {state.out_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            # ＝ トレーニングルームモード（椅子検知 ＋ 座り判定）＝
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT member_id, name, face_embedding FROM members WHERE face_embedding IS NOT NULL")
            registered_members = cursor.fetchall()
            conn.close()

            faces = face_app.get(frame)
            detected_user_name = "Guest"
            detected_member_id = None
            matched_face_center = None

            # 1. 顔による1:Nマッチング
            for face in faces:
                highest_sim = 0.0
                best_match_name = "Guest"
                best_match_id = None

                for member in registered_members:
                    db_embedding = np.frombuffer(member['face_embedding'], dtype=np.float32)
                    sim = np.dot(db_embedding, face.embedding) / (np.linalg.norm(db_embedding) * np.linalg.norm(face.embedding))
                    if sim > highest_sim:
                        highest_sim = sim
                        best_match_name = member['name']
                        best_match_id = member['member_id']

                if highest_sim > FACE_MATCH_THRESHOLD:
                    detected_user_name = best_match_name
                    detected_member_id = best_match_id
                    matched_face_center = ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)
                    break

            # 2. 椅子と人（骨格つき）を検出。どちらもトラッカーなしのルーム専用モデルで、1フレーム1回
            detected_seats = []
            seat_det = room_model(frame, classes=SEAT_CLASSES, conf=ROOM_CONF, verbose=False)[0]
            if seat_det.boxes is not None and len(seat_det.boxes) > 0:
                detected_seats = list(seat_det.boxes.xyxy.cpu().numpy())

            persons = []   # (人ボックス, 骨格の座標, 骨格の信頼度)
            pose_det = pose_model(frame, conf=0.3, verbose=False)[0]
            if pose_det.boxes is not None and len(pose_det.boxes) > 0 and pose_det.keypoints is not None:
                p_boxes = pose_det.boxes.xyxy.cpu().numpy()
                kxy_all = pose_det.keypoints.xy.cpu().numpy()
                kcf_all = (pose_det.keypoints.conf.cpu().numpy() if pose_det.keypoints.conf is not None
                           else np.ones(kxy_all.shape[:2]))
                persons = [(p_boxes[i], kxy_all[i], kcf_all[i]) for i in range(len(p_boxes))]
            person_boxes = [p[0] for p in persons]

            # 椅子は、座った人に隠れて検出が途切れても、直前の位置を覚えて使う
            seat_boxes = update_seat_memory(seat_memory, detected_seats, person_boxes, current_time)

            # 座り判定（椅子との重なり＋腰の位置＋脚の曲がり）
            best = (False, -1.0, "no-person")
            sit_box = None
            for box, kxy_i, kcf_i in persons:
                res = evaluate_sitting(box, kxy_i, kcf_i, seat_boxes)
                if (res[0], res[1]) > (best[0], best[1]):
                    best = res
                    sit_box = box
            is_sitting, overlap, why = best
            active_machine_key = "bench_press"  # デフォルトのマシン

            # 顔の持ち主が座っている人と別人なら名前を使わず、顔で特定できないときは服の色で推定
            detected_user_name, detected_member_id, reid_used, reid_text = body_reid.resolve(
                frame, sit_box, is_sitting, detected_user_name, detected_member_id, matched_face_center, current_time)

            # デバッグ表示：人=緑、椅子=オレンジ、腰・膝=赤い点
            for box, kxy_i, kcf_i in persons:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                for idx in (11, 12, 13, 14):
                    if kcf_i[idx] >= KP_CONF:
                        cv2.circle(frame, (int(kxy_i[idx][0]), int(kxy_i[idx][1])), 4, (0, 0, 255), -1)
            for m in seat_memory:
                b = m["box"]
                color = (0, 165, 255) if m["live"] else (0, 255, 255)   # 検出中=オレンジ / 記憶のみ=黄
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
            n_live = sum(1 for m in seat_memory if m["live"])
            cv2.putText(frame, f"persons={len(persons)} seats={n_live}(+{len(seat_memory) - n_live} remembered) "
                               f"overlap={max(overlap, 0.0):.2f} why={why}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if reid_text:
                cv2.putText(frame, reid_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # 判定が変わったらターミナルにも出す（1秒に1回まで）
            log_key = (is_sitting, why)
            if log_key != sit_log["key"] and current_time - sit_log["time"] >= 1.0:
                print(f"🪑 座り判定: {is_sitting} (why={why}, overlap={max(overlap, 0.0):.2f}, "
                      f"persons={len(persons)}, seats={n_live}+{len(seat_memory) - n_live}記憶)")
                sit_log["key"], sit_log["time"] = log_key, current_time

            # 顔認識が外れても、直前に特定した会員を保持（座っている間は保持し続ける）
            detected_user_name, detected_member_id, identity_held = identity_holder.update(
                detected_user_name, detected_member_id, is_sitting, current_time)
            held_mark = " (face lost)" if identity_held else (" (color)" if reid_used else "")

            # 5. 会員が検知され、座っている状態が SIT_CONFIRM_SECONDS 続いたときだけタイマー開始
            sitting_now = detected_user_name != "Guest" and is_sitting
            confirmed, sit_since = sit_confirmer.update(sitting_now, current_time)
            if confirmed:
                state.update_machine_occupancy(active_machine_key, detected_user_name, current_time,
                                               started_at=sit_since, member_id=detected_member_id)
                cv2.putText(frame, f"Sitting & Using: {detected_user_name}{held_mark}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif sitting_now:
                cv2.putText(frame, f"Sitting? confirming {current_time - sit_since:.1f}/{SIT_CONFIRM_SECONDS:.1f}s",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                status_text = f"User: {detected_user_name}{held_mark} | Sitting: {is_sitting}"
                cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            state.release_idle_machines(current_time)
            state.push_update()
            cv2.putText(frame, "MODE: [ROOM (Chair Occupancy Mode)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        with render_lock:
            current_output_frame = frame.copy()

        time.sleep(0.01)

    cap.release()

def console_switcher_thread():
    global active_mode
    while True:
        try:
            input()
            with mode_lock:
                if active_mode == "entrance":
                    active_mode = "room"
                    print("\n🔄 【モード切替】 ➔ 【トレーニングルームモード】")
                else:
                    active_mode = "entrance"
                    print("\n🔄 【モード切替】 ➔ 【入口ゲートモード】")
            state.push_update()
        except Exception:
            break

# =========================================================================
# 🚀 FastAPI ルーティング & API設定
# =========================================================================
connected_websockets = []

@app.get("/register", response_class=HTMLResponse)
def get_register_page(request: Request):
    return templates.TemplateResponse(request, "register.html", {})

@app.post("/api/register")
def api_register(data: RegisterRequest):
    try:
        # メールアドレスを正規化（全角→半角、前後の空白除去）。SMTPは半角ASCIIしか扱えない
        email = unicodedata.normalize("NFKC", data.email).strip()
        if not email.isascii() or "@" not in email:
            return {"status": "error", "message": "メールアドレスは半角英数字で入力してください。"}

        header, encoded = data.image.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        faces = face_app.get(frame)

        if len(faces) == 0:
            return {"status": "error", "message": "顔が検出されませんでした。もう一度撮影してください。"}
        
        face_embedding = faces[0].embedding.tobytes()

        member_id = "mem_" + uuid.uuid4().hex[:8]
        qr_token = "token_" + uuid.uuid4().hex

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO members (member_id, name, email, qr_token, face_embedding) VALUES (?, ?, ?, ?, ?)",
            (member_id, data.name, email, qr_token, sqlite3.Binary(face_embedding))
        )
        conn.commit()
        conn.close()

        import qrcode
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_token)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)

        msg = MIMEMultipart()
        msg['Subject'] = '【ジムセキュリティ】会員登録完了と専用QRコードのお知らせ'
        msg['From'] = SENDER_EMAIL
        msg['To'] = email

        body = MIMEText(f"{data.name} 様\n\nジムの新規ご登録ありがとうございます。\nあなた専用の入館用QRコードを発行いたしました。")
        msg.attach(body)

        img_attachment = MIMEImage(img_io.read(), name="gym_qr_code.png")
        msg.attach(img_attachment)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())

        return {"status": "success", "member_id": member_id}

    except Exception as e:
        print(f"登録エラー: {e}")
        traceback.print_exc()  # どの行で失敗したかをターミナルに出す
        return {"status": "error", "message": str(e)}

@app.get("/video")
async def video_feed():
    async def generate_async_stream():
        while True:
            with render_lock:
                frame_to_send = current_output_frame
            if frame_to_send is not None:
                ret, buffer = cv2.imencode('.jpg', frame_to_send)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.04)

    return StreamingResponse(
        generate_async_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close"
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

async def ws_broadcast_loop():
    while True:
        while not state.update_queue.empty():
            data = state.update_queue.get()
            for ws in connected_websockets:
                try:
                    await ws.send_text(json.dumps(data))
                except Exception:
                    pass
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ws_broadcast_loop())

# ==========================================
# 🏠 メインダッシュボード（ログイン保護付き）
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    
    return templates.TemplateResponse(request, "index.html", {})

# ==========================================
# 🔑 管理者認証・ログイン/ログアウトAPI
# ==========================================
@app.post("/api/admin/login")
def api_admin_login(request: Request, data: AdminAuthRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE username = ? AND password = ?", (data.username, data.password))
    user = cursor.fetchone()
    conn.close()

    if user:
        request.session["admin_user"] = data.username
        return {"status": "success"}
    else:
        return {"status": "error", "message": "IDまたはパスワードが違います"}

@app.get("/admin/logout")
async def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/admin/login", status_code=303)

# =========================================================================
# 👥 管理者用：会員管理・一覧用エンドポイント
# =========================================================================
@app.get("/admin/members", response_class=HTMLResponse)
def get_admin_members_page(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "members.html", {})

@app.get("/api/admin/members")
def api_get_members():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT member_id, name, email, qr_token FROM members")
    members = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"members": members}

@app.delete("/api/admin/members/{member_id}")
def api_delete_member(member_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM members WHERE member_id = ?", (member_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"会員 {member_id} を削除しました。"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/admin/members/{member_id}/detail")
def api_get_member_detail(member_id: str, request: Request):
    """会員ごとの詳細：基本情報、入館履歴、マシンの利用履歴、集計"""
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT member_id, name, email, face_embedding IS NOT NULL AS has_face FROM members WHERE member_id = ?",
                   (member_id,))
    member = cursor.fetchone()
    if member is None:
        conn.close()
        return JSONResponse({"status": "error", "message": "会員が見つかりません。"}, status_code=404)

    # 入館ログには「名前 (会員ID)」の形で残っている。LIKEの特殊文字(_ %)はエスケープして検索
    escaped = member_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%({escaped})"
    visit_where = "direction = 'IN' AND is_alert = 0 AND member_id LIKE ? ESCAPE '\\'"
    cursor.execute(f"SELECT timestamp FROM passing_logs WHERE {visit_where} ORDER BY id DESC LIMIT 50", (pattern,))
    visits = [row["timestamp"] for row in cursor.fetchall()]
    cursor.execute(f"SELECT COUNT(*) AS c, MAX(timestamp) AS last FROM passing_logs WHERE {visit_where}", (pattern,))
    v = cursor.fetchone()

    cursor.execute("SELECT machine, start_time, end_time, duration FROM occupancy_logs WHERE member_id = ? ORDER BY id DESC LIMIT 50",
                   (member_id,))
    usages = []
    for row in cursor.fetchall():
        info = MACHINE_LIMITS.get(row["machine"], {})
        usages.append({
            "machine": info.get("name", row["machine"] or "-"),
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration": row["duration"],
            "overtime": bool(info) and (row["duration"] or 0) > info["limit"],
        })
    cursor.execute("SELECT COUNT(*) AS c, COALESCE(SUM(duration), 0) AS total FROM occupancy_logs WHERE member_id = ?", (member_id,))
    u = cursor.fetchone()
    conn.close()

    # 今、マシンを使っているか
    current = None
    for key, m_state in state.machine_states.items():
        if m_state.get("member_id") == member_id and m_state["user"] != "-":
            current = {"machine": MACHINE_LIMITS[key]["name"], "duration": m_state["duration"], "status": m_state["status"]}

    return {
        "status": "success",
        "member": {"member_id": member["member_id"], "name": member["name"], "email": member["email"],
                   "has_face": bool(member["has_face"])},
        "summary": {"visit_count": v["c"], "last_visit": v["last"],
                    "usage_count": u["c"], "total_usage_seconds": u["total"]},
        "current": current,
        "visits": visits,
        "usages": usages,
    }

# =========================================================================
# 📋 管理者用：入退館・違反ログ一覧エンドポイント
# =========================================================================
@app.get("/admin/logs", response_class=HTMLResponse)
def get_admin_logs_page(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "logs.html", {})

def _member_label(names, raw):
    """会員ID(または名前)を「名前 (会員ID)」の表示に。会員が削除済みなら、そのまま返す"""
    if raw in names:
        return f"{names[raw]} ({raw})"
    return raw

def _member_id_from_label(label):
    """入館ログの「名前 (会員ID)」から会員IDを取り出す。取れなければ None"""
    m = re.match(r"^.*\(([^()]+)\)$", label or "")
    return m.group(1) if m else None

def _member_stats(cursor, member_id):
    """会員の基本情報と集計。会員がいなければ None"""
    if not member_id:
        return None
    cursor.execute("SELECT member_id, name, email, face_embedding IS NOT NULL AS has_face FROM members WHERE member_id = ?",
                   (member_id,))
    m = cursor.fetchone()
    if m is None:
        return None
    escaped = member_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    cursor.execute("SELECT COUNT(*) AS c, MAX(timestamp) AS last FROM passing_logs "
                   "WHERE direction = 'IN' AND is_alert = 0 AND member_id LIKE ? ESCAPE '\\'", (f"%({escaped})",))
    v = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) AS c, COALESCE(SUM(duration), 0) AS total FROM occupancy_logs WHERE member_id = ?", (member_id,))
    u = cursor.fetchone()
    return {"member_id": m["member_id"], "name": m["name"], "email": m["email"], "has_face": bool(m["has_face"]),
            "visit_count": v["c"], "last_visit": v["last"], "usage_count": u["c"], "total_usage_seconds": u["total"]}

def _fmt_seconds(sec):
    sec = int(round(sec or 0))
    return f"{sec // 60}分{sec % 60}秒" if sec >= 60 else f"{sec}秒"

# main.py の適当なルーティング付近（例: @app.get("/", ... の下など）に追加
@app.get("/live", response_class=HTMLResponse)
async def read_live(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "live.html", {})

@app.get("/api/admin/logs")
def api_get_logs(request: Request):
    """ゲートの入退館ログと、マシンの占有ログ（現在利用中を含む）を新しい順にまとめて返す"""
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT member_id, name FROM members")
    names = {row["member_id"]: row["name"] for row in cursor.fetchall()}

    logs = []
    cursor.execute("SELECT id, timestamp, direction, member_id, is_alert FROM passing_logs ORDER BY id DESC LIMIT 100")
    for r in cursor.fetchall():
        logs.append({"kind": "gate", "ref": str(r["id"]), "display_id": f"G-{r['id']}", "type": "gate",
                     "timestamp": r["timestamp"], "event": r["direction"], "member": r["member_id"],
                     "status": "alert" if r["is_alert"] == 1 else "normal", "duration": None})

    cursor.execute("SELECT id, member_id, machine, start_time, duration FROM occupancy_logs ORDER BY id DESC LIMIT 100")
    for r in cursor.fetchall():
        info = MACHINE_LIMITS.get(r["machine"], {})
        over = bool(info) and (r["duration"] or 0) > info["limit"]
        logs.append({"kind": "machine", "ref": str(r["id"]), "display_id": f"M-{r['id']}", "type": "machine",
                     "timestamp": r["start_time"], "event": info.get("name", r["machine"] or "-"),
                     "member": _member_label(names, r["member_id"]),
                     "status": "overtime" if over else "normal", "duration": r["duration"]})
    conn.close()
    logs.sort(key=lambda x: x["timestamp"] or "", reverse=True)
    logs = logs[:100]

    # 現在利用中のマシン（まだ記録されていないので、状態から作る）。先頭に並べる
    live = []
    for key, m in state.machine_states.items():
        if m["user"] != "-" and m["start_time"] is not None:
            live.append({"kind": "active", "ref": key, "display_id": "LIVE", "type": "machine",
                         "timestamp": datetime.fromtimestamp(m["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                         "event": MACHINE_LIMITS[key]["name"],
                         "member": _member_label(names, m.get("member_id") or m["user"]),
                         "status": "overtime" if m["status"] == "overtime" else "active",
                         "duration": m["duration"]})
    return {"status": "success", "logs": live + logs}

@app.get("/api/admin/logs/{kind}/{ref}")
def api_get_log_detail(kind: str, ref: str, request: Request):
    """ログ1件の詳細。kind: gate（入退館）/ machine（占有の記録）/ active（現在利用中）"""
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)
    not_found = JSONResponse({"status": "error", "message": "ログが見つかりません。"}, status_code=404)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    fields, title, member_id = [], "", None

    if kind in ("gate", "machine"):
        try:
            log_id = int(ref)
        except ValueError:
            conn.close()
            return not_found

    if kind == "gate":
        cursor.execute("SELECT id, timestamp, direction, member_id, is_alert, note FROM passing_logs WHERE id = ?", (log_id,))
        r = cursor.fetchone()
        if r is None:
            conn.close()
            return not_found
        member_id = _member_id_from_label(r["member_id"])
        title = f"{'入館' if r['direction'] == 'IN' else '退館'}ログ G-{r['id']}"
        fields = [
            {"label": "日時", "value": r["timestamp"]},
            {"label": "種別", "value": "入館 (IN)" if r["direction"] == "IN" else "退館 (OUT)"},
            {"label": "記録された会員", "value": r["member_id"]},
            {"label": "結果", "value": "警告" if r["is_alert"] == 1 else "正常",
             "tone": "danger" if r["is_alert"] == 1 else "ok"},
            {"label": "判定の理由", "value": r["note"] or "（記録なし：この機能の追加前のログです）"},
        ]
    elif kind == "machine":
        cursor.execute("SELECT id, member_id, machine, start_time, end_time, duration FROM occupancy_logs WHERE id = ?", (log_id,))
        r = cursor.fetchone()
        if r is None:
            conn.close()
            return not_found
        member_id = r["member_id"]
        info = MACHINE_LIMITS.get(r["machine"], {})
        over = bool(info) and (r["duration"] or 0) > info["limit"]
        title = f"マシン利用ログ M-{r['id']}"
        fields = [
            {"label": "マシン", "value": info.get("name", r["machine"] or "-")},
            {"label": "開始", "value": r["start_time"]},
            {"label": "終了", "value": r["end_time"]},
            {"label": "利用時間", "value": _fmt_seconds(r["duration"])},
            {"label": "上限", "value": _fmt_seconds(info["limit"]) if info else "-"},
            {"label": "判定", "value": "時間超過" if over else "範囲内", "tone": "danger" if over else "ok"},
        ]
    elif kind == "active":
        m = state.machine_states.get(ref)
        if m is None or m["user"] == "-" or m["start_time"] is None:
            conn.close()
            return not_found
        member_id = m.get("member_id")
        over = m["status"] == "overtime"
        title = f"現在利用中：{MACHINE_LIMITS[ref]['name']}"
        fields = [
            {"label": "マシン", "value": MACHINE_LIMITS[ref]["name"]},
            {"label": "利用者", "value": m["user"]},
            {"label": "開始", "value": datetime.fromtimestamp(m["start_time"]).strftime("%Y-%m-%d %H:%M:%S")},
            {"label": "経過時間", "value": _fmt_seconds(m["duration"])},
            {"label": "上限", "value": _fmt_seconds(MACHINE_LIMITS[ref]["limit"])},
            {"label": "状態", "value": "時間超過" if over else "利用中", "tone": "danger" if over else "ok"},
        ]
    else:
        conn.close()
        return not_found

    member = _member_stats(cursor, member_id)
    conn.close()
    return {"status": "success", "kind": kind, "title": title, "fields": fields, "member": member}

# =========================================================================
# 🔑 管理者用：登録・ログイン画面
# =========================================================================
@app.get("/admin/register", response_class=HTMLResponse)
def get_admin_register_page(request: Request):
    return templates.TemplateResponse(request, "admin_register.html", {})

@app.post("/api/admin/register")
def api_admin_register(data: AdminAuthRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (data.username, data.password))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "このIDはすでに使われています"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/admin/login", response_class=HTMLResponse)
def get_admin_login_page(request: Request):
    return templates.TemplateResponse(request, "admin_login.html", {})

# =========================================================================
# 🏁 起動確認
# =========================================================================
if __name__ == "__main__":
    camera_thread = threading.Thread(target=camera_processing_loop, daemon=True)
    camera_thread.start()
    
    switcher_thread = threading.Thread(target=console_switcher_thread, daemon=True)
    switcher_thread.start()
    
    print("\n🚀 サーバーを椅子検知・座り判定モードで起動しました！")
    print("👉 ブラウザで http://localhost:8000/ を開いてダッシュボードを確認してください。")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")