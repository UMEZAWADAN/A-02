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
STREAM_JPEG_QUALITY = 80        # ライブ映像のJPEG品質（既定の95より少し軽くする）
CO_TRAILING_WINDOW = 10.0
LIMIT_SECONDS = 10.0
FACE_TIMEOUT = 10.0            # 顔の類似度（直近の最大値）を有効とみなす秒数
FACE_MATCH_THRESHOLD = 0.40   # 顔が一致したとみなすコサイン類似度（入口の1:1判定。ログを見て調整）
ROOM_FACE_THRESHOLD = 0.40    # ルームモードの1:N判定用。部屋のカメラは遠い・斜めで類似度が低めになりやすいので、画面の best= を見て下げる

# 📧 メール送信設定（実際の運用に合わせて書き換えてください）
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "chocozap123456@gmail.com"  # 送信元のメールアドレス
SENDER_PASSWORD = "rfrlsheyiuzhyubg"   # アプリパスワード等

# 🚨 違反メール：マシンの利用時間が上限を超えた会員に、利用の終了時にメールで違反内容を知らせる
VIOLATION_MAIL_ENABLED = True       # False にするとメールを送らない
VIOLATION_MAIL_COOLDOWN = 600.0     # 同じ会員には、この秒数のあいだ違反メールを重ねて送らない（識別の揺れで続けて送られるのを防ぐ）

# 🎞️ 映像の解像度（カメラ取得・YOLO・顔検出・ライブ表示すべてこのサイズで処理する）
FRAME_W, FRAME_H = 640, 480    # 元の解像度。高精度にしたいときは 1280, 720 などに上げる（その分CPU負荷が大きく増える）
LINE_X = FRAME_W // 2          # 入口の通過判定ライン（画面中央）
YOLO_IMGSZ = 640               # YOLOの推論サイズ（ultralyticsの既定値）。FRAME_W を上げたら合わせて上げる
FACE_DET_SIZE = (640, 640)     # 顔検出の入力サイズ（幅, 高さ）。入力が640x480なら640で十分（1280にしても精度は上がらず重くなるだけ）
UI = FRAME_W / 640.0           # 画面上の文字・線の拡大率（HUDの見た目を従来と同じ比率に保つ）
THICK = max(1, int(round(2 * UI * 0.75)))   # ボックス・ラインの太さ
CAMERA_ID = 0  # 内蔵カメラ1台

# 🪑 椅子・座り判定の設定
SEAT_CLASSES = [56]           # COCO: 56=chair（普通の椅子のみ。ベンチ等も拾いたい場合は 13=bench, 57=couch を追加）
ROOM_MODEL_PATH = "yolo11n.pt"  # 椅子の検出が不安定なら "yolo11s.pt" や "yolo11m.pt" に変更（精度↑・速度↓）
ROOM_PUSH_INTERVAL = 0.5      # ルームモードで画面へ状態を送る最小間隔（秒）。毎フレーム送るのをやめて負荷を下げる
SEAT_RELEASE_GRACE = 5.0      # 座っていない状態がこの秒数続いたら、利用終了（マシンを空きに戻す）
ROOM_CONF = 0.20              # ルームモードの検出しきい値（低めにして椅子の取りこぼしを減らす）
SIT_OVERLAP_RATIO = 0.20      # 人の下半身ボックスが椅子ボックスと重なっている割合の下限（画面の overlap= を見て調整）
SIT_CONFIRM_SECONDS = 2.0     # 「座っている」がこの秒数続いて初めてタイマー開始（椅子にかすっただけを除外）
SIT_BREAK_TOLERANCE = 1.0     # 座り判定がこの秒数以内の途切れなら、座り続けているとみなす
POSE_MODEL_PATH = "yolo11n-pose.pt"  # 骨格推定モデル（初回起動時に自動ダウンロード）
KP_CONF = 0.4                 # 骨格の点（腰・膝など）を「見えている」とみなす信頼度
SIT_THIGH_DROP_RATIO = 0.60   # (膝−腰の縦の差)÷(胴の長さ)。立つと約1.0（遠近で縮んでも0.7前後）、座ると0〜0.5。これ以上なら立っているとみなす（従来は0.75）
IDENTITY_HOLD_SECONDS = 15.0  # 顔認識が外れても、直前に特定した会員をこの秒数は保持（座っている間は保持し続ける）
SEAT_MEMORY_SECONDS = 10.0    # 椅子が検出できなくなっても、この秒数は直前の位置を覚えておく（座った人に隠れる対策）
SEAT_OCCLUDED_RATIO = 0.15    # 人が椅子ボックスの面積のこれ以上に重なっていたら「隠れている」とみなし、椅子の記憶を保持
HIP_SEAT_MARGIN = 0.10        # 腰が椅子ボックスの上端よりこの割合だけ上でも許容
HIP_SEAT_MIN_DEPTH_NOKNEE = 0.20   # 膝が見えないとき、腰は椅子ボックスの上端からこの割合より下にあること（座ると腰は椅子の中ほど、立つと上端付近）。-0.10 で従来どおり
SHOULDER_ABOVE_SEAT_MAX = 0.40     # 腰が見えない（上半身のみ）とき、肩が椅子ボックスの上端より（椅子の高さ×この割合）以上高ければ立っているとみなす。座ると約0.15、立つと約0.55
STANDING_ASPECT_RATIO = 3.0   # 骨格が取れないときの予備：人ボックスの縦÷横がこれ以上なら立っているとみなす

# 🚪 入口：「顔認証」と「線の通過」を分けて判定する設定
ENTRANCE_CROSS_MARGIN_PX = round(FRAME_W * 0.03)      # 認証済みの本人が、線からこの距離だけ向こう側（右）で見つかったら通過（入館）とする（幅の3%）
ENTRANCE_REASSOC_SECONDS = 3.0       # 本人を見失っても、この秒数以内なら直前の位置に近い枠を本人として探し直す
ENTRANCE_REASSOC_BASE_PX = round(FRAME_W * 0.12)      # 探し直しで許容する移動距離（基本。幅の12%）
ENTRANCE_REASSOC_PX_PER_SEC = round(FRAME_W * 0.20)   # 探し直しで許容する移動距離（見失った秒数あたりの追加。幅の20%）
ENTRANCE_OTHER_FACE_SIM = 0.15       # 会員との類似度がこれ未満の顔を含む枠は、別人として探し直しの候補から外す
ENTRANCE_BANNER_SECONDS = 4.0        # 認証NG・失効のメッセージを画面に出す秒数

# 🚪 退館：QRは使わず、顔だけで誰が出たかを識別する（全会員との1:N照合）
EXIT_FACE_THRESHOLD = 0.40    # 退館時の顔識別のしきい値。別人に誤認されるなら上げる、識別できないなら下げる（画面の会員ID＋類似度を見て調整）

# 👕 顔が見えないときの服装による補助推定（ルームモード）
CLOTHES_FALLBACK_ENABLED = True   # False にすると服装による推定を完全に無効化（顔だけで識別）
CLOTHES_MAX_DIST = 20.0           # 服装の色の差（Lab空間の距離。上半身・下半身の大きいほう）がこれ以上なら別人扱い。小さいほど厳しい
CLOTHES_MIN_MARGIN = 8.0          # 候補が複数いるとき、1位と2位の差がこれ未満なら決めない（曖昧なので Guest のまま）

CLOTHES_RECHECK_SECONDS = 1.5     # 服装で推定した本人の服が、入口の記録と合わない状態がこの秒数続いたら推定を解除する

RECENT_ENTRY_TTL = 900.0      # 入口で記録した服装情報を、服装による補助判定の候補に残す秒数（15分）

def put_text(img, text, org, font, fs, color, th=2, hud=True):
    """cv2.putText と同じ引数。hud=True なら位置・文字サイズ・太さを解像度に合わせて拡大する。
    人ボックスに付けるラベルなど、実座標を渡す場合は hud=False にする。"""
    x, y = (int(org[0] * UI), int(org[1] * UI)) if hud else (int(org[0]), int(org[1]))
    cv2.putText(img, text, (x, y), font, fs * UI, color, max(1, int(round(th * UI))))

def open_camera():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap

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
# 🚨 違反メール（マシンの利用時間超過）
# =========================================================================
_violation_mail_last = {}   # {member_id: 最後に送信した時刻}
_violation_mail_lock = threading.Lock()

def _send_overtime_mail(member_id, machine_key, start_str, end_str, duration):
    """時間超過した会員に、違反内容をメールで送る。送れたら True。"""
    from email.header import Header
    info = MACHINE_LIMITS.get(machine_key, {})
    limit = info.get("limit", 0.0)
    machine_name = info.get("name", machine_key)
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute("SELECT name, email FROM members WHERE member_id = ?", (member_id,)).fetchone()
        finally:
            conn.close()
    except Exception as e:
        print(f"⚠️ 違反メール：会員情報の取得に失敗: {e}")
        return False
    if row is None or not row[1]:
        print(f"⚠️ 違反メールを送れません：会員 {member_id} のメールアドレスが見つかりません")
        return False
    name, email = row

    body = (
        f"{name} 様\n\n"
        "ご利用中のマシンで、利用時間の上限を超過しましたのでお知らせします。\n\n"
        "■ 違反内容：マシンの利用時間超過\n"
        f"■ マシン：{machine_name}\n"
        f"■ 利用開始：{start_str}\n"
        f"■ 利用終了：{end_str}\n"
        f"■ 利用時間：{_fmt_seconds(duration)}\n"
        f"■ 上限：{_fmt_seconds(limit)}\n"
        f"■ 超過時間：{_fmt_seconds(duration - limit)}\n\n"
        "他の会員様がお待ちの場合がありますので、上限時間内でのご利用にご協力をお願いいたします。"
    )
    msg = MIMEMultipart()
    msg["Subject"] = Header("【ジムセキュリティ】マシン利用時間超過のお知らせ", "utf-8")
    msg["From"] = SENDER_EMAIL
    msg["To"] = email
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())
    except Exception as e:
        print(f"⚠️ 違反メールの送信に失敗（{name} / {member_id}）: {e}")
        return False
    print(f"📧 【違反メール送信】 {name}（{member_id}）宛：{machine_name} の時間超過（{_fmt_seconds(duration)} / 上限 {_fmt_seconds(limit)}）")
    return True

def notify_overtime_async(member_id, machine_key, start, end, duration):
    """時間超過を会員にメールで知らせる。別スレッドで送るので、カメラ処理は止まらない。"""
    if not VIOLATION_MAIL_ENABLED:
        return
    if not member_id:
        print("⚠️ 時間超過ですが会員IDが分からないため、違反メールは送りません")
        return
    now = time.time()
    with _violation_mail_lock:
        last = _violation_mail_last.get(member_id)
        if last is not None and now - last < VIOLATION_MAIL_COOLDOWN:
            print(f"ℹ️ 違反メールは直近に送信済みのためスキップ: {member_id}")
            return
        _violation_mail_last[member_id] = now

    def worker():
        ok = _send_overtime_mail(member_id, machine_key, start.strftime("%Y-%m-%d %H:%M:%S"),
                                 end.strftime("%Y-%m-%d %H:%M:%S"), duration)
        if not ok:   # 送れなかったときは、次の機会に再送できるよう記録を戻す
            with _violation_mail_lock:
                _violation_mail_last.pop(member_id, None)

    threading.Thread(target=worker, daemon=True).start()

# =========================================================================
# 💾 データベース書き込みのキュー化（ロック競合防止用）
# =========================================================================
db_write_queue = Queue()

def db_writer_worker():
    """データベースへの書き込みを1つのスレッドで直列化して実行する常駐ワーカー"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    while True:
        try:
            task = db_write_queue.get()
            if task is None:
                break
            sql, params = task
            cursor.execute(sql, params)
            conn.commit()
            db_write_queue.task_done()
        except Exception as e:
            print(f"⚠️ DB書き込みエラー: {e}")
            try:
                conn.rollback()
            except:
                pass

# アプリ起動時に専用スレッドを開始
threading.Thread(target=db_writer_worker, daemon=True).start()

def execute_db_write(sql, params=()):
    """データベースへ書き込む際は、この関数を呼び出す（キューに追加する）"""
    db_write_queue.put((sql, params))

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
        
        # 館内にいる会員（QR＋顔で正規に入館し、まだ退館していない人）。退館時の識別はこの人たちに限る
        self.inside_members = {}   # {member_id: 入館時刻}
        self.last_clothes_dist = None   # 直近の服装推定で採用した色の差（診断表示用）
        self._clothes_log_time = 0.0
        self.clothes_diag = "not evaluated"   # 直近の服装照合の結果（画面に表示。cv2は日本語を描けないので英数字）
        self._restore_inside_members()

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
        self.recent_entries = [] # 直近の入室者リスト

    def _restore_inside_members(self):
        """再起動で「館内にいる人」が消えないよう、直近24時間のログから復元する。"""
        try:
            from datetime import timedelta
            since = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT timestamp, direction, member_id, is_alert FROM passing_logs WHERE timestamp >= ? ORDER BY id",
                    (since,)).fetchall()
            finally:
                conn.close()
        except Exception as e:
            print(f"⚠️ 館内の会員の復元に失敗: {e}")
            return
        for ts, direction, label, alert in rows:
            m = re.match(r"^.*\(([^()]+)\)$", label or "")
            if not m:
                continue
            if direction == "IN" and not alert:
                self.inside_members[m.group(1)] = ts
            elif direction == "OUT":
                self.inside_members.pop(m.group(1), None)

    def mark_inside(self, member_id):
        """正規に入館した会員を「館内にいる」として記録する。"""
        with self._lock:
            self.inside_members[member_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def leave_if_inside(self, member_id):
        """館内にいる会員なら退館扱いにして True。入館記録がなければ False。"""
        with self._lock:
            return self.inside_members.pop(member_id, None) is not None

    def add_recent_entry(self, member_id, name, color_feat, current_time):
        with self._lock:
            self.recent_entries = [e for e in self.recent_entries if current_time - e["time"] <= RECENT_ENTRY_TTL]
            self.recent_entries.append({
                "member_id": member_id,
                "name": name,
                "color_feat": color_feat,
                "time": current_time
            })

    def match_recent_entry_by_feature(self, kxy, kconf, frame, current_time):
        """顔が見えない人を、入口で記録した服装から推定する。
        ・上半身と下半身の色分布が、どちらも十分近い会員だけを候補にする。
        ・候補が複数いて1位と2位の差が小さいときは、曖昧なので決めない。"""
        if not CLOTHES_FALLBACK_ENABLED:
            self.clothes_diag = "off"
            return None, None
        with self._lock:
            # 古い記録は、新しい入館者がいなくても照合の時点で候補から外す
            candidates = [e for e in self.recent_entries if current_time - e["time"] <= RECENT_ENTRY_TTL]
            if not candidates:
                self.clothes_diag = "no clothes record (not captured at entrance, or >15min)"
                return None, None

            target = get_box_color_feature(frame, kxy, kconf)
            if target is None:
                self.clothes_diag = "no torso color (shoulders not seen / mostly skin)"
                return None, None

            # 会員ごとに一番近い記録を使う（同じ人が複数回入館していても、1位と2位の比較に影響させない）
            per_member = {}
            for entry in candidates:
                d = clothes_distance(entry["color_feat"], target)
                cur = per_member.get(entry["member_id"])
                if cur is None or d < cur[0]:
                    per_member[entry["member_id"]] = (d, entry)
            ranked = sorted(per_member.values(), key=lambda x: x[0])
            best_d, best = ranked[0]
            second_d = ranked[1][0] if len(ranked) > 1 else None

            self.clothes_diag = (f"cand={len(ranked)} best={best['member_id']} d={best_d:.1f}"
                                 + (f" 2nd={second_d:.1f}" if second_d is not None else ""))
            reason = None
            if best_d >= CLOTHES_MAX_DIST:
                reason = f"服装が違う（最小の差 {best_d:.2f} ≥ 上限 {CLOTHES_MAX_DIST}）"
            elif second_d is not None and (second_d - best_d) < CLOTHES_MIN_MARGIN:
                reason = f"服装が似た候補が複数いて決められない（差 {best_d:.2f} / {second_d:.2f}）"

            if reason is not None:
                self.clothes_diag += " -> reject"
                if current_time - self._clothes_log_time >= 3.0:   # 毎フレームは出さない
                    print(f"👕 服装による推定は見送り：{reason}")
                    self._clothes_log_time = current_time
                return None, None

            self.clothes_diag += " -> match"
            self.last_clothes_dist = best_d
            return best["name"], best["member_id"]

    def clothes_distance_to_member(self, member_id, kxy, kconf, frame, current_time):
        """この会員の入口での服装記録のうち、いまの人の服装に最も近いものとの差。比べられなければ None。"""
        with self._lock:
            feats = [e["color_feat"] for e in self.recent_entries
                     if e["member_id"] == member_id and current_time - e["time"] <= RECENT_ENTRY_TTL]
        if not feats:
            return None
        target = get_box_color_feature(frame, kxy, kconf)
        if target is None:
            return None
        return min(clothes_distance(f, target) for f in feats)

    def should_process_qr(self, qr_data: str, now: float) -> bool:
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
            
            execute_db_write(
                "INSERT INTO passing_logs (timestamp, direction, member_id, is_alert, note) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), direction, member_id, is_alert, note)
           )
        self.push_update()

    def _log_occupancy(self, m_key, m_state):
        if m_state["user"] == "-" or m_state["start_time"] is None or m_state["duration"] <= 0:
            return
        start = datetime.fromtimestamp(m_state["start_time"])
        end = datetime.fromtimestamp(m_state["start_time"] + m_state["duration"])
        who = m_state.get("member_id") or m_state["user"]
        try:
            execute_db_write(
                "INSERT INTO occupancy_logs (member_id, start_time, end_time, duration, machine) VALUES (?, ?, ?, ?, ?)",
                (who, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"),
                float(m_state["duration"]), m_key)
            )
        except Exception as e:
            print(f"⚠️ 占有ログの保存に失敗: {e}")
        if m_state["duration"] > MACHINE_LIMITS[m_key]["limit"]:
            notify_overtime_async(m_state.get("member_id"), m_key, start, end, m_state["duration"])

    def update_machine_occupancy(self, active_machine_key, user_name, current_time, started_at=None, member_id=None):
        with self._lock:
            for m_key in self.machine_states.keys():
                if m_key == active_machine_key and user_name != "Guest":
                    m_state = self.machine_states[m_key]
                    m_state["last_seen"] = current_time
                    if m_state["user"] != user_name:
                        self._log_occupancy(m_key, m_state)
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
        with self._lock:
            for m_key, m_state in self.machine_states.items():
                if m_state["user"] != "-" and m_state["last_seen"] is not None \
                        and (current_time - m_state["last_seen"]) > SEAT_RELEASE_GRACE:
                    print(f"🪑 【利用終了】 {m_state['user']} / 占有 {m_state['duration']:.0f}秒")
                    self._log_occupancy(m_key, m_state)
                    m_state.update({"user": "-", "duration": 0.0, "status": "free",
                                    "start_time": None, "last_seen": None, "member_id": None})

    def build_payload(self):
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

        # 「追跡中ユーザー／滞在時間／ステータス」は、利用中のマシンの状態から求める
        in_use = [m for m in machines_payload if m["status"] in ("using", "overtime")]
        if not in_use:
            room_user, room_time, room_overtime = "Guest", 0, False
        else:
            room_user = in_use[0]["user"] if len(in_use) == 1 else "Multiple"
            room_time = max(m["duration"] for m in in_use)
            room_overtime = any(m["status"] == "overtime" for m in in_use)

        return {
            "active_mode": current_active_mode,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "last_qr": self.last_scanned_qr,
            "co_trailing_alert": self.co_trailing_alert,
            "user_name": room_user,
            "accumulated_time": room_time,
            "is_overtime": room_overtime,
            "machines": machines_payload,
            "logs": logs_payload
        }

    def push_update(self):
        self.update_queue.put(self.build_payload())

state = SystemStateManager()

current_output_frame = None
render_lock = threading.Lock()

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

class MemberFaceCache:
    """顔データ付き会員の一覧をメモリに保持する（毎フレームのDB読み込みを避ける）。
    会員の登録・削除時は invalidate() で即座に破棄し、念のため ttl 秒ごとにも再読み込みする。"""
    def __init__(self, ttl=5.0):
        self._lock = threading.Lock()
        self._ttl = ttl
        self._loaded_at = 0.0
        self._dirty = True
        self._ids, self._names, self._matrix = [], [], None

    def invalidate(self):
        with self._lock:
            self._dirty = True

    def get(self):
        now = time.time()
        with self._lock:
            if self._dirty or (now - self._loaded_at) > self._ttl:
                self._reload()
                self._loaded_at = now
                self._dirty = False
            return self._ids, self._names, self._matrix

    def _reload(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT member_id, name, face_embedding FROM members WHERE face_embedding IS NOT NULL"
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:
            print(f"⚠️ 会員顔データの読み込みに失敗（前回のキャッシュを継続使用）: {e}")
            return
        ids, names, vecs, dim = [], [], [], None
        for r in rows:
            v = np.frombuffer(r["face_embedding"], dtype=np.float32)
            n = np.linalg.norm(v)
            if n == 0:
                continue
            if dim is None:
                dim = v.shape[0]
            if v.shape[0] != dim:
                continue
            ids.append(r["member_id"])
            names.append(r["name"])
            vecs.append(v / n)   # 正規化済みで保持 → 内積がそのままコサイン類似度
        self._ids, self._names = ids, names
        self._matrix = np.stack(vecs) if vecs else None

member_cache = MemberFaceCache()

def _iou(a, b):
    inter = _box_inter(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0

def find_pose_for_box(pose_result, box, min_iou=0.3):
    """骨格推定の結果から、指定した人の枠に一番重なる人の (keypoints xy, conf) を返す。見つからなければ (None, None)。"""
    if (pose_result is None or pose_result.boxes is None or len(pose_result.boxes) == 0
            or pose_result.keypoints is None):
        return None, None
    p_boxes = pose_result.boxes.xyxy.cpu().numpy()
    kxy = pose_result.keypoints.xy.cpu().numpy()
    kcf = (pose_result.keypoints.conf.cpu().numpy() if pose_result.keypoints.conf is not None
           else np.ones(kxy.shape[:2]))
    best_i, best_iou = None, min_iou
    for i in range(len(p_boxes)):
        v = _iou(box, p_boxes[i])
        if v >= best_iou:
            best_i, best_iou = i, v
    if best_i is None:
        return None, None
    return kxy[best_i], kcf[best_i]

def clothes_roi(frame_shape, kxy, kconf):
    """服の色を測る範囲（胸〜腹の中央）を、骨格の肩・腰の位置から決める。肩が見えていなければ None。
    人の枠の割合で決めると、近づいたときに顔や首（肌）を測ってしまい、服を替えても一致してしまうため。"""
    if kxy is None or kconf is None:
        return None
    if kconf[5] < KP_CONF or kconf[6] < KP_CONF:
        return None
    ls, rs = kxy[5], kxy[6]
    sw = abs(float(ls[0]) - float(rs[0]))
    if sw < FRAME_W * 0.05:      # 肩幅が極端に狭い（真横向き・遠すぎ）と、胸の範囲が決められない
        return None
    sy = (float(ls[1]) + float(rs[1])) / 2
    if kconf[11] >= KP_CONF and kconf[12] >= KP_CONF:
        hy = (float(kxy[11][1]) + float(kxy[12][1])) / 2
    else:
        hy = sy + 1.1 * sw       # 腰が見えない（机で隠れる等）ときは、肩幅から胴の長さを見積もる
    if hy - sy < 0.5 * sw:
        return None
    x1 = min(float(ls[0]), float(rs[0])) + 0.25 * sw
    x2 = max(float(ls[0]), float(rs[0])) - 0.25 * sw
    y1 = sy + 0.25 * (hy - sy)
    y2 = sy + 0.70 * (hy - sy)
    h_img, w_img = frame_shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2

def get_box_color_feature(frame, kxy, kconf):
    """服の色：胸まわりの範囲から、肌色のピクセルを除いた代表色（Lab色空間の中央値）[L,a,b]。
    骨格が取れない、または範囲のほとんどが肌色のときは None（＝服では判定しない）。"""
    r = clothes_roi(frame.shape, kxy, kconf)
    if r is None:
        return None
    roi = frame[r[1]:r[3], r[0]:r[2]]
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)   # 手・首・顔などの肌色
    keep = ~skin
    if keep.sum() < 50 or keep.sum() < 0.4 * keep.size:
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[keep].astype(np.float32)
    return np.median(lab, axis=0)

def lab_to_bgr(lab):
    bgr = cv2.cvtColor(np.uint8([[np.clip(lab, 0, 255)]]), cv2.COLOR_LAB2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_swatch(frame, slot, label, lab_feat):
    """右上に色の見本（四角）を並べる（slot 0 が右端）。測れていないときは × を描く。"""
    size, step = int(36 * UI), int(60 * UI)
    x, y = FRAME_W - (slot + 1) * step, int(10 * UI)
    if lab_feat is None:
        cv2.rectangle(frame, (x, y), (x + size, y + size), (140, 140, 140), 1)
        cv2.line(frame, (x, y), (x + size, y + size), (140, 140, 140), 1)
        cv2.line(frame, (x + size, y), (x, y + size), (140, 140, 140), 1)
    else:
        cv2.rectangle(frame, (x, y), (x + size, y + size), lab_to_bgr(lab_feat), -1)
        cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 1)
    cv2.putText(frame, str(label)[:8], (x, y + size + int(14 * UI)), cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * UI, (255, 255, 255), 1)

_CLOTHES_WEIGHT = np.array([0.5, 1.0, 1.0], dtype=np.float32)   # 明るさ(L)は照明で変わりやすいので重みを半分にする

def clothes_distance(a, b):
    """2つの服装（上半身の色）の差。大きいほど別の服。"""
    if a is None or b is None or a.shape != b.shape:
        return 255.0
    return float(np.linalg.norm((a - b) * _CLOTHES_WEIGHT))

def judge_entry(time_since_qr, last_scanned_qr, has_embedding, best_sim):
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

def identify_face_1n(embedding, member_ids, member_names, member_matrix):
    """顔の特徴量を全会員と照合し、最も似ている会員 (member_id, name, sim) を返す。
    顔データ付きの会員がいない／次元が合わないときは None。"""
    if member_matrix is None:
        return None
    fe = np.asarray(embedding, dtype=np.float32)
    fn = np.linalg.norm(fe)
    if fn == 0 or fe.shape[0] != member_matrix.shape[1]:
        return None
    sims = member_matrix @ (fe / fn)
    idx = int(np.argmax(sims))
    return member_ids[idx], member_names[idx], float(sims[idx])

def sit_overlap_ratio(p_box, seat_boxes):
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
    def __init__(self):
        self.since = None
        self.last_seen = 0.0
        self.had_gap = False   # 前回「座っている」と判定したあと、「座っていない」判定のフレームが挟まったか

    def update(self, sitting_now, now):
        if not sitting_now:
            self.had_gap = True
            return False, self.since
        # 途切れとみなすのは、「座っていない」フレームが挟まり、かつ SIT_BREAK_TOLERANCE を超えて空いたときだけ。
        # （座っている判定が連続しているのに、1フレームの処理が遅いだけでタイマーが巻き戻らないようにする）
        if self.since is None or (self.had_gap and (now - self.last_seen) > SIT_BREAK_TOLERANCE):
            self.since = now
        self.had_gap = False
        self.last_seen = now
        return (now - self.since) >= SIT_CONFIRM_SECONDS, self.since

class IdentityHolder:
    def __init__(self):
        self.name = None
        self.member_id = None
        self.time = 0.0
        self.estimated = False   # True: 顔ではなく服装から推定した本人

    def has_identity(self, now):
        """直前に特定した会員を、いま保持しているか"""
        return self.name is not None and (now - self.time) <= IDENTITY_HOLD_SECONDS

    def update(self, recognized_name, recognized_id, is_sitting, now, estimated=False):
        """戻り値: (名前, 会員ID, 保持中か, 服装による推定か)"""
        if recognized_name != "Guest":
            self.name, self.member_id, self.time, self.estimated = recognized_name, recognized_id, now, estimated
            return recognized_name, recognized_id, False, estimated
        if self.has_identity(now):
            if is_sitting:
                self.time = now
            return self.name, self.member_id, True, self.estimated
        self.name, self.member_id, self.estimated = None, None, False
        return "Guest", None, False, False

def _box_inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

def _box_area(a):
    return max(1.0, (a[2] - a[0]) * (a[3] - a[1]))

def update_seat_memory(memory, detected, person_boxes, now):
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
            contained = inter / _box_area(d)
            score = max(iou, contained if contained >= 0.7 else 0.0)
            if score >= 0.3 and score > best_score:
                best_j, best_score = j, score
        if best_j is not None:
            used.add(best_j)
            m["live"] = True
            m["last_seen"] = now
            if not occluded:
                m["box"] = detected[best_j]
        elif occluded:
            m["last_seen"] = now
    for j, d in enumerate(detected):
        if j not in used:
            memory.append({"box": d, "last_seen": now, "live": True})
    memory[:] = [m for m in memory if now - m["last_seen"] <= SEAT_MEMORY_SECONDS]
    return [m["box"] for m in memory]

def _visible_mean(kxy, kconf, idxs):
    pts = [kxy[i] for i in idxs if kconf[i] >= KP_CONF and (kxy[i][0] > 0 or kxy[i][1] > 0)]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

def _hip_on_seat(hip, seat, margin=None):
    sx1, sy1, sx2, sy2 = seat
    margin = HIP_SEAT_MARGIN if margin is None else margin
    top = sy1 - margin * (sy2 - sy1)
    return sx1 <= hip[0] <= sx2 and top <= hip[1] <= sy2

def evaluate_sitting(box, kxy, kconf, seat_boxes):
    overlap = sit_overlap_ratio(box, seat_boxes) if seat_boxes else 0.0
    if overlap < SIT_OVERLAP_RATIO:
        return False, overlap, "overlap"

    hip = _visible_mean(kxy, kconf, (11, 12))
    shoulder = _visible_mean(kxy, kconf, (5, 6))
    if hip is not None:
        if not any(_hip_on_seat(hip, s) for s in seat_boxes):
            return False, overlap, "hip-outside"
        knee = _visible_mean(kxy, kconf, (13, 14))
        if knee is not None:
            torso = (hip[1] - shoulder[1]) if shoulder is not None else (box[3] - box[1]) * 0.3
            torso = max(torso, 1.0)
            drop = (knee[1] - hip[1]) / torso
            if drop >= SIT_THIGH_DROP_RATIO:
                return False, overlap, f"standing-legs(drop={drop:.2f})"
            return True, overlap, f"pose(drop={drop:.2f})"
        # 膝が見えない（近くに立って足が画面外、など）：脚で確かめられないので、腰の高さで確かめる。
        # 座っていれば腰は椅子の中ほど、立っていれば椅子ボックスの上端付近になる。
        if not any(_hip_on_seat(hip, s, margin=-HIP_SEAT_MIN_DEPTH_NOKNEE) for s in seat_boxes):
            return False, overlap, "hip-shallow(no-knee)"
        return True, overlap, "pose(no-knee)"

    # 腰が見えない（上半身のみ）：肩が椅子に対して高すぎれば立っているとみなす
    if shoulder is not None and not any(
            shoulder[1] >= s[1] - SHOULDER_ABOVE_SEAT_MAX * (s[3] - s[1]) for s in seat_boxes):
        return False, overlap, "standing-high(no-hip)"
    h, w = box[3] - box[1], max(box[2] - box[0], 1.0)
    if h / w >= STANDING_ASPECT_RATIO:
        return False, overlap, "standing-aspect"
    return True, overlap, f"box(h/w={h / w:.1f})"

# =========================================================================
# 🚪 入口の認証セッション（QR → 顔の一致 → 線の通過 を1人分まとめて管理）
# =========================================================================
class EntranceSession:
    """QRを読んだ会員1人分の状態を管理する（入口モード用）。

    ・会員の顔と一致した時点で「顔認証OK」にする（線をまたぐ瞬間を待たない）。
    ・認証後は、顔 → 追跡ID → 直前の位置に近い枠 の順で本人を探し直す。
      そのため、処理が遅れて枠のIDが付け替わっても本人を見失いにくい。
    ・線の手前（左）にいた本人が、線から一定距離だけ向こう側（右）で見つかった時点で通過（入館）とする。
      線をまたいだ瞬間のフレームは必要ない。"""

    def __init__(self, member_id, label, embedding):
        self.member_id = member_id
        self.label = label
        self.embedding = embedding
        self.authed = False
        self.auth_time = None
        self.match_sim = 0.0         # 会員と一致した顔の最大類似度
        self.best_sim_seen = None    # QR読み取り後に見えた全ての顔の最大類似度（不一致の診断用）
        self.track_id = None
        self.pos = None              # 本人を最後に確認した位置 (x, y)
        self.last_box = None
        self.last_seen = None
        self.armed = False           # 線の手前（左）にいたことを確認済みか
        self.done = False
        self.last_face_sims = []     # [(face, 会員との類似度)] 描画用

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _has_other_face(self, box):
        """この枠の中に、会員とは明らかに別人の顔があるか。"""
        for face, sim in self.last_face_sims:
            fx, fy = self._center(face.bbox)
            if box[0] <= fx <= box[2] and box[1] <= fy <= box[3] and sim < ENTRANCE_OTHER_FACE_SIM:
                return True
        return False

    def update(self, now, qr_time, tracked, faces):
        """1フレーム分の更新。tracked は [(track_id, box), ...]。
        戻り値: None / "authed"（顔認証OK）/ "crossed"（通過=入館）/ "expired"（時間切れ）"""
        if self.done:
            return None
        if now - qr_time > CO_TRAILING_WINDOW:
            self.done = True
            return "expired"

        # 1) 会員の顔と似ている顔を探す
        self.last_face_sims = []
        face_hit = None
        for face in faces:
            sim = cosine_sim(self.embedding, face.embedding)
            self.last_face_sims.append((face, sim))
            if self.best_sim_seen is None or sim > self.best_sim_seen:
                self.best_sim_seen = sim
            if sim > FACE_MATCH_THRESHOLD and (face_hit is None or sim > face_hit[2]):
                fc = self._center(face.bbox)
                face_hit = (fc[0], fc[1], sim)

        # 2) 本人の位置を決める
        pos = box = tid = None
        if face_hit is not None:
            fx, fy, sim = face_hit
            self.match_sim = max(self.match_sim, sim)
            for t, b in tracked:
                if b[0] <= fx <= b[2] and b[1] <= fy <= b[3]:
                    box, tid = b, t
                    break
            pos = self._center(box) if box is not None else (fx, fy)   # 枠が無くても顔の位置を使う
        elif self.authed:
            # 顔が見えないとき：同じ追跡IDの枠 → なければ直前の位置に近い枠
            for t, b in tracked:
                if t == self.track_id:
                    box, tid = b, t
                    break
            if box is None and self.pos is not None and now - self.last_seen <= ENTRANCE_REASSOC_SECONDS:
                allowed = ENTRANCE_REASSOC_BASE_PX + ENTRANCE_REASSOC_PX_PER_SEC * (now - self.last_seen)
                best_d = None
                for t, b in tracked:
                    c = self._center(b)
                    d = float(np.hypot(c[0] - self.pos[0], c[1] - self.pos[1]))
                    if d > allowed or (best_d is not None and d >= best_d) or self._has_other_face(b):
                        continue
                    box, tid, best_d = b, t, d
            if box is not None:
                pos = self._center(box)

        if pos is not None:
            self.pos, self.last_seen = pos, now
            if box is not None:
                self.last_box = box
            if tid is not None:
                self.track_id = tid

        # 3) 顔認証
        event = None
        if not self.authed:
            if face_hit is None:
                return None
            self.authed = True
            self.auth_time = now
            event = "authed"

        # 4) 通過の判定（線の手前にいた本人が、向こう側で見つかったら通過）
        if pos is not None:
            if pos[0] < LINE_X:
                self.armed = True
            elif self.armed and pos[0] >= LINE_X + ENTRANCE_CROSS_MARGIN_PX:
                self.done = True
                return "crossed"
        return event

# =========================================================================
# 🎥 カメラ処理ループ
# =========================================================================
def camera_processing_loop():
    global current_output_frame, yolo_model
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis

    yolo_model = YOLO("yolo11n.pt")
    room_model = YOLO(ROOM_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    
    print("👤 顔認識モデルを読み込み中...")
    face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)

    cap = open_camera()
    while not cap.isOpened():
        print("❌ エラー: カメラを開けませんでした。3秒後に再試行します...")
        time.sleep(3)
        cap.release()
        cap = open_camera()
    print(f">> カメラ解像度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
          f"（目標 {FRAME_W}x{FRAME_H} / YOLO imgsz={YOLO_IMGSZ} / 顔検出 {FACE_DET_SIZE[0]}x{FACE_DET_SIZE[1]}）")

    track_history = {}
    face_sim_history = {}
    exit_id_history = {}   # 退館用：追跡IDごとに、顔で識別した会員 {member_id, name, sim, time}
    last_push_time = 0.0
    last_err_log = 0.0
    prev_frame_time = None   # 1フレームの処理時間の計測用（遅いとタイマー系の判定に影響するため画面に出す）
    loop_dt = 0.0
    last_slow_log = 0.0
    last_face_log = 0.0
    sit_confirmer = SitConfirmer()
    seat_memory = []
    identity_holder = IdentityHolder()
    clothes_mismatch_since = None   # 服装で推定した本人の服が合わなくなり始めた時刻

    def entrance_clothes_feature(frame, box, member_id):
        """入館した人の服の色（胸まわり）を測る。骨格推定を1回だけ実行する。測れなければ None。"""
        try:
            res = pose_model(frame, conf=0.3, imgsz=YOLO_IMGSZ, verbose=False)[0]
            kxy, kcf = find_pose_for_box(res, box)
            feat = get_box_color_feature(frame, kxy, kcf)
        except Exception as e:
            print(f"⚠️ 入口の服装の測定に失敗: {e}")
            return None
        if feat is None:
            print(f"👕 入口の服装を測れませんでした（肩が写っていない、または胸まわりが肌色ばかり）: {member_id}")
        else:
            b, g, r = lab_to_bgr(feat)
            print(f"👕 入口の服装を記録: {member_id} 色(RGB)=({r},{g},{b})")
        return feat
    sit_log = {"key": None, "time": 0.0}
    entrance_session = None   # 入口：QRを読んだ会員1人分の認証セッション
    auth_banner = {"text": "", "color": (200, 200, 200), "until": 0.0}   # 認証NG・失効・入館OKの一時表示
    print("\n====================================================")
    print(" 📹 カメラ処理ループが稼働しました（椅子検知＋座り判定モード）。")
    print("====================================================\n")

    while cap.isOpened():
        try:
            current_time = time.time()
            if prev_frame_time is not None:
                loop_dt = 0.8 * loop_dt + 0.2 * (current_time - prev_frame_time)
            prev_frame_time = current_time
            success, frame = cap.read()
            if not success:
                time.sleep(0.03)
                continue

            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))   # カメラが指定解像度に非対応のときの保険
            with mode_lock:
                mode = active_mode
            if mode != "entrance":
                entrance_session = None

            if mode == "entrance":
                qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame)
                if qr_bbox is not None and len(qr_bbox) > 0:
                    pts = qr_bbox[0].astype(int)
                    for i in range(4):
                        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), THICK)
                    if qr_data:
                        if state.should_process_qr(qr_data, current_time):
                            state.process_qr(qr_data, current_time)
                            if (state.accepted_token == qr_data and state.last_member_id is not None
                                    and state.last_member_embedding is not None):
                                # 有効なQR：この会員の顔認証セッションを開始（顔データ未登録の会員は従来どおり通過時に失敗扱い）
                                entrance_session = EntranceSession(state.last_member_id, state.last_scanned_qr,
                                                                   state.last_member_embedding)
                                auth_banner["until"] = 0.0
                            elif state.last_scanned_qr == "Invalid QR":
                                entrance_session = None

                faces = face_app.get(frame)

                yolo_results = yolo_model.track(frame, persist=True, classes=[0], imgsz=YOLO_IMGSZ, verbose=False)
                cv2.line(frame, (LINE_X, 0), (LINE_X, FRAME_H), (255, 0, 0), THICK)
                put_text(frame, "GATE LINE", (LINE_X + 10, int(30 * UI)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, hud=False)

                for tid in [t for t, r in face_sim_history.items() if current_time - r["time"] > FACE_TIMEOUT * 4]:
                    del face_sim_history[tid]
                for tid in [t for t, r in exit_id_history.items() if current_time - r["time"] > FACE_TIMEOUT * 4]:
                    del exit_id_history[tid]
                member_ids, member_names, member_matrix = member_cache.get()   # 退館時の1:N識別用

                tracked = []   # [(track_id, box), ...]
                if yolo_results[0].boxes.id is not None:
                    tracked = list(zip(yolo_results[0].boxes.id.cpu().numpy().astype(int),
                                       yolo_results[0].boxes.xyxy.cpu().numpy()))

                # --- 顔認証セッションの更新（顔が会員と一致した時点で認証OK。線を越えた本人を探し直して通過を判定） ---
                skip_tid = None      # このIDの通過は下の従来判定ではなく、ここで扱う
                face_sims = []
                if entrance_session is not None:
                    sess = entrance_session
                    ev = sess.update(current_time, state.last_qr_time, tracked, faces)
                    skip_tid = sess.track_id
                    face_sims = sess.last_face_sims
                    member_name = sess.label.split(" (")[0] if " (" in sess.label else sess.label
                    if ev == "authed":
                        print(f"✅ 【顔認証OK】 {sess.label}（類似度 {sess.match_sim:.2f}）… このまま線を越えると入館になります")
                        if sess.last_box is not None:
                            sess.clothes_feat = entrance_clothes_feature(frame, sess.last_box, sess.member_id)
                    elif ev == "crossed":
                        print(f"{member_name}：認証成功（類似度: {sess.match_sim:.2f}）… 顔認証後に線の向こう側で確認")
                        state.register_pass("IN", sess.label, 0,
                                            note=f"QRと顔が一致（顔の類似度 {sess.match_sim:.2f}）／顔認証後、線の向こう側で本人を確認")
                        state.mark_inside(sess.member_id)
                        feat = (entrance_clothes_feature(frame, sess.last_box, sess.member_id)
                                if sess.last_box is not None else None)
                        if feat is None:
                            feat = getattr(sess, "clothes_feat", None)   # 顔認証OKの時点で測れていれば、それを使う
                        if feat is not None:
                            state.add_recent_entry(sess.member_id, member_name, feat, current_time)
                        auth_banner.update(text=f"IN OK: {sess.member_id}", color=(0, 255, 0),
                                           until=current_time + ENTRANCE_BANNER_SECONDS)
                    elif ev == "expired":
                        if sess.authed:
                            print(f"⌛ 【認証失効】 {sess.label}：顔認証後、{CO_TRAILING_WINDOW:.0f}秒以内に線を越えなかったため失効しました")
                            auth_banner.update(text=f"AUTH EXPIRED: {sess.member_id}", color=(0, 165, 255),
                                               until=current_time + ENTRANCE_BANNER_SECONDS)
                        elif sess.best_sim_seen is not None:
                            print(f"❌ 【顔認証NG】 {sess.label}：QRと顔が一致しませんでした（最大類似度 {sess.best_sim_seen:.2f} ≤ しきい値 {FACE_MATCH_THRESHOLD}）")
                            state.set_alert(True)
                            auth_banner.update(text=f"AUTH NG (face mismatch): {sess.member_id}", color=(0, 0, 255),
                                               until=current_time + ENTRANCE_BANNER_SECONDS)
                        else:
                            print(f"⌛ 【顔認証タイムアウト】 {sess.label}：{CO_TRAILING_WINDOW:.0f}秒以内に顔が見つかりませんでした")
                            auth_banner.update(text=f"AUTH TIMEOUT (no face): {sess.member_id}", color=(0, 165, 255),
                                               until=current_time + ENTRANCE_BANNER_SECONDS)
                    if sess.done:
                        entrance_session = None

                # 顔の枠（会員と一致した顔=黄、それ以外=灰）。認証セッション中は類似度も表示する
                sim_by_face = {id(f): sv for f, sv in face_sims}
                for face in faces:
                    fsim = sim_by_face.get(id(face))
                    matched = fsim is not None and fsim > FACE_MATCH_THRESHOLD
                    fb = face.bbox.astype(int)
                    cv2.rectangle(frame, (int(fb[0]), int(fb[1])), (int(fb[2]), int(fb[3])),
                                  (0, 255, 255) if matched else (160, 160, 160), 2)
                    if fsim is not None:
                        put_text(frame, f"face:{fsim:.2f}", (int(fb[0]), max(int(15 * UI), int(fb[1]) - int(8 * UI))),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if matched else (200, 200, 200), 1, hud=False)

                if yolo_results[0].boxes.id is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                    track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, track_ids):
                        x_center = int((box[0] + box[2]) / 2)
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), THICK)
                        put_text(frame, f"ID:{track_id}", (int(box[0]), int(box[3]) + int(18 * UI)),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, hud=False)

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

                        # 退館用：この人の枠内の顔を全会員と照合（1:N）。QRは使わず、最も似た会員を線を越えるまで保持する
                        for face in faces:
                            fcx = (face.bbox[0] + face.bbox[2]) / 2
                            fcy = (face.bbox[1] + face.bbox[3]) / 2
                            if not (box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3]):
                                continue
                            ident = identify_face_1n(face.embedding, member_ids, member_names, member_matrix)
                            if ident is None or ident[2] <= EXIT_FACE_THRESHOLD:
                                continue
                            erec = exit_id_history.get(track_id)
                            if erec is None or current_time - erec["time"] > FACE_TIMEOUT or ident[2] > erec["sim"]:
                                exit_id_history[track_id] = {"member_id": ident[0], "name": ident[1],
                                                             "sim": ident[2], "time": current_time}
                            else:
                                erec["time"] = current_time
                        erec = exit_id_history.get(track_id)
                        if erec is not None and current_time - erec["time"] <= FACE_TIMEOUT:
                            put_text(frame, f"{erec['member_id']} {erec['sim']:.2f}",
                                     (int(box[0]), int(box[3]) + int(36 * UI)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, hud=False)

                        rec = face_sim_history.get(track_id)
                        if rec is not None and current_time - rec["time"] <= FACE_TIMEOUT:
                            put_text(frame, f"sim:{rec['best']:.2f}", (int(box[0]), max(int(15 * UI), int(box[1]) - int(8 * UI))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, hud=False)

                        if track_id in track_history:
                            prev_x = track_history[track_id]
                            if skip_tid is not None and track_id == skip_tid and prev_x < LINE_X and x_center >= LINE_X:
                                pass   # 認証セッションの本人の通過は、上の判定（線の向こう側で本人を確認）で扱う
                            elif prev_x < LINE_X and x_center >= LINE_X:
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
                                # 💡 【修正ポイント1】 インデントを正しく分離
                                if ok:
                                    member_name = state.last_scanned_qr.split(" (")[0] if " (" in state.last_scanned_qr else state.last_scanned_qr
                                    print(f"{member_name}：認証成功（類似度: {best_sim:.2f}）")
                                    state.register_pass("IN", state.last_scanned_qr, 0, note=f"QRと顔が一致（顔の類似度 {best_sim:.2f}）")
                                    state.mark_inside(state.last_member_id)
                                    color_feat = entrance_clothes_feature(frame, box, state.last_member_id)
                                    if color_feat is not None:
                                        state.add_recent_entry(state.last_member_id, member_name, color_feat, current_time)
                                else:
                                    print(f"未登録または別人：認証失敗（理由: {reason}）")
                                    state.set_alert(True)
                                    state.register_pass("IN", "Unknown / Face Mismatch", 1, note=reason)
                            elif prev_x > LINE_X and x_center <= LINE_X:
                                # 退館：QRは不要。線を越えるまでに顔で識別できた会員を記録する
                                erec = exit_id_history.get(track_id)
                                if erec is not None and current_time - erec["time"] <= FACE_TIMEOUT:
                                    if state.leave_if_inside(erec["member_id"]):
                                        out_label = f"{erec['name']} ({erec['member_id']})"
                                        print(f"{erec['name']}：退館（顔の類似度: {erec['sim']:.2f}）")
                                        state.register_pass("OUT", out_label, 0,
                                                            note=f"顔で識別（類似度 {erec['sim']:.2f}）")
                                    else:
                                        out_reason = "入館記録がない（QR認証なしで入館した可能性）"
                                        print(f"退館者を Unknown として記録します（理由: {out_reason}）")
                                        state.register_pass("OUT", "Unknown", 0,
                                                            note=f"{out_reason}／顔は {erec['name']} ({erec['member_id']}) に類似（{erec['sim']:.2f}）")
                                else:
                                    if member_matrix is None:
                                        out_reason = "顔データが登録された会員がいないため識別できない"
                                    else:
                                        out_reason = f"退館前に顔を識別できなかった（しきい値 {EXIT_FACE_THRESHOLD}）"
                                    print(f"退館者を識別できませんでした（理由: {out_reason}）")
                                    state.register_pass("OUT", "Unknown", 0, note=out_reason)
                                state.set_alert(False)

                        track_history[track_id] = x_center

                put_text(frame, "MODE: [ENTRANCE (IN: QR+Face / OUT: Face)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                put_text(frame, f"IN: {state.in_count} | OUT: {state.out_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if entrance_session is not None:
                    remain = max(0.0, CO_TRAILING_WINDOW - (current_time - state.last_qr_time))
                    if entrance_session.authed:
                        put_text(frame, f"AUTH OK: {entrance_session.member_id} sim={entrance_session.match_sim:.2f} "
                                        f"-> go through ({remain:.0f}s)", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        put_text(frame, f"QR OK: {entrance_session.member_id} - show your face ({remain:.0f}s)",
                                 (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif current_time < auth_banner["until"]:
                    put_text(frame, auth_banner["text"], (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, auth_banner["color"], 2)

            else:
                # ＝ トレーニングルームモード（椅子検知 ＋ 座り判定）＝
                member_ids, member_names, member_matrix = member_cache.get()

                faces = face_app.get(frame)
                detected_user_name = "Guest"
                detected_member_id = None
                matched_face_center = None
                best_face_sim = None   # 今のフレームで見えた顔のうち、会員との最高類似度（診断表示用）
                face_note = "no-face" if len(faces) == 0 else ("no-members" if member_matrix is None else "no-match")

                # 1. 顔による1:Nマッチング（キャッシュ済みの正規化行列との内積で一括計算）
                for face in faces:
                    if member_matrix is None:
                        break
                    fe = np.asarray(face.embedding, dtype=np.float32)
                    fn = np.linalg.norm(fe)
                    if fn == 0 or fe.shape[0] != member_matrix.shape[1]:
                        continue
                    sims = member_matrix @ (fe / fn)
                    best_idx = int(np.argmax(sims))
                    sim_val = float(sims[best_idx])
                    if best_face_sim is None or sim_val > best_face_sim:
                        best_face_sim = sim_val
                    if sim_val > ROOM_FACE_THRESHOLD:
                        face_note = "matched"
                        detected_user_name = member_names[best_idx]
                        detected_member_id = member_ids[best_idx]   # 【バグ修正】一致した会員のIDを使う
                        matched_face_center = ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)
                        break

                # 2. 椅子と人（骨格つき）を検出
                detected_seats = []
                seat_det = room_model(frame, classes=SEAT_CLASSES, conf=ROOM_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
                if seat_det.boxes is not None and len(seat_det.boxes) > 0:
                    detected_seats = list(seat_det.boxes.xyxy.cpu().numpy())

                persons = []
                pose_det = pose_model(frame, conf=0.3, imgsz=YOLO_IMGSZ, verbose=False)[0]
                if pose_det.boxes is not None and len(pose_det.boxes) > 0 and pose_det.keypoints is not None:
                    p_boxes = pose_det.boxes.xyxy.cpu().numpy()
                    kxy_all = pose_det.keypoints.xy.cpu().numpy()
                    kcf_all = (pose_det.keypoints.conf.cpu().numpy() if pose_det.keypoints.conf is not None
                               else np.ones(kxy_all.shape[:2]))
                    persons = [(p_boxes[i], kxy_all[i], kcf_all[i]) for i in range(len(p_boxes))]
                person_boxes = [p[0] for p in persons]

                seat_boxes = update_seat_memory(seat_memory, detected_seats, person_boxes, current_time)

                # 座り判定
                best = (False, -1.0, "no-person")
                sit_box = None
                sit_kxy = sit_kcf = None
                for box, kxy_i, kcf_i in persons:
                    res = evaluate_sitting(box, kxy_i, kcf_i, seat_boxes)
                    if (res[0], res[1]) > (best[0], best[1]):
                        best = res
                        sit_box = box
                        sit_kxy, sit_kcf = kxy_i, kcf_i
                is_sitting, overlap, why = best
                active_machine_key = "bench_press"

                # 顔認識で特定した人が椅子の範囲にいるかチェック
                if detected_user_name != "Guest":
                    seat_h = (sit_box[3] - sit_box[1]) if sit_box is not None else 0
                    inside = (matched_face_center is not None
                              and sit_box is not None
                              and sit_box[0] <= matched_face_center[0] <= sit_box[2]
                              and (sit_box[1] - seat_h) <= matched_face_center[1] <= sit_box[3])
                    if sit_box is not None and not inside:
                        detected_user_name, detected_member_id = "Guest", None
                        face_note = "matched-but-outside-seat"

                # 服装による補助判定：顔で一度も特定できていない（保持中の本人もいない）座っている人にだけ使う。
                # 一度決まれば IdentityHolder が保持するので、毎フレーム選び直さない。
                estimated_by_clothes = False
                if (detected_user_name == "Guest" and is_sitting and sit_box is not None
                        and not identity_holder.has_identity(current_time)):
                    fb_name, fb_id = state.match_recent_entry_by_feature(sit_kxy, sit_kcf, frame, current_time)
                    if fb_name is not None:
                        detected_user_name = fb_name
                        detected_member_id = fb_id
                        estimated_by_clothes = True
                        print(f"👕 【服装から推定】 顔が見えないため入口の服装記録から {fb_name} と推定しました"
                              f"（色の差 {state.last_clothes_dist:.2f} / 上限 {CLOTHES_MAX_DIST}）")

                # 服装で推定した本人は、保持中も服装を確認し続ける。着替えなどで入口の記録と合わなくなったら推定を解除する
                recheck_d = None
                if (CLOTHES_FALLBACK_ENABLED and identity_holder.estimated and identity_holder.has_identity(current_time)
                        and not estimated_by_clothes and detected_user_name == "Guest" and sit_box is not None):
                    recheck_d = state.clothes_distance_to_member(identity_holder.member_id, sit_kxy, sit_kcf, frame, current_time)
                    if recheck_d is not None and recheck_d >= CLOTHES_MAX_DIST:
                        if clothes_mismatch_since is None:
                            clothes_mismatch_since = current_time
                        elif current_time - clothes_mismatch_since >= CLOTHES_RECHECK_SECONDS:
                            print(f"👕 【推定を解除】 服装が入口の記録と合わなくなりました"
                                  f"（色の差 {recheck_d:.1f} ≥ 上限 {CLOTHES_MAX_DIST}）")
                            identity_holder.name = identity_holder.member_id = None
                            identity_holder.estimated = False
                            clothes_mismatch_since = None
                    else:
                        clothes_mismatch_since = None
                else:
                    clothes_mismatch_since = None

                if estimated_by_clothes:
                    clothes_txt = f"USED d={state.last_clothes_dist:.1f}"
                elif detected_user_name != "Guest":
                    clothes_txt = "skip (face matched)"
                elif not is_sitting or sit_box is None:
                    clothes_txt = "skip (not sitting)"
                elif identity_holder.has_identity(current_time):
                    clothes_txt = (f"held by clothes d={recheck_d:.1f}" if recheck_d is not None
                                   else "skip (identity held)")
                else:
                    clothes_txt = state.clothes_diag
                put_text(frame, f"clothes: {clothes_txt} (max {CLOTHES_MAX_DIST:.0f})",
                         (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

                # 服の色を測っている範囲（水色の枠）と色の見本を出す。NOW=いま測った色、IDつき=入口で記録した色。
                # 見本が肌色や背景の色なら、測る場所がずれている。
                now_feat = get_box_color_feature(frame, sit_kxy, sit_kcf) if sit_kxy is not None else None
                roi_box = clothes_roi(frame.shape, sit_kxy, sit_kcf) if sit_kxy is not None else None
                if roi_box is not None:
                    cv2.rectangle(frame, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (255, 255, 0), 2)
                draw_swatch(frame, 0, "NOW", now_feat)
                for n, e in enumerate(list(state.recent_entries)[-3:], start=1):
                    draw_swatch(frame, n, e["member_id"], e["color_feat"])

                # デバッグ表示：人=緑、椅子=オレンジ、腰・膝=赤い点
                for box, kxy_i, kcf_i in persons:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), THICK)
                    for idx in (11, 12, 13, 14):
                        if kcf_i[idx] >= KP_CONF:
                            cv2.circle(frame, (int(kxy_i[idx][0]), int(kxy_i[idx][1])), int(4 * UI), (0, 0, 255), -1)
                for m in seat_memory:
                    b = m["box"]
                    color = (0, 165, 255) if m["live"] else (0, 255, 255)
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, THICK)
                n_live = sum(1 for m in seat_memory if m["live"])
                put_text(frame, f"persons={len(persons)} seats={n_live}(+{len(seat_memory) - n_live} remembered) "
                                   f"overlap={max(overlap, 0.0):.2f} why={why}",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                log_key = (is_sitting, why.split("(")[0])
                if log_key != sit_log["key"] and current_time - sit_log["time"] >= 1.0:
                    print(f"🪑 座り判定: {is_sitting} (why={why}, overlap={max(overlap, 0.0):.2f}, "
                          f"persons={len(persons)}, seats={n_live}+{len(seat_memory) - n_live}記憶)")
                    sit_log["key"], sit_log["time"] = log_key, current_time

                # 顔認識が外れても、直前に特定した会員を保持（IdentityHolder）
                detected_user_name, detected_member_id, identity_held, identity_estimated = identity_holder.update(
                    detected_user_name, detected_member_id, is_sitting, current_time,
                    estimated=estimated_by_clothes)
                held_mark = " (estimated)" if identity_estimated else (" (face lost)" if identity_held else "")

                # --- 診断表示: 顔判定のどこで止まっているか／誰として扱っているか／1フレームの処理時間 ---
                if detected_user_name == "Guest":
                    id_source = "none"
                elif identity_estimated:
                    id_source = "clothes"
                elif identity_held:
                    id_source = "held"
                else:
                    id_source = "face"
                sim_txt = f"{best_face_sim:.2f}" if best_face_sim is not None else "-"
                put_text(frame, f"face={face_note} faces={len(faces)} best={sim_txt}/{ROOM_FACE_THRESHOLD:.2f} "
                                   f"id={id_source} loop={loop_dt:.2f}s",
                            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                if is_sitting and current_time - last_face_log >= 3.0:
                    print(f"👤 顔判定: {face_note} / 顔の数={len(faces)} / 最高類似度={sim_txt}（閾値 {ROOM_FACE_THRESHOLD:.2f}）"
                          f" / 識別元={id_source} / 1フレーム={loop_dt:.2f}秒")
                    last_face_log = current_time
                if loop_dt > SIT_BREAK_TOLERANCE and current_time - last_slow_log >= 10.0:
                    print(f"⚠️ 1フレームの処理に約{loop_dt:.1f}秒かかっています（顔認識＋椅子検出＋骨格推定をCPUで実行中）。"
                          f"座り判定の反応が遅れる原因になります。")
                    last_slow_log = current_time

                # 会員が検知され、座っている状態が SIT_CONFIRM_SECONDS 続いたときだけタイマー開始
                sitting_now = detected_user_name != "Guest" and is_sitting
                confirmed, sit_since = sit_confirmer.update(sitting_now, current_time)
                if confirmed:
                    state.update_machine_occupancy(active_machine_key, detected_user_name, current_time,
                                                   started_at=sit_since, member_id=detected_member_id)
                    put_text(frame, f"Sitting & Using: {detected_user_name}{held_mark}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif sitting_now:
                    put_text(frame, f"Sitting? confirming [{detected_member_id or '-'}]{held_mark} "
                                       f"{current_time - sit_since:.1f}/{SIT_CONFIRM_SECONDS:.1f}s",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    status_text = f"User: {detected_user_name}{held_mark} | Sitting: {is_sitting}"
                    put_text(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                state.release_idle_machines(current_time)
                if current_time - last_push_time >= ROOM_PUSH_INTERVAL:
                    state.push_update()
                    last_push_time = current_time
                put_text(frame, "MODE: [ROOM (Chair Occupancy Mode)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            with render_lock:
                current_output_frame = frame.copy()

            time.sleep(0.01)

        except Exception as e:
            # 1フレームの失敗でスレッドが死なないようにする（ログは5秒に1回まで）
            if time.time() - last_err_log > 5.0:
                print(f"⚠️ カメラ処理でエラー（処理は継続します）: {e}")
                traceback.print_exc()
                last_err_log = time.time()
            time.sleep(0.1)
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

        # メール送信に成功してから会員を登録する（失敗時に「登録済みなのにQRが届かない」状態を防ぐ）
        # 直接 connect せず、execute_db_write に置き換える
        execute_db_write(
            "INSERT INTO members (member_id, name, email, qr_token, face_embedding) VALUES (?, ?, ?, ?, ?)",
            (member_id, data.name, email, qr_token, sqlite3.Binary(face_embedding))
        )
        member_cache.invalidate()

        return {"status": "success", "member_id": member_id}

    except Exception as e:
        print(f"登録エラー: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/video")
async def video_feed():
    async def generate_async_stream():
        while True:
            with render_lock:
                frame_to_send = current_output_frame
            if frame_to_send is not None:
                ret, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
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
        # 接続直後に最新の状態を配信キューへ入れる（次のイベントを待たずに画面が埋まる）。
        # 通常の更新と同じキューを通すので、古いデータが後から届いて上書きすることもない。
        await asyncio.to_thread(state.push_update)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

async def ws_broadcast_loop():
    while True:
        try:
            while not state.update_queue.empty():
                text = json.dumps(state.update_queue.get())
                for ws in list(connected_websockets):
                    try:
                        await ws.send_text(text)
                    except Exception:
                        if ws in connected_websockets:
                            connected_websockets.remove(ws)
        except Exception as e:
            print(f"⚠️ WebSocket配信でエラー（配信は継続します）: {e}")
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ws_broadcast_loop())

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "index.html", {})

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
        member_cache.invalidate()
        return {"status": "success", "message": f"会員 {member_id} を削除しました。"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/admin/members/{member_id}/detail")
def api_get_member_detail(member_id: str, request: Request):
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

@app.get("/admin/logs", response_class=HTMLResponse)
def get_admin_logs_page(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "logs.html", {})

def _member_label(names, raw):
    if raw in names:
        return f"{names[raw]} ({raw})"
    return raw

def _member_id_from_label(label):
    m = re.match(r"^.*\(([^()]+)\)$", label or "")
    return m.group(1) if m else None

def _member_stats(cursor, member_id):
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

@app.get("/live", response_class=HTMLResponse)
async def read_live(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "live.html", {})

@app.get("/api/admin/logs")
def api_get_logs(request: Request):
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
            {"label": "判定の理由", "value": r["note"] or "（記録なし）"},
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

if __name__ == "__main__":
    camera_thread = threading.Thread(target=camera_processing_loop, daemon=True)
    camera_thread.start()
    
    switcher_thread = threading.Thread(target=console_switcher_thread, daemon=True)
    switcher_thread.start()
    
    print("\n🚀 サーバーを椅子検知・座り判定モードで起動しました！")
    print("👉 ブラウザで http://localhost:8000/ を開いてダッシュボードを確認してください。")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")