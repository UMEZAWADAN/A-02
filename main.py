# 標準ライブラリと外部ライブラリの読み込み。
# cv2=カメラ映像とQR、numpy=顔特徴量の計算、pymysql=MySQL、fastapi=Webサーバー本体。
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
import pymysql
import pymysql.cursors
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
from fastapi.staticfiles import StaticFiles

# .env ファイルの中身（DBのパスワードやメールの設定など）を環境変数として読み込む。
from dotenv import load_dotenv
load_dotenv()

# パスワードハッシュ化の設定
from passlib.context import CryptContext
# パスワードは平文のまま保存せず、bcrypt でハッシュ化して保存する。
# deprecated="auto" は、古い方式のハッシュが混ざっていても検証できるようにする指定。
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """平文のパスワードをハッシュ化する"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """入力されたパスワードが、ハッシュ値と一致するか検証する"""
    return pwd_context.verify(plain_password, hashed_password)

# パスワードの最低文字数。validate_password_strength で使う。
PASSWORD_MIN_LENGTH = 8

def validate_password_strength(password: str):
    """パスワード強度を検証する。問題なければ None、問題があればエラーメッセージを返す。
    条件：8文字以上／英大文字／英小文字／数字／特殊文字を含む"""
    # 全角文字・空白・制御文字は不可（半角の英数字と記号 ! 〜 ~ のみ）
    if not re.fullmatch(r"[!-~]*", password):
        return "パスワードは半角の英数字・記号のみで入力してください(全角文字・空白は使えません)。"
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"パスワードは{PASSWORD_MIN_LENGTH}文字以上で入力してください。"
    if not re.search(r"[a-z]", password):
        return "パスワードには英小文字を含めてください。"
    if not re.search(r"[A-Z]", password):
        return "パスワードには英大文字を含めてください。"
    if not re.search(r"[0-9]", password):
        return "パスワードには数字を含めてください。"
    if not re.search(r"[!-/:-@\[-`{-~]", password):
        return "パスワードには半角の特殊文字(!@#$% など)を含めてください。"
    # bcrypt は72バイトまでしか扱えないので、上限も設ける
    if len(password.encode("utf-8")) > 72:
        return "パスワードが長すぎます(72バイト以内にしてください)。"
    return None



# テンプレートエンジンの準備
# FastAPI本体と、HTMLテンプレート(templates フォルダ)・静的ファイル(static フォルダ)の設定。
app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# SECRET_KEY は .env で必ず設定すること（未設定だと固定の予備の値になり、Cookieの署名が誰でも作れてしまう）。
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key")
# ログイン状態をCookieに保存するための設定。管理者は session["admin_user"]、会員は session["member_id"] で区別する。
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# 以下の2つは現在どこからも使われていない（以前の実装の名残）。消しても動作は変わらない。
clients = []
last_detected_qr = "" # 直近のQRコードを保持する変数

# =========================================================================
# システム設定値（調整可能）
# =========================================================================
# MySQL 接続設定（環境変数で上書きできる。パスワードはコードに直接書かないこと）
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "gym_security")
DB_CONFIG = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
                 database=DB_NAME, charset="utf8mb4")

def get_conn(dict_cursor=False):
    """MySQL への接続を新しく開く。dict_cursor=True なら行を row["name"] のように読める。"""
    return pymysql.connect(
        **DB_CONFIG,
        cursorclass=pymysql.cursors.DictCursor if dict_cursor else pymysql.cursors.Cursor,
    )

def _exec(conn, sql, params=()):
    """conn.execute() の代わり（pymysql の接続には execute が無いため）。カーソルを返す。"""
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur
# 同じQRを読み続けているとき、この秒数以内の再読み取りは「同じ提示の続き」とみなして、もう一度認証し直さない。
DUPLICATE_QR_WINDOW = 5.0
STREAM_JPEG_QUALITY = 80        # ライブ映像のJPEG品質（既定の95より少し軽くする）
# QR認証が有効な秒数。この間に線を越えた別の人は「共連れ」の疑いになる。認証セッションの寿命でもある。
CO_TRAILING_WINDOW = 10.0
# 以前使っていた共通の利用時間上限。今はマシンごとの MACHINE_LIMITS を使うので、この値は参照されていない。
LIMIT_SECONDS = 10.0
FACE_TIMEOUT = 10.0            # 顔の類似度（直近の最大値）を有効とみなす秒数
FACE_MATCH_THRESHOLD = 0.60   # 顔が一致したとみなすコサイン類似度（入口の1:1判定。ログを見て調整）
ROOM_FACE_THRESHOLD = 0.60    # ルームモードの1:N判定用。部屋のカメラは遠い・斜めで類似度が低めになりやすいので、画面の best= を見て下げる

# メール送信設定（環境変数から安全に読み込む）
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "chocozap123456@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")   # コード内からパスワードを排除

# 違反メール：マシンの利用時間が上限を超えた会員に、利用の終了時にメールで違反内容を知らせる
VIOLATION_MAIL_ENABLED = True       # False にするとメールを送らない
VIOLATION_MAIL_COOLDOWN = 600.0     # 同じ会員には、この秒数のあいだ違反メールを重ねて送らない（識別の揺れで続けて送られるのを防ぐ）

# 映像の解像度（カメラ取得・YOLO・顔検出・ライブ表示すべてこのサイズで処理する）
FRAME_W, FRAME_H = 640, 480    # 元の解像度。高精度にしたいときは 1280, 720 などに上げる（その分CPU負荷が大きく増える）
LINE_X = FRAME_W // 2          # 入口の通過判定ライン（画面中央）
YOLO_IMGSZ = 640               # YOLOの推論サイズ（ultralyticsの既定値）。FRAME_W を上げたら合わせて上げる
FACE_DET_SIZE = (640, 640)     # 顔検出の入力サイズ（幅, 高さ）。入力が640x480なら640で十分（1280にしても精度は上がらず重くなるだけ）
UI = FRAME_W / 640.0           # 画面上の文字・線の拡大率（HUDの見た目を従来と同じ比率に保つ）
THICK = max(1, int(round(2 * UI * 0.75)))   # ボックス・ラインの太さ
CAMERA_ID = 0  # 内蔵カメラ1台

# 椅子・座り判定の設定
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

# 入口：「顔認証」と「線の通過」を分けて判定する設定
ENTRANCE_CROSS_MARGIN_PX = round(FRAME_W * 0.03)      # 認証済みの本人が、線からこの距離だけ向こう側（右）で見つかったら通過（入館）とする（幅の3%）
ENTRANCE_REASSOC_SECONDS = 3.0       # 本人を見失っても、この秒数以内なら直前の位置に近い枠を本人として探し直す
ENTRANCE_REASSOC_BASE_PX = round(FRAME_W * 0.12)      # 探し直しで許容する移動距離（基本。幅の12%）
ENTRANCE_REASSOC_PX_PER_SEC = round(FRAME_W * 0.20)   # 探し直しで許容する移動距離（見失った秒数あたりの追加。幅の20%）
ENTRANCE_OTHER_FACE_SIM = 0.15       # 会員との類似度がこれ未満の顔を含む枠は、別人として探し直しの候補から外す
ENTRANCE_BANNER_SECONDS = 4.0        # 認証NG・失効のメッセージを画面に出す秒数
ENTRANCE_PRECROSS_GRACE = 3.0        # QRは読んだが顔認証がまだの人がいるとき、線を越えた人を共連れと決める前に待つ秒数（その間に顔が一致すれば本人の通過として扱う）
LINE_HYSTERESIS_PX = ENTRANCE_CROSS_MARGIN_PX         # 通過判定ラインの前後この距離を越えるまで、左右の状態を切り替えない（枠の揺れで行き来と誤判定しない）
TRACK_LINK_SECONDS = 1.5             # 追跡IDを見失っても、この秒数以内に近くへ現れた新しい枠なら同じ人として履歴を引き継ぐ
TRACK_LINK_BASE_PX = round(FRAME_W * 0.12)            # 引き継ぎで許容する移動距離（基本。幅の12%）
TRACK_LINK_PX_PER_SEC = round(FRAME_W * 0.20)         # 引き継ぎで許容する移動距離（見失った秒数あたりの追加。幅の20%）
TRACK_HISTORY_TTL = 30.0             # 追跡IDごとの履歴をこの秒数だけ残す（それより古いものは捨てる）

# 退館：QRは使わず、顔だけで誰が出たかを識別する（全会員との1:N照合）
EXIT_FACE_THRESHOLD = 0.40    # 退館時の顔識別のしきい値。別人に誤認されるなら上げる、識別できないなら下げる（画面の会員ID＋類似度を見て調整）

# 顔が見えないときの服装による補助推定（ルームモード）
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
    

# カメラを開いて、解像度を FRAME_W x FRAME_H で要求する。
# カメラによっては指定どおりにならないので、読み出す側でも resize している。
def open_camera():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap

# カメラの役割（モード）。'entrance'=入口 / 'room'=トレーニングルーム / 'esthe'=エステ室 / 'register'=会員登録中（カメラを手放す）。
# カメラ処理スレッドとWebの処理の両方から書き換えるので、読み書きは mode_lock を取ってから行う。
active_mode = "entrance"
mode_lock = threading.Lock()

# マシン・椅子の管理情報
# マシンごとの表示名と、利用時間の上限（秒）。上限を超えて座り続けると「超過」になり、違反メールが送られる。
# 単位は秒。実際のジムで使うなら、もっと長い時間に直すこと。
MACHINE_LIMITS = {
    "bench_press": {"name": "ベンチプレス", "limit": 15.0},
    "squat_rack": {"name": "スクワットラック", "limit": 20.0}
}

# =========================================================================
# リクエスト用データモデル（先頭側に定義）
# =========================================================================
# 会員登録API(/api/register)が受け取るJSON。image は撮影した顔写真で、base64の data URL 形式。
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    image: str

# 管理者ログインAPIが受け取るJSON。
class AdminAuthRequest(BaseModel):
    username: str
    password: str

# 会員ログインAPIが受け取るJSON。
class MemberLoginRequest(BaseModel):
    email: str
    password: str

# =========================================================================
# データベース自動初期化
# =========================================================================
def _ensure_database():
    """データベース自体が無ければ作る（CREATE 権限が無い場合は何もしない）。"""
    cfg = dict(DB_CONFIG)
    cfg.pop("database")
    try:
        conn = pymysql.connect(**cfg)
        try:
            conn.cursor().execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4")
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f" データベースの自動作成をスキップ: {e}")

# テーブルに列が無ければ追加する。古い版で作ったDBを、作り直さずに今の形へ移行するために使う。
def _ensure_column(cursor, table, column, ddl):
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
        (table, column))
    if cursor.fetchone()[0] == 0:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

# 起動時に呼ばれて、必要なテーブルをすべて作る（すでにあれば何もしない）。
# members=会員、admins=管理者、passing_logs=入退館ログ、occupancy_logs=マシン利用ログ、esthe_reservations=エステ予約。
def init_database():
    _ensure_database()
    conn = get_conn()
    cursor = conn.cursor()
    # ID・トークン・パスワードは大文字小文字を区別して比較したいので utf8mb4_bin にする
    BIN = "CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"

    # 管理者テーブル。パスワードはハッシュ値で入る。
    # 管理者の新規登録画面は廃止したので、管理者を追加するときはDBへ直接入れる（パスワードは get_password_hash で作ったハッシュにすること）。
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS admins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) {BIN} UNIQUE NOT NULL,
            password VARCHAR(255) {BIN} NOT NULL
        ) DEFAULT CHARSET=utf8mb4
    """)

    # 会員テーブル。qr_token は入館用QRに埋め込む文字列、face_embedding は顔写真から作った特徴量（バイト列）。
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS members (
            member_id VARCHAR(32) {BIN} PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(255) NOT NULL,
            password VARCHAR(255) {BIN} NOT NULL,
            qr_token VARCHAR(128) {BIN} UNIQUE NOT NULL,
            face_embedding BLOB
        ) DEFAULT CHARSET=utf8mb4
    """)
    _ensure_column(cursor, "members", "password", f"VARCHAR(255) {BIN} NOT NULL") # 既存DBへの保険

    # 入館(IN)・退館(OUT)の記録。member_id 列には「名前 (会員ID)」の形の文字列を入れる（共連れなど会員でないときは別の文字列）。
    # is_alert=1 は警告つきの記録。
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS passing_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp VARCHAR(19),
            direction VARCHAR(8),
            member_id VARCHAR(255) {BIN},
            is_alert INT,
            note TEXT,
            INDEX idx_passing_ts (timestamp)
        ) DEFAULT CHARSET=utf8mb4
    """)
    # マシンの利用記録（誰が・いつからいつまで・何秒・どのマシンか）。duration の単位は秒。
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS occupancy_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            member_id VARCHAR(255) {BIN},
            start_time VARCHAR(19),
            end_time VARCHAR(19),
            duration DOUBLE,
            machine VARCHAR(50),
            INDEX idx_occupancy_member (member_id)
        ) DEFAULT CHARSET=utf8mb4
    """)
    # エステ室の予約。status は reserved(予約済み) → in_progress(利用中) → completed(完了) と進む。
    # 予約したのに来なかった場合は no_show、終了時刻を過ぎても居座った場合は overtime、会員が取り消した場合は cancelled になる。
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS esthe_reservations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            member_id VARCHAR(32) {BIN} NOT NULL,
            start_time DATETIME NOT NULL,
            end_time DATETIME NOT NULL,
            status VARCHAR(20) DEFAULT 'reserved', -- 'reserved', 'in_progress', 'completed', 'no_show', 'overtime'
            used_at DATETIME DEFAULT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_esthe_time (start_time, end_time)
        ) DEFAULT CHARSET=utf8mb4
    """)
    # 以前に作った（列が足りない）テーブルにも対応
    _ensure_column(cursor, "occupancy_logs", "machine", "VARCHAR(50)")
    _ensure_column(cursor, "passing_logs", "note", "TEXT")
    conn.commit()
    conn.close()

# 起動時にDBを初期化する（uvicorn が動き出す前に、モジュールを読み込んだ時点で1回だけ実行される）。
init_database()
print(f">> 使用DB: MySQL {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

print(">> [1/3] YOLO（人・椅子検出）モデルを読み込み中...")
# YOLOのモデルは重いので、カメラ処理スレッドの中で読み込む。ここでは入れ物だけ用意しておく。
# QRコードの検出と読み取りは、OpenCV標準の検出器を使う。
yolo_model = None
qr_detector = cv2.QRCodeDetector()

# =========================================================================
# 違反メール（マシンの利用時間超過）
# =========================================================================
_violation_mail_last = {}   # {member_id: 最後に送信した時刻}
# カメラ処理のスレッドとメール送信のスレッドの両方から触るので、ロックで守る。
_violation_mail_lock = threading.Lock()

def _send_overtime_mail(member_id, machine_key, start_str, end_str, duration):
    """時間超過した会員に、違反内容をメールで送る。送れたら True。"""
    from email.header import Header
    # 上限秒数とマシン名を設定から引く。設定に無いマシンなら、上限0・名前はキーそのままになる。
    info = MACHINE_LIMITS.get(machine_key, {})
    limit = info.get("limit", 0.0)
    machine_name = info.get("name", machine_key)
    try:
        conn = get_conn()
        try:
            # メールの宛先と宛名にする名前を、会員テーブルから取る。
            row = _exec(conn, "SELECT name, email FROM members WHERE member_id = %s", (member_id,)).fetchone()
        finally:
            conn.close()
    except Exception as e:
        print(f" 違反メール：会員情報の取得に失敗: {e}")
        return False
    # 会員が削除済み、またはメールアドレスが空なら送れない。
    if row is None or not row[1]:
        print(f" 違反メールを送れません：会員 {member_id} のメールアドレスが見つかりません")
        return False
    name, email = row

    # 超過した時間は「利用時間 - 上限」で求めている。
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
    # 日本語の件名は Header で UTF-8 に包まないと文字化けする。
    msg = MIMEMultipart()
    msg["Subject"] = Header("【ジムセキュリティ】マシン利用時間超過のお知らせ", "utf-8")
    msg["From"] = SENDER_EMAIL
    msg["To"] = email
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        # Gmailのサーバーに接続 → TLSで暗号化 → ログイン → 送信、の順。timeout は、応答が無いときに延々と待たないための指定。
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())
    except Exception as e:
        print(f" 違反メールの送信に失敗（{name} / {member_id}）: {e}")
        return False
    print(f" 【違反メール送信】 {name}（{member_id}）宛：{machine_name} の時間超過（{_fmt_seconds(duration)} / 上限 {_fmt_seconds(limit)}）")
    return True

def notify_overtime_async(member_id, machine_key, start, end, duration):
    """時間超過を会員にメールで知らせる。別スレッドで送るので、カメラ処理は止まらない。"""
    if not VIOLATION_MAIL_ENABLED:
        return
    if not member_id:
        print(" 時間超過ですが会員IDが分からないため、違反メールは送りません")
        return
    # 同じ会員に短い間隔で何通も送らないための確認。顔の識別が揺れて、同じ人の利用が何度も区切られても、メールは1通で済む。
    now = time.time()
    with _violation_mail_lock:
        last = _violation_mail_last.get(member_id)
        if last is not None and now - last < VIOLATION_MAIL_COOLDOWN:
            print(f" 違反メールは直近に送信済みのためスキップ: {member_id}")
            return
        _violation_mail_last[member_id] = now

    # 実際の送信は別スレッドで行う。SMTPは数秒かかることがあり、カメラのループを止めたくないため。
    def worker():
        ok = _send_overtime_mail(member_id, machine_key, start.strftime("%Y-%m-%d %H:%M:%S"),
                                 end.strftime("%Y-%m-%d %H:%M:%S"), duration)
        if not ok:   # 送れなかったときは、次の機会に再送できるよう記録を戻す
            with _violation_mail_lock:
                _violation_mail_last.pop(member_id, None)

    threading.Thread(target=worker, daemon=True).start()

def _send_notification_mail(member_id, subject, body):
    """指定した会員に任意の件名・本文のメールを送る共通ヘルパー"""
    try:
        conn = get_conn()
        try:
            row = _exec(conn, "SELECT name, email FROM members WHERE member_id = %s", (member_id,)).fetchone()
        finally:
            conn.close()
    except Exception as e:
        print(f" 通知メール：会員情報の取得に失敗: {e}")
        return False

    if row is None or not row[1]:
        print(f" 通知メールを送れません：会員 {member_id} のメールアドレスが見つかりません")
        return False
    name, email = row

    # 本文の前に宛名、後ろに署名を付けて、どの通知も同じ体裁にする。
    full_body = f"{name} 様\n\n{body}\n\nジムセキュリティ管理システム"
    
    from email.header import Header
    msg = MIMEMultipart()
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = SENDER_EMAIL
    msg["To"] = email
    msg.attach(MIMEText(full_body, "plain", "utf-8"))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())
    except Exception as e:
        print(f" 通知メールの送信に失敗（{name} / {member_id}）: {e}")
        return False
    print(f" 【通知メール送信成功】 {name}（{member_id}）宛：{subject}")
    return True

def notify_esthe_over_async(member_id, start_str, end_str):
    """エステ室の長期占有超過メールを非同期送信"""
    if not VIOLATION_MAIL_ENABLED or not member_id: return
    def worker():
        _send_notification_mail(
            member_id,
            "【ジムセキュリティ】エステ室の利用時間超過に関するお知らせ",
            f"エステ室のご利用時間が終了時刻を過ぎて留まっていることが検知されました。\n\n■ 予約時間：{start_str} 〜 {end_str}\n\n速やかにご退出をお願いいたします。"
        )
    threading.Thread(target=worker, daemon=True).start()

def notify_esthe_noshow_async(member_id, start_str):
    """エステ無断キャンセルメールを非同期送信"""
    if not VIOLATION_MAIL_ENABLED or not member_id: return
    def worker():
        _send_notification_mail(
            member_id,
            "【ジムセキュリティ】エステ室のご予約についての確認（無断キャンセル）",
            f"ご予約いただいていたエステ室の時間にお越しいただけなかったため、無断キャンセル（No-Show）として処理されました。\n\n■ 予約開始時間：{start_str}\n\nペナルティポイントが加算されますのでご注意ください。"
        )
    threading.Thread(target=worker, daemon=True).start()

def notify_cotrailing_async(member_id, timestamp_str):
    """共連れ・不正入館の警告メールを非同期送信"""
    if not VIOLATION_MAIL_ENABLED or not member_id: return
    def worker():
        _send_notification_mail(
            member_id,
            "【ジムセキュリティ】入館ルール違反（共連れ）に関する警告",
            f"入館ゲートにおいて、お客様の認証直後に未登録の同行者を侵入させた（共連れ）ことが検知されました。\n\n■ 検知日時：{timestamp_str}\n\n利用規約違反となりますので、今後のご利用には十分ご注意ください。"
        )
    threading.Thread(target=worker, daemon=True).start()
# =========================================================================
# データベース書き込みのキュー化（ロック競合防止用）
# =========================================================================
# 書き込みの要求をいったんここに溜めて、専用スレッドが順番に実行する。
# カメラ処理が書き込み待ちで固まらないようにするのと、MySQLへの同時書き込みの競合を避ける目的。
db_write_queue = Queue()

def db_writer_worker():
    """データベースへの書き込みを1つのスレッドで直列化して実行する常駐ワーカー"""
    while True:
        # 要求が来るまでここで待つ。None が来たらスレッドを終わらせる（今のところ None を入れている箇所は無い）。
        task = db_write_queue.get()
        if task is None:
            db_write_queue.task_done()
            break
        # 1件ごとに接続を開いて、実行・コミットして閉じる。失敗しても例外はログに出すだけで、ワーカー自体は止めない。
        sql, params = task
        try:
            conn = get_conn()
            try:
                conn.cursor().execute(sql, params)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print(f" DB書き込みエラー: {e}")
        finally:
            db_write_queue.task_done()

# アプリ起動時に専用スレッドを開始
threading.Thread(target=db_writer_worker, daemon=True).start()

def execute_db_write(sql, params=()):
    """データベースへ書き込む際は、この関数を呼び出す（キューに追加する）"""
    db_write_queue.put((sql, params))

# =========================================================================
# システム状態管理（ステート）用クラス
# =========================================================================
# システム全体の状態（入退館の人数、QRの認証状況、館内にいる会員、マシンの利用状況など）をまとめて持つクラス。
# カメラ処理のスレッドとWeb側の処理の両方から触るので、書き換えるときは self._lock で守っている。
class SystemStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        # 画面に出す入館・退館の人数（起動してからの累計）。
        self.in_count = 0
        self.out_count = 0
        # 直近に読んだQRの結果。会員なら「名前 (会員ID)」、失敗なら "Invalid QR"、まだ何も読んでいなければ "None"。
        self.last_scanned_qr = "None"
        self.last_qr_time = 0.0
        self.accepted_token = None   # 直近に認証成功したQRトークン
        self.rejected_token = None   # 直近に拒否した（未登録/読み取り崩れ）QRの文字列
        self.rejected_time = 0.0
        # 共連れ・不正入館の警告を画面に出すかどうか。
        self.co_trailing_alert = False
        
        # 館内にいる会員（QR＋顔で正規に入館し、まだ退館していない人）。退館時の識別はこの人たちに限る
        self.inside_members = {}   # {member_id: 入館時刻}
        self.last_clothes_dist = None   # 直近の服装推定で採用した色の差（診断表示用）
        self._clothes_log_time = 0.0
        self.clothes_diag = "not evaluated"   # 直近の服装照合の結果（画面に表示。cv2は日本語を描けないので英数字）
        # 起動時に、直近24時間のログから「いま館内にいる人」を復元しておく。
        self._restore_inside_members()

        # 入口の1:1照合用に直近の会員情報と顔特徴量を保持
        self.last_member_id = None
        self.last_member_embedding = None

        # ルーム監視用の状態管理
        # 以前のルーム監視で使っていた入れ物。今は machine_states を使うので、参照されていない。
        self.active_users = {} 
        
        # マシンごとの利用状況。user=表示名、member_id=会員ID、start_time=座り始めた時刻(UNIX秒)、
        # duration=経過秒、status=free/using/overtime、last_seen=最後に座っているのを確認した時刻。
        self.machine_states = {
            k: {"user": "-", "duration": 0.0, "status": "free", "start_time": None, "last_seen": None, "member_id": None} 
            for k in MACHINE_LIMITS.keys()
        }
        
        # 画面(WebSocket)へ送る更新データの待ち行列。ws_broadcast_loop が取り出して配信する。
        self.update_queue = Queue()
        self.recent_entries = [] # 直近の入室者リスト

    def _restore_inside_members(self):
        """再起動で「館内にいる人」が消えないよう、直近24時間のログから復元する。"""
        try:
            from datetime import timedelta
            # 24時間より古いログは見ない（それより前に入館して退館の記録が無い人は、もう帰ったものとみなす）。
            since = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            conn = get_conn()
            try:
                # 自分の予約を、開始が新しい順に最大20件取る。
                rows = _exec(conn, 
                    "SELECT timestamp, direction, member_id, is_alert FROM passing_logs WHERE timestamp >= %s ORDER BY id",
                    (since,)).fetchall()
            finally:
                conn.close()
        except Exception as e:
            print(f" 館内の会員の復元に失敗: {e}")
            return
        # ログを古い順に再生して、IN なら館内に追加、OUT なら外す。最後に残った人が「いま館内にいる人」になる。
        for ts, direction, label, alert in rows:
            # member_id 列は「名前 (会員ID)」の形。末尾の括弧の中身が会員ID。この形でないログは読み飛ばす。
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

    def add_recent_entry(self, member_id, name, color_feat, body_ratio, current_time):
        with self._lock:
            # 古くなった記録を捨ててから、新しい記録を足す。
            self.recent_entries = [e for e in self.recent_entries if current_time - e["time"] <= RECENT_ENTRY_TTL]
            self.recent_entries.append({
                "member_id": member_id,
                "name": name,
                "color_feat": color_feat,
                "body_ratio": body_ratio,
                "time": current_time
            })

    def match_recent_entry_by_feature(self, kxy, kconf, frame, current_time):
        """顔が見えない人を、入口で記録した服装と体格から推定する。"""
        if not CLOTHES_FALLBACK_ENABLED:
            self.clothes_diag = "off"
            return None, None
        with self._lock:
            candidates = [e for e in self.recent_entries if current_time - e["time"] <= RECENT_ENTRY_TTL]
            if not candidates:
                self.clothes_diag = "no clothes record"
                return None, None

            # いま見えている人の服の色と体格の比率を測る。どちらかが測れなければ（骨格が取れていない等）判断しない。
            target_color = get_box_color_feature(frame, kxy, kconf)
            target_ratio = get_body_proportion_feature(kxy, kconf)
            
            if target_color is None or target_ratio is None:
                self.clothes_diag = "no torso/pose for feature"
                return None, None

            # 会員ごとに「服の色がいちばん近い記録」だけを残す（同じ人が何度も入館していても、1人として数えるため）。
            per_member = {}
            for entry in candidates:
                # 1. 服の色の差
                d_color = clothes_distance(entry["color_feat"], target_color)
                
                # 2. 体格（骨格比率）の差
                entry_ratio = entry.get("body_ratio")
                d_ratio = abs(entry_ratio - target_ratio) if entry_ratio is not None else 0.0
                
                # 体格が違いすぎる場合は除外
                if d_ratio > 0.15:
                    continue
                    
                cur = per_member.get(entry["member_id"])
                if cur is None or d_color < cur[0]:
                    per_member[entry["member_id"]] = (d_color, entry)
                    
            if not per_member:
                self.clothes_diag = "rejected by body ratio mismatch"
                return None, None

            # 色の差が小さい順に並べる。1位を採用の候補、2位を比較の相手にする。
            ranked = sorted(per_member.values(), key=lambda x: x[0])
            best_d, best = ranked[0]
            second_d = ranked[1][0] if len(ranked) > 1 else None

            # 判定ロジック（上限や2位との差のチェック）
            if best_d >= CLOTHES_MAX_DIST:
                self.clothes_diag = f"color dist too large ({best_d:.1f})"
                return None, None
            elif second_d is not None and (second_d - best_d) < CLOTHES_MIN_MARGIN:
                self.clothes_diag = "too close to 2nd candidate"
                return None, None

            self.clothes_diag = f"match! d={best_d:.1f}"
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
            # 直前に認証したのと同じQRを DUPLICATE_QR_WINDOW 秒以内にまた読んだら、同じ提示の続きなので処理しない。
            # そのとき last_qr_time だけ更新して、認証の有効期限を延ばす（かざし続けているあいだは切れない）。
            if qr_data == self.accepted_token and (now - self.last_qr_time) <= DUPLICATE_QR_WINDOW:
                self.last_qr_time = now
                return False
            # 拒否済みの文字列も、しばらくは再判定しない（毎フレームDBを引いて、失敗ログが溢れるのを防ぐ）。
            if qr_data == self.rejected_token and (now - self.rejected_time) <= DUPLICATE_QR_WINDOW:
                return False
            return True

    def process_qr(self, qr_data: str, current_time: float):
        with self._lock:
            conn = get_conn(dict_cursor=True)
            cursor = conn.cursor()
            # QRの文字列(qr_token)から会員を探す。
            cursor.execute("SELECT member_id, name, face_embedding FROM members WHERE qr_token = %s", (qr_data,))
            member = cursor.fetchone()
            conn.close()

            if member:
                self.accepted_token = qr_data
                self.rejected_token = None
                self.last_scanned_qr = f"{member['name']} ({member['member_id']})"
                # 入口の顔照合に使うため、この会員のIDと顔特徴量を覚えておく（保存してあるバイト列を float32 の配列に戻す）。
                self.last_member_id = member['member_id']
                if member['face_embedding']:
                    self.last_member_embedding = np.frombuffer(member['face_embedding'], dtype=np.float32)
                else:
                    self.last_member_embedding = None
                print(f" 【QR認証成功】 歓迎: {member['name']} (ID: {member['member_id']})")
            else:
                self.rejected_token = qr_data
                self.rejected_time = current_time
                # 有効な認証が残っているあいだに、QRの読み取りが崩れて別の文字列になることがある。
                # そのときは認証を取り消さず、読み取りを無視して継続する（正規の会員が通れなくなるのを防ぐ）。
                auth_still_valid = (
                    self.last_member_id is not None
                    and (current_time - self.last_qr_time) <= CO_TRAILING_WINDOW
                )
                if auth_still_valid:
                    print(f" 【QR読み取り無視】 有効な認証の直後に未登録の文字列を検出（認証は維持）: {qr_data!r}")
                    return
                self.accepted_token = None
                self.last_scanned_qr = "Invalid QR"
                self.last_member_id = None
                self.last_member_embedding = None
                print(f" 【QR認証失敗】 無効なトークンです: {qr_data!r} (長さ {len(qr_data)})")

            # 認証の成否にかかわらず時刻を更新し、前回の警告表示は消して、画面に最新の状態を送る。
            self.last_qr_time = current_time
            self.co_trailing_alert = False
            self.push_update()

    # 共連れ警告の表示をON/OFFして、画面に知らせる。
    def set_alert(self, state: bool):
        with self._lock:
            self.co_trailing_alert = state
            self.push_update()

    # 入館・退館を1件記録する。人数のカウントを増やし、通行ログをDBに書き込む（書き込みはキュー経由）。
    def register_pass(self, direction: str, member_id: str, is_alert: int, note: str = None):
        with self._lock:
            if direction == "IN":
                self.in_count += 1
            else:
                self.out_count += 1
            
            execute_db_write(
                "INSERT INTO passing_logs (timestamp, direction, member_id, is_alert, note) VALUES (%s, %s, %s, %s, %s)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), direction, member_id, is_alert, note)
           )
        self.push_update()

    # マシンの利用が1回終わったときに、利用ログをDBへ保存する。上限を超えていたら違反メールも送る。
    # 呼び出す側（update_machine_occupancy など）が self._lock を持っている前提で動く。
    def _log_occupancy(self, m_key, m_state):
        # 誰も使っていない・開始時刻が無い・時間が0以下のものは記録しない。
        if m_state["user"] == "-" or m_state["start_time"] is None or m_state["duration"] <= 0:
            return
        # 開始時刻と、開始＋経過秒の終了時刻を求める（UNIX秒 → 日時）。
        start = datetime.fromtimestamp(m_state["start_time"])
        end = datetime.fromtimestamp(m_state["start_time"] + m_state["duration"])
        # 会員IDが分かっていればそれを、無ければ表示名を記録する。
        who = m_state.get("member_id") or m_state["user"]
        try:
            # キュー経由の安全な書き込みに置き換え
            execute_db_write(
                "INSERT INTO occupancy_logs (member_id, start_time, end_time, duration, machine) VALUES (%s, %s, %s, %s, %s)",
                (who, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"),
                float(m_state["duration"]), m_key)
            )
        except Exception as e:
            print(f" 占有ログの保存に失敗: {e}")
        # 上限を超えていたら、会員に違反メールを送る。
        if m_state["duration"] > MACHINE_LIMITS[m_key]["limit"]:
            notify_overtime_async(m_state.get("member_id"), m_key, start, end, m_state["duration"])
            
    # 「このマシンをこの人が使っている」と報告されるたびに呼ばれて、利用者と経過時間を更新する。
    # 利用者が入れ替わったら、前の人の利用を保存してから新しい人の計測を始める。Guest（誰か分からない人）は計測しない。
    def update_machine_occupancy(self, active_machine_key, user_name, current_time, started_at=None, member_id=None):
        with self._lock:
            for m_key in self.machine_states.keys():
                if m_key == active_machine_key and user_name != "Guest":
                    m_state = self.machine_states[m_key]
                    m_state["last_seen"] = current_time
                    # 利用者が変わった（または空きの状態から使い始めた）。前の利用を記録して、新しい人の計測を始める。
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

                    # 上限を超えていれば overtime、そうでなければ using にする。
                    limit = MACHINE_LIMITS[m_key]["limit"]
                    if m_state["duration"] > limit:
                        m_state["status"] = "overtime"
                    else:
                        m_state["status"] = "using"

    # 利用者が最後に確認されてから SEAT_RELEASE_GRACE 秒たったマシンは、立ち去ったとみなして空きに戻す。
    # このとき利用ログも保存される。
    def release_idle_machines(self, current_time):
        with self._lock:
            for m_key, m_state in self.machine_states.items():
                if m_state["user"] != "-" and m_state["last_seen"] is not None \
                        and (current_time - m_state["last_seen"]) > SEAT_RELEASE_GRACE:
                    print(f" 【利用終了】 {m_state['user']} / 占有 {m_state['duration']:.0f}秒")
                    self._log_occupancy(m_key, m_state)
                    m_state.update({"user": "-", "duration": 0.0, "status": "free",
                                    "start_time": None, "last_seen": None, "member_id": None})

    # 画面（ダッシュボード）へ送るデータを組み立てる。マシンの状態、直近の通行ログ、エステ室の状況が入る。
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

        conn = get_conn(dict_cursor=True)
        cursor = conn.cursor()
        # 画面の「直近の通行」欄のために、最新の5件を取る。
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

        # エステ室の現在の状態をデータベースから取得してpayloadに含める
        esthe_user_name = "-"
        esthe_status_str = "FREE"
        try:
            conn_e = get_conn(dict_cursor=True)
            cur_e = conn_e.cursor()
            # 利用中(in_progress)か超過(overtime)の予約を1件取り、利用者の名前と状態を画面に出す。
            cur_e.execute("""
                SELECT r.id, m.name, r.status, r.end_time 
                FROM esthe_reservations r
                JOIN members m ON r.member_id = m.member_id
                WHERE r.status IN ('in_progress', 'overtime')
                LIMIT 1
            """)
            active_esthe = cur_e.fetchone()
            conn_e.close()

            if active_esthe:
                esthe_user_name = active_esthe["name"]
                if active_esthe["status"] == "overtime":
                    esthe_status_str = "OVERTIME"
                else:
                    esthe_status_str = "USING"
        except Exception as e:
            print(f" エステ状態の取得エラー: {e}")

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
            "logs": logs_payload,
            # エステ室の現在の利用者と状態
            "esthe_user": esthe_user_name,
            "esthe_status": esthe_status_str
        }

    # 最新の状態を組み立てて、画面への配信キューに入れる。
    def push_update(self):
        self.update_queue.put(self.build_payload())

# アプリ全体で共有する状態管理のインスタンス。
state = SystemStateManager()

# カメラ処理のスレッドが描き終えた最新の1フレーム。/video がこれを配信する。読み書きは render_lock を取ってから行う。
current_output_frame = None
render_lock = threading.Lock()

# 2つの顔特徴量のコサイン類似度（-1〜1。1に近いほど同じ人）。どちらかがゼロベクトルなら 0 を返す。
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

    # キャッシュが破棄済み、または ttl 秒以上たっていたらDBから読み直す。
    # 呼び出す側は (会員IDの配列, 名前の配列, 特徴量の行列) を受け取る。
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
            conn = get_conn(dict_cursor=True)
            try:
                rows = _exec(conn, 
                    "SELECT member_id, name, face_embedding FROM members WHERE face_embedding IS NOT NULL"
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:
            print(f" 会員顔データの読み込みに失敗（前回のキャッシュを継続使用）: {e}")
            return
        ids, names, vecs, dim = [], [], [], None
        for r in rows:
            v = np.frombuffer(r["face_embedding"], dtype=np.float32)
            n = np.linalg.norm(v)
            if n == 0:
                continue
            # 最初の1件の次元数を基準にして、違う次元の特徴量（別のモデルで作ったもの等）は混ぜない。
            if dim is None:
                dim = v.shape[0]
            if v.shape[0] != dim:
                continue
            ids.append(r["member_id"])
            names.append(r["name"])
            vecs.append(v / n)   # 正規化済みで保持 → 内積がそのままコサイン類似度
        self._ids, self._names = ids, names
        self._matrix = np.stack(vecs) if vecs else None

# 全体で共有する会員の顔キャッシュ。登録・削除したときに invalidate() する。
member_cache = MemberFaceCache()

# 2つの長方形(x1,y1,x2,y2)の IoU（重なっている面積 ÷ 合わせた面積）。1に近いほどぴったり重なっている。
def _iou(a, b):
    inter = _box_inter(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0

def find_pose_for_box(pose_result, box, min_iou=0.3):
    """骨格推定の結果から、指定した人の枠に一番重なる人の (keypoints xy, conf) を返す。見つからなければ (None, None)。"""
    if (pose_result is None or pose_result.boxes is None or len(pose_result.boxes) == 0
            or pose_result.keypoints is None):
        return None, None
    # 骨格推定の結果のうち、指定した人の枠といちばん重なる1人を選ぶ。min_iou に届かなければ別人とみなして見つからない扱いにする。
    p_boxes = pose_result.boxes.xyxy.cpu().numpy()
    kxy = pose_result.keypoints.xy.cpu().numpy()
    # 骨格点の信頼度が無いモデルの場合は、全部 1.0（見えている）として扱う。
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
    # 骨格点の番号は COCO 形式: 5=左肩、6=右肩、11=左腰、12=右腰（ほか 13=左膝、14=右膝）。
    # 肩がはっきり見えていないと、胸の位置が決められない。
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
    # 測る範囲は、横が両肩の内側（肩幅の中央50%）、縦が肩から腰までの25%〜70%（胸からみぞおちのあたり）。
    # 腕・首・襟といった、服の色以外が入りにくい場所を選んでいる。
    x1 = min(float(ls[0]), float(rs[0])) + 0.25 * sw
    x2 = max(float(ls[0]), float(rs[0])) - 0.25 * sw
    y1 = sy + 0.25 * (hy - sy)
    y2 = sy + 0.70 * (hy - sy)
    # 画像の外にはみ出さないよう切り詰める。小さすぎる範囲は測らない。
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
    # 肌色の判定には YCrCb が向いている（明るさと色味が分かれているため）。
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)   # 手・首・顔などの肌色
    # 肌色でない画素だけを使う。使える画素が少なすぎる、または範囲の6割以上が肌色なら当てにならないので None を返す。
    keep = ~skin
    if keep.sum() < 50 or keep.sum() < 0.4 * keep.size:
        return None
    # 服の色は Lab 色空間の中央値で代表させる（平均より外れ値に強く、Lab は人の見た目の色の差に近い）。
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[keep].astype(np.float32)
    return np.median(lab, axis=0)

def get_body_proportion_feature(kxy, kconf):
    """骨格データから体格の比率（肩幅÷胴の長さなど）を計算する"""
    if kxy is None or kconf is None:
        return None
    # 肩（5, 6）と腰（11, 12）の信頼度が十分かチェック
    if kconf[5] < KP_CONF or kconf[6] < KP_CONF or kconf[11] < KP_CONF or kconf[12] < KP_CONF:
        return None
    shoulder_w = abs(float(kxy[5][0]) - float(kxy[6][0]))
    torso_h = abs(float((kxy[5][1] + kxy[6][1]) / 2) - float((kxy[11][1] + kxy[12][1]) / 2))
    if torso_h < 1.0:
        return None
    return shoulder_w / torso_h

# Lab の色を、画面に描くための BGR の色に変換する。
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
    # 比べられないとき（片方が無い、形が違う）は、最大級の差(255)を返して「別の服」として扱う。
    if a is None or b is None or a.shape != b.shape:
        return 255.0
    return float(np.linalg.norm((a - b) * _CLOTHES_WEIGHT))

# 入館の判定結果を (OKか, 理由) で返す関数。今の入口の処理（EntranceSession）からは呼ばれていない、以前の実装の名残。
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
    # 会員側は正規化済みなので、行列との内積がそのままコサイン類似度になる。いちばん高い会員を返す。
    sims = member_matrix @ (fe / fn)
    idx = int(np.argmax(sims))
    return member_ids[idx], member_names[idx], float(sims[idx])

# 人の体の下半分が、椅子の枠とどれだけ重なっているか（0〜1）。座っていれば下半身が椅子と重なる。
# 椅子が複数あれば、いちばん重なっているものの値を使う。
def sit_overlap_ratio(p_box, seat_boxes):
    px1, py1, px2, py2 = p_box
    # 人の枠の縦の中央より下を「下半身」とみなす。
    mid_y = py1 + (py2 - py1) * 0.5
    lower_area = max(1.0, (px2 - px1) * (py2 - mid_y))
    best = 0.0
    for sx1, sy1, sx2, sy2 in seat_boxes:
        ix1, iy1 = max(px1, sx1), max(mid_y, sy1)
        ix2, iy2 = min(px2, sx2), min(py2, sy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        best = max(best, inter / lower_area)
    return best

# 「座っている」判定が一定時間つづいたかを確認するクラス。
# 椅子の前をかすめただけ、判定が一瞬ぶれただけで、タイマーが始まったり巻き戻ったりしないようにする。
class SitConfirmer:
    def __init__(self):
        self.since = None
        self.last_seen = 0.0
        self.had_gap = False   # 前回「座っている」と判定したあと、「座っていない」判定のフレームが挟まったか

    def update(self, sitting_now, now):
        # 座っていないと判定されたら、「途切れた」という印だけ付けて、確定にはしない。
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

# 顔や服装で特定した会員を、一定時間おぼえておくクラス。
# 座っているあいだは顔が下や横を向いて見えなくなりがちなので、一度特定できたら IDENTITY_HOLD_SECONDS 秒は同じ人として扱い続ける。
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
        # いまのフレームで特定できたら、その人で上書きして時刻を更新する。
        if recognized_name != "Guest":
            self.name, self.member_id, self.time, self.estimated = recognized_name, recognized_id, now, estimated
            return recognized_name, recognized_id, False, estimated
        # 特定できなかったが、直前の特定がまだ有効期間内なら、その人のままにする。座っているあいだは期限を延ばし続ける。
        if self.has_identity(now):
            if is_sitting:
                self.time = now
            return self.name, self.member_id, True, self.estimated
        # 期限切れ。誰とも分からない状態に戻す。
        self.name, self.member_id, self.estimated = None, None, False
        return "Guest", None, False, False

# 2つの枠が重なっている部分の面積。重なっていなければ 0。
def _box_inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

# 枠の面積。0 で割ってしまわないよう、最小を 1 にしている。
def _box_area(a):
    return max(1.0, (a[2] - a[0]) * (a[3] - a[1]))

# 椅子の位置を覚えておく処理。人が座ると椅子が人に隠れて検出されなくなるので、いちど見つけた椅子は SEAT_MEMORY_SECONDS 秒は覚えておく。
# memory は [{box, last_seen, live}] のリストで、毎フレーム、検出結果(detected)と突き合わせて更新する。
# 戻り値は、いま有効な椅子の枠の一覧。
def update_seat_memory(memory, detected, person_boxes, now):
    for m in memory:
        m["live"] = False
    used = set()
    for m in memory:
        # この記憶の椅子が、人に隠れている（人の枠が椅子に十分重なっている）かどうか。
        occluded = any(_box_inter(p, m["box"]) / _box_area(m["box"]) >= SEAT_OCCLUDED_RATIO for p in person_boxes)
        best_j, best_score = None, 0.0
        # 検出された椅子の中から、この記憶と同じ椅子とみなせるものを探す。
        # IoU が高いか、検出枠が記憶の枠にほぼ収まっていれば同じ椅子とする。
        for j, d in enumerate(detected):
            if j in used:
                continue
            inter = _box_inter(m["box"], d)
            iou = inter / (_box_area(m["box"]) + _box_area(d) - inter)
            contained = inter / _box_area(d)
            score = max(iou, contained if contained >= 0.7 else 0.0)
            if score >= 0.3 and score > best_score:
                best_j, best_score = j, score
        # 見つかった: 位置を更新する（ただし人に隠れて枠が欠けて見えているときは、元の枠を残す）。
        if best_j is not None:
            used.add(best_j)
            m["live"] = True
            m["last_seen"] = now
            if not occluded:
                m["box"] = detected[best_j]
        # 見つからないが人に隠れているだけなので、記憶の期限を延ばす。
        elif occluded:
            m["last_seen"] = now
    # どの記憶にも対応しなかった検出は、新しい椅子として覚える。
    for j, d in enumerate(detected):
        if j not in used:
            memory.append({"box": d, "last_seen": now, "live": True})
    # 期限切れの椅子を忘れる。
    memory[:] = [m for m in memory if now - m["last_seen"] <= SEAT_MEMORY_SECONDS]
    return [m["box"] for m in memory]

# 指定した骨格点のうち、信頼度が十分で画面内にある点だけの平均位置を返す（左右の腰など）。1つも見えなければ None。
def _visible_mean(kxy, kconf, idxs):
    pts = [kxy[i] for i in idxs if kconf[i] >= KP_CONF and (kxy[i][0] > 0 or kxy[i][1] > 0)]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

# 腰の位置が椅子の枠の上にあるか（横は枠の内側、縦は枠の上端から margin 分だけ上〜下端まで）。
def _hip_on_seat(hip, seat, margin=None):
    sx1, sy1, sx2, sy2 = seat
    margin = HIP_SEAT_MARGIN if margin is None else margin
    top = sy1 - margin * (sy2 - sy1)
    return sx1 <= hip[0] <= sx2 and top <= hip[1] <= sy2

# 人が椅子に座っているかを判定して、(座っているか, 椅子との重なり, 理由の文字列) を返す。
# 順に、1) 下半身が椅子と十分重なっている 2) 腰が椅子の上にある 3) 膝と腰の高さの差が小さい（脚が曲がっている）を確かめる。
# 骨格が一部しか取れていないときは、見えている点だけで代わりの判定をする。
def evaluate_sitting(box, kxy, kconf, seat_boxes):
    overlap = sit_overlap_ratio(box, seat_boxes) if seat_boxes else 0.0
    if overlap < SIT_OVERLAP_RATIO:
        return False, overlap, "overlap"

    # 骨格点の番号: 11,12=腰、5,6=肩、13,14=膝。
    hip = _visible_mean(kxy, kconf, (11, 12))
    shoulder = _visible_mean(kxy, kconf, (5, 6))
    if hip is not None:
        if not any(_hip_on_seat(hip, s) for s in seat_boxes):
            return False, overlap, "hip-outside"
        knee = _visible_mean(kxy, kconf, (13, 14))
        if knee is not None:
            torso = (hip[1] - shoulder[1]) if shoulder is not None else (box[3] - box[1]) * 0.3
            torso = max(torso, 1.0)
            # 膝が腰よりどれだけ下にあるかを、胴の長さで割る。立っていると約1、座ると小さくなる。しきい値以上なら立っているとみなす。
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
# 入口の認証セッション（QR → 顔の一致 → 線の通過 を1人分まとめて管理）
# =========================================================================
class EntranceSession:
    """QRを読んだ会員1人分の状態を管理する（入口モード用）。

    ・会員の顔と一致した時点で「顔認証OK」にする（線をまたぐ瞬間を待たない）。
    ・認証後は、顔 → 追跡ID → 直前の位置に近い枠 の順で本人を探し直す。
      そのため、処理が遅れて枠のIDが付け替わっても本人を見失いにくい。
    ・線の手前（左）にいた本人が、線から一定距離だけ向こう側（右）で見つかった時点で通過（入館）とする。
      線をまたいだ瞬間のフレームは必要ない。"""

    def __init__(self, member_id, label, embedding, qr_time=0.0, token=None):
        self.member_id = member_id
        self.label = label
        self.embedding = embedding
        self.qr_time = qr_time       # このQRを読んだ（または読み続けている）時刻。有効時間はここから数える
        self.token = token           # 読み取ったQRの文字列（かざし続けているときの延長用）
        self.consumed = False        # 共連れの判定にすでに使った（二重に警報を出さないため）
        self.clothes_feat = (None, None)   # 顔認証OK時点で測った (服の色, 体格)
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

    # 枠 (x1,y1,x2,y2) の中心の座標。
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

    def grant_valid(self, now):
        """このQRが、共連れ（同じQRで続けて入る人）の判定にまだ使えるか。
        本人が通過して認証セッションが終わったあとも、有効時間内は使える。"""
        return (not self.consumed) and (now - self.qr_time) <= CO_TRAILING_WINDOW

    def update(self, now, tracked, faces, claimed=()):
        """1フレーム分の更新。tracked は [(track_id, box), ...]。
        claimed は、ほかの認証セッションが本人として追っている追跡ID（探し直しの候補から外す）。
        戻り値: None / "authed"（顔認証OK）/ "crossed"（通過=入館）/ "expired"（時間切れ）"""
        if self.done:
            return None
        # ほかの会員の認証セッションが追っている枠は、自分の本人候補から外す。
        others = set(claimed)
        others.discard(self.track_id)
        # QRを読んでから有効時間が過ぎたら、この認証セッションは時間切れで終わる。
        if now - self.qr_time > CO_TRAILING_WINDOW:
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
                    if (t in others or d > allowed or (best_d is not None and d >= best_d)
                            or self._has_other_face(b)):
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
# 線の通過検出・共連れ判定
# =========================================================================
class LineTracker:
    """追跡IDごとに、入口ラインの左右どちら側にいるかを覚えて、通過（左→右 "in" / 右→左 "out"）を検出する。

    ・ライン±margin_px を越えるまで左右を切り替えない（ヒステリシス）。
      ライン付近で枠が揺れても「入館→誤退館→誤警報」にならない。
    ・追跡IDが付け替わっても、見失った枠の近くに現れた新しい枠へ履歴を引き継ぐ。
    ・古い履歴は ttl 秒で捨てる（辞書が増え続けない）。"""

    def __init__(self, line_x, margin_px, ttl=TRACK_HISTORY_TTL):
        self.line_x = line_x
        self.margin = margin_px
        self.ttl = ttl
        self._h = {}   # {track_id: {"side": -1(左)/1(右), "x", "y", "time"}}

    def update(self, tid, x, y, now):
        rec = self._h.get(tid)
        if rec is None:
            # 初めて見た枠は、いまの位置で左右だけ決める（通過とはしない）
            self._h[tid] = {"side": -1 if x < self.line_x else 1, "x": x, "y": y, "time": now}
            return None
        event = None
        # 左側にいた枠が、ラインより margin 以上右に来たら「入館方向(in)」、逆なら「退館方向(out)」とする。
        # ライン付近で揺れているだけのあいだは左右が切り替わらないので、通過にはならない。
        if rec["side"] < 0 and x >= self.line_x + self.margin:
            rec["side"], event = 1, "in"
        elif rec["side"] > 0 and x <= self.line_x - self.margin:
            rec["side"], event = -1, "out"
        rec["x"], rec["y"], rec["time"] = x, y, now
        return event

    def link_new_tracks(self, positions, now):
        """positions: {track_id: (x, y)}（いまのフレームに見えている枠）。
        履歴のない新しい枠を、最近見失った枠のうち近いものと対応づけ、履歴を引き継ぐ。
        戻り値: {新しいID: 古いID}（ほかの履歴を引き継ぐときに使う）"""
        # new_ids=いまのフレームで初めて見えた枠、lost_ids=直前まで見えていたのに今は見えない枠。
        # 新しい枠が、見失った枠の近くに現れたのなら、同じ人とみなす。
        new_ids = [t for t in positions if t not in self._h]
        lost_ids = [t for t, r in self._h.items() if t not in positions and now - r["time"] <= TRACK_LINK_SECONDS]
        pairs = []
        for n in new_ids:
            nx, ny = positions[n]
            for o in lost_ids:
                r = self._h[o]
                allowed = TRACK_LINK_BASE_PX + TRACK_LINK_PX_PER_SEC * (now - r["time"])
                d = float(np.hypot(nx - r["x"], ny - r["y"]))
                if d <= allowed:
                    pairs.append((d, n, o))
        # 距離が近い組み合わせから順に確定していく（1つの枠を2人に割り当てないため）。
        pairs.sort()
        mapping, used_new, used_old = {}, set(), set()
        for d, n, o in pairs:
            if n in used_new or o in used_old:
                continue
            used_new.add(n)
            used_old.add(o)
            mapping[n] = o
        # 履歴の持ち主を、古いIDから新しいIDに付け替える。
        for n, o in mapping.items():
            self._h[n] = self._h.pop(o)
        return mapping

    # 長いこと見えていない追跡IDの履歴を捨てる。
    def cleanup(self, now):
        for t in [t for t, r in self._h.items() if now - r["time"] > self.ttl]:
            del self._h[t]


def pick_suspect_session(candidates, center, now):
    """共連れの手引きをした可能性が高いQR所有者を選ぶ。
    直前まで位置が分かっていた本人のうち、通過した人に一番近い人 → なければ最後にQRを読んだ人。"""
    near = [s for s in candidates
            if s.pos is not None and s.last_seen is not None and now - s.last_seen <= ENTRANCE_REASSOC_SECONDS]
    if near:
        return min(near, key=lambda s: float(np.hypot(s.pos[0] - center[0], s.pos[1] - center[1])))
    return max(candidates, key=lambda s: s.qr_time)


class CrossingJudge:
    """線を左→右にまたいだ「どの認証セッションの本人でもない枠」を、共連れ／未認証侵入として判定する。

    ・QRは読んだが顔認証がまだの人がいるとき、その人自身が先に線を越えた可能性がある。
      すぐには決めず ENTRANCE_PRECROSS_GRACE 秒待ち、その間に顔が一致して本人の枠だと分かれば共連れにしない。
    ・QRが有効な人が複数いても、共連れとして減点するのは1人だけ（通過した人に近い本人。いなければ最後にQRを読んだ人）。
      判定に使ったQRは consumed にして二重に警報を出さない。"""

    def __init__(self):
        self.pending = []   # [{"tid", "center", "time", "cands"}]

    def reset(self):
        self.pending = []

    # 追跡IDが付け替わったとき、保留中の判定の対象IDも新しいほうに合わせる。
    def remap(self, old_id, new_id):
        for p in self.pending:
            if p["tid"] == old_id:
                p["tid"] = new_id

    @staticmethod
    def _co_trailing(cands, center, now, tid):
        suspect = pick_suspect_session(cands, center, now)
        suspect.consumed = True
        return ("co_trailing", suspect, tid)

    def on_cross_in(self, tid, center, now, sessions, owned_by_tid):
        """線を左→右にまたいだ枠 tid を判定する。owned_by_tid は {追跡ID: 本人として追っているセッション}。
        戻り値: 確定した判定のリスト [("co_trailing", 疑いのセッション, tid) / ("intrusion", None, tid)]"""
        # 越えたのが認証セッションの本人なら、正規の通過なので何もしない（本人の入館の判定は EntranceSession 側で行う）。
        owner = owned_by_tid.get(tid)
        if owner is not None:
            owner.armed = True   # 線の左から来たことが確かなので、本人の通過判定に使える
            return []
        # まだ有効なQRを持っている人（共連れの手引きをした可能性がある人）を集める。1人もいなければ、QRなしの侵入。
        cands = [s for s in sessions if s.grant_valid(now)]
        if not cands:
            return [("intrusion", None, tid)]
        # 顔認証がまだの人がいると、いま通ったのはその人本人かもしれない。猶予時間のあいだ、判定を保留する。
        if any((not s.done) and (not s.authed) for s in cands):
            self.pending.append({"tid": tid, "center": center, "time": now, "cands": cands})
            return []
        return [self._co_trailing(cands, center, now, tid)]

    def poll(self, now, owned_by_tid):
        """保留中の通過を見直す。本人の枠と分かれば取り下げ、猶予が過ぎれば確定する。"""
        # 保留中の通過を1件ずつ見直す。
        decided, keep = [], []
        for p in self.pending:
            owner = owned_by_tid.get(p["tid"])
            if owner is not None:
                owner.armed = True
                continue
            if now - p["time"] >= ENTRANCE_PRECROSS_GRACE:
                cands = [s for s in p["cands"] if not s.consumed]
                decided.append(self._co_trailing(cands, p["center"], now, p["tid"]) if cands
                               else ("intrusion", None, p["tid"]))
            else:
                keep.append(p)
        self.pending = keep
        return decided

# =========================================================================
# カメラ処理ループ
# =========================================================================
# カメラの映像を読み続けて、モードごとの判定を行うメインのループ（専用のスレッドで動く）。
# entrance=入口の入退館 / esthe=エステ室の入室と占有の監視 / room=トレーニングルームの座り判定とマシン利用時間 / register=カメラを手放して待機。
def camera_processing_loop():
    global current_output_frame, yolo_model
    # 重いライブラリはここで読み込む（Webサーバーの起動を遅らせないため）。
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis

    # 使うモデルは4つ。人の検出(入口・エステ用)、人と椅子の検出(ルーム用)、骨格推定、顔認識(insightface)。
    yolo_model = YOLO("yolo11n.pt")
    room_model = YOLO(ROOM_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    print(" 顔認識モデルを読み込み中...")
    face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)

    # カメラが開けるまで、3秒おきにやり直す。
    cap = open_camera()
    while not cap.isOpened():
        print(" エラー: カメラを開けませんでした。3秒後に再試行します...")
        time.sleep(3)
        cap.release()
        cap = open_camera()

    print(f">> カメラ解像度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
          f"（目標 {FRAME_W}x{FRAME_H} / YOLO imgsz={YOLO_IMGSZ} / 顔検出 {FACE_DET_SIZE[0]}x{FACE_DET_SIZE[1]}）")

    print(">> カメラ処理ループ開始")

    line_tracker = LineTracker(LINE_X, LINE_HYSTERESIS_PX)   # 追跡IDごとの線の左右（ヒステリシス付き）
    crossing_judge = CrossingJudge()                         # 本人の枠でない通過者の共連れ／侵入判定
    # 追跡IDごとの顔照合の記録 {追跡ID: {best:最大の類似度, time, member_id}}。
    face_sim_history = {}
    exit_id_history = {}   # 退館用：追跡IDごとに、顔で識別した会員 {member_id, name, sim, time}
    last_push_time = 0.0
    last_err_log = 0.0
    prev_frame_time = None   # 1フレームの処理時間の計測用（遅いとタイマー系の判定に影響するため画面に出す）
    loop_dt = 0.0
    last_slow_log = 0.0
    last_face_log = 0.0
    # 注意: 座り判定の確定タイマー(sit_confirmer)と本人の保持(identity_holder)は、それぞれ1つだけ作っていて、人ごとには持っていない。
    # ルームに複数人いるときは、人が入れ替わるたびに状態が混ざる。複数人を正確に扱うなら、人ごとに持たせる必要がある。
    sit_confirmer = SitConfirmer()
    seat_memory = []
    identity_holder = IdentityHolder()
    clothes_mismatch_since = None   # 服装で推定した本人の服が合わなくなり始めた時刻

    # 入館した人の服の色と体格の比率を、骨格推定で測って返す。測れなければ (None, None)。
    # あとでルームで顔が見えないときに、本人を推定するための材料になる。
    def entrance_body_and_clothes_feature(frame, box, member_id):
        try:
            res = pose_model(frame, conf=0.3, imgsz=YOLO_IMGSZ, verbose=False)[0]
            kxy, kcf = find_pose_for_box(res, box)
            color_feat = get_box_color_feature(frame, kxy, kcf)
            body_ratio = get_body_proportion_feature(kxy, kcf)
            return color_feat, body_ratio
        except Exception:
            return None, None
    
    def report_unauthorized_pass(kind, suspect, tid, sessions, now):
        """CrossingJudge の判定結果を、画面・ログ・アラートに反映する。"""
        if kind == "co_trailing":
            suspect_member = suspect.label   # 例: "Aさん (mem_01)"
            reason_note = f"共連れ検知 (Co-trailing): {suspect_member} のQR認証直後に未認証の人物(ID:{tid})が侵入しました"
            if not suspect.authed:
                reason_note += "（QR所有者の顔認証は完了していませんでした）"
            print(f" 【共連れ検知】 主犯/手助け: {suspect_member} | 侵入者ID: {tid}")
            state.set_alert(True)
            state.register_pass("IN", f"Co-trailing by {suspect_member}", 1, note=reason_note)
            notify_cotrailing_async(suspect.member_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            print(f" 【未認証侵入】 QRなしでの通過を検知 (ID:{tid})")
            state.set_alert(True)
            state.register_pass("IN", "Unknown / No QR", 1, note="QRなし・単独での未認証通過")
            state.last_scanned_qr = "None"

    # sit_log は今のところ参照されていない。
    sit_log = {"key": None, "time": 0.0}
    entrance_sessions = []    # 入口：QRを読んだ会員ごとの認証セッション
    esthe_qr_seen = {}        # エステ：QR文字列 → 最後に処理した時刻（同じQRを毎フレーム処理しないため）
    auth_banner = {"text": "", "color": (200, 200, 200), "until": 0.0}
    
    print("\n====================================================")
    print("  カメラ処理ループが稼働しました（椅子検知＋座り判定モード）。")
    print("====================================================\n")

    while cap.isOpened():
        try:
            current_time = time.time()
            
            with mode_lock:
                # 今のモードを読む（Webの画面を開いたときにも切り替わるため、ロックを取ってから読んでいる）。
                mode = active_mode
        
            # 【registerモード】ブラウザでの顔撮影のため、Python側のカメラを一時解放する
            if mode == "register":
                if cap.isOpened():
                    cap.release()
                    print(" [registerモード] ブラウザでの顔撮影のため、Python側のカメラを一時解放しました")
                time.sleep(0.5)
                continue

            # registerモードから他のモードに戻ったとき、カメラが閉じられていれば再オープンする
            if not cap.isOpened():
                print(" カメラを再接続します...")
                cap = open_camera()
                while not cap.isOpened():
                    time.sleep(1)
                    cap = open_camera()

            # 1フレームの処理時間を、移動平均でならして測る（処理が重いと、タイマー系の判定がずれるため）。
            if prev_frame_time is not None:
                loop_dt = 0.8 * loop_dt + 0.2 * (current_time - prev_frame_time)
            prev_frame_time = current_time
            
            # カメラから1フレーム読む。失敗したら少し待ってやり直す。
            success, frame = cap.read()
            if not success:
                time.sleep(0.03)
                continue

            # カメラが指定の解像度で出してくれなかったときは、ここで揃える（以降の座標の計算は FRAME_W/FRAME_H が前提）。
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # 入口モード以外のあいだは、入口の認証セッションと共連れの保留判定を捨てる（モードを戻したとき、古い状態が残らないように）。
            if mode != "entrance":
                entrance_sessions = []
                crossing_judge.reset()

            # エステ以外のモードのあいだは、エステ用の時刻を現在に合わせ続ける。
            # （エステモードに戻った直後に、古い時刻のせいで「無人が続いた」と誤判定して予約を完了にしないため）
            if mode != "esthe":
                esthe_last_person_time = current_time

            if mode == "entrance":
                # 入口: まずQRコードを探す。見つかったら緑の枠で囲む。
                qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame)
                if qr_bbox is not None and len(qr_bbox) > 0:
                    pts = qr_bbox[0].astype(int)
                    for i in range(4):
                        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), THICK)
                    if qr_data:
                        # 新しく提示されたQRなら会員を照合して、認証セッション（顔の一致と線の通過を見届ける係）を作る。
                        if state.should_process_qr(qr_data, current_time):
                            state.process_qr(qr_data, current_time)
                            if (state.accepted_token == qr_data and state.last_member_id is not None
                                    and state.last_member_embedding is not None):
                                # 同じ会員のセッションがすでにあれば、作り直す。
                                entrance_sessions = [s for s in entrance_sessions if s.member_id != state.last_member_id]
                                entrance_sessions.append(EntranceSession(state.last_member_id, state.last_scanned_qr,
                                                                         state.last_member_embedding,
                                                                         qr_time=current_time, token=qr_data))
                                auth_banner["until"] = 0.0
                            elif state.last_scanned_qr == "Invalid QR":
                                entrance_sessions = []
                        else:
                            # 同じQRをかざし続けているあいだは、認証の有効時間を延ばす。
                            for s in entrance_sessions:
                                if s.token == qr_data and not s.done and not s.consumed:
                                    s.qr_time = current_time

                # 画面に映っている全員の顔を検出して、特徴量を出す。
                faces = face_app.get(frame)

                # 人(class 0)を検出して、追跡IDを付ける。persist=True にすると、前のフレームとのIDの対応を引き継ぐ。
                yolo_results = yolo_model.track(frame, persist=True, classes=[0], imgsz=YOLO_IMGSZ, verbose=False)
                # 通過判定のラインを青で描く。
                cv2.line(frame, (LINE_X, 0), (LINE_X, FRAME_H), (255, 0, 0), THICK)
                put_text(frame, "GATE LINE", (LINE_X + 10, int(30 * UI)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, hud=False)

                # 長いこと見えていない追跡IDの記録を捨てる。
                for tid in [t for t, r in face_sim_history.items() if current_time - r["time"] > FACE_TIMEOUT * 4]:
                    del face_sim_history[tid]
                for tid in [t for t, r in exit_id_history.items() if current_time - r["time"] > FACE_TIMEOUT * 4]:
                    del exit_id_history[tid]
                # 退館のときの顔識別のために、全会員の顔特徴量を取り出す（キャッシュ済みなのでDBは毎回は引かない）。
                member_ids, member_names, member_matrix = member_cache.get()

                # 追跡の結果を、(追跡ID, 枠) のリストにまとめる。
                tracked = []
                if yolo_results[0].boxes.id is not None:
                    tracked = list(zip(yolo_results[0].boxes.id.cpu().numpy().astype(int).tolist(),
                                       yolo_results[0].boxes.xyxy.cpu().numpy()))

                # 追跡IDが付け替わっても同じ人なら履歴を引き継ぐ（見失った直後に近くへ現れた新しい枠を、同じ人とみなす）。
                positions = {t: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for t, b in tracked}
                for new_id, old_id in line_tracker.link_new_tracks(positions, current_time).items():
                    for hist in (face_sim_history, exit_id_history):
                        if old_id in hist and new_id not in hist:
                            hist[new_id] = hist.pop(old_id)
                    for s in entrance_sessions:
                        if s.track_id == old_id:
                            s.track_id = new_id
                    crossing_judge.remap(old_id, new_id)
                line_tracker.cleanup(current_time)

                # ここから、QRを読んだ会員ごとの認証セッションを、1フレーム分だけ進める。
                face_sims = []
                owned_by_tid = {}
                # claimed は、すでにどこかの会員の本人として追われている枠。
                claimed = {s.track_id for s in entrance_sessions if not s.done and s.track_id is not None}
                for sess in entrance_sessions:
                    if sess.done:
                        continue
                    ev = sess.update(current_time, tracked, faces, claimed=claimed)
                    if sess.track_id is not None:
                        claimed.add(sess.track_id)
                    face_sims.extend(sess.last_face_sims)
                    # label は「名前 (会員ID)」の形。名前の部分だけを取り出す。
                    member_name = sess.label.split(" (")[0] if " (" in sess.label else sess.label
                    # ev は、このフレームで起きた出来事: authed=顔が一致、crossed=線を越えて入館、expired=時間切れ。
                    if ev == "authed":
                        # 顔が一致した時点の服装を先に測っておく（入館の瞬間に測れなかったときの予備にする）。
                        if sess.last_box is not None:
                            sess.clothes_feat = entrance_body_and_clothes_feature(frame, sess.last_box, sess.member_id)
                    elif ev == "crossed":
                        # 入館が成立した: 通行ログに記録して、館内にいる会員として登録する。
                        state.register_pass("IN", sess.label, 0,
                                            note=f"QRと顔が一致（顔の類似度 {sess.match_sim:.2f}）／顔認証後、線の向こう側で本人を確認")
                        state.mark_inside(sess.member_id)
                        # 入館の瞬間の服装と体格を測る。測れないときは、顔が一致した時点で測った値で代用する。
                        if sess.last_box is not None:
                            color_feat, body_ratio = entrance_body_and_clothes_feature(frame, sess.last_box, sess.member_id)
                        else:
                            color_feat, body_ratio = None, None
                        early_color, early_ratio = getattr(sess, "clothes_feat", (None, None))
                        if color_feat is None:
                            color_feat = early_color
                        if body_ratio is None:
                            body_ratio = early_ratio
                        if color_feat is not None:
                            state.add_recent_entry(sess.member_id, member_name, color_feat, body_ratio, current_time)
                        
                        auth_banner.update(text=f"IN OK: {sess.member_id}", color=(0, 255, 0),
                                           until=current_time + ENTRANCE_BANNER_SECONDS)
                    # 時間切れ。顔認証までできていれば単なる失効、顔は見えたのに会員と合わなければ警告、顔が映らなければタイムアウトと表示する。
                    elif ev == "expired":
                        if sess.authed:
                            auth_banner.update(text=f"AUTH EXPIRED: {sess.member_id}", color=(0, 165, 255),
                                               until=current_time + ENTRANCE_BANNER_SECONDS)
                        elif sess.best_sim_seen is not None:
                            state.set_alert(True)
                            auth_banner.update(text=f"AUTH NG (face mismatch): {sess.member_id}", color=(0, 0, 255),
                                               until=current_time + ENTRANCE_BANNER_SECONDS)
                        else:
                            auth_banner.update(text=f"AUTH TIMEOUT (no face): {sess.member_id}", color=(0, 165, 255),
                                               until=current_time + ENTRANCE_BANNER_SECONDS)
                    # このフレームで本人として追えている枠を記録する（通過した枠が、共連れと間違われないようにするため）。
                    if sess.track_id is not None and (not sess.done or ev == "crossed"):
                        owned_by_tid[sess.track_id] = sess

                # 顔の枠に、会員との類似度を重ねて表示する（しきい値を超えたら黄色で表示）。
                sim_by_face = {}
                for f, sv in face_sims:
                    if sv > sim_by_face.get(id(f), -1.0):
                        sim_by_face[id(f)] = sv
                for face in faces:
                    fsim = sim_by_face.get(id(face))
                    matched = fsim is not None and fsim > FACE_MATCH_THRESHOLD
                    fb = face.bbox.astype(int)
                    cv2.rectangle(frame, (int(fb[0]), int(fb[1])), (int(fb[2]), int(fb[3])),
                                  (0, 255, 255) if matched else (160, 160, 160), 2)
                    if fsim is not None:
                        put_text(frame, f"face:{fsim:.2f}", (int(fb[0]), max(int(15 * UI), int(fb[1]) - int(8 * UI))),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if matched else (200, 200, 200), 1, hud=False)

                # 検出された人ごとに、枠の描画、顔の照合、ラインの通過判定を行う。
                if yolo_results[0].boxes.id is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                    track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, track_ids):
                        x_center = int((box[0] + box[2]) / 2)
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), THICK)
                        put_text(frame, f"ID:{track_id}", (int(box[0]), int(box[3]) + int(18 * UI)),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, hud=False)

                        # 直近にQRを読んだ会員の顔と、この人の枠の中の顔を比べて、追跡IDごとに最大の類似度を残す（画面表示・診断用）。
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

                        for face in faces:
                            fcx = (face.bbox[0] + face.bbox[2]) / 2
                            fcy = (face.bbox[1] + face.bbox[3]) / 2
                            if not (box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3]):
                                continue
                            # 退館用: この人の顔を全会員と照合して、いちばん似ている会員を追跡IDに結びつけて残す。EXIT_FACE_THRESHOLD 以下なら識別しない。
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

                        # ラインのどちら側にいるかを更新する。越えた瞬間なら、'in'（左→右）か 'out'（右→左）が返ってくる。
                        crossing = line_tracker.update(int(track_id), x_center, (box[1] + box[3]) / 2, current_time)

                        # 左から右へ越えた。本人の枠でなければ、共連れか未認証の侵入かを判定する。
                        if crossing == "in":
                            for kind, suspect, tid_ in crossing_judge.on_cross_in(
                                    int(track_id), (x_center, (box[1] + box[3]) / 2), current_time,
                                    entrance_sessions, owned_by_tid):
                                report_unauthorized_pass(kind, suspect, tid_, entrance_sessions, current_time)

                        # 右から左へ越えた＝退館。顔で識別できた会員が館内にいると記録されていれば、その人の退館として記録する。
                        # 識別できなかったときは「不明」の退館として記録する。
                        elif crossing == "out":
                            erec = exit_id_history.get(track_id)
                            if erec is not None and current_time - erec["time"] <= FACE_TIMEOUT:
                                member_id = erec["member_id"]
                                name = erec["name"]
                                member_label = f"{name} ({member_id})"
                                if state.leave_if_inside(member_id):
                                    state.register_pass("OUT", member_label, 0, note=f"顔認証による退館 (類似度 {erec['sim']:.2f})")
                                    print(f"【退館検知-成功】 {member_label} が退館しました（類似度: {erec['sim']:.2f}）")
                                else:
                                    # 顔は識別できたが、館内にいる記録が無い会員。履歴を残すため、注記つきで退館を記録する。
                                    state.register_pass("OUT", member_label, 0,
                                                        note=f"顔認証で識別（類似度 {erec['sim']:.2f}）／館内の入館記録なし")
                                    print(f"【退館検知-入館記録なし】 {member_label} の退館を記録しました（類似度: {erec['sim']:.2f}）")
                            else:
                                state.register_pass("OUT", "Unknown / Guest", 0, note="顔未識別での退館")
                                print("【退館検知】 顔を識別できない人物が退館しました（Unknown / Guest）")

                # 保留にしていた通過を見直して、猶予が過ぎたものを確定させる。
                for kind, suspect, tid_ in crossing_judge.poll(current_time, owned_by_tid):
                    report_unauthorized_pass(kind, suspect, tid_, entrance_sessions, current_time)

                # 有効期限の切れた認証セッションを片づける。
                entrance_sessions = [s for s in entrance_sessions if current_time - s.qr_time <= CO_TRAILING_WINDOW]

                put_text(frame, "MODE: [ENTRANCE (IN: QR+Face / OUT: Face)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                put_text(frame, f"IN: {state.in_count} | OUT: {state.out_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif mode == "esthe":
                put_text(frame, "MODE: [ESTHE (QR Entry & Occupancy Monitor)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # (1) エステ室用のQRコード読み取りと入室処理
                qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame)
                if qr_bbox is not None and len(qr_bbox) > 0:
                    pts = qr_bbox[0].astype(int)
                    for i in range(4):
                        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (255, 0, 255), THICK)
                    # エステ室のQRは、有効な予約があればその場で「利用中」にする（顔の照合はしない）。
                    # 入口用の should_process_qr は使わない（エステ側では認証済みトークンを記録しないため、毎フレーム判定されてしまう）。
                    # 同じQRは DUPLICATE_QR_WINDOW 秒以内なら再処理しない。
                    if qr_data and (current_time - esthe_qr_seen.get(qr_data, 0.0)) > DUPLICATE_QR_WINDOW:
                        esthe_qr_seen[qr_data] = current_time
                        if len(esthe_qr_seen) > 50:
                            for k in [k for k, t in esthe_qr_seen.items() if current_time - t > DUPLICATE_QR_WINDOW]:
                                del esthe_qr_seen[k]
                        conn = get_conn(dict_cursor=True)
                        try:
                            # QRの文字列から会員を特定する。
                            member = _exec(conn, "SELECT member_id, name FROM members WHERE qr_token = %s", (qr_data,)).fetchone()
                            if member:
                                m_id = member["member_id"]
                                m_name = member["name"]
                                # 開始の少し前（ESTHE_EARLY_ENTRY_SECONDS）から終了までの予約を「利用中(in_progress)」にする
                                res = _exec(conn, f"""
                                    SELECT id FROM esthe_reservations
                                    WHERE member_id = %s AND status = 'reserved'
                                      AND start_time <= DATE_ADD(NOW(), INTERVAL {int(ESTHE_EARLY_ENTRY_SECONDS)} SECOND)
                                      AND end_time >= NOW()
                                    LIMIT 1
                                """, (m_id,)).fetchone()
                                if res:
                                    _exec(conn, "UPDATE esthe_reservations SET status = 'in_progress', used_at = NOW() WHERE id = %s", (res["id"],))
                                    conn.commit()
                                    print(f" 【エステ入室成功】 {m_name} (ID: {m_id}) が入室しました")
                                else:
                                    print(f" 【エステ入室拒否】 有効な予約時間帯ではありません: {m_name}")
                        except Exception as e:
                            print(f" エステ入室処理エラー: {e}")
                        finally:
                            conn.close()

                # (2) エステ室の利用時間超過（長時間占有）を10秒おきにチェックする処理
                # 前回チェックした時刻。初回だけ変数が無いので、ここで 0 にしておく。
                if 'last_esthe_check' not in locals():
                    last_esthe_check = 0.0
                
                # ESTHE_CHECK_INTERVAL 秒おきに、無断キャンセルと時間超過を調べる（毎フレームDBを引かないため）。
                if current_time - last_esthe_check > ESTHE_CHECK_INTERVAL:
                    last_esthe_check = current_time
                    try:
                        conn = get_conn(dict_cursor=True)
                        # 連続予約の引き継ぎ：利用中の枠が終わったとき、同じ会員が直後の枠も予約していれば、
                        # QRをかざし直さなくても、そのまま次の枠を「利用中」にする（前の枠は完了にする）。
                        # 超過判定より先に行うので、連続予約の切れ目で超過メールが送られることはない。
                        handover_rows = _exec(conn, """
                            SELECT a.id AS cur_id, b.id AS next_id, a.member_id
                            FROM esthe_reservations a
                            JOIN esthe_reservations b
                              ON b.member_id = a.member_id
                             AND b.start_time = a.end_time
                             AND b.status = 'reserved'
                            WHERE a.status = 'in_progress'
                              AND a.end_time <= NOW()
                              AND b.end_time > NOW()
                        """).fetchall()
                        for h in handover_rows:
                            _exec(conn, "UPDATE esthe_reservations SET status = 'completed' WHERE id = %s AND status = 'in_progress'", (h["cur_id"],))
                            _exec(conn, "UPDATE esthe_reservations SET status = 'in_progress', used_at = NOW() WHERE id = %s AND status = 'reserved'", (h["next_id"],))
                            conn.commit()
                            print(f" 【エステ連続予約】 会員ID {h['member_id']} の予約ID {h['cur_id']} → {h['next_id']} へ引き継ぎました（QR再認証なし）")

                        # 無断キャンセル（予約したのに来ていない）も、スコア画面を開かなくてもここで検知する
                        mark_esthe_noshows(conn)

                        # 利用中なのに終了時刻を過ぎている予約を探して、overtime に変え、会員に超過メールを送る。
                        overdue_res = _exec(conn, """
                            SELECT r.id, r.member_id, r.start_time, r.end_time, m.name
                            FROM esthe_reservations r
                            JOIN members m ON r.member_id = m.member_id
                            WHERE r.status = 'in_progress' AND r.end_time < NOW()
                        """).fetchall()
                        
                        for r in overdue_res:
                            _exec(conn, "UPDATE esthe_reservations SET status = 'overtime' WHERE id = %s", (r["id"],))
                            conn.commit()
                            print(f" 【エステ超過検知】 会員 {r['name']} ({r['member_id']}) が終了時刻 ({r['end_time']}) を過ぎてもエステ室に留まっています。")
                            
                            start_str = r["start_time"].strftime("%Y-%m-%d %H:%M:%S")
                            end_str = r["end_time"].strftime("%Y-%m-%d %H:%M:%S")
                            notify_esthe_over_async(r["member_id"], start_str, end_str)
                            
                        conn.close()
                    except Exception as e:
                        print(f" エステ超過チェックエラー: {e}")

                # (3) エステ室から人がいなくなったら自動で「完了(completed)」にする処理
                # 最後に人が映っていた時刻。初回だけ変数が無いので、ここで今の時刻にしておく。
                if 'esthe_last_person_time' not in locals():
                    esthe_last_person_time = current_time

                # YOLOでエステ室内の「人（クラス0）」を検知
                esthe_yolo_res = yolo_model(frame, classes=[0], imgsz=YOLO_IMGSZ, verbose=False)[0]
                if len(esthe_yolo_res.boxes) > 0:
                    esthe_last_person_time = current_time  # 人が映っていればタイマーをリセット
                    for box in esthe_yolo_res.boxes.xyxy.cpu().numpy():
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
                else:
                    # 人が誰もいない状態が一定秒数続いたら、利用中(in_progress)と超過(overtime)の予約を自動完了にする
                    if current_time - esthe_last_person_time > ESTHE_EMPTY_COMPLETE_SECONDS:
                        try:
                            conn = get_conn(dict_cursor=True)
                            in_progress_res = _exec(conn, """
                                SELECT id, member_id FROM esthe_reservations
                                WHERE status IN ('in_progress', 'overtime')
                            """).fetchall()
                            
                            for r in in_progress_res:
                                _exec(conn, "UPDATE esthe_reservations SET status = 'completed' WHERE id = %s", (r["id"],))
                                conn.commit()
                                print(f" 【エステ自動退出】 室内から人がいなくなったため、予約ID {r['id']} (会員ID: {r['member_id']}) を完了にしました。")
                            conn.close()
                        except Exception as e:
                            print(f" エステ自動完了処理エラー: {e}")
            
            else:
                # ＝ トレーニングルームモード（椅子検知・座り判定・マシン利用時間計測） ＝
                put_text(frame, "MODE: [ROOM (Seat & Machine Monitor)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

                # 1. 人と椅子の検出（YOLO）
                room_results = room_model(frame, classes=[0] + SEAT_CLASSES, imgsz=YOLO_IMGSZ, conf=ROOM_CONF, verbose=False)[0]
                
                person_boxes = []
                seat_boxes = []
                # 検出結果を、人(class 0)と椅子(SEAT_CLASSES)に振り分ける。椅子はオレンジの枠で描く。
                for box_data in room_results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = box_data
                    box = [int(x1), int(y1), int(x2), int(y2)]
                    if int(cls_id) == 0:
                        person_boxes.append(box)
                    elif int(cls_id) in SEAT_CLASSES:
                        seat_boxes.append(box)
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 140, 0), THICK)

                # 椅子の位置をメモリで保持（影やオクルージョン対策）
                active_seats = update_seat_memory(seat_memory, seat_boxes, person_boxes, current_time)

                # 2. 骨格推定（姿勢・座り判定用）
                pose_res = pose_model(frame, conf=0.3, imgsz=YOLO_IMGSZ, verbose=False)[0]

                # 3. 各マシンの状態を初期化
                for m_key in MACHINE_LIMITS.keys():
                    st = state.machine_states[m_key]
                    st["live_user"] = "-"
                    st["live_member_id"] = None

                # 4. 人ごとの判定と紐付け
                for p_box in person_boxes:
                    cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 255, 0), THICK)

                    # 骨格データの取得
                    kxy, kconf = find_pose_for_box(pose_res, p_box)

                    # 座っているかの評価
                    is_sitting, overlap, reason = evaluate_sitting(p_box, kxy, kconf, active_seats)

                    # 本人の特定（顔認識または服装補助）
                    # 注意: 人1人ごとに、フレーム全体の顔検出をやり直している。人が増えるとその分だけ重くなる。
                    # 気になるなら、人のループの外で1回だけ検出して使い回す形に直せる。
                    faces = face_app.get(frame)
                    recognized_name, recognized_id = "Guest", None
                    
                    for face in faces:
                        fcx = (face.bbox[0] + face.bbox[2]) / 2
                        fcy = (face.bbox[1] + face.bbox[3]) / 2
                        if p_box[0] <= fcx <= p_box[2] and p_box[1] <= fcy <= p_box[3]:
                            member_ids, member_names, member_matrix = member_cache.get()
                            ident = identify_face_1n(face.embedding, member_ids, member_names, member_matrix)
                            if ident is not None and ident[2] > ROOM_FACE_THRESHOLD:
                                recognized_id, recognized_name, _ = ident
                                break

                    # 顔で見つからない場合は服装による補助推定
                    estimated_by_clothes = False
                    if recognized_name == "Guest" and CLOTHES_FALLBACK_ENABLED:
                        r_name, r_id = state.match_recent_entry_by_feature(kxy, kconf, frame, current_time)
                        if r_name is not None:
                            recognized_name, recognized_id = r_name, r_id
                            estimated_by_clothes = True

                    # アイデンティティの保持クラスを通す
                    name, member_id, holding, est_flag = identity_holder.update(recognized_name, recognized_id, is_sitting, current_time, estimated=estimated_by_clothes)

                    # 状態の表示とマシンへの割り当て
                    status_str = f"{name}"
                    if is_sitting:
                        confirmed, since_t = sit_confirmer.update(True, current_time)
                        if confirmed:
                            status_str += " (Sitting)"
                            # 注意: 実際にはマシンの位置との照合をしておらず、座っている人を全マシンの利用者として登録している。
                            # マシンごとに使う人を分けたいなら、椅子とマシンの対応づけが別途必要。
                            for m_key in MACHINE_LIMITS.keys():
                                state.update_machine_occupancy(m_key, name, current_time, started_at=since_t, member_id=member_id)
                        else:
                            status_str += " (Confirming...)"
                    else:
                        sit_confirmer.update(False, current_time)
                        status_str += " (Standing)"

                    color = (0, 255, 0) if is_sitting else (0, 165, 255)
                    put_text(frame, status_str, (p_box[0], max(15, p_box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, hud=False)

                # アイドル状態のマシン（誰も使っていない状態が続いたもの）を解放する
                state.release_idle_machines(current_time)

            with render_lock:
                current_output_frame = frame.copy()

            time.sleep(0.01)

        except Exception as e:
            if time.time() - last_err_log > 5.0:
                print(f" カメラ処理でエラー（処理は継続します）: {e}")
                traceback.print_exc()
                last_err_log = time.time()
            time.sleep(0.1)
    cap.release()

# コンソールで Enter キーを押すたびに、入口 → ルーム → エステ → 入口 … とモードを切り替える（動作確認用）。
def console_switcher_thread():
    global active_mode
    while True:
        try:
            input()
            with mode_lock:
                if active_mode == "entrance":
                    active_mode = "room"
                    print("\n 【モード切替】 ➔ 【トレーニングルームモード】")
                elif active_mode == "room":
                    active_mode = "esthe"
                    print("\n 【モード切替】 ➔ 【エステ室・予約照合モード】")
                else:
                    active_mode = "entrance"
                    print("\n 【モード切替】 ➔ 【入口ゲートモード】")
            state.push_update()
        except Exception:
            break

# =========================================================================
# FastAPI ルーティング & API設定
# =========================================================================
# 画面を開いているブラウザ(WebSocket)の一覧。ここへ状態を配信する。
connected_websockets = []

# 登録画面への導線（ポータルから押されたときだけセッション許可を与える）
@app.get("/api/start-register")
def api_start_register(request: Request):
    # 登録ページ(/register)は、ここを通ったときだけ開けるようにしている。URLを直接打っても入れない。
    request.session["allow_register"] = True
    return RedirectResponse(url="/register", status_code=303)


# 以前カメラの競合防止に用意したロック。現在はどこからも使われていない。
camera_lock = threading.Lock()

@app.get("/register", response_class=HTMLResponse)
def get_register_page(request: Request):
    # 通行証（allow_register）がなければポータルへ追い返す（直打ちブロック）
    if not request.session.get("allow_register"):
        return RedirectResponse(url="/portal", status_code=303)
    
    # 一度アクセスしたら通行証を消す
    request.session.pop("allow_register", None)

    # レジスター画面を開いたときは、確実にカメラモードを "register" にして解放を促す
    with mode_lock:
        global active_mode
        active_mode = "register"

    # 万が一キャプチャが開きっぱなしなら、ここで強制的に解放をかける
    # （カメラ処理ループ側の `active_mode == "register"` の判定とも連動します）

    return templates.TemplateResponse(request, "register.html", {})

# エステ予約用リクエストモデル（誰の予約かは、ログイン中の会員のセッションから決める）
class EstheReserveRequest(BaseModel):
    start_time: str  # 例: "2026-11-06 14:00:00"

class EstheCancelRequest(BaseModel):
    reservation_id: int

# エステ予約の設定値（すべて秒単位）。動作確認しやすいように短くしてある。実運用では分単位に直すこと。
ESTHE_SLOT_SECONDS = 20            # 予約1枠の長さ（秒）。開始時刻もこの刻み（秒が 00・20・40）に揃える
ESTHE_EARLY_ENTRY_SECONDS = 20     # 開始のこの秒数前から入室できる
ESTHE_CHECK_INTERVAL = 3.0         # 時間超過・無断キャンセルをチェックする間隔（秒）
ESTHE_EMPTY_COMPLETE_SECONDS = 15.0   # 無人がこの秒数続いたら自動で完了にする

def mark_esthe_noshows(conn, member_id=None):
    """予約したのに入室しないまま、枠の終了時刻を過ぎた予約を no_show にして、メールを送る。member_id を渡すとその会員だけが対象。"""
    where = "status = 'reserved' AND end_time < NOW()"
    params = ()
    if member_id is not None:
        where += " AND member_id = %s"
        params = (member_id,)
    cur = conn.cursor(pymysql.cursors.Cursor)
    cur.execute(f"SELECT id, member_id, start_time FROM esthe_reservations WHERE {where}", params)
    targets = cur.fetchall()
    for rid, mid, start in targets:
        # 他の処理が先に更新していたら（rowcount が 0）、メールを重ねて送らない
        cur.execute("UPDATE esthe_reservations SET status = 'no_show' WHERE id = %s AND status = 'reserved'", (rid,))
        if cur.rowcount:
            print(f" 【エステ無断キャンセル検知】 会員ID {mid} / 予約開始 {start}")
            notify_esthe_noshow_async(mid, str(start))
    conn.commit()

# 会員ログインが必要なAPIで、未ログインのときに返す共通の応答（HTTP 401）。
def _login_required_json():
    return JSONResponse({"status": "error", "message": "ログインが必要です。ポータルからログインしてください。"}, status_code=401)

ESTHE_TIME_FMT = "%H:%M:%S"   # 枠が秒単位なので、画面に出す時刻も秒まで出す

# エステ予約登録API（会員ログイン必須）
@app.post("/api/esthe/reserve")
def api_reserve_esthe(request: Request, data: EstheReserveRequest):
    # 予約をする人は、画面から送らせずにログイン中のセッションから決める（他人の名前で予約できないようにするため）。
    member_id = request.session.get("member_id")
    if not member_id:
        return _login_required_json()
    try:
        from datetime import timedelta
        try:
            # 日時の文字列を解釈する。形式が違えばエラーを返す。
            start_dt = datetime.strptime(data.start_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return {"status": "error", "message": "日時の形式が正しくありません。"}
        # 開始時刻は枠の刻みに揃える（秒が 00・20・40）
        if start_dt.second % ESTHE_SLOT_SECONDS != 0:
            return {"status": "error", "message": f"{ESTHE_SLOT_SECONDS}秒刻みの時刻（秒が00・20・40）を選んでください。"}
        if start_dt <= datetime.now():
            return {"status": "error", "message": "過去の日時は予約できません。"}
        # 終了時刻は、開始時刻 ＋ 1枠の長さ。
        end_dt = start_dt + timedelta(seconds=ESTHE_SLOT_SECONDS)

        conn = get_conn(dict_cursor=True)
        try:
            # 会員が実在するか（退会済みのセッションが残っている場合の対策）
            if _exec(conn, "SELECT 1 FROM members WHERE member_id = %s", (member_id,)).fetchone() is None:
                request.session.pop("member_id", None)
                return _login_required_json()

            # 既に埋まっていないか（時間帯が重なる、予約済み・利用中の枠は取れない）
            if _exec(conn,
                     "SELECT id FROM esthe_reservations WHERE start_time < %s AND end_time > %s "
                     "AND status IN ('reserved', 'in_progress')",
                     (end_dt, start_dt)).fetchone():
                return {"status": "error", "message": "この時間帯は既に予約されています。"}

            # 会員の予約はその場で確定させる（直後の一覧表示に反映されるよう、書き込みキューを使わず直接 INSERT する）
            _exec(conn,
                  "INSERT INTO esthe_reservations (member_id, start_time, end_time) VALUES (%s, %s, %s)",
                  (member_id, start_dt, end_dt))
            conn.commit()
        finally:
            conn.close()
        return {"status": "success", "message": f"{start_dt.strftime('%m/%d ' + ESTHE_TIME_FMT)} のエステを予約しました。"}
    except Exception as e:
        print(f" エステ予約エラー: {e}")
        return {"status": "error", "message": "予約に失敗しました。時間をおいてもう一度お試しください。"}

# 自分のエステ予約の一覧（新しい順に最大20件）
@app.get("/api/esthe/my-reservations")
def api_my_esthe_reservations(request: Request):
    member_id = request.session.get("member_id")
    if not member_id:
        return _login_required_json()
    conn = get_conn(dict_cursor=True)
    try:
        rows = _exec(conn,
            "SELECT id, start_time, end_time, status FROM esthe_reservations "
            "WHERE member_id = %s ORDER BY start_time DESC LIMIT 20", (member_id,)).fetchall()
    finally:
        conn.close()
    now = datetime.now()
    reservations = [{
        "id": r["id"],
        "start_time": r["start_time"].strftime("%Y-%m-%d " + ESTHE_TIME_FMT),
        "end_time": r["end_time"].strftime(ESTHE_TIME_FMT),
        "status": r["status"],
        # 予約済みで、まだ始まっていないものだけキャンセルできる
        "cancellable": r["status"] == "reserved" and r["start_time"] > now,
    } for r in rows]
    return {"status": "success", "reservations": reservations}

# 自分のエステ予約のキャンセル
@app.post("/api/esthe/cancel")
def api_cancel_esthe(request: Request, data: EstheCancelRequest):
    member_id = request.session.get("member_id")
    if not member_id:
        return _login_required_json()
    conn = get_conn()
    try:
        # 本人の予約で、予約済み・開始前のものだけをキャンセルする（条件を満たさなければ 0 件更新）
        cur = _exec(conn,
            "UPDATE esthe_reservations SET status = 'cancelled' "
            "WHERE id = %s AND member_id = %s AND status = 'reserved' AND start_time > NOW()",
            (data.reservation_id, member_id))
        changed = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    if changed == 0:
        return {"status": "error", "message": "この予約はキャンセルできません（開始済み、またはすでに取り消し済みです）。"}
    return {"status": "success", "message": "予約をキャンセルしました。"}

# 会員ログイン画面。すでにログイン済みならマイページ(/portal)へ戻す。
@app.get("/login", response_class=HTMLResponse)
def get_login_page(request: Request):
    # すでにログイン済みならマイページへ自動転送
    if request.session.get("member_id"):
        return RedirectResponse(url="/portal", status_code=303)
    return templates.TemplateResponse(request, "login.html", {})

# 会員のマイページ。未ログインならログイン画面へ飛ばす。
@app.get("/portal", response_class=HTMLResponse)
def get_user_portal(request: Request):
    if not request.session.get("member_id"):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "portal.html", {})

# 利用履歴ページ。未ログインなら /portal へ戻す（そこからログイン画面に回される）。
@app.get("/history", response_class=HTMLResponse)
def get_history_page(request: Request):
    if not request.session.get("member_id"):
        return RedirectResponse(url="/portal", status_code=303)
    return templates.TemplateResponse(request, "history.html", {})

@app.post("/api/register")
# 会員登録API。入力チェック → 顔の特徴量の取り出し → 会員IDとQRトークンの発行 → QR画像をメールで送る → DBに保存、の順に進める。
# 途中で失敗すると会員は作られない（メール送信に失敗したときも、DBには入らない）。
def api_register(data: RegisterRequest):
    # デバッグ用の出力。名前とメールアドレスがログに残るので、運用では消すこと。
    print(f" 受信した名前: {data.name}")
    print(f" 受信したメール: {data.email}")
    try:
        # 全角で入力されたメールアドレスを半角に直し、前後の空白を取り除く。
        email = unicodedata.normalize("NFKC", data.email).strip()
        if not email.isascii() or "@" not in email:
            return {"status": "error", "message": "メールアドレスは半角英数字で入力してください。"}

        # パスワード強度チェック（8文字以上・大文字・小文字・数字・特殊文字）
        pw_error = validate_password_strength(data.password)
        if pw_error:
            return {"status": "error", "message": pw_error}

        # パスワードをハッシュ化する（平文のままDBに入れない）。
        hashed_password = get_password_hash(data.password)

        # 画像は「data:image/jpeg;base64,xxxx」の形で届く。カンマより後ろを取り出してデコードし、OpenCVの画像にする。
        header, encoded = data.image.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 顔の特徴量を取り出す。リクエストのたびにモデルを読み込むので、登録には少し時間がかかる。
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        faces = face_app.get(frame)

        if len(faces) == 0:
            return {"status": "error", "message": "顔が検出されませんでした。もう一度撮影してください。"}
        
        # 複数の顔が写っていたら先頭の1つを使う。特徴量はバイト列にして、DBの BLOB 列に保存する。
        face_embedding = faces[0].embedding.tobytes()

        # 会員IDと、入館用QRに埋め込むトークンを乱数で作る。QRのトークンは推測されにくいように長めにしている。
        member_id = "mem_" + uuid.uuid4().hex[:8]
        qr_token = "token_" + uuid.uuid4().hex

        # トークンをQRコードの画像(PNG)にして、メモリ上に作る。
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_token)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)

        msg = MIMEMultipart()
        # 登録完了のメールに、QR画像を添付して送る。
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

        # メール送信まで成功してから会員を保存する（途中で失敗すれば会員は作られない）。パスワードはハッシュ化したものを入れる。
        execute_db_write(
            "INSERT INTO members (member_id, name, email, password, qr_token, face_embedding) VALUES (%s, %s, %s, %s, %s, %s)",
            (member_id, data.name, email, hashed_password, qr_token, face_embedding)
        )
        # 会員が増えたので、顔キャッシュを破棄して、次の照合ですぐ読み直させる。
        member_cache.invalidate()

        return {"status": "success", "member_id": member_id}

    except Exception as e:
        print(f"登録エラー: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# エステ予約の画面。会員ログインが必要。
@app.get("/esthe", response_class=HTMLResponse)
def get_esthe_page(request: Request):
    if not request.session.get("member_id"):
        return RedirectResponse(url="/portal", status_code=303)

    return templates.TemplateResponse(request, "esthe.html", {})

@app.get("/video")
# カメラの映像をMJPEG（JPEG画像を連続で送る形式）で配信する。ブラウザの <img> に直接つなげて表示できる。
# 0.04秒おき（約25コマ/秒）で、最新のフレームを送る。
async def video_feed():
    async def generate_async_stream():
        while True:
            with render_lock:
                frame_to_send = current_output_frame
            if frame_to_send is not None:
                # 最新のフレームをJPEGに圧縮して送る。
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

# 画面とリアルタイムにつなぐWebSocket。つながった直後に最新の状態を1回送り、その後の更新は ws_broadcast_loop が配信する。
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

# 待ち行列に溜まった更新データを取り出して、接続中のすべての画面へ0.1秒おきに配信する。送れなくなった接続は一覧から外す。
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
            print(f" WebSocket配信でエラー（配信は継続します）: {e}")
        await asyncio.sleep(0.1)

# =========================================================================
# 会員有料度（貢献度）スコア関連のルーティング・API
# =========================================================================

# 会員1人ぶんのスコアは、来館・マシン利用の実績から加点し、違反（マシンの超過、共連れ、エステの無断キャンセル・超過）で減点して出す。
def calculate_member_score(conn, member_id):
    """会員1人あたりの有料度・貢献度スコア（0〜100点）および内訳を計算する（共連れ・超過の減点対応）"""
    cursor = conn.cursor(pymysql.cursors.Cursor)
    # passing_logs の member_id は「名前 (会員ID)」の形なので、末尾が「(会員ID)」で終わるものを LIKE で探す。
    # 会員IDの中の % や _ は LIKE の特殊文字なので、先にエスケープしておく。
    escaped = member_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%({escaped})"

    # 1. 正常な来館回数 (INログかつ is_alert = 0)
    cursor.execute(
        "SELECT COUNT(*) FROM passing_logs WHERE direction = 'IN' AND is_alert = 0 AND member_id LIKE %s",
        (pattern,)
    )
    normal_visit_count = cursor.fetchone()[0] or 0

    # 2. 共連れ・不正入館などのアラート回数 (is_alert = 1)
    cursor.execute(
        "SELECT COUNT(*) FROM passing_logs WHERE direction = 'IN' AND is_alert = 1 AND member_id LIKE %s",
        (pattern,)
    )
    alert_count = cursor.fetchone()[0] or 0

    # 3. マシン利用データの集計（超過回数のカウント）
    cursor.execute(
        "SELECT duration, machine FROM occupancy_logs WHERE member_id = %s",
        (member_id,)
    )
    usages = cursor.fetchall()
    total_duration_sec = sum(u[0] or 0 for u in usages)

    # マシンごとの上限を超えて使った回数を数える（上限が0のマシンは対象外）。
    overtime_count = 0
    for duration, machine_key in usages:
        info = MACHINE_LIMITS.get(machine_key, {})
        limit = info.get("limit", 0.0)
        if limit > 0 and (duration or 0) > limit:
            overtime_count += 1
    
    # 無断キャンセルの判定（共通ヘルパー。カメラ処理側の10秒おきのチェックでも同じ処理が走る）
    mark_esthe_noshows(conn, member_id)

    # エステ無断キャンセル回数
    cursor.execute(
        "SELECT COUNT(*) FROM esthe_reservations WHERE member_id = %s AND status = 'no_show'",
        (member_id,)
    )
    esthe_noshow_count = cursor.fetchone()[0] or 0

    # エステ時間超過回数
    cursor.execute(
        "SELECT COUNT(*) FROM esthe_reservations WHERE member_id = %s AND status = 'overtime'",
        (member_id,)
    )
    esthe_overtime_count = cursor.fetchone()[0] or 0

    # --- スコア計算 ---
    # 来館点は1回5点で上限40点。利用点はマシンを10分(600秒)使うごとに5点で上限40点。
    # 加点の合計は最大80点なので、ランクSになれるのは減点なしで加点を満点まで貯めたときだけ。
    visit_score = min(40, normal_visit_count * 5)
    usage_score = min(40, int((total_duration_sec / 600) * 5))

    # 減点の内訳（エステの違反も対象）:
    # - マシン超過: 1回 -10pt
    # - 共連れアラート: 1回 -20pt
    # - エステ無断キャンセル: 1回 -15pt
    # - エステ時間超過: 1回 -10pt
    penalty_points = (overtime_count * 10) + (alert_count * 20) + (esthe_noshow_count * 15) + (esthe_overtime_count * 10)

    # 加点から減点を引いて、0〜100の範囲に収める。
    total_score = max(0, min(100, (visit_score + usage_score) - penalty_points))

    # ランク判定
    # ランクは、80点以上=S、60点以上=A、40点以上=B、それ未満=C。
    if total_score >= 80:
        rank = "S"
    elif total_score >= 60:
        rank = "A"
    elif total_score >= 40:
        rank = "B"
    else:
        rank = "C"

    return {
        "total_score": total_score,
        "rank": rank,
        "visit_count": normal_visit_count,
        "alert_count": alert_count,
        "total_duration_fmt": _fmt_seconds(total_duration_sec),
        "overtime_count": overtime_count,
        "esthe_noshow_count": esthe_noshow_count, #  画面表示用に返す場合に追加
        "esthe_overtime_count": esthe_overtime_count, #  追加
        "penalty_points": penalty_points
    }


@app.get("/admin/scores", response_class=HTMLResponse)
def get_admin_scores_page(request: Request):
    """有料度スコア一覧ページ表示"""
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "scores.html", {})


@app.get("/api/admin/scores")
def api_get_member_scores(request: Request):
    """全会員の有料度スコアランキング取得API"""
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)

    conn = get_conn(dict_cursor=True)
    cursor = conn.cursor()
    cursor.execute("SELECT member_id, name, email FROM members")
    members = cursor.fetchall()

    score_list = []
    for m in members:
        score_info = calculate_member_score(conn, m["member_id"])
        score_list.append({
            "member_id": m["member_id"],
            "name": m["name"],
            "email": m["email"],
            **score_info
        })

    conn.close()

    # スコアが高い順に並び替え
    score_list.sort(key=lambda x: x["total_score"], reverse=True)

    return {"status": "success", "scores": score_list}

# =========================================================================
# 会員ポータル（portal.html）用：会員ログイン・マイページAPI
# 管理者ログイン（session["admin_user"]）とは別に、session["member_id"] で会員を管理する。
# =========================================================================
@app.post("/api/member/login")
def api_member_login(request: Request, data: MemberLoginRequest):
    # メールアドレスを半角に直して照合する（登録のときと同じ整え方）。
    email = unicodedata.normalize("NFKC", data.email).strip()
    if not email or not data.password:
        return JSONResponse({"status": "error", "message": "メールアドレスとパスワードを入力してください。"}, status_code=400)

    conn = get_conn(dict_cursor=True)
    try:
        # 該当するメールアドレスの会員情報を取得（複数ある場合はループや複数件対応も可）
        member = _exec(conn,
            "SELECT member_id, name, password FROM members WHERE email = %s LIMIT 1", (email,)).fetchone()
    finally:
        conn.close()

    # 会員が存在しない、またはパスワードが一致しない場合
    if member is None or not verify_password(data.password, member["password"]):
        return JSONResponse({"status": "error", "message": "メールアドレスまたはパスワードが違います。"}, status_code=400)

    # ログイン成功。会員IDをセッション(Cookie)に入れる。以降のAPIは、これでログイン中の会員を判断する。
    request.session["member_id"] = member["member_id"]
    return {"status": "success"}

@app.post("/api/member/logout")
def api_member_logout(request: Request):
    """会員のログイン状態だけを解除する（管理者のログインには影響しない）。"""
    request.session.pop("member_id", None)
    return {"status": "success"}

@app.get("/api/member/me")
def api_member_me(request: Request):
    """ログイン中の会員のマイページ情報（ランク・来店・利用履歴・次回のエステ予約）を返す。"""
    member_id = request.session.get("member_id")
    if not member_id:
        return JSONResponse({"status": "error", "message": "ログインしていません。"}, status_code=401)

    conn = get_conn(dict_cursor=True)
    try:
        member = _exec(conn,
            "SELECT member_id, name, email FROM members WHERE member_id = %s", (member_id,)).fetchone()
        if member is None:
            # 退会（削除）された会員のセッションが残っていた場合
            request.session.pop("member_id", None)
            return JSONResponse({"status": "error", "message": "会員が見つかりません。"}, status_code=401)

        score = calculate_member_score(conn, member_id)

        rows = _exec(conn,
            "SELECT machine, start_time, duration FROM occupancy_logs "
            "WHERE member_id = %s ORDER BY id DESC LIMIT 10", (member_id,)).fetchall()
        usages = [{
            "machine": MACHINE_LIMITS.get(r["machine"], {}).get("name", r["machine"] or "-"),
            "start_time": r["start_time"],
            "duration": r["duration"] or 0,
        } for r in rows]

        nxt = _exec(conn,
            "SELECT start_time FROM esthe_reservations "
            "WHERE member_id = %s AND status = 'reserved' AND start_time >= NOW() "
            "ORDER BY start_time LIMIT 1", (member_id,)).fetchone()
    finally:
        conn.close()

    return {
        "status": "success",
        "member": {"member_id": member["member_id"], "name": member["name"], "email": member["email"]},
        "score": score,
        "usages": usages,
        "next_esthe": nxt["start_time"].strftime("%Y-%m-%d %H:%M") if nxt else None,
    }

# サーバーが起動したときに、画面への配信ループをバックグラウンドで走らせる。
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ws_broadcast_loop())

# 管理者のダッシュボード（トップページ）。管理者ログインが必要。開くと、カメラのモードを入口に戻す。
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    
    # ページアクセス時にモードを自動切り替え
    with mode_lock:
        global active_mode
        active_mode = "entrance"
        
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/api/admin/login")
def api_admin_login(request: Request, data: AdminAuthRequest):
    conn = get_conn(dict_cursor=True) #  カラム名でアクセスできるようにする
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE username = %s", (data.username,))
    user = cursor.fetchone()
    conn.close()

    # 取得したハッシュ化パスワードと入力されたパスワードを安全に検証
    if user and verify_password(data.password, user["password"]):
        request.session["admin_user"] = data.username
        return {"status": "success"}
    else:
        return JSONResponse({"status": "error", "message": "IDまたはパスワードが違います"}, status_code=400)

# 管理者のログアウト。session を全部消すので、同じブラウザの会員ログインも一緒に消える。
@app.get("/admin/logout")
async def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/admin/login", status_code=303)

@app.get("/admin/members", response_class=HTMLResponse)
def get_admin_members_page(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "members.html", {})

# 会員一覧API・削除APIも、他の /api/admin/ 系と同じく管理者ログインが必要。
@app.get("/api/admin/members")
def api_get_members(request: Request):
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)
    conn = get_conn(dict_cursor=True)
    cursor = conn.cursor()
    cursor.execute("SELECT member_id, name, email, qr_token FROM members")
    members = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"members": members}

@app.delete("/api/admin/members/{member_id}")
def api_delete_member(member_id: str, request: Request):
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM members WHERE member_id = %s", (member_id,))
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

    conn = get_conn(dict_cursor=True)
    cursor = conn.cursor()
    cursor.execute("SELECT member_id, name, email, face_embedding IS NOT NULL AS has_face FROM members WHERE member_id = %s",
                   (member_id,))
    member = cursor.fetchone()
    if member is None:
        conn.close()
        return JSONResponse({"status": "error", "message": "会員が見つかりません。"}, status_code=404)

    escaped = member_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%({escaped})"
    # 警告のない入館だけを、来館回数として数える。
    visit_where = "direction = 'IN' AND is_alert = 0 AND member_id LIKE %s"
    cursor.execute(f"SELECT timestamp FROM passing_logs WHERE {visit_where} ORDER BY id DESC LIMIT 50", (pattern,))
    visits = [row["timestamp"] for row in cursor.fetchall()]
    cursor.execute(f"SELECT COUNT(*) AS c, MAX(timestamp) AS last FROM passing_logs WHERE {visit_where}", (pattern,))
    v = cursor.fetchone()

    cursor.execute("SELECT machine, start_time, end_time, duration FROM occupancy_logs WHERE member_id = %s ORDER BY id DESC LIMIT 50",
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
    cursor.execute("SELECT COUNT(*) AS c, COALESCE(SUM(duration), 0) AS total FROM occupancy_logs WHERE member_id = %s", (member_id,))
    u = cursor.fetchone()
    conn.close()

    # いま利用中のマシンがあれば、メモリ上の状態から取り出して画面に出す。
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

@app.get("/api/admin/recent-alerts")
def api_get_recent_alerts(request: Request):
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)
    
    conn = get_conn(dict_cursor=True)
    cursor = conn.cursor()
    # 警告フラグが立った passing_logs や、違反などのログを新しい順に取得
    cursor.execute("""
        SELECT timestamp, 'passing' AS type, member_id AS detail, note AS message 
        FROM passing_logs WHERE is_alert = 1 
        UNION ALL
        SELECT start_time AS timestamp, 'machine' AS type, member_id AS detail, machine AS message 
        FROM occupancy_logs 
        ORDER BY timestamp DESC LIMIT 5
    """)
    alerts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"status": "success", "alerts": alerts}

@app.get("/admin/logs", response_class=HTMLResponse)
def get_admin_logs_page(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "logs.html", {})

# マシンログの member_id（会員IDか表示名）を、「名前 (会員ID)」の形に整える。会員が見つからなければそのまま返す。
def _member_label(names, raw):
    if raw in names:
        return f"{names[raw]} ({raw})"
    return raw

# 「名前 (会員ID)」から、末尾の括弧の中の会員IDを取り出す。この形でなければ None。
def _member_id_from_label(label):
    # 末尾の括弧の中身が会員ID。
    m = re.match(r"^.*\(([^()]+)\)$", label or "")
    return m.group(1) if m else None

# 1人の会員の、来館回数・最後の来館・マシンの利用回数・合計の利用秒数をまとめて返す。会員がいなければ None。
def _member_stats(cursor, member_id):
    if not member_id:
        return None
    cursor.execute("SELECT member_id, name, email, face_embedding IS NOT NULL AS has_face FROM members WHERE member_id = %s",
                   (member_id,))
    m = cursor.fetchone()
    if m is None:
        return None
    escaped = member_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    cursor.execute("SELECT COUNT(*) AS c, MAX(timestamp) AS last FROM passing_logs "
                   "WHERE direction = 'IN' AND is_alert = 0 AND member_id LIKE %s", (f"%({escaped})",))
    v = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) AS c, COALESCE(SUM(duration), 0) AS total FROM occupancy_logs WHERE member_id = %s", (member_id,))
    u = cursor.fetchone()
    return {"member_id": m["member_id"], "name": m["name"], "email": m["email"], "has_face": bool(m["has_face"]),
            "visit_count": v["c"], "last_visit": v["last"], "usage_count": u["c"], "total_usage_seconds": u["total"]}

# 秒数を「3分20秒」「45秒」のような読みやすい文字列にする。
def _fmt_seconds(sec):
    sec = int(round(sec or 0))
    return f"{sec // 60}分{sec % 60}秒" if sec >= 60 else f"{sec}秒"

# ライブ映像の画面。管理者ログインが必要。
@app.get("/live", response_class=HTMLResponse)
async def read_live(request: Request):
    if "admin_user" not in request.session:
        return RedirectResponse(url="/admin/login", status_code=303)
        
    return templates.TemplateResponse(request, "live.html", {})

@app.get("/api/admin/logs")
def api_get_logs(request: Request):
    if "admin_user" not in request.session:
        return JSONResponse({"status": "error", "message": "ログインが必要です。"}, status_code=401)

    conn = get_conn(dict_cursor=True)
    cursor = conn.cursor()
    cursor.execute("SELECT member_id, name FROM members")
    names = {row["member_id"]: row["name"] for row in cursor.fetchall()}

    logs = []

    cursor.execute("SELECT id, timestamp, direction, member_id, is_alert FROM passing_logs ORDER BY id DESC LIMIT 100")
    for r in cursor.fetchall():
        d = r["direction"]
        # IN/OUT以外の種別はエステ関連のログとして分類する（表示用の番号は G=ゲート、E=エステ）。
        if d in ("IN", "OUT"):
            l_type = "gate"
            prefix = "G"
        else:
            l_type = "esthe"
            prefix = "E"
            
        logs.append({"kind": "gate", "ref": str(r["id"]), "display_id": f"{prefix}-{r['id']}", "type": l_type,
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
    # 通行ログとマシンのログを混ぜて、新しい順に並べ直し、100件に絞る。
    logs.sort(key=lambda x: x["timestamp"] or "", reverse=True)
    logs = logs[:100]

    # いま利用中のマシンを、ログ一覧の先頭に「LIVE」として並べる。
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

    conn = get_conn(dict_cursor=True)
    cursor = conn.cursor()
    fields, title, member_id = [], "", None

    # ログの種類(kind)ごとに、詳細の項目を組み立てる。gate=入退館、machine=マシンの利用記録、active=いま利用中。
    if kind in ("gate", "machine"):
        try:
            log_id = int(ref)
        except ValueError:
            conn.close()
            return not_found

    if kind == "gate":
        cursor.execute("SELECT id, timestamp, direction, member_id, is_alert, note FROM passing_logs WHERE id = %s", (log_id,))
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
        cursor.execute("SELECT id, member_id, machine, start_time, end_time, duration FROM occupancy_logs WHERE id = %s", (log_id,))
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

    # 関係する会員がいれば、その会員の来館・利用の集計も添える。
    member = _member_stats(cursor, member_id)
    conn.close()
    return {"status": "success", "kind": kind, "title": title, "fields": fields, "member": member}

@app.get("/admin/login", response_class=HTMLResponse)
def get_admin_login_page(request: Request):
    return templates.TemplateResponse(request, "admin_login.html", {})

# 直接実行したときの起動処理。カメラ処理とコンソールのモード切り替えを別スレッドで動かしてから、Webサーバー(uvicorn)を起動する。
# 127.0.0.1 で待ち受けるので、このPC以外からは開けない。
if __name__ == "__main__":
    camera_thread = threading.Thread(target=camera_processing_loop, daemon=True)
    camera_thread.start()
    
    switcher_thread = threading.Thread(target=console_switcher_thread, daemon=True)
    switcher_thread.start()
    
    print("\n サーバーを椅子検知・座り判定モードで起動しました！")
    print(" ブラウザで http://localhost:8000/ を開いてダッシュボードを確認してください。")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")