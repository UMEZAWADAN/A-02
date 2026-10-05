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
import smtplib
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from queue import Queue
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import uvicorn

# テンプレートエンジンの準備
app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
FACE_TIMEOUT = 5.0

# 📧 メール送信設定（実際の運用に合わせて書き換えてください）
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"  # 送信元のメールアドレス
SENDER_PASSWORD = "your_app_password"   # アプリパスワード等

LINE_X = 320
CAMERA_ID = 0  # 内蔵カメラ1台

# 🎛️ カメラの役割切り替え用フラグ ('entrance' または 'room')
active_mode = "entrance"
mode_lock = threading.Lock()

# 🏋️‍♂️ マシンエリアの定義
MACHINE_AREAS = {
    "bench_press": {"name": "ベンチプレス", "box": [50, 100, 250, 350], "limit": 15.0},
    "squat_rack": {"name": "スクワットラック", "box": [390, 100, 590, 350], "limit": 20.0}
}

# =========================================================================
# 💾 データベース自動初期化
# =========================================================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
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
    conn.commit()
    conn.close()

init_database()

print(">> [1/3] YOLO（人流解析）モデルを読み込み中...")
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
        self.co_trailing_alert = False
        
        # ルーム監視用の状態管理（1:N識別対応）
        self.active_users = {} # 検出中のユーザー情報を保持 {track_or_face_id: {"name": ..., "accumulated_time": ..., ...}}
        
        self.machine_states = {
            k: {"user": "-", "duration": 0.0, "status": "free", "start_time": None} 
            for k in MACHINE_AREAS.keys()
        }
        
        self.update_queue = Queue()

    def process_qr(self, qr_data: str, current_time: float):
        with self._lock:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT member_id, name FROM members WHERE qr_token = ?", (qr_data,))
            member = cursor.fetchone()
            conn.close()

            if member:
                self.last_scanned_qr = f"{member['name']} ({member['member_id']})"
                print(f"🔓 【QR認証成功】 歓迎: {member['name']} (ID: {member['member_id']})")
            else:
                self.last_scanned_qr = "Invalid QR"
                print(f"❌ 【QR認証失敗】 無効なトークンです: {qr_data}")

            self.last_qr_time = current_time
            self.co_trailing_alert = False
            self.push_update()

    def set_alert(self, state: bool):
        with self._lock:
            self.co_trailing_alert = state
            self.push_update()

    def register_pass(self, direction: str, member_id: str, is_alert: int):
        with self._lock:
            if direction == "IN":
                self.in_count += 1
            else:
                self.out_count += 1
            
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute(
                "INSERT INTO passing_logs (timestamp, direction, member_id, is_alert) VALUES (?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), direction, member_id, is_alert)
            )
            conn.commit()
            conn.close()
            self.push_update()

    def update_machine_occupancy(self, active_machine_key, user_name, current_time):
        with self._lock:
            for m_key in self.machine_states.keys():
                if m_key == active_machine_key and user_name != "Guest":
                    m_state = self.machine_states[m_key]
                    if m_state["user"] != user_name:
                        m_state["user"] = user_name
                        m_state["start_time"] = current_time
                        m_state["duration"] = 0.0
                    else:
                        if m_state["start_time"] is not None:
                            m_state["duration"] = current_time - m_state["start_time"]
                    
                    limit = MACHINE_AREAS[m_key]["limit"]
                    if m_state["duration"] > limit:
                        m_state["status"] = "overtime"
                    else:
                        m_state["status"] = "using"

    def push_update(self):
        machines_payload = []
        for key, info in MACHINE_AREAS.items():
            st = self.machine_states[key]
            machines_payload.append({
                "name": info["name"],
                "user": st["user"],
                "duration": int(st["duration"]),
                "status": st["status"]
            })

        # DBから直近の履歴を取得
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

# =========================================================================
# 🎥 カメラ処理ループ（1台のカメラでモードに応じてAI処理を切り替え）
# =========================================================================
def camera_processing_loop():
    global current_output_frame, yolo_model
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis

    yolo_model = YOLO("yolo11n.pt")
    
    print("👤 顔認識モデルを読み込み中...")
    face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ エラー: カメラを開けませんでした。")
        return

    track_history = {}
    print("\n====================================================")
    print(" 📹 カメラ処理ループが稼働しました。")
    print(" 🖥️  ブラウザ (http://localhost:8000/) で映像を確認してください。")
    print(" ⌨️  【ターミナルで Enter キーを押す】と、")
    print("     [ 入口モード ] ⇄ [ ルームモード ] が切り替わります！")
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

        # -----------------------------------------------------------------
        # モード A: 入口ゲート処理 (QR & YOLO)
        # -----------------------------------------------------------------
        if mode == "entrance":
            qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame)
            if qr_bbox is not None and len(qr_bbox) > 0:
                pts = qr_bbox[0].astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
                if qr_data:
                    if qr_data != state.last_scanned_qr or (current_time - state.last_qr_time) > DUPLICATE_QR_WINDOW:
                        state.process_qr(qr_data, current_time)

            yolo_results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
            cv2.line(frame, (LINE_X, 0), (LINE_X, 480), (255, 0, 0), 2)
            cv2.putText(frame, "GATE LINE", (LINE_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if yolo_results[0].boxes.id is not None:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x_center = int((box[0] + box[2]) / 2)
                    y_center = int((box[1] + box[3]) / 2)
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                    if track_id in track_history:
                        prev_x = track_history[track_id]
                        if prev_x < LINE_X and x_center >= LINE_X:
                            time_since_qr = current_time - state.last_qr_time
                            if time_since_qr <= CO_TRAILING_WINDOW and state.last_scanned_qr != "None" and state.last_scanned_qr != "Invalid QR":
                                state.register_pass("IN", state.last_scanned_qr, 0)
                                print(f"✅ [入館許可] 会員 {state.last_scanned_qr}")
                            else:
                                state.set_alert(True)
                                state.register_pass("IN", "Unknown", 1)
                                print("🚨 [共連れ検知 または 未認証]")
                        elif prev_x > LINE_X and x_center <= LINE_X:
                            state.register_pass("OUT", "Unknown", 0)
                            state.set_alert(False)

                    track_history[track_id] = x_center

            cv2.putText(frame, "MODE: [ENTRANCE]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"IN: {state.in_count} | OUT: {state.out_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # -----------------------------------------------------------------
        # モード B: トレーニングルーム処理 (1:N 顔認識 & マシンROI) 【ステップ4】
        # -----------------------------------------------------------------
        else:
            for m_key, m_info in MACHINE_AREAS.items():
                bx1, by1, bx2, by2 = m_info["box"]
                st = state.machine_states[m_key]
                color = (0, 0, 255) if st["status"] == "overtime" else (255, 165, 0)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, f"{m_info['name']} ({int(st['duration'])}s)", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # データベースから全会員の顔特徴量をロード
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT member_id, name, face_embedding FROM members WHERE face_embedding IS NOT NULL")
            registered_members = cursor.fetchall()
            conn.close()

            faces = face_app.get(frame)
            active_machine_key = None
            detected_user_name = "Guest"

            for face in faces:
                best_match_name = "Guest"
                highest_sim = 0.0

                # 登録されている全会員とコサイン類似度を計算 (1:N 照合)
                for member in registered_members:
                    db_embedding = np.frombuffer(member['face_embedding'], dtype=np.float32)
                    sim = np.dot(db_embedding, face.embedding) / (np.linalg.norm(db_embedding) * np.linalg.norm(face.embedding))
                    
                    if sim > highest_sim:
                        highest_sim = sim
                        best_match_name = member['name']

                # 類似度が閾値 (例: 0.55以上) を超えた場合に本人と認定
                if highest_sim > 0.55:
                    detected_user_name = best_match_name
                    target_face_box = face.bbox.astype(int)
                    
                    fx = int((target_face_box[0] + target_face_box[2]) / 2)
                    fy = int(target_face_box[3])
                    
                    for m_key, m_info in MACHINE_AREAS.items():
                        bx1, by1, bx2, by2 = m_info["box"]
                        if bx1 <= fx <= bx2 and by1 <= fy <= by2:
                            active_machine_key = m_key
                            break

                    # マシン利用状況の更新
                    state.update_machine_occupancy(active_machine_key, detected_user_name, current_time)
                    
                    # 画面に名前と類似度を描画
                    x1, y1, x2, y2 = target_face_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detected_user_name} ({int(highest_sim*100)}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            state.push_update()
            cv2.putText(frame, "MODE: [ROOM (1:N Matching)]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Web配信用のフレームを更新
        with render_lock:
            current_output_frame = frame.copy()

        time.sleep(0.01)

    cap.release()

# =========================================================================
# ⌨️ ターミナルでの Enter キー監視スレッド（モード切替用）
# =========================================================================
def console_switcher_thread():
    global active_mode
    while True:
        try:
            input()
            with mode_lock:
                if active_mode == "entrance":
                    active_mode = "room"
                    print("\n🔄 【モード切替】 ➔ 【トレーニングルームモード】 に切り替えました（1:N 顔認識マシン監視）")
                else:
                    active_mode = "entrance"
                    print("\n🔄 【モード切替】 ➔ 【入口ゲートモード】 に切り替えました（QR & 人流）")
            state.push_update()
        except Exception:
            break

# =========================================================================
# 🚀 FastAPI サーバー & ストリーミング・登録API設定
# =========================================================================
connected_websockets = []

class RegisterRequest(BaseModel):
    name: str
    email: str
    image: str

@app.get("/register", response_class=HTMLResponse)
def get_register_page(request: Request):
    return templates.TemplateResponse(request, "register.html", {})

@app.post("/api/register")
def api_register(data: RegisterRequest):
    try:
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
            (member_id, data.name, data.email, qr_token, sqlite3.Binary(face_embedding))
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
        msg['To'] = data.email

        body = MIMEText(f"{data.name} 様\n\nジムの新規ご登録ありがとうございます。\nあなた専用の入館用QRコードを発行いたしました。\n添付のQRコードを受付のカメラにかざしてご入館ください。")
        msg.attach(body)

        img_attachment = MIMEImage(img_io.read(), name="gym_qr_code.png")
        msg.attach(img_attachment)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, data.email, msg.as_string())

        return {"status": "success", "member_id": member_id}

    except Exception as e:
        print(f"登録エラー: {e}")
        return {"status": "error", "message": str(e)}

def generate_stream():
    while True:
        with render_lock:
            if current_output_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_output_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

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
            # 非同期で少し待機し、CPU負荷を下げつつ次のフレームへ
            await asyncio.sleep(0.04)

    return StreamingResponse(
        generate_async_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close"  # 👈 ここがポイント：切断時にコネクションを強制終了させる
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

@app.get("/", response_class=HTMLResponse)
def get_dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

# =========================================================================
# 👥 管理者用：会員管理・一覧用エンドポイント
# =========================================================================
@app.get("/admin/members", response_class=HTMLResponse)
def get_admin_members_page(request: Request):
    return templates.TemplateResponse(request, "members.html", {})

@app.get("/api/admin/members")
def api_get_members():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # セキュリティや一覧性のために顔特徴量(face_embedding)以外の情報を取得
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

# =========================================================================
# 📋 管理者用：入退館・違反ログ一覧エンドポイント
# =========================================================================
@app.get("/admin/logs", response_class=HTMLResponse)
def get_admin_logs_page(request: Request):
    return templates.TemplateResponse(request, "logs.html", {})

@app.get("/api/admin/logs")
def api_get_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # データベースから最新の入退館ログを最大100件取得
    cursor.execute("SELECT id, timestamp, direction, member_id, is_alert FROM passing_logs ORDER BY id DESC LIMIT 100")
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"logs": logs}

# =========================================================================
# 🏁 起動確認
# =========================================================================
if __name__ == "__main__":
    camera_thread = threading.Thread(target=camera_processing_loop, daemon=True)
    camera_thread.start()
    
    switcher_thread = threading.Thread(target=console_switcher_thread, daemon=True)
    switcher_thread.start()
    
    print("\n🚀 サーバーを完全にアップグレードしました！")
    print("👉 ブラウザで http://localhost:8000/ を開いてダッシュボードを確認してください。")
    print("👉 会員登録ページ: http://localhost:8000/register")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")