import asyncio
import cv2
import json
import numpy as np
import os
import sqlite3
import threading
import time
from datetime import datetime
from queue import Queue
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi import Request

# テンプレートエンジンの準備
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# =========================================================================
# ⚙️ システム設定値（調整可能）
# =========================================================================
DB_PATH = "gym_security.db"
DUPLICATE_QR_WINDOW = 5.0  # 同じQRコードの重複読み取りを無視する時間（秒）
CO_TRAILING_WINDOW = 5.0   # QR認証後、この秒数以内にラインを通過しなければならない（秒）
LIMIT_SECONDS = 10.0       # 長時間占有と判定するデモ用制限時間（秒）
FACE_TIMEOUT = 5.0         # 画面から顔が消えてから離席と判定する時間（秒）

# カメラ画面の「中央の縦線」のX座標（横幅640pxの真ん中）
LINE_X = 320

# 📷 カメラデバイス設定
CAMERA_ENTRANCE_ID = 0  # 入口用カメラのインデックス
CAMERA_ROOM_ID = 1      # ルーム用カメラのインデックス

# カメラの台数状態（自動判定）
HAS_SECOND_CAMERA = False

# =========================================================================
# 💾 データベース自動初期化
# =========================================================================
def init_database():
    """SQLiteデータベースと必要な履歴テーブルを初期化する"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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

# =========================================================================
# 🧠 AIモデル・ツールの初期化
# =========================================================================
print(">> [1/3] YOLO（人流解析）モデルを読み込み中...")
yolo_model = None
qr_detector = cv2.QRCodeDetector()

# =========================================================================
# 🎥 複数カメラ対応のハブシステム
# =========================================================================
entrance_latest_frame = None
room_latest_frame = None
frame_lock = threading.Lock()

def camera_hub_thread():
    """カメラの接続状況に応じて映像を各処理へ分配するスレッド"""
    global entrance_latest_frame, room_latest_frame, HAS_SECOND_CAMERA
    
    cap_entrance = cv2.VideoCapture(CAMERA_ENTRANCE_ID)
    cap_room = cv2.VideoCapture(CAMERA_ROOM_ID)
    
    HAS_SECOND_CAMERA = cap_room.isOpened()
    
    if not HAS_SECOND_CAMERA:
        cap_room.release()
        print("📹 【カメラ1台モード】トレーニングルームのAIは完全にオフ。入口（QR・YOLO）のみで稼働します。")
    else:
        print("📹 【カメラ2台モード】入口用とルーム用のカメラを完全分離して稼働します。")

    while cap_entrance.isOpened():
        # 入口用カメラの読み込み
        success_ent, frame_ent = cap_entrance.read()
        if success_ent:
            frame_ent = cv2.resize(frame_ent, (640, 480))
            with frame_lock:
                entrance_latest_frame = frame_ent.copy()

        # ルーム用カメラの読み込み（2台ある場合のみ取得）
        if HAS_SECOND_CAMERA:
            success_room, frame_room = cap_room.read()
            if success_room:
                frame_room = cv2.resize(frame_room, (640, 480))
                with frame_lock:
                    room_latest_frame = frame_room.copy()
        else:
            # 1台のときはルーム用カメラの映像は取得せず、ダミーとして真っ黒な画面か、またはプレビュー用にメッセージを出す等の処理にできる
            # 今回はシンプルにルーム用最新フレームはNoneにしておきます
            with frame_lock:
                room_latest_frame = None

        time.sleep(0.03)

    cap_entrance.release()
    if HAS_SECOND_CAMERA:
        cap_room.release()

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
        
        self.first_user_embedding = None
        self.first_user_name = "Guest"
        self.accumulated_time = 0.0
        self.last_check_time = None
        
        self.update_queue = Queue()

    def process_qr(self, qr_data: str, current_time: float):
        with self._lock:
            self.last_scanned_qr = qr_data
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

    def push_update(self):
        data = {
            "in_count": self.in_count,
            "out_count": self.out_count,
            "last_qr": self.last_scanned_qr,
            "co_trailing_alert": self.co_trailing_alert,
            "user_name": self.first_user_name,
            "accumulated_time": int(self.accumulated_time),
            "is_overtime": self.accumulated_time > LIMIT_SECONDS if self.first_user_embedding is not None else False
        }
        self.update_queue.put(data)

state = SystemStateManager()

entrance_output_frame = None
room_output_frame = None
render_lock = threading.Lock()

# =========================================================================
# 🏃 スレッド1：入口ゲートのAI処理 (YOLO人数カウント ＆ QR認証)
# =========================================================================
def entrance_processing_loop():
    global entrance_output_frame, yolo_model
    from ultralytics import YOLO
    
    yolo_model = YOLO("yolo11n.pt")
    track_history = {}
    print("🏃 入口ゲート（YOLO + QR）処理スレッドが稼働しました。")

    while True:
        current_time = time.time()
        frame = None
        
        with frame_lock:
            if entrance_latest_frame is not None:
                frame = entrance_latest_frame.copy()
                
        if frame is None:
            time.sleep(0.03)
            continue

        # 1. QRコード検出処理
        qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame)
        if qr_bbox is not None and len(qr_bbox) > 0:
            pts = qr_bbox[0].astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            if qr_data:
                if qr_data != state.last_scanned_qr or (current_time - state.last_qr_time) > DUPLICATE_QR_WINDOW:
                    state.process_qr(qr_data, current_time)
                    print(f"🔓 【QR認証成功】 会員ID: {qr_data}")
                    state.accumulated_time = 0.0
                    state.last_check_time = current_time

        # 2. YOLOによる人流トラッキング
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
                cv2.circle(frame, (x_center, y_center), 4, (0, 0, 255), -1)

                if track_id in track_history:
                    prev_x = track_history[track_id]

                    if prev_x < LINE_X and x_center >= LINE_X:
                        time_since_qr = current_time - state.last_qr_time
                        if time_since_qr <= CO_TRAILING_WINDOW and state.last_scanned_qr != "None":
                            state.register_pass("IN", state.last_scanned_qr, 0)
                            print(f"✅ [入館許可] 会員 {state.last_scanned_qr} が入館しました。")
                        else:
                            state.set_alert(True)
                            state.register_pass("IN", "Unknown", 1)
                            print("🚨 [共連れ検知] 不正入館の疑いあり！")

                    elif prev_x > LINE_X and x_center <= LINE_X:
                        state.register_pass("OUT", "Unknown", 0)
                        state.set_alert(False)

                track_history[track_id] = x_center

        cv2.putText(frame, f"IN: {state.in_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {state.out_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Last QR: {state.last_scanned_qr}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if state.co_trailing_alert:
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 5)
            cv2.putText(frame, "CO-TRAILING ALERT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        with render_lock:
            entrance_output_frame = frame.copy()

        time.sleep(0.01)

# =========================================================================
# 👤 スレッド2：トレーニングルームのAI処理 (カメラ2台のときのみ稼働)
# =========================================================================
def room_processing_loop():
    """トレーニングエリアの顔識別ループ（カメラ1台のときは完全オフ）"""
    global room_output_frame
    
    # カメラが2台になるまで完全に待機
    while not HAS_SECOND_CAMERA:
        time.sleep(1.0)
        # 1台のときはルーム用画面に「ROOM CAMERA OFF」と表示した黒画面などを出す
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "ROOM CAMERA OFF (1-Camera Mode)", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        with render_lock:
            room_output_frame = dummy_frame

    # 2台目のカメラがある場合のみInsightFaceをロード
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("👤 トレーニングルーム（顔認識）処理スレッドが稼働しました。")

    while True:
        current_time = time.time()
        frame = None
        
        with frame_lock:
            if room_latest_frame is not None:
                frame = room_latest_frame.copy()
                
        if frame is None:
            time.sleep(0.03)
            continue

        faces = face_app.get(frame)
        user_detected_this_frame = False
        target_face_box = None

        for face in faces:
            if state.first_user_embedding is None and state.last_scanned_qr != "None" and (current_time - state.last_qr_time) < 10.0:
                state.first_user_embedding = face.embedding
                state.first_user_name = state.last_scanned_qr
                state.accumulated_time = 0.0
                state.last_check_time = current_time
                print(f"👤 【顔自動登録】 '{state.first_user_name}' を自動追跡対象に設定しました。")

            if state.first_user_embedding is not None:
                sim = np.dot(state.first_user_embedding, face.embedding) / (np.linalg.norm(state.first_user_embedding) * np.linalg.norm(face.embedding))
                if sim > 0.6:
                    user_detected_this_frame = True
                    target_face_box = face.bbox.astype(int)
                    break

        if user_detected_this_frame:
            if state.last_check_time is not None:
                delta_time = current_time - state.last_check_time
                state.accumulated_time += delta_time
            state.last_check_time = current_time

            if target_face_box is not None:
                x1, y1, x2, y2 = target_face_box
                if state.accumulated_time > LIMIT_SECONDS:
                    color = (0, 0, 255)
                    text = f"{state.first_user_name}: OVER TIME ({int(state.accumulated_time)}s)"
                else:
                    color = (0, 255, 0)
                    text = f"{state.first_user_name}: OK ({int(state.accumulated_time)}s)"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            if state.last_check_time is not None and (current_time - state.last_check_time) > FACE_TIMEOUT:
                if state.first_user_embedding is not None:
                    conn = sqlite3.connect(DB_PATH)
                    conn.cursor().execute(
                        "INSERT INTO occupancy_logs (member_id, start_time, end_time, duration) VALUES (?, ?, ?, ?)",
                        (state.first_user_name, 
                         datetime.fromtimestamp(current_time - state.accumulated_time).strftime("%Y-%m-%d %H:%M:%S"),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         state.accumulated_time)
                    )
                    conn.commit()
                    conn.close()
                    print(f"🚪 [占有終了] {state.first_user_name} が離席。ログを保存しました。")
                
                state.first_user_embedding = None
                state.first_user_name = "Guest"
                state.accumulated_time = 0.0
                state.last_check_time = None
                state.push_update()

        with render_lock:
            room_output_frame = frame.copy()

        time.sleep(0.01)

# =========================================================================
# 🚀 FastAPI サーバー & リアルタイム Web フロントエンド
# =========================================================================
connected_websockets = []

def generate_entrance_stream():
    while True:
        with render_lock:
            if entrance_output_frame is not None:
                ret, buffer = cv2.imencode('.jpg', entrance_output_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

def generate_room_stream():
    while True:
        with render_lock:
            if room_output_frame is not None:
                ret, buffer = cv2.imencode('.jpg', room_output_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.get("/video/entrance")
def video_entrance():
    return StreamingResponse(generate_entrance_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/room")
def video_room():
    return StreamingResponse(generate_room_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        initial_data = {
            "in_count": state.in_count,
            "out_count": state.out_count,
            "last_qr": state.last_scanned_qr,
            "co_trailing_alert": state.co_trailing_alert,
            "user_name": state.first_user_name,
            "accumulated_time": int(state.accumulated_time),
            "is_overtime": state.accumulated_time > LIMIT_SECONDS if state.first_user_embedding is not None else False
        }
        await websocket.send_text(json.dumps(initial_data))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
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
# 🏁 統合Webサーバー & AI処理スレッドの同時起動
# =========================================================================
if __name__ == "__main__":
    hub_thread = threading.Thread(target=camera_hub_thread, daemon=True)
    hub_thread.start()
    
    entrance_thread = threading.Thread(target=entrance_processing_loop, daemon=True)
    entrance_thread.start()
    
    room_thread = threading.Thread(target=room_processing_loop, daemon=True)
    room_thread.start()
    
    print("\n🚀 全システムが正常起動しました！")
    print("👉 ブラウザで http://localhost:8000/ を開き、管理画面を確認してください。")
    print("※ サーバーを終了するにはターミナルで Ctrl+C を押してください。")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")