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

# グローバル変数として追加
clients = []
last_detected_qr = "" # 直近のQRコードを保持する変数（もし名前が違ったら既存のものに合わせてください）

# =========================================================================
# ⚙️ システム設定値（調整可能）
# =========================================================================
DB_PATH = "gym_security.db"
DUPLICATE_QR_WINDOW = 5.0
CO_TRAILING_WINDOW = 5.0
LIMIT_SECONDS = 10.0
FACE_TIMEOUT = 5.0

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
        
        self.first_user_embedding = None
        self.first_user_name = "Guest"
        self.accumulated_time = 0.0
        self.last_check_time = None
        
        self.machine_states = {
            k: {"user": "-", "duration": 0.0, "status": "free", "start_time": None} 
            for k in MACHINE_AREAS.keys()
        }
        
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
            "user_name": self.first_user_name,
            "accumulated_time": int(self.accumulated_time),
            "is_overtime": self.accumulated_time > LIMIT_SECONDS if self.first_user_embedding is not None else False,
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
                        print(f"🔓 【QR認証成功】 会員ID: {qr_data}")
                        state.accumulated_time = 0.0
                        state.last_check_time = current_time

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
                            if time_since_qr <= CO_TRAILING_WINDOW and state.last_scanned_qr != "None":
                                state.register_pass("IN", state.last_scanned_qr, 0)
                                print(f"✅ [入館許可] 会員 {state.last_scanned_qr}")
                            else:
                                state.set_alert(True)
                                state.register_pass("IN", "Unknown", 1)
                                print("🚨 [共連れ検知]")
                        elif prev_x > LINE_X and x_center <= LINE_X:
                            state.register_pass("OUT", "Unknown", 0)
                            state.set_alert(False)

                    track_history[track_id] = x_center

            cv2.putText(frame, "MODE: [ENTRANCE]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"IN: {state.in_count} | OUT: {state.out_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # -----------------------------------------------------------------
        # モード B: トレーニングルーム処理 (顔認識 & マシンROI)
        # -----------------------------------------------------------------
        else:
            for m_key, m_info in MACHINE_AREAS.items():
                bx1, by1, bx2, by2 = m_info["box"]
                st = state.machine_states[m_key]
                color = (0, 0, 255) if st["status"] == "overtime" else (255, 165, 0)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, f"{m_info['name']} ({int(st['duration'])}s)", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            faces = face_app.get(frame)
            user_detected_this_frame = False
            target_face_box = None
            active_machine_key = None

            for face in faces:
                if state.first_user_embedding is None and state.last_scanned_qr != "None" and (current_time - state.last_qr_time) < 15.0:
                    state.first_user_embedding = face.embedding
                    state.first_user_name = state.last_scanned_qr
                    state.accumulated_time = 0.0
                    state.last_check_time = current_time
                    print(f"👤 【顔自動登録】 '{state.first_user_name}' を紐付けました。")

                if state.first_user_embedding is not None:
                    sim = np.dot(state.first_user_embedding, face.embedding) / (np.linalg.norm(state.first_user_embedding) * np.linalg.norm(face.embedding))
                    if sim > 0.6:
                        user_detected_this_frame = True
                        target_face_box = face.bbox.astype(int)
                        
                        fx = int((target_face_box[0] + target_face_box[2]) / 2)
                        fy = int(target_face_box[3])
                        
                        for m_key, m_info in MACHINE_AREAS.items():
                            bx1, by1, bx2, by2 = m_info["box"]
                            if bx1 <= fx <= bx2 and by1 <= fy <= by2:
                                active_machine_key = m_key
                                break
                        break

            if user_detected_this_frame:
                if state.last_check_time is not None:
                    state.accumulated_time += (current_time - state.last_check_time)
                state.last_check_time = current_time

                state.update_machine_occupancy(active_machine_key, state.first_user_name, current_time)
                state.push_update()

                if target_face_box is not None:
                    x1, y1, x2, y2 = target_face_box
                    color = (0, 0, 255) if state.accumulated_time > LIMIT_SECONDS else (0, 255, 0)
                    text = f"{state.first_user_name} ({int(state.accumulated_time)}s)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                if state.last_check_time is not None and (current_time - state.last_check_time) > FACE_TIMEOUT:
                    state.first_user_embedding = None
                    state.first_user_name = "Guest"
                    state.accumulated_time = 0.0
                    state.last_check_time = None
                    state.push_update()

            cv2.putText(frame, "MODE: [ROOM]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
            input()  # ターミナルで Enter キーが押されるのを待つ
            with mode_lock:
                if active_mode == "entrance":
                    active_mode = "room"
                    print("\n🔄 【モード切替】 ➔ 【トレーニングルームモード】 に切り替えました（マシン監視）")
                else:
                    active_mode = "entrance"
                    print("\n🔄 【モード切替】 ➔ 【入口ゲートモード】 に切り替えました（QR & 人流）")
            # 切り替え直後に状態を強制ブロードキャスト
            state.push_update()
        except Exception:
            break

# =========================================================================
# 🚀 FastAPI サーバー & ストリーミング設定
# =========================================================================
connected_websockets = []

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
def video_feed():
    return StreamingResponse(generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            # ブラウザからのメッセージを待つ（または単に接続維持）
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
# 🏁 起動処理
# =========================================================================
if __name__ == "__main__":
    camera_thread = threading.Thread(target=camera_processing_loop, daemon=True)
    camera_thread.start()
    
    switcher_thread = threading.Thread(target=console_switcher_thread, daemon=True)
    switcher_thread.start()
    
    print("\n🚀 サーバーを起動しました！")
    print("👉 ブラウザで http://localhost:8000/ を開いて映像を確認してください。")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")