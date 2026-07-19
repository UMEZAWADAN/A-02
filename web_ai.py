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

# =========================================================================
# 💾 データベース自動初期化
# =========================================================================
def init_database():
    """SQLiteデータベースと必要な履歴テーブルを初期化する"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 1. 通行履歴
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS passing_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            direction TEXT,
            member_id TEXT,
            is_alert INTEGER
        )
    """)
    # 2. 占有履歴
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
# 遅延を避けるためスレッド起動時に読み込む（ここではプレースホルダー宣言）
yolo_model = None

print(">> [2/3] InsightFace（顔識別）を初期化中...")
# 読み込み時間を考慮し、スレッド起動時に遅延初期化します

print(">> [3/3] OpenCV QRコード検出器を準備中...")
qr_detector = cv2.QRCodeDetector()

# =========================================================================
# 🎥 カメラキャプチャのハブシステム（カメラの奪い合いを防止する仕組み）
# =========================================================================
# PC1台でテストする場合、2つの処理が同時にカメラを要求するとエラーになります。
# そのため、カメラ映像を「1つのスレッド」で読み込み、他の処理に分配します。
latest_frame = None
frame_lock = threading.Lock()

def camera_hub_thread():
    """USBカメラから映像をキャプチャし続けるスレッド"""
    global latest_frame
    cap = cv2.VideoCapture(0)
    print("📹 カメラハブスレッドが起動しました。")
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # 640x480にリサイズして処理を安定化
            frame = cv2.resize(frame, (640, 480))
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.03)  # 約30FPS
    cap.release()

# =========================================================================
# 💾 システム状態管理（ステート）用クラス
# =========================================================================
class SystemStateManager:
    """システム全体のリアルタイムデータを安全に一元管理するクラス"""
    def __init__(self):
        self._lock = threading.Lock()
        
        # ゲート状態
        self.in_count = 0
        self.out_count = 0
        self.last_scanned_qr = "None"
        self.last_qr_time = 0.0
        self.co_trailing_alert = False
        
        # 滞在時間（部屋）状態
        self.first_user_embedding = None
        self.first_user_name = "Guest"
        self.accumulated_time = 0.0
        self.last_check_time = None
        
        # WebSocket送信用データキュー
        self.update_queue = Queue()

    def process_qr(self, qr_data: str, current_time: float):
        """QRコードの認証成功処理"""
        with self._lock:
            self.last_scanned_qr = qr_data
            self.last_qr_time = current_time
            self.co_trailing_alert = False  # 新しいQRで警告解除
            self.push_update()

    def set_alert(self, state: bool):
        """共連れ警告フラグの更新"""
        with self._lock:
            self.co_trailing_alert = state
            self.push_update()

    def register_pass(self, direction: str, member_id: str, is_alert: int):
        """通行履歴をDBに記録"""
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
        """最新ステータスをWebSocket配信キューにプッシュ"""
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

# グローバルな状態管理オブジェクト
state = SystemStateManager()

# フロントエンドに配信するための「映像バッファ」
entrance_output_frame = None
room_output_frame = None
render_lock = threading.Lock()

# =========================================================================
# 🏃 スレッド1：入口ゲートのAI処理 (YOLO人数カウント ＆ QR認証)
# =========================================================================
def entrance_processing_loop():
    """入口の共連れ検知とQRコードを処理する無限ループ"""
    global entrance_output_frame, yolo_model
    from ultralytics import YOLO
    
    yolo_model = YOLO("yolo11n.pt")
    track_history = {}
    print("🏃 入口ゲート（YOLO + QR）処理スレッドが稼働しました。")

    while True:
        current_time = time.time()
        frame = None
        
        # ハブから最新フレームを取得
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                
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

                    # 💡【顔自動登録連携用のフック】
                    # 入口でQRを通した瞬間に顔を自動登録するため、現在のフレームをそのまま顔識別側へ「予約」する
                    state.accumulated_time = 0.0
                    state.last_check_time = current_time

        # 2. YOLOによる人流トラッキング
        yolo_results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
        
        # ゲートラインの描画
        cv2.line(frame, (LINE_X, 0), (LINE_X, 480), (255, 0, 0), 2)
        cv2.putText(frame, "GATE LINE", (LINE_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)

                # バウンディングボックスの描画
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.circle(frame, (x_center, y_center), 4, (0, 0, 255), -1)

                if track_id in track_history:
                    prev_x = track_history[track_id]

                    # 💡【IN方向への横断】
                    if prev_x < LINE_X and x_center >= LINE_X:
                        # 突合判定
                        time_since_qr = current_time - state.last_qr_time
                        if time_since_qr <= CO_TRAILING_WINDOW and state.last_scanned_qr != "None":
                            # 正常通過
                            state.register_pass("IN", state.last_scanned_qr, 0)
                            print(f"✅ [入館許可] 会員 {state.last_scanned_qr} が入館しました。")
                        else:
                            # 共連れ検出
                            state.set_alert(True)
                            state.register_pass("IN", "Unknown", 1)
                            print("🚨 [共連れ検知] 不正入館の疑いあり！")

                    # 💡【OUT方向への横断】
                    elif prev_x > LINE_X and x_center <= LINE_X:
                        state.register_pass("OUT", "Unknown", 0)
                        state.set_alert(False)

                track_history[track_id] = x_center

        # 3. カウンター情報のUI描画
        cv2.putText(frame, f"IN: {state.in_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {state.out_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Last QR: {state.last_scanned_qr}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if state.co_trailing_alert:
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 5)
            cv2.putText(frame, "CO-TRAILING ALERT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 配信バッファに書き出し
        with render_lock:
            entrance_output_frame = frame.copy()

        time.sleep(0.01)

# =========================================================================
# 👤 スレッド2：トレーニングルームのAI処理 (InsightFace顔識別 ＆ 長時間占有タイマー)
# =========================================================================
def room_processing_loop():
    """トレーニングエリアの顔識別とタイムアウト判定ループ"""
    global room_output_frame
    from insightface.app import FaceAnalysis
    
    face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("👤 トレーニングルーム（顔認識）処理スレッドが稼働しました。")

    while True:
        current_time = time.time()
        frame = None
        
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                
        if frame is None:
            time.sleep(0.03)
            continue

        faces = face_app.get(frame)
        user_detected_this_frame = False
        target_face_box = None

        for face in faces:
            # 🔴 QRに連動した顔の「自動ロックオン」
            if state.first_user_embedding is None and state.last_scanned_qr != "None" and (current_time - state.last_qr_time) < 10.0:
                state.first_user_embedding = face.embedding
                state.first_user_name = state.last_scanned_qr
                state.accumulated_time = 0.0
                state.last_check_time = current_time
                print(f"👤 【顔自動登録】 '{state.first_user_name}' を自動追跡対象に設定しました。")

            # 照合
            if state.first_user_embedding is not None:
                sim = np.dot(state.first_user_embedding, face.embedding) / (np.linalg.norm(state.first_user_embedding) * np.linalg.norm(face.embedding))
                if sim > 0.6:
                    user_detected_this_frame = True
                    target_face_box = face.bbox.astype(int)
                    break

        # タイマー累積
        if user_detected_this_frame:
            if state.last_check_time is not None:
                delta_time = current_time - state.last_check_time
                state.accumulated_time += delta_time
            state.last_check_time = current_time

            # 枠の描画
            if target_face_box is not None:
                x1, y1, x2, y2 = target_face_box
                if state.accumulated_time > LIMIT_SECONDS:
                    color = (0, 0, 255)  # 赤
                    text = f"{state.first_user_name}: OVER TIME ({int(state.accumulated_time)}s)"
                else:
                    color = (0, 255, 0)  # 緑
                    text = f"{state.first_user_name}: OK ({int(state.accumulated_time)}s)"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # 離席判定（5秒以上消えたらセッションリセットしてDBに書き出し）
            if state.last_check_time is not None and (current_time - state.last_check_time) > FACE_TIMEOUT:
                if state.first_user_embedding is not None:
                    # DBへログ保存
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

        # 未登録「Guest」の描画
        if state.first_user_embedding is not None:
            for face in faces:
                sim = np.dot(state.first_user_embedding, face.embedding) / (np.linalg.norm(state.first_user_embedding) * np.linalg.norm(face.embedding))
                if sim <= 0.6:
                    box = face.bbox.astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (128, 128, 128), 1)
                    cv2.putText(frame, "Guest", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # 配信バッファに書き出し
        with render_lock:
            room_output_frame = frame.copy()

        time.sleep(0.01)

# =========================================================================
# 🚀 FastAPI サーバー & リアルタイム Web フロントエンド（一体型）
# =========================================================================
app = FastAPI()
connected_websockets = []

# カメラ映像ストリーミング用ジェネレータ
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
        # 接続時に初期状態を送信
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
    """データ更新時に接続されているすべてのブラウザへ一斉送信する"""
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
def get_dashboard():
    # メンバーにそのまま引き渡せる、Tailwind CSS を採用した本格的なダークテーマ管理画面
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>無人店舗 AI Security Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            @keyframes pulse-red {
                0%, 100% { background-color: rgba(239, 68, 68, 0.15); }
                50% { background-color: rgba(239, 68, 68, 0.4); }
            }
            .alert-active {
                animation: pulse-red 1.5s infinite;
                border-color: #ef4444 !important;
            }
        </style>
    </head>
    <body class="bg-gray-950 text-gray-100 min-h-screen font-sans">
        
        <!-- ヘッダー -->
        <header class="border-b border-gray-800 bg-gray-900 px-6 py-4 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <i class="fa-solid fa-shield-halved text-lime-400 text-3xl"></i>
                <h1 class="text-xl font-bold tracking-wider">AI無人店舗セキュリティシステム</h1>
            </div>
            <div class="flex items-center gap-2">
                <span class="inline-block w-3 h-3 bg-green-500 rounded-full animate-ping"></span>
                <span class="text-sm text-gray-400">システム常時監視中</span>
            </div>
        </header>

        <!-- メイングリッド -->
        <main class="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            <!-- 左・中：ライブモニター (2カラム分) -->
            <div class="lg:col-span-2 flex flex-col gap-6">
                
                <!-- 入口ゲート -->
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-4">
                    <div class="flex justify-between items-center mb-3">
                        <h2 class="text-lg font-semibold flex items-center gap-2">
                            <i class="fa-solid fa-door-open text-cyan-400"></i> エリア①：入口ゲート監視 (YOLO & QR)
                        </h2>
                        <span class="text-xs bg-cyan-950 text-cyan-400 px-2.5 py-1 rounded-full">カメラ01</span>
                    </div>
                    <div class="relative aspect-video rounded-xl overflow-hidden bg-black border border-gray-800">
                        <img src="/video/entrance" class="w-full h-full object-cover" alt="Entrance Camera Feed">
                    </div>
                </div>

                <!-- トレーニングルーム -->
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-4">
                    <div class="flex justify-between items-center mb-3">
                        <h2 class="text-lg font-semibold flex items-center gap-2">
                            <i class="fa-solid fa-dumbbell text-lime-400"></i> エリア②：マシン占有監視 (Face & Timer)
                        </h2>
                        <span class="text-xs bg-lime-950 text-lime-400 px-2.5 py-1 rounded-full">カメラ02</span>
                    </div>
                    <div class="relative aspect-video rounded-xl overflow-hidden bg-black border border-gray-800">
                        <img src="/video/room" class="w-full h-full object-cover" alt="Room Camera Feed">
                    </div>
                </div>

            </div>

            <!-- 右：ステータス＆コントロールパネル (1カラム分) -->
            <div class="flex flex-col gap-6">
                
                <!-- 共連れアラートカード -->
                <div id="alert_card" class="bg-gray-900 border border-gray-800 rounded-2xl p-6 flex flex-col items-center justify-center transition-all duration-300">
                    <i id="alert_icon" class="fa-solid fa-triangle-exclamation text-gray-600 text-5xl mb-4"></i>
                    <h3 id="alert_title" class="text-xl font-bold text-gray-400 mb-1">不正通行監視</h3>
                    <p id="alert_desc" class="text-sm text-gray-500 text-center">異常は検知されていません</p>
                </div>

                <!-- リアルタイムカウンター数 -->
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 grid grid-cols-2 gap-4">
                    <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 text-center">
                        <span class="text-sm text-gray-400 block mb-1">館内入場数 (IN)</span>
                        <strong id="val_in" class="text-4xl font-extrabold text-emerald-400">0</strong>
                    </div>
                    <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 text-center">
                        <span class="text-sm text-gray-400 block mb-1">退館数 (OUT)</span>
                        <strong id="val_out" class="text-4xl font-extrabold text-rose-500">0</strong>
                    </div>
                </div>

                <!-- 最後のQR認証ログ -->
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6">
                    <h4 class="text-sm text-gray-400 mb-3 flex items-center gap-2">
                        <i class="fa-solid fa-qrcode text-cyan-400"></i> 直近のQR読み取り
                    </h4>
                    <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 flex justify-between items-center">
                        <div>
                            <span class="text-xs text-gray-500 block">会員ID</span>
                            <strong id="val_qr" class="text-lg font-bold text-gray-200">None</strong>
                        </div>
                        <i class="fa-solid fa-circle-check text-emerald-500 text-xl"></i>
                    </div>
                </div>

                <!-- 占有エリア状態 -->
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6">
                    <h4 class="text-sm text-gray-400 mb-3 flex items-center gap-2">
                        <i class="fa-solid fa-user-clock text-lime-400"></i> マシン占有状況
                    </h4>
                    <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 flex flex-col gap-3">
                        <div class="flex justify-between items-center">
                            <span class="text-sm text-gray-300">利用中の会員:</span>
                            <strong id="room_user" class="text-gray-100 font-semibold">Guest</strong>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="text-sm text-gray-300">連続占有時間:</span>
                            <strong id="room_time" class="text-lg font-bold text-lime-400">0 秒</strong>
                        </div>
                        <div class="w-full bg-gray-800 rounded-full h-2 overflow-hidden mt-1">
                            <div id="time_progress" class="bg-lime-400 h-2 w-0 transition-all duration-300"></div>
                        </div>
                    </div>
                </div>

            </div>

        </main>

        <!-- リアルタイム通知WebSocketスクリプト -->
        <script>
            const ws = new WebSocket("ws://" + window.location.host + "/ws");
            
            // 警告用ビープ音
            function playBeep() {
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioCtx.createOscillator();
                oscillator.type = 'sawtooth';
                oscillator.frequency.setValueAtTime(440, audioCtx.currentTime); // A4
                oscillator.connect(audioCtx.destination);
                oscillator.start();
                oscillator.stop(audioCtx.currentTime + 0.3);
            }

            let prevAlertState = false;

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);

                // 各種ステータスの更新
                document.getElementById("val_in").innerText = data.in_count;
                document.getElementById("val_out").innerText = data.out_count;
                document.getElementById("val_qr").innerText = data.last_qr;
                document.getElementById("room_user").innerText = data.user_name;
                document.getElementById("room_time").innerText = data.accumulated_time + " 秒";

                // プログレスバーの更新 (制限10秒基準)
                const percent = Math.min(100, (data.accumulated_time / 10.0) * 100);
                const progressBar = document.getElementById("time_progress");
                progressBar.style.width = percent + "%";

                if (data.is_overtime) {
                    progressBar.className = "bg-red-500 h-2 transition-all duration-300";
                    document.getElementById("room_time").className = "text-lg font-bold text-red-500";
                } else {
                    progressBar.className = "bg-lime-400 h-2 transition-all duration-300";
                    document.getElementById("room_time").className = "text-lg font-bold text-lime-400";
                }

                // 🚨 共連れアラート処理
                const alertCard = document.getElementById("alert_card");
                const alertIcon = document.getElementById("alert_icon");
                const alertTitle = document.getElementById("alert_title");
                const alertDesc = document.getElementById("alert_desc");

                if (data.co_trailing_alert) {
                    alertCard.classList.add("alert-active");
                    alertIcon.className = "fa-solid fa-circle-exclamation text-red-500 text-5xl mb-4 animate-bounce";
                    alertTitle.className = "text-xl font-bold text-red-500 mb-1";
                    alertTitle.innerText = "🚨 共連れ不正入館！";
                    alertDesc.innerText = "QR未認証者の侵入を検知しました。確認してください。";
                    alertDesc.className = "text-sm text-red-300 text-center";
                    
                    if (!prevAlertState) {
                        playBeep();
                    }
                    prevAlertState = true;
                } else {
                    alertCard.classList.remove("alert-active");
                    alertIcon.className = "fa-solid fa-triangle-exclamation text-emerald-500 text-5xl mb-4";
                    alertTitle.className = "text-xl font-bold text-emerald-400 mb-1";
                    alertTitle.innerText = "監視状態：正常";
                    alertDesc.innerText = "異常は検知されていません";
                    alertDesc.className = "text-sm text-gray-500 text-center";
                    prevAlertState = false;
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# =========================================================================
# 🏁 統合Webサーバー & AI処理スレッドの同時起動
# =========================================================================
if __name__ == "__main__":
    # 1. 1台のカメラを裏側で回し続けるスレッド
    hub_thread = threading.Thread(target=camera_hub_thread, daemon=True)
    hub_thread.start()
    
    # 2. 入口ゲート(YOLO+QR)処理スレッド
    entrance_thread = threading.Thread(target=entrance_processing_loop, daemon=True)
    entrance_thread.start()
    
    # 3. トレーニングルーム(顔識別)処理スレッド
    room_thread = threading.Thread(target=room_processing_loop, daemon=True)
    room_thread.start()
    
    # 4. FastAPIサーバーをポート8000番で起動
    print("\n🚀 全システムが正常起動しました！")
    print("👉 ブラウザで http://localhost:8000/ を開き、管理画面を確認してください。")
    print("※ サーバーを終了するにはターミナルで Ctrl+C を押してください。")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")