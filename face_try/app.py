"""Canlı yüz tanıma web arayüzü.

Çalıştır:  python app.py   ->  http://127.0.0.1:8000
"""

import os
import threading
import time
import webbrowser

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from recognizer import FACES_DIR, ROOT, FaceEngine

CAMERA_INDEX = int(os.environ.get("CAMERA", 0))


class Live:
    def __init__(self):
        self.engine = FaceEngine()
        if not self.engine.load_gallery():
            self.engine.build_gallery()
        self.lock = threading.Lock()
        self.raw = None
        self.jpeg = None
        self.jpeg_seq = 0  # stream sadece yeni kare geldiğinde göndersin
        self.results = []
        self.fps = 0.0
        self.running = True
        self.status = "hazır"
        # Kamera okuma ayrı thread'de: işleme döngüsü kareyi beklemez, hep en güncel kareyi alır
        self.cam_cond = threading.Condition()
        self.cam_frame = None
        threading.Thread(target=self.capture, daemon=True).start()
        threading.Thread(target=self.loop, daemon=True).start()

    def capture(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 720p'de yüksek FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # eski kareler birikmesin
        while self.running:
            ok, frame = cap.read()
            if not ok:
                self.status = "kamera açılamadı"
                time.sleep(0.5)
                continue
            with self.cam_cond:
                self.cam_frame = frame
                self.cam_cond.notify()
        cap.release()

    def loop(self):
        last = time.time()
        while self.running:
            with self.cam_cond:
                while self.cam_frame is None and self.running:
                    self.cam_cond.wait(0.5)
                frame, self.cam_frame = self.cam_frame, None
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)  # ayna görüntü
            results = self.engine.process(frame)
            drawn = self.engine.draw(frame, results)
            ok, buf = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 80])
            now = time.time()
            with self.lock:
                self.raw, self.results = frame, results
                if ok:
                    self.jpeg = buf.tobytes()
                    self.jpeg_seq += 1
                self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now


live = Live()
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(ROOT, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/stream")
def stream():
    def gen():
        sent = -1
        while True:
            with live.lock:
                frame, seq = live.jpeg, live.jpeg_seq
            if frame and seq != sent:
                sent = seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.01)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def state():
    with live.lock:
        faces = [
            {
                "name": r["name"],
                "score": round(r["score"], 3),
                "stability": round(r["stability"], 2),
                "scores": {k: round(v, 3) for k, v in r["scores"].items()},
            }
            for r in live.results
        ]
        fps = live.fps
    eng = live.engine
    return {
        "fps": round(fps, 1),
        "status": live.status,
        "faces": faces,
        "people": {n: len(v) // 2 for n, v in eng.gallery.items()},
        "threshold": eng.threshold,
    }


@app.get("/api/thumb/{name}")
def thumb(name: str):
    img = live.engine.thumbs.get(name)
    if img is None:
        return Response(status_code=404)
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")


class Settings(BaseModel):
    threshold: float


@app.post("/api/settings")
def settings(s: Settings):
    live.engine.threshold = float(np.clip(s.threshold, 0.2, 0.9))
    return {"threshold": live.engine.threshold}


class Enroll(BaseModel):
    name: str


@app.post("/api/enroll")
def enroll(req: Enroll):
    """Kameradaki (tek) yüzü verilen kişinin klasörüne fotoğraf olarak ekler."""
    name = "".join(c for c in req.name.strip().lower() if c.isalnum() or c in "-_")
    if not name:
        return JSONResponse({"ok": False, "msg": "Geçersiz isim"}, status_code=400)
    with live.lock:
        frame = None if live.raw is None else live.raw.copy()
        n_faces = len(live.results)
    if frame is None or n_faces != 1:
        return JSONResponse({"ok": False, "msg": "Kadrajda tam olarak bir yüz olmalı"}, status_code=400)
    folder = os.path.join(FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"cam_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imencode(".jpg", frame)[1].tofile(path)
    return {"ok": True, "msg": f"{name} için fotoğraf kaydedildi. Galeriyi yenilemeyi unutma."}


@app.post("/api/rebuild")
def rebuild():
    live.status = "galeri yenileniyor"
    # Takip modeli canlı döngüde kullanılıyor; kayıt için ayrı motor açıp galeriyi aktar
    tmp = FaceEngine()
    people = tmp.build_gallery()
    live.engine.gallery, live.engine.thumbs = tmp.gallery, tmp.thumbs
    live.status = "hazır"
    return {"ok": True, "people": people}


@app.post("/api/quit")
def quit_app():
    """Arayüzde 'q' tuşu: kamerayı bırak ve sunucuyu kapat."""
    live.running = False

    def stop():
        time.sleep(0.5)  # kamera döngüsünün kapanması ve yanıtın gitmesi için
        os._exit(0)

    threading.Thread(target=stop, daemon=True).start()
    return {"ok": True}


if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
